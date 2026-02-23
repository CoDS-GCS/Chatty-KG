"""
Simple Chatty-KG HTTP server (single active chat) using Python's built-in http.server.

Design goals (per discussion):
- Single active chat (global state)
- No sessions dictionary
- Chat history is kept by Chatty-KG's kg_graph_state (and we pass a copy into AgentState each turn)
- Synchronous request handling, demo-friendly

Endpoints
---------
POST /start_chat
  Body JSON (optional fields have defaults):
  {
    "knowledge_graph": "dbpedia",
    "session_id": "0",
    "n_limit_VQuery": 600,
    "n_max_Vs": 1,
    "n_limit_EQuery": 25,
    "n_max_Es": 21,
    "n_max_answers": 41,
    "filtration_enabled": true
  }

POST /ask
  Body JSON:
  {
    "question": "Who directed Inception?",
    "system_mode": "Dialogue"   // optional, default "Dialogue"
  }

GET /reset
  Resets the active chat (clears kg_graph_state and counters)

Run
---
From your repo root so imports resolve, for example:
  python src/multi_agent/server/simple_chattykg_http_server.py --host 0.0.0.0 --port 8899

Notes
-----
- This server assumes only one request at a time (demo).
- If the process restarts, history is lost (by design).
"""

import argparse
import io
import json
import traceback
import re
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import Any, Dict, Optional
from chattykg.json_logger import JsonLogger

# Chatty-KG imports (must run from repo root, or ensure PYTHONPATH includes src/)
from multi_agent.shared.state import AgentState
from multi_agent.utils.graph_builder import build_langgraph
from multi_agent.modules.State import State



# ----------------------------
# Global single-chat state
# ----------------------------
GRAPH = None                      # built once at startup
KG_GRAPH_STATE: Optional[State] = None
ACTIVE_SESSION_ID: str = "0"
QUESTION_ID: int = 0

DEFAULT_KG = "dbpedia"
SYSTEM_MODE = "Dialogue"

LIMIT_VQUERY = 600
MAX_VS = 1
LIMIT_EQUERY = 25
MAX_ES = 21
MAX_ANSWERS = 41
FILTRATION_ENABLED = True

def _extract_executed_relation_uris(sparql_query):
    """
    Extract predicate URIs from executed SPARQL query string(s).
    Returns a set of predicate URIs.
    """
    if not sparql_query:
        return set()

    queries = sparql_query if isinstance(sparql_query, list) else [sparql_query]
    predicate_uris = set()

    pattern = re.compile(r'(?:<[^>]+>|\?[A-Za-z_]\w*)\s+<([^>]+)>\s+(?:<[^>]+>|\?[A-Za-z_]\w*)')

    for q in queries:
        if not isinstance(q, str):
            continue
        for pred in pattern.findall(q):
            predicate_uris.add(pred)

    return predicate_uris

def _extract_qir_intermediate(query_graph, sparql_query=None):
    """
    Return intermediate outputs:
    - qir: [{subject, predicate, object}]
    - entities: [{label, uris}]
    - relations: [{label, uris: [{relation, entity, direction, score}]}]

    relations are filtered to only predicates actually executed in SPARQL.
    """
    if query_graph is None:
        return [], [], []

    executed_relation_uris = _extract_executed_relation_uris(sparql_query)

    qir_triples = []

    # label -> set(uris)
    entities_map = {}
    # predicate label -> list of relation candidate objects
    relations_map = {}

    for u, v, attrs in query_graph.edges(data=True):
        attrs = attrs or {}

        u_attrs = query_graph.nodes[u] if u in query_graph.nodes else {}
        v_attrs = query_graph.nodes[v] if v in query_graph.nodes else {}

        subject = u_attrs.get("label", u)
        obj = v_attrs.get("label", v)
        predicate = attrs.get("relation", "")

        qir_triples.append({
            "subject": subject,
            "predicate": predicate,
            "object": obj
        })

        # Collect entity URIs (variables may have no URIs)
        for label, node_attrs in [(subject, u_attrs), (obj, v_attrs)]:
            if label not in entities_map:
                entities_map[label] = set()

            for uri in node_attrs.get("uris", []) or []:
                entities_map[label].add(uri)

        # Initialize relation bucket
        if predicate not in relations_map:
            relations_map[predicate] = []

        # Filter relation candidates to only executed predicates, while preserving metadata
        for item in attrs.get("uris", []) or []:
            # Expected format: [predicate_uri, entity_uri, direction_bool, score]
            if not isinstance(item, (list, tuple)) or len(item) < 1:
                continue

            predicate_uri = item[0]
            if executed_relation_uris and predicate_uri not in executed_relation_uris:
                continue

            relation_obj = {
                "relation": item[0] if len(item) > 0 else None,
                "entity": item[1] if len(item) > 1 else None,
                "direction": ("incoming" if item[2] else "outgoing") if len(item) > 2 else None,
                "score": item[3] if len(item) > 3 else None,
            }

            relations_map[predicate].append(relation_obj)

    entities = []
    for label in sorted(entities_map.keys()):
        entities.append({
            "label": label,
            "uris": sorted(entities_map[label])  # [] for variables / unmatched
        })

    relations = []
    for label in sorted(relations_map.keys()):
        # Optional: sort by score descending (None last)
        rel_items = relations_map[label]
        rel_items.sort(key=lambda x: (x["score"] is not None, x["score"] if x["score"] is not None else -1), reverse=True)

        relations.append({
            "label": label,
            "uris": rel_items
        })

    return qir_triples, entities, relations

def _json_response(handler: BaseHTTPRequestHandler, status: int, payload: Dict[str, Any]) -> None:
    data = json.dumps(payload).encode("utf-8")
    handler.send_response(status)
    handler.send_header("Content-Type", "application/json; charset=utf-8")
    handler.send_header("Access-Control-Allow-Origin", "*")
    handler.send_header("Content-Length", str(len(data)))
    handler.end_headers()
    handler.wfile.write(data)


def _read_json(handler: BaseHTTPRequestHandler) -> Dict[str, Any]:
    try:
        content_length = int(handler.headers.get("Content-Length", "0"))
        raw = handler.rfile.read(content_length) if content_length > 0 else b"{}"
        return json.load(io.BytesIO(raw))
    except Exception:
        raise ValueError("Failed to parse JSON body")


def _ensure_graph_initialized() -> None:
    global GRAPH
    if GRAPH is None:
        GRAPH = build_langgraph()


def _start_chat(kg_name: str):
    global KG_GRAPH_STATE, QUESTION_ID
    QUESTION_ID = 0
    KG_GRAPH_STATE = State(
        knowledge_graph=kg_name,
        n_limit_VQuery=LIMIT_VQUERY,
        n_max_Vs=MAX_VS,
        n_limit_EQuery=LIMIT_EQUERY,
        n_max_Es=MAX_ES,
        n_max_answers=MAX_ANSWERS,
        filtration_enabled=FILTRATION_ENABLED,
        json_logger=JsonLogger()
    )
    return {
        "status": "ok",
        "message": "chat_started",
        "knowledge_graph": kg_name,
        "session_id": ACTIVE_SESSION_ID,
        "question_id": QUESTION_ID,
    }


def _reset_chat() -> Dict[str, Any]:
    """Resets the active chat."""
    global KG_GRAPH_STATE, QUESTION_ID
    KG_GRAPH_STATE = None
    QUESTION_ID = 0
    return {"status": "ok", "message": "chat_reset"}


def _ask(payload):
    global KG_GRAPH_STATE, QUESTION_ID, ACTIVE_SESSION_ID

    question = (payload.get("question") or "").strip()
    if not question:
        raise ValueError("Missing required field: 'question'")

    if KG_GRAPH_STATE is None:
        kg_name = payload.get("knowledge_graph", DEFAULT_KG)
        _start_chat(kg_name)

    try:
        history = KG_GRAPH_STATE.get_chat_history(ACTIVE_SESSION_ID) or []
    except Exception:
        history = []

    state = AgentState(
        session_id=ACTIVE_SESSION_ID,
        question_id=QUESTION_ID,
        question=question,
        chat_history=history.copy(),
        system_mode=SYSTEM_MODE,
        kg_graph_state=KG_GRAPH_STATE,
        query_done=False,
        qir_done=False,
        has_been_resolved=False,
        route="chat_agent",
        sparql_query=None,
        query_result=None,
        resolved_question=None,
        ambiguity_resolver_done=False,
        query_graph=None,
        matching_done=False,
        ambiguity_resolver_tries=0
    )

    # --- NEW: capture SPARQL list length before this turn ---
    pre_sparql_len = 0
    try:
        existing_sparql = getattr(KG_GRAPH_STATE, "sparql_query", None)
        if isinstance(existing_sparql, list):
            pre_sparql_len = len(existing_sparql)
        elif isinstance(existing_sparql, str):
            pre_sparql_len = 1
    except Exception:
        pre_sparql_len = 0

    raw_state = GRAPH.invoke(state)
    final_state = AgentState(**dict(raw_state))

    # --- NEW: keep only current-turn SPARQL queries ---
    all_sparql_queries = getattr(final_state, "sparql_query", None)
    if isinstance(all_sparql_queries, list):
        current_turn_sparql = all_sparql_queries[pre_sparql_len:]
    elif isinstance(all_sparql_queries, str):
        # if a single string is returned, treat it as current turn
        current_turn_sparql = [all_sparql_queries]
    else:
        current_turn_sparql = []

    query_graph = getattr(final_state, "query_graph", None)

    # IMPORTANT: use current-turn queries here, not all historical queries
    qir_triples, entities, relations = _extract_qir_intermediate(query_graph, current_turn_sparql)

    kg_state = final_state.kg_graph_state
    executed_query_count = kg_state.get_num_executed_queries()
    answer_value = final_state.query_result
    answer_count = len(answer_value) if isinstance(answer_value, list) else (0 if answer_value in (None, "", {}) else 1)

    timing = {
        "understanding_time": kg_state.get_understanding_time(),
        "linking_time": kg_state.get_linking_time(),
        "query_selection_time": kg_state.get_query_selection_time(),
        "execution_time": kg_state.get_execution_time(),
    }

    resp = {
        "status": "ok",
        "question_id": QUESTION_ID,
        "answer": final_state.query_result,
        "sparql_query": current_turn_sparql,   # <-- changed
        "resolved_question": getattr(final_state, "resolved_question", None),
        "qir": qir_triples,
        "entities": entities,
        "relations": relations,                # now filtered against current-turn query only
        "answer_count": answer_count,
        "executed_query_count": executed_query_count,
        "timing": timing,
    }
    QUESTION_ID += 1
    return resp

class ChattyKGHandler(BaseHTTPRequestHandler):
    def do_OPTIONS(self):
        # CORS preflight support
        self.send_response(200)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()

    def do_GET(self):
        if self.path in ("/", "/info"):
            payload = {
                "status": "ok",
                "message": "Chatty-KG simple server",
                "active_session_id": ACTIVE_SESSION_ID,
                "question_id": QUESTION_ID,
                "chat_started": KG_GRAPH_STATE is not None,
            }
            _json_response(self, 200, payload)
            return

        if self.path == "/reset":
            payload = _reset_chat()
            _json_response(self, 200, payload)
            return

        _json_response(self, 404, {"status": "error", "error": f"Unknown path: {self.path}"})

    def do_POST(self):
        try:
            payload = _read_json(self)
        except ValueError as e:
            _json_response(self, 400, {"status": "error", "error": str(e)})
            traceback.print_exc()
            return

        try:
            if self.path == "/start_chat":
                kg = payload.get("knowledge_graph", DEFAULT_KG)
                resp = _start_chat(kg)
                _json_response(self, 200, resp)
                return

            if self.path == "/ask":
                resp = _ask(payload)
                _json_response(self, 200, resp)
                return

            _json_response(self, 404, {"status": "error", "error": f"Unknown path: {self.path}"})

        except Exception as e:
            tb = traceback.format_exc()
            traceback.print_exc()
            _json_response(self, 500, {"status": "error", "error": str(e), "traceback": tb})


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8899)
    args = parser.parse_args()

    _ensure_graph_initialized()

    server = HTTPServer((args.host, args.port), ChattyKGHandler)
    print(f"Server started http://{args.host}:{args.port}")
    print("Endpoints: POST /start_chat, POST /ask, GET /reset, GET /info")

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
        print("Server stopped.")


if __name__ == "__main__":
    main()
