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

    # Lazy init: start chat on first ask
    if KG_GRAPH_STATE is None:
        kg_name = payload.get("knowledge_graph", DEFAULT_KG)
        _start_chat(kg_name)

    # history comes from kg_graph_state; pass a copy
    try:
        history = KG_GRAPH_STATE.get_chat_history(ACTIVE_SESSION_ID) or []
    except Exception:
        history = []

    state = AgentState(
        session_id=ACTIVE_SESSION_ID,
        question_id=QUESTION_ID,
        question=question,
        chat_history=history.copy(),
        system_mode=SYSTEM_MODE,          # fixed
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

    raw_state = GRAPH.invoke(state)
    final_state = AgentState(**dict(raw_state))

    resp = {
        "status": "ok",
        "question_id": QUESTION_ID,
        "answer": final_state.query_result,
        "sparql_query": getattr(final_state, "sparql_query", None),
        "resolved_question": getattr(final_state, "resolved_question", None),
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
