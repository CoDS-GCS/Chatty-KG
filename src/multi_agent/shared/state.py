from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List
import networkx as nx
from multi_agent.modules.State import State

# -------------------------
# Agent State Definition
# -------------------------
@dataclass
class AgentState:
    question: str
    session_id: str = "12345"
    question_id: str = "1"
    chat_history: List[Dict[str, str]] = field(default_factory=list)
    resolved_question: Optional[str] = None
    sparql_query: Optional[str] = None
    query_result: Optional[str] = None
    kg_graph_state: Optional[State] = None
    ambiguity_resolver_done: bool = False
    qir_done: bool = False
    has_been_resolved: bool = False
    query_done: bool = False
    matching_done: bool = False
    route: Optional[str] = "chat_agent"
    query_graph: Optional[nx.MultiGraph] = field(default_factory=nx.MultiGraph)