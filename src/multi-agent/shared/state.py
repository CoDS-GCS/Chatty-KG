from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List


# -------------------------
# Agent State Definition
# -------------------------
@dataclass
class AgentState:
    question: str
    chat_history: List[Dict[str, str]] = field(default_factory=list)
    resolved_question: Optional[str] = None
    sparql_query: Optional[str] = None
    query_result: Optional[str] = None
    ambiguity_resolver_done: bool = False
    qir_done: bool = False
    has_been_resolved: bool = False
    query_done: bool = False
    qir: Optional[Dict[str, Any]] = None
    route: Optional[str] = "chat_agent"