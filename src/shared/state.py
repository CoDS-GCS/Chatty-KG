from pydantic import BaseModel
from typing import List, Dict, Optional, Any

class AgentState(BaseModel):
    question: str
    chat_history: List[Dict[str, str]]
    query_done: bool = False
    qir_done: bool = False
    has_been_resolved: bool = False
    route: str = "chat_agent"
    sparql_query: Optional[str] = None
    query_result: Optional[Any] = None
    qir: Optional[Dict] = None
    resolved_question: Optional[str] = None
    ambiguity_resolver_done: bool = False 