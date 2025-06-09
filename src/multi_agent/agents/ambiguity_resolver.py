import json
from typing import List, Dict, Tuple
from multi_agent.shared.state import AgentState
import re
from langchain.schema import SystemMessage, HumanMessage, AIMessage
import os
import sys

# Add the src directory to Python path
src_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.insert(0, src_dir)

from ..modules.modules import (
    rephrase_question
)



def ambiguity_resolver_agent(state: AgentState) -> AgentState:
    print("\n🧩 Ambiguity Resolver Agent:")
    

    if state.ambiguity_resolver_tries > 1:
        resolved = state.resolved_question
    else:
        resolved = state.question

    print(" - Resolving:", resolved)

    # First add some chat history
    # chat_history = state.kg_graph_state.get_by_session_id(state.session_id)
    # for message in state.chat_history:
    #     if message["role"] == "user":
    #         chat_history.add_messages([
    #             HumanMessage(content=message["content"])
    #         ])
    #     else:
    #         chat_history.add_messages([
    #             AIMessage(content=message["content"])
    #         ])
    
    # Now test rephrasing
    resolved = rephrase_question(state.session_id, resolved, state.kg_graph_state)
    

    if resolved == state.question:
        print(" - No change after resolution. Ending.")
    else:
        print(" - Resolved question:", resolved)
        
    state.resolved_question = resolved
    state.has_been_resolved = True
    state.ambiguity_resolver_done = True
    state.route = "qir_agent"
    print(" - Routing back to QIR Agent.")
    return state