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
    classify_question,
    get_qir_from_question,
    rephrase_question,
    perform_linking,
    query_selection_execution
)
from ..modules.State import State


#Todo: Orogat: How to make state a global variable between all files?
state_global = State(
            knowledge_graph="dbpedia",  # Using dbpedia as it's a valid knowledge graph
            n_limit_VQuery=600,
            n_max_Vs=1,
            n_limit_EQuery=25,
            n_max_Es=21,
            n_max_answers=41,
            filtration_enabled=True
        )


def ambiguity_resolver_agent(state: AgentState) -> AgentState:
    print("\n🧩 Ambiguity Resolver Agent:")
    print(" - Resolving:", state.question)

    resolved = state.question

    # First add some chat history
    chat_history = state_global.get_by_session_id(state.session_id)
    for message in state.chat_history:
        if message["role"] == "user":
            chat_history.add_messages([
                HumanMessage(content=message["content"])
            ])
        else:
            chat_history.add_messages([
                AIMessage(content=message["content"])
            ])
    
    # Now test rephrasing
    resolved = rephrase_question(state.session_id, resolved, state_global)
    

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