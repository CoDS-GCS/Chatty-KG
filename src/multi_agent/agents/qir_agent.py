import json
from typing import List, Tuple, Dict
from langchain.schema import SystemMessage, HumanMessage, AIMessage
from multi_agent.shared.state import AgentState
from langchain.schema import SystemMessage, HumanMessage
import re
import sys
import os

# Add the src directory to Python path
src_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.insert(0, src_dir)

from ..modules.modules import (
    classify_question,
    get_qir_from_question
)
from ..modules.State import State


def is_question_self_contained(question: str, kg_graph_state: State) -> bool:
    result = classify_question(question, kg_graph_state).lower().strip()
    return not "non-self-contained" in result.lower()
    # return result == "self-contained"

def qir_agent(state: AgentState) -> AgentState:
    print("\n🔍 QIR Agent:")
    # question = state.question
    question = state.resolved_question if state.resolved_question else state.question

    if state.system_mode.lower() == "dialogue":
        is_standalone = is_question_self_contained(question, state.kg_graph_state)
        if not is_standalone and state.ambiguity_resolver_tries<3:
            state.ambiguity_resolver_tries += 1
            print(
                f" - Question is not self-contained. Routing (try {state.ambiguity_resolver_tries}/3) to Ambiguity Resolver...")
            state.route = "ambiguity_resolver_agent"
            return state
        state.resolved_question = question
    else:
        print(" - Question is self-contained.")
        state.resolved_question = question
        # # TODO: This should be the QIR code
        # session_id = state.session_id
        # # First add some chat history
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
        #
        #
        # Test with a follow-up question
    question = state.resolved_question
    question_id = state.question_id
    result = get_qir_from_question(question, question_id, state.kg_graph_state)

    query_graph = state.kg_graph_state.get_query_graph()
    state.query_graph = query_graph
    # print(f" - Query graph: {state.query_graph}")
    #
    # print(" - [GRAPH NODES WITH URIs:]")
    # for node in query_graph.nodes(data=True):
    #     print(f"\t\t{node}")
    #
    # print(f" - [GRAPH EDGES WITH URIs:]")
    # for edge in query_graph.edges(data=True):
    #     print(f"\t\t{edge}")


    state.qir_done = True
    state.route = "chat_agent"
    print(" - QIR complete. Routing back to Chat Agent.")
    return state


