import json
from typing import List, Dict, Tuple
from langchain.schema import SystemMessage, HumanMessage
import os
import sys
from multi_agent.shared.state import AgentState
from chattykg.question import Question
# Add the src directory to Python path
src_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.insert(0, src_dir)

from ..modules.modules import (
    perform_linking
)


def matching_agent(state: AgentState) -> AgentState:
    print("\n🔄 Matching Agent:")
    print(" - Using QIR:", state.query_graph)

    # step 1: Get the URIs of the entities and edges
    question = Question(
        question_text=state.question,
        question_id=state.question_id
    )
    state.kg_graph_state.set_question(question)
    state.kg_graph_state.set_query_graph(state.query_graph)
    perform_linking(state.kg_graph_state)
    print(f" --> Step 1: Getting the URIs of the entities and edges")
    query_graph = state.kg_graph_state.get_query_graph()
    print(" - [GRAPH NODES WITH URIs:]")
    for node in query_graph.nodes(data=True):
        print(f"\t\t{node}")

    print(f" - [GRAPH EDGES WITH URIs:]")
    for edge in query_graph.edges(data=True):
        print(f"\t\t{edge}")


    state.matching_done = True
    state.route = "query_agent"
    return state
