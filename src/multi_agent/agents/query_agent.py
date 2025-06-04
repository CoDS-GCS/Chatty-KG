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
    query_selection_execution
)


def query_agent(state: AgentState) -> AgentState:
    print("\n🧾 Query Agent:")

    # step 1: Get the URIs of the entities and edges
    question = Question(
        question_text=state.question,
        question_id=state.question_id
    )
    state.kg_graph_state.set_question(question)
    state.kg_graph_state.set_query_graph(state.query_graph)

    # Pass to the matching agent
    if not state.matching_done:
        state.route = "matching_agent"
        print(" - Matching agent is not done. Routing to Matching Agent...")
        return state



    # step 2: Construct, Select, and Execute the SPARQL query
    print(f" - Matching agent is done. Constructing and Selecting the SPARQL query")
    query_selection_execution(state.kg_graph_state)
    query = state.kg_graph_state.get_sparql_query()
    state.sparql_query = query
    print(f" - SPARQL Query: {query}")

    # step 3: Fromat the result
    print(f" - Executing the SPARQL query")
    result = state.kg_graph_state.get_answer_values()
    print(f" - Number of possible answers: {len(state.kg_graph_state.get_answer_values())}")
    print(f" - Query Result: {result}")


    state.query_result = result
    state.query_done = True
    state.route = "chat_agent"
    return state
