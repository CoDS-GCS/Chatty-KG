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

def prepare_evaluation_result(answers, is_boolean):
    answers = [
        answer.json() for answer in answers
    ]
    if is_boolean:
        bool_value = False
        for answer in answers:
            bool_value = answer['boolean'] or bool_value
        output = [{'boolean': bool_value}]
    else:
        all_bindings = list()
        for answer in answers:
            if answer['results'] and answer['results']['bindings']:
                all_bindings.extend(answer['results']['bindings'])

        for binding in all_bindings:
            key = list(binding.keys())[0]
        output = [{'results': {'bindings': all_bindings}}]
    return output


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
    state.evaluation_result = prepare_evaluation_result(state.kg_graph_state.get_answers(), state.kg_graph_state.is_boolean_question())
    state.query_done = True
    state.route = "chat_agent"
    return state
