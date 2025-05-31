import json
from typing import List, Dict, Tuple
from langchain.schema import SystemMessage, HumanMessage
from shared.state import AgentState


def query_agent(state: AgentState) -> AgentState:
    print("\n🧾 Query Agent:")
    print(" - Using QIR:", state.qir)
    
    # TODO: This should be the query agent code
    query = "SELECT ?var WHERE { ?var is_president_of Egypt }"

    state.sparql_query = query
    print(f" - SPARQL Query: {query}")

    # TODO: This should be the query result (run the query and get the result)
    result = "Egypt"

    print(f" - Query Result: {result}")


    state.query_result = result
    state.query_done = True
    state.route = "chat_agent"
    return state
