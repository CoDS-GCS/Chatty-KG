import json
from langchain.schema import SystemMessage, HumanMessage
from shared.state import AgentState
from agents.qir_agent import qir_agent
from agents.query_agent import query_agent
from langgraph.graph import END


def chat_agent(state: AgentState) -> AgentState:
    print("\n🧠 Chat Agent:")

    if state.query_done:
        print(" - Query done. Ending flow and returning to main loop.")
        state.route = END  # This will stop the graph
    elif state.qir_done:
        print(" - QIR complete. Routing to Query Agent...")
        state.route = "query_agent"
    else:
        print(" - Received question:", state.question)
        print(" - Routing to QIR Agent...")
        state.route = "qir_agent"
    return state
