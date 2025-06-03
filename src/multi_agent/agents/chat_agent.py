import json
from langchain.schema import SystemMessage, HumanMessage
from multi_agent.shared.state import AgentState
from multi_agent.agents.qir_agent import qir_agent
from multi_agent.agents.query_agent import query_agent
from langgraph.graph import END


# Test Example:
# Dialogue Questions:
# - What is the birth place of Antony Cheng?, Answer: Vancouver, British Columbia, Canada
#  - What is his country of residence?, Answer: http://dbpedia.org/resource/Richmond_Hill,_Ontario, http://dbpedia.org/resource/Canada

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
