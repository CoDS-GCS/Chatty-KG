import json
from langchain.schema import SystemMessage, HumanMessage
from multi_agent.shared.state import AgentState
from multi_agent.agents.qir_agent import qir_agent
from multi_agent.agents.query_agent import query_agent
from langgraph.graph import END

from ..modules.modules import (
    update_history, reformalize_final_answer
)



def chat_agent(state: AgentState) -> AgentState:
    print("\n🧠 Chat Agent:")

    if state.query_done:
        print(" - Query done. Ending flow and returning to main loop.")
        
        #Final answer
        # Reformalize the final answer using LLM
        if state.answer_mode == "Formulated":
            final_answer = reformalize_final_answer(state.resolved_question, state.kg_graph_state.get_answer_values())
            state.final_answer = final_answer  # store in state for main loop or UI


        update_history(state.session_id, state.question, state.resolved_question, state.kg_graph_state)
        state.route = END  # This will stop the graph
    elif state.qir_done:
        print(" - QIR complete. Routing to Query Agent...")
        state.route = "query_agent"
    else:
        print(" - Received question:", state.question)
        print(" - Routing to QIR Agent...")
        state.route = "qir_agent"
    return state
