import json
from langchain.schema import SystemMessage, HumanMessage
from multi_agent.shared.state import AgentState
from multi_agent.agents.qir_agent import qir_agent
from multi_agent.agents.query_agent import query_agent
from langgraph.graph import END
from chatbot.prompts import FINAL_ANSWER_REFORMALIZATION_PROMPT
from langchain_openai import ChatOpenAI

from ..modules.modules import (
    update_history
)

# Test Example:
# Dialogue Questions:
# - What is the birth place of Antony Cheng?, Answer: Vancouver, British Columbia, Canada
#  - What is his country of residence?, Answer: http://dbpedia.org/resource/Richmond_Hill,_Ontario, http://dbpedia.org/resource/Canada

def reformalize_final_answer(question: str, answers: list) -> str:
    print(f"reformalize_final_answer: \nquestion: {question}\nanswer:{answers}")
    llm = ChatOpenAI(
        model_name="gpt-3.5-turbo",
        temperature=0.3,  # low randomness for consistent reformalization
    )
    chain = FINAL_ANSWER_REFORMALIZATION_PROMPT | llm
    result = chain.invoke({"question": question, "answers": answers})
    # Extract only the text
    if hasattr(result, "content"):
        return result.content
    return str(result)


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
