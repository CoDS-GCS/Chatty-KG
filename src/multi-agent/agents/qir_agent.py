import json
from typing import List, Tuple, Dict
from langchain.schema import SystemMessage, HumanMessage
from shared.state import AgentState
import re

def qir_agent(state: AgentState) -> AgentState:
    print("\n🔍 QIR Agent:")
    question = state.question

    if  not state.ambiguity_resolver_done and re.search(r'\b(it|he|she|they|this|that|them)\b', question.lower()):
        print(" - Processing question:", question)
        print(" - Question is not self-contained. Routing to Ambiguity Resolver...")
        state.route = "ambiguity_resolver_agent"
    
    else:
        if state.ambiguity_resolver_done:
            print(f" - Converting question [{state.resolved_question}] to Query Intermediate Representation...")
        else:
            print(" - Question is self-contained.")
            state.resolved_question = question
        
        # TODO: This should be the QIR code
        state.qir = {
            "triples": [("?var", "is_president_of", "Egypt")]
        }

        state.qir_done = True
        state.route = "chat_agent"
        print(" - QIR complete. Tripples: ", state.qir["triples"], "Routing back to Chat Agent.")
    return state