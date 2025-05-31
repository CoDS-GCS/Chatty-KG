import json
from typing import List, Dict, Tuple
from shared.state import AgentState
import re

def ambiguity_resolver_agent(state: AgentState) -> AgentState:
    print("\n🧩 Ambiguity Resolver Agent:")
    print(" - Resolving:", state.question)

    replacements = {
        r"\bit\b": "Egypt",
        r"\bhe\b": "the president",
        r"\bshe\b": "the minister",
        r"\bthey\b": "the people",
        r"\bthem\b": "the group",
        r"\bthis\b": "the topic",
        r"\bthat\b": "the context"
    }

    resolved = state.question
    # TODO: This should be the ambiguity resolver code
    for pattern, replacement in replacements.items():
        resolved = re.sub(pattern, replacement, resolved, flags=re.IGNORECASE)
    
    state.ambiguity_resolver_done = True

    if resolved == state.question:
        print(" - No change after resolution. Ending.")
    else:
        print(" - Resolved question:", resolved)
        
    state.resolved_question = resolved
    state.has_been_resolved = True
    state.route = "qir_agent"
    print(" - Routing back to QIR Agent.")
    return state