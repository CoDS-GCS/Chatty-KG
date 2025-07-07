from langgraph.graph import StateGraph, END
from multi_agent.agents.chat_agent import chat_agent
from multi_agent.agents.qir_agent import qir_agent
from multi_agent.agents.ambiguity_resolver import ambiguity_resolver_agent
from multi_agent.agents.query_agent import query_agent
from multi_agent.agents.matching_agent import matching_agent
from multi_agent.shared.state import AgentState

# -------------------------
# Build LangGraph
# -------------------------
def build_langgraph():

    builder = StateGraph(state_schema=AgentState)

    builder.add_node("chat_agent", chat_agent)
    builder.add_node("qir_agent", qir_agent)
    builder.add_node("ambiguity_resolver_agent", ambiguity_resolver_agent)
    builder.add_node("query_agent", query_agent)
    builder.add_node("matching_agent", matching_agent)

    builder.set_entry_point("chat_agent")

    builder.add_conditional_edges(
        "chat_agent",
        lambda state: END if state.query_done else ("query_agent" if state.qir_done else "qir_agent")
    )
    builder.add_conditional_edges("qir_agent", lambda state: state.route)
    builder.add_conditional_edges("ambiguity_resolver_agent",lambda state: state.route)
    builder.add_conditional_edges("query_agent", lambda state: state.route)
    builder.add_conditional_edges("matching_agent", lambda state: state.route)

    builder.set_finish_point("chat_agent")

    graph = builder.compile()
    return graph