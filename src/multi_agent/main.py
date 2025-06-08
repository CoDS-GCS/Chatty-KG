from multi_agent.shared.state import AgentState
from multi_agent.utils.graph_builder import build_langgraph
import pprint
from multi_agent.modules.State import State






if __name__ == "__main__":
    pp = pprint.PrettyPrinter(indent=2)
    print("Welcome to Chatty-KG (LangGraph-powered). Type 'exit' to quit.")

    chat_history = []

    question_id = 1
    while True:
        user_input = input("\nYou: ").strip()
        if user_input.lower() == "exit":
            print("Goodbye!")
            break

        kg_graph_state = State(
            knowledge_graph="dbpedia",  # Using dbpedia as it's a valid knowledge graph
            n_limit_VQuery=600,
            n_max_Vs=1,
            n_limit_EQuery=25,
            n_max_Es=21,
            n_max_answers=41,
            filtration_enabled=True
        )

        # Create a clean state for this turn
        state = AgentState(
            session_id="12345",
            question_id=question_id,
            question=user_input,
            chat_history=chat_history.copy(),
            system_mode = "Dialogue", # Use 'Dialogue' to activate the Resolver Ambiguity Agent; any other value to deactivate it.
            kg_graph_state=kg_graph_state,
            query_done=False,
            qir_done=False,
            has_been_resolved=False,
            route="chat_agent",
            sparql_query=None,
            query_result=None,
            resolved_question=None,
            ambiguity_resolver_done=False,
            query_graph=None,
            matching_done=False,
            ambiguity_resolver_tries=0
        )

        question_id += 1

        # Run the graph and extract final state
        graph = build_langgraph()
        raw_state = graph.invoke(state)
        final_state = AgentState(**dict(raw_state))  # Flatten and de-duplicate

        # Log interaction
        chat_history.append({"role": "user", "content": user_input})
        chat_history.append({
            "role": "assistant",
            "content": f"Answer: {final_state.query_result}"
        })

        # Output final answer and state
        print("\n📦 Final Agent State:")
        pp.pprint(final_state)

        print("\n🎯 Finished. Ask another question or type 'exit'.")