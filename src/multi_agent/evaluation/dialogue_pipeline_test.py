import json
import os
import time
import traceback

from multi_agent.shared.state import AgentState
from multi_agent.utils.graph_builder import build_langgraph
from multi_agent.modules.State import State


kg_related_variables = {
    "yago": ("http://206.12.95.86:8892/sparql", "../../chatbot/evaluation/data/yago_e11_20_5_original.json"),
    "dblp": ("http://206.12.95.86:8894/sparql", "../../chatbot/evaluation/data/dblp_e11_20_5_original.json"),
    "dbpedia": ("http://206.12.95.86:8890/sparql", "../../chatbot/evaluation/data/dbpedia_e11_20_5_original.json"),
}

# How to add Kgs
if __name__ == '__main__':
    kg_name = 'dblp'
    endpoint, dataset_file_name = kg_related_variables[kg_name]
    graph = build_langgraph()
    id = 0


    with open(dataset_file_name, 'r') as f:
        data = json.load(f)
        output = []

    dialogue_num = 0
    for obj in data["data"]:
        questions = obj["dialogue"]
        chat_history = []
        # if dialogue_num == 1:
        #     break
        kg_graph_state = State(
            knowledge_graph=kg_name,
            n_limit_VQuery=600,
            n_max_Vs=1,
            n_limit_EQuery=25,
            n_max_Es=21,
            n_max_answers=41,
            filtration_enabled=True
        )
        for d_question in questions:
            # Create initial agent state
            state = AgentState(
                session_id=str(dialogue_num),
                question_id=id,
                question=d_question,
                chat_history=chat_history,
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
                matching_done=False
            )

            try:
                raw_state = graph.invoke(state)
                final_state = AgentState(**dict(raw_state))

                result = final_state.evaluation_result

                output.append({
                    'id': id,
                    'question': d_question,
                    'answers': result
                })
                chat_history = state.kg_graph_state.get_chat_history(str(dialogue_num))
                # print("=========================")
                # print(chat_history)
                # chat_history.append({"role": "user", "content": d_question})
                # chat_history.append({
                #     "role": "assistant",
                #     "content": f"Answer: {final_state.query_result}"
                # })

            except Exception as e:
                print(f"[ERROR] while processing Q{id}: {d_question}")
                traceback.print_exc()
                # output.append({
                #     'id': id,
                #     'question': d_question,
                #     'answers': [],
                #     'error': str(e)
                # })

            id += 1
        dialogue_num += 1

    # Save output
    dataset_id = os.path.basename(dataset_file_name).split('.')[0]
    result = {
        "dataset": {"id": f'{dataset_id}_dialogue'},
        "questions": output
    }

    timestr = time.strftime("%Y%m%d-%H%M%S")
    output_file_name = f'output/{kg_name}_dialogue_answers_{timestr}.json'
    os.makedirs("output", exist_ok=True)

    with open(output_file_name, encoding='utf-8', mode='w') as rfobj:
        json.dump(result, rfobj, indent=4)
        rfobj.write('\n')
