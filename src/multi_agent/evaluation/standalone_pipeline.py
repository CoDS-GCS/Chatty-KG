import json
import os
import time
import sys
sys.path.append('../')
sys.path.append('../..')
from chattykg.chattykg import ChattyKG

from multi_agent.shared.state import AgentState
from multi_agent.utils.graph_builder import build_langgraph
from multi_agent.modules.State import State
import traceback

# max_Vs = 1
# max_Es = 21
# max_answers = 41
# limit_VQuery = 600
# limit_EQuery = 300

# os.environ["OPENAI_API_KEY"] = ""
# pair of end point, dataset file name
kg_related_variables = {"yago": ("http://206.12.95.86:8892/sparql", "../../chatbot/evaluation/data/yago_e11_20_5_original.json"),
                        "dblp": ("http://206.12.95.86:8894/sparql", "../../chatbot/evaluation/data/dblp_e11_20_5_original.json"),
                        "dbpedia": ("http://206.12.95.86:8890/sparql", "../../chatbot/evaluation/data/dbpedia_e11_20_5_original.json"),
                        "wikidata": ("https://query.wikidata.org/sparql", "../../chatbot/evaluation/data/wikidata_subgraph_summarized_20_5_simplified.json"),
                        }

if __name__ == '__main__':
    kg_name = 'yago'
    endpoint, dataset_file_name = kg_related_variables[kg_name]
    graph = build_langgraph()
    id = 0
    output = list()
    with open(dataset_file_name, 'r') as f:
        data = json.load(f)

    for obj in data["data"]:
        questions = obj["original"]
        # if id > 5:
        #     break
        for question in questions:
            kg_graph_state = State(
                knowledge_graph=kg_name,
                n_limit_VQuery=600,
                n_max_Vs=1,
                n_limit_EQuery=25,
                n_max_Es=21,
                n_max_answers=41,
                filtration_enabled=True
            )

            state = AgentState(
                session_id=id,
                question_id=id,
                question=question,
                chat_history=[],
                system_mode="Standalone",
                answer_mode="Not Formulated",
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
            try:
                raw_state = graph.invoke(state)
                final_state = AgentState(**dict(raw_state))

                result = final_state.evaluation_result

                output.append({
                    'id': id,
                    'question': question,
                    'answers': result
                })
            except Exception as e:
                print(f"[ERROR] while processing Q{id}: {question}")
                traceback.print_exc()
            id += 1

    dataset_id = dataset_file_name.split('/')[-1]
    dot_index = dataset_id.find('.')
    dataset_id = dataset_id[:dot_index]
    result = {"dataset": {"id": f'{dataset_id}_standalone'}, "questions": output}

    timestr = time.strftime("%Y%m%d-%H%M%S")
    output_file_name = f'output/{kg_name}_standalone_answers_{timestr}.json'
    with open(output_file_name, encoding='utf-8', mode='w') as rfobj:
        json.dump(result, rfobj, indent=4)
        rfobj.write('\n')



