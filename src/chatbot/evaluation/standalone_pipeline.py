import json
import os
import time
import sys
sys.path.append('../')
sys.path.append('../..')
from kgqan.kgqan import KGQAn

max_Vs = 1
max_Es = 21
max_answers = 41
limit_VQuery = 600
limit_EQuery = 300

os.environ["OPENAI_API_KEY"] = ""
# pair of end point, dataset file name
kg_related_variables = {"yago": ("http://206.12.95.86:8892/sparql", "data/yago_e11_20_5_original.json"),
                        "dblp": ("http://206.12.95.86:8894/sparql", "data/dblp_e11_20_5_original.json"),
                        "dbpedia": ("http://206.12.95.86:8890/sparql", "data/dbpedia_e11_20_5_original.json"),
                        }

if __name__ == '__main__':
    kg_name = 'dbpedia'
    endpoint, dataset_file_name = kg_related_variables[kg_name]
    id = 0
    kgqan = KGQAn(n_max_answers=max_answers, n_max_Vs=max_Vs, n_max_Es=max_Es, n_limit_VQuery=limit_VQuery, n_limit_EQuery=limit_EQuery)
    output = list()
    with open(dataset_file_name, 'r') as f:
        data = json.load(f)

    for obj in data["data"]:
        questions = obj["original"]
        for question in questions:
            answers, _, _, understanding_time, linking_time, execution_time, query_selection_time, num_queries_executed, is_boolean \
                = kgqan.ask(question_text=question, question_id=id, knowledge_graph=kg_name)

            if is_boolean:
                bool_value = False
                for answer in answers:
                    bool_value = answer['boolean'] or bool_value
                user_values = [bool_value]
                answer = [{'boolean': bool_value}]
            else:
                all_bindings = list()
                for answer in answers:
                    if answer['results'] and answer['results']['bindings']:
                        all_bindings.extend(answer['results']['bindings'])

                answer = [{'results': {'bindings': all_bindings}}]
            output.append({'id': id, 'question': question, 'answers': answer})
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



