import json
import os
import time
from chatbot.KGChatbot import KGChatbot


os.environ["OPENAI_API_KEY"] = ""
# pair of end point, dataset file name
kg_related_variables = {"yago": ("http://206.12.95.86:8892/sparql", "data/yago_e11_20_5_original.json"),
                        "dblp": ("http://206.12.95.86:8894/sparql", "data/dblp_e11_20_5_original.json"),
                        "dbpedia": ("http://206.12.95.86:8890/sparql", "data/dbpedia_e11_20_5_original.json"),
                        }



if __name__ == '__main__':
    kg_name = 'dblp'
    endpoint, dataset_file_name = kg_related_variables[kg_name]
    id = 0
    with open(dataset_file_name, 'r') as f:
        data = json.load(f)
        output = list()
    for obj in data["data"]:
        # questions = obj["original"]
        questions = obj["dialogue"]
        chatbot = KGChatbot(kg_name, '')
        for d_question in questions:
            answer, _ = chatbot.ask_question("1", d_question)
            # answer, _ = chatbot.ask_question("1", "Is Batman: The Dark Knight an ongoing series?")
            output.append({'id': id, 'question': d_question, 'answers': answer})
            id += 1

    dataset_id = dataset_file_name.split('/')[-1]
    dot_index = dataset_id.find('.')
    dataset_id = dataset_id[:dot_index]
    result = {"dataset": {"id": f'{dataset_id}_dialogue'}, "questions": output}
    # result = {"dataset": {"id": f'{dataset_id}_standalone'}, "questions": output}

    timestr = time.strftime("%Y%m%d-%H%M%S")
    output_file_name = f'output3/Boolean/{kg_name}_dialogue_answers_{timestr}.json'
    # output_file_name = f'output2/{kg_name}_standalone_answers_{timestr}.json'
    with open(output_file_name, encoding='utf-8', mode='w') as rfobj:
        json.dump(result, rfobj, indent=4)
        rfobj.write('\n')