import json
import os
import time
import requests
from chatbot.KGChatbot import KGChatbot
import urllib.parse


os.environ["OPENAI_API_KEY"] = ""
# pair of end point, dataset file name
kg_related_variables = {"yago": ("http://206.12.95.86:8892/sparql", "data/yago_e11_20_5_original.json"),
                        "dblp": ("http://206.12.95.86:8894/sparql", "data/dblp_e11_20_5_original.json"),
                        "dbpedia": ("http://206.12.95.86:8890/sparql", "data/dbpedia_e11_20_5_original.json"),
                        }

def normalize(answer):
    # Check if the answer is a whole number (integer in string format)
    if answer.isdigit():
        return answer + ".0"
    else:
        # Strip whitespace and decode URL-encoded characters
        return urllib.parse.unquote(answer.strip())

def fscore(precision, recall):
    if precision == 0 and recall == 0:
        return 0
    return 2 * (precision * recall) / (precision + recall)

def compute_results(results):
    sum_precision = 0
    sum_recall = 0

    for comp in results:

        if comp['processed']:
            # OUT OF SCOPE questions
            if comp['total_gold'] == 0:
                if comp['total_user'] == 0:
                    precision = 1
                    recall = 1
                    f1 = 1
                else:
                    precision = 0
                    recall = 0
                    f1 = 0
            # all other questions
            else:
                if comp['total_user'] == 0:
                    precision = 1
                else:
                    precision = comp['correct'] / comp['total_user']
                recall = comp['correct'] / comp['total_gold']
                f1 = fscore(precision, recall)
        else:
            precision = 0
            recall = 0
            f1 = 0

        sum_precision += precision
        sum_recall += recall

    measures = {'processed': {}}

    # Measures on processed questions only
    number_of_processed_questions = 0
    for item in results:
        if item['processed']:
            number_of_processed_questions += 1

    macro_precision1 = 0 if number_of_processed_questions == 0 else sum_precision / number_of_processed_questions
    macro_recall1 = 0 if number_of_processed_questions == 0 else sum_recall / number_of_processed_questions

    measures['processed']['macro'] = {
        'precision': macro_precision1,
        'recall': macro_recall1,
        'f1': fscore(macro_precision1, macro_recall1)
    }

    return measures

def process_answers_user(answers_user):
    result = set()
    print(answers_user)
    for answer in answers_user:
        result.add(normalize(answer))
    # for answer in answers_user:
    #     for key, value in answer.items():
    #         result.add(normalize(value["value"]))

    return list(result)

def process_gold_answers(golden_answer):
    result = set()
    if "results" in golden_answer and "bindings" in golden_answer["results"]:
        for bind in golden_answer["results"]["bindings"]:
            for key, value in bind.items():
                value =  value['value']
                result.add(normalize(value))
    elif "boolean" in golden_answer:
        result.add(golden_answer["boolean"])

    return list(result)


def compare_single_answer(answers_user, answers_gold):
    results = {}
    answers_user = process_answers_user(answers_user)
    answers_gold = process_gold_answers(answers_gold)

    # Get the correct answers (gold) for the specific id
    results['total_gold'] = len(answers_gold)

    # Check if the user has provided answers for the specific id
    if len(answers_user) > 0:
        results['processed'] = True
        results['total_user'] = len(answers_user)
        results['correct'] = len([x for x in answers_user if x in answers_gold])
    else:
        results['processed'] = False

    return results

def evaluate_SPARQL_query(endpoint, query):
    payload = {
        "default-graph-uri": "",
        "query": query,
        "format": "application/json",
        "CXML_redir_for_subjs": "121",
        "CXML_redir_for_hrefs": "",
        "timeout": "40000",
        "debug": "on",
        "run": "+Run+Query+",
    }
    query_response = requests.get(endpoint, params=payload)
    if query_response.status_code in [414]:
        return '{"head":{"vars":[]}, "results":{"bindings": []}, "status":414 }'
    return query_response.text


def get_answers(endpoint, queries):
    answers = []
    for query in queries:
        result = evaluate_SPARQL_query(endpoint, query)
        result_json = json.loads(result)
        answers.append(result_json)
    return answers

if __name__ == '__main__':
    kg_name = 'dbpedia'
    endpoint, dataset_file_name = kg_related_variables[kg_name]
    # host = "http://206.12.95.86:8899"
    with open(dataset_file_name, 'r') as f:
        data = json.load(f)
    output = list()
    results = list()
    for obj in data["data"]:
        # dialogue = obj["dialogue"]
        dialogue = obj["original"]
        queries = obj["queries"]
        answers = get_answers(endpoint, queries)
        chatbot = KGChatbot(kg_name, '')

        for dialogue_question, golden_answer in zip(dialogue, answers):
            answer, values = chatbot.ask_question("1", dialogue_question)
            result = compare_single_answer(values, golden_answer)
            results.append(result)
            question_output = {"question": dialogue_question, "queries": queries, "golden_answer": golden_answer,
                               "answer": answer, "result": result}
            output.append(question_output)

    evaluation_metrics = compute_results(results)
    output = {"Output": output, "Metrics": evaluation_metrics}
    timestr = time.strftime("%Y%m%d-%H%M%S")
    output_file_name = f'output/{kg_name}_dialogue_answers_{timestr}.json'
    # output_file_name = f'output/{kg_name}_answers_{timestr}.json'
    with open(output_file_name, encoding='utf-8', mode='w') as rfobj:
        json.dump(output, rfobj, indent=4)
        rfobj.write('\n')