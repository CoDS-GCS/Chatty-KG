import os
import json
import time

import requests
import traceback
import logging
import urllib

from multi_agent.shared.state import AgentState
from multi_agent.utils.graph_builder import build_langgraph
from multi_agent.modules.State import State

logger = logging.getLogger("chatbot_vs_baseline")
logger.setLevel(logging.INFO)
file_handler = logging.FileHandler("evaluation.log")
file_handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
logger.addHandler(file_handler)

kg_endpoint = "https://query.wikidata.org/sparql"
dataset_file = "../../chatbot/evaluation/data/wikidata_subgraph_summarized_20_5_simplified.json"
kg_name = 'wikidata'

def get_uri_label(kg_endpoint, uri):
    sparql_query = f"""
    SELECT ?label WHERE {{
        <{uri}> rdfs:label ?label .
        FILTER (lang(?label) = "en" || lang(?label) = "")
    }} LIMIT 1
    """

    response = evaluate_SPARQL_query_wikidata(kg_endpoint, sparql_query)
    # print(uri)
    # print(response)
    time.sleep(5)
    result_json = json.loads(response)

    bindings = result_json.get("results", {}).get("bindings", [])
    if bindings:
        return bindings[0]["label"]["value"]

    return None


def get_label(kg_endpoint, binding):
    """
    Extract label for a binding value based on its type.
    """
    if binding["type"] == "uri":
        # Option 1: Use pre-fetched label from SPARQL (if available)
        if "label" in binding:
            return binding["label"]["value"]
        # Option 2: Fallback to URI defragmentation
        # return defrag_uri_without_space(binding["value"])
        return get_uri_label(kg_endpoint, binding["value"])
    elif binding["type"] == "typed-literal":
        return binding["value"]  # Add datatype-specific formatting if needed
    elif binding["type"] == "literal":
        # Handle plain literals with optional language tags
        if "xml:lang" in binding:
            return f"{binding['value']} ({binding['xml:lang']})"
        return binding["value"]
    return ""

def process_sparql_results(kg_endpoint, json_data):
    """
    Process SPARQL query JSON results and extract labels.
    """
    results = []
    if "boolean" in json_data:
        results = ["yes", "true", True] if json_data.get("boolean") == True else ["no", "false", False]
        return results
    for result in json_data.get("results", {}).get("bindings", []):
        for var, binding in result.items():
            results.append(get_label(kg_endpoint, binding))
    return results


def evaluate_SPARQL_query_wikidata(endpoint_url, query: str):
    headers = {"Accept": "application/sparql-results+json"}
    query_response = requests.get(endpoint_url, params={"query": query}, headers=headers)
    if query_response.status_code in [414]:
        return '{"head":{"vars":[]}, "results":{"bindings": []}, "status":414 }'
    return query_response.text

def get_answers_wikidata(endpoint, queries):
    answers = []
    for query in queries:
        time.sleep(3)
        result = evaluate_SPARQL_query_wikidata(endpoint, query)
        print("===================")
        print(result)
        print("===================")
        if not result:
            result = "{}"
        result_json = json.loads(result)
        answers.append(result_json)
    return answers

def get_ground_truths(kg_name, kg_endpoint, dataset_file_name, outputfile):
    try:
        logger.info(f"Using Knowledge Graph: {kg_name} at endpoint: {kg_endpoint}")

        data = None
        # Load dataset
        with open(dataset_file_name, "r", encoding="utf-8") as dataset_file:
            data = json.load(dataset_file)

        results = []

        for idx, obj in enumerate(data.get("data", [])):
            queries = obj.get("queries")
            answers = get_answers_wikidata(kg_endpoint, queries)
            answers_labels = [process_sparql_results(kg_endpoint, a) for a in answers]
            results.append(
                {**obj, "ground_truths": answers, "ground_truths_labels": answers_labels}
            )

        parent_dir = os.path.dirname(outputfile)
        if not os.path.exists(parent_dir):
            os.makedirs(parent_dir, exist_ok=True)
        with open(outputfile, "w") as f:
            json.dump(results, f, indent=4)


    except Exception as e:
        traceback.print_exc()
        logger.error(f"An error occurred during the experiment: {e}", exc_info=True)

# ======================== Run Evaluation ======================
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

def process_gold_answers(golden_answer):
    result = set()
    if "results" in golden_answer and "bindings" in golden_answer["results"]:
        for bind in golden_answer["results"]["bindings"]:
            for key, value in bind.items():
                value = value['value']
                result.add(normalize(value))
    elif "boolean" in golden_answer:
        result.add(golden_answer["boolean"])

    return list(result)
def normalize(answer):
    # Check if the answer is a whole number (integer in string format)
    if isinstance(answer, bool):
        return answer
    if answer.isdigit():
        return answer + ".0"
    else:
        # Strip whitespace and decode URL-encoded characters
        return urllib.parse.unquote(answer.strip())

def process_answers_user(answers_user):
    result = set()
    print(answers_user)
    if not answers_user:
        return []
    for answer in answers_user:
        result.add(normalize(answer))
    # for answer in answers_user:
    #     for key, value in answer.items():
    #         result.add(normalize(value["value"]))

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

def run_chatbot_evaluation(experiment_name, kg_name, kg_endpoint, dataset_file_name, dialogue_mode=True):
    try:
        logger.info(f"Starting experiment: {experiment_name}")
        logger.info(f"Using Knowledge Graph: {kg_name} at endpoint: {kg_endpoint}")

        # Load dataset
        with open(dataset_file_name, "r", encoding="utf-8") as dataset_file:
            data = json.load(dataset_file)

        graph = build_langgraph()

        # Initialize chatbot
        # chatbot = KGChatbot(kg_name, kg_endpoint)
        output = []
        results = []
        dialogue_num = 0

        # Process each dialogue in the dataset
        for idx, obj in enumerate(data.get("data", [])):
            questions = obj.get("original")
            queries = obj.get("queries")
            answers = get_answers_wikidata(kg_endpoint, queries)
            if dialogue_mode:
                questions = obj.get("dialogue", [])
            chat_history = []
            kg_graph_state = State(
                knowledge_graph=kg_name,
                n_limit_VQuery=600,
                n_max_Vs=1,
                n_limit_EQuery=25,
                n_max_Es=21,
                n_max_answers=41,
                filtration_enabled=True
            )
            for question, golden_answer in zip(questions, answers):
                state = AgentState(
                    session_id=str(dialogue_num),
                    question_id=str(idx),
                    question=question,
                    chat_history=chat_history,
                    system_mode="Dialogue",
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
                raw_state = graph.invoke(state)
                final_state = AgentState(**dict(raw_state))

                answer = final_state.evaluation_result
                values = final_state.evaluation_user_values

                # output.append({
                #     'id': id,
                #     'question': question,
                #     'answers': answer
                # })
                chat_history = state.kg_graph_state.get_chat_history(str(dialogue_num))
                print(f"QUE- {question}")
                print(f"ANS:VAL - {answer}:{values}")
                result = compare_single_answer(values, golden_answer)
                results.append(result)
                question_output = {"id": idx, "question": question, "queries": queries, "golden_answer": golden_answer,
                                   "answer": answer, "result": result}
                output.append(question_output)
            dialogue_num += 1

        # Prepare result structure
        dataset_id = os.path.splitext(os.path.basename(dataset_file_name))[0]
        evaluation_metrics = compute_results(results)
        output = {"Output": output, "Metrics": evaluation_metrics}
        evaluation_results = {"dataset": {"id": f"{dataset_id}_{"dialogue" if dialogue_mode else "original"}"},
                              "data": output}

        # Save results to output file
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        output_file_name = f"output/dialogue3/{experiment_name}.json"
        parent_dir = os.path.dirname(output_file_name)
        if not os.path.exists(parent_dir):
            os.makedirs(parent_dir, exist_ok=True)
        with open(output_file_name, "w", encoding="utf-8") as output_file:
            json.dump(evaluation_results, output_file, indent=4)

        logger.info(f"Experiment completed successfully. Results saved to {output_file_name}")

    except Exception as e:
        traceback.print_exc()
        logger.error(f"An error occurred during the experiment: {e}", exc_info=True)

if __name__ == '__main__':
    # Obtaining the ground_truth
    # outputfile = "../../chatbot/logs/chatbot/chattykgv2_golden/wikidata_e11_20_5_original_3.json"
    # get_ground_truths(kg_name, kg_endpoint, dataset_file, outputfile)
    dialogue_mode = True
    experiment_name = f"exp-{kg_name}-{"dialogue" if dialogue_mode else "original"}-new-data"
    run_chatbot_evaluation(experiment_name, kg_name, kg_endpoint, dataset_file, dialogue_mode)