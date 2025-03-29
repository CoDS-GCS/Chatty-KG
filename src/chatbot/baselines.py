import requests
import json
import urllib
import os
import time


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
    if not answers_user:
        return []
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

class APIClient:
    def __init__(self, base_url, client_name, headers):
        self.base_url = base_url
        self.client_name = client_name
        self.headers = headers

    def create_request_body(self, question, history_questions=None, history_answers=None, **kwargs):
        """
        Dynamically create the request body for the API.
        """
        body = {
            "question": question,
            "history_questions": history_questions or [],
            "history_answers": history_answers or []
        }
        body.update(kwargs)  # Add any additional parameters
        return body

    def send_post_request(self, endpoint, body):
        """
        Send a POST request to the specified endpoint with the provided body.
        """
        url = f"{self.base_url}/{endpoint}"
        response = requests.post(url, headers=self.headers, json=body)
        return response

    def process_response(self, response, question, history_questions, history_answers):
        """
        Process the API response and return the result.
        """
        if response.status_code == 200:
            response_data = response.json()
            print(f"Response for '{question}': {response_data}")
            return {
                "question": question,
                "answer": response_data.get("answer", {}),
                "history_questions": history_questions[:],
                "history_answers": history_answers[:],
                **response_data  # Include all other keys in the response
            }
        else:
            print(f"Error for '{question}': {response.status_code}, {response.text}")
            return None
    
    def post_processing(self, id_value, result):
        """
        post process the response to compare against golden answer
        """
        question = result.get("question", "")
        answer = result.get("answer", {})
        structured_representation = result.get("structured_representation", {})
        
        answer_binding = [{
            "variable": {
                "type": "literal",
                "value": answer.get('label', '')  # Use 'label' as the answer value
            }
        }]
        
        transformed_result = [
            {
                "head": {
                    "link": [],  # Assuming no links are provided
                    "vars": [
                        "variable"
                    ]
                },
                "results": {
                    "distinct": False,  # Default value (can be changed)
                    "ordered": True,    # Default value (can be changed)
                    "bindings": answer_binding  # The binding for the answer
                }
            }
        ]
        return transformed_result

    def evaluate(self, kg_name, input_file, kg_endpoint, dialogue_mode=False, output_file="results.json", **kwargs):
        """
        Evaluate the API with the provided data, including dialogue mode and questions.
        """

        data = None
        with open(input_file, "r") as f:
            data = json.load(f)
        
        if not data:
            raise ValueError(f"No data in input file : {input_file}")
        results = []
        output = []
        output2 = []
        output3 = []
        qid = 0
        for idx, obj in enumerate(data.get("data", [])):
            questions = obj.get("original")
            queries = obj.get("queries")
            answers = get_answers(kg_endpoint, queries)
            if dialogue_mode:
                questions = obj.get("dialogue", [])
            
            history_questions = []
            history_answers = []
            
            for question, golden_answer, query in zip(questions, answers, queries):
                # Create the request body
                body = self.create_request_body(
                    question, history_questions, history_answers, **kwargs
                )
                body["variant"] = "efficient"

                # Send the POST request
                response = self.send_post_request("answer-question", body)

                # Process the response
                result = self.process_response(response, question, history_questions, history_answers)
                result["api_answer"] = result.get("answer",None)
                result["answer"] = self.post_processing(qid,result)
                if result:
                    comparison_result = compare_single_answer(result["api_answer"]["label"], golden_answer)
                    results.append(comparison_result)

                    # Build the output object for this question
                    question_output = {
                        "id": qid,
                        "question": question,
                        "query": query,
                        "golden_answer": golden_answer,
                        "answer": result["answer"],
                        "api_answer": result["api_answer"],
                        "result": comparison_result
                    }
                    output.append(question_output)
                    question_output2 = {
                        "id": qid,
                        "question": [{"language":"en","string":question}],
                        "query": {"sparql": query},
                        "golden_answers": [golden_answer],
                        "answers": result["answer"]
                    }
                    output2.append(question_output2)

                    if self.client_name == "explaignn":
                        question_output3 = {
                            "id": qid,
                            "question": [{"language":"en","string":question}],
                            "query": {"sparql": query},
                            "golden_answers": [golden_answer],
                            "answers": [self.post_processing(qid,ans.get("answer")) for ans in result["ranked_answers"]]
                        }
                        output3.append(question_output3)

                    # Update history
                    history_questions.append(question)
                    history_answers.append(result.get("api_answer", {}).get("label", ""))

                qid += 1

        dataset_id = os.path.splitext(os.path.basename(input_file))[0]
        output2_data = {}
        output2_data = {"questions": output2}
        output2_data["dataset"] = {"id":f"{dataset_id}_{"dialogue" if dialogue_mode else "original"}"}
        # Save results to file
        parent_dir = os.path.dirname(output_file)
        if not os.path.exists(parent_dir):
            os.makedirs(parent_dir, exist_ok=True)
        with open(output_file, "w") as f:
            json.dump(output2_data, f, indent=4)

        if self.client_name == "explaignn":
            name, extension = output_file.rsplit(".", 1)
            output_file = f"{name}_extended.{extension}"
            output3_data = {}
            output3_data = {"questions": output3}
            output3_data["dataset"] = {"id":f"{dataset_id}_{"dialogue" if dialogue_mode else "original"}"}
            # Save results to file
            parent_dir = os.path.dirname(output_file)
            if not os.path.exists(parent_dir):
                os.makedirs(parent_dir, exist_ok=True)
            with open(output_file, "w") as f:
                json.dump(output3_data, f, indent=4)
        
        # Prepare result structure
        evaluation_metrics = compute_results(results)
        output = {"Output": output, "Metrics": evaluation_metrics}
        evaluation_results = {"dataset": {"id": f"{dataset_id}_{"dialogue" if dialogue_mode else "original"}"}, "data": output}

        # Save results to output file
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        output_file_name = f"logs/{self.client_name}/{kg_name}_{timestamp}.json"
        parent_dir = os.path.dirname(output_file_name)
        if not os.path.exists(parent_dir):
            os.makedirs(parent_dir, exist_ok=True)
        with open(output_file_name, "w", encoding="utf-8") as f:
            json.dump(evaluation_results, f, indent=4)


class ConvinseClient(APIClient):
    def __init__(self):
        base_url = "https://convinse.mpi-inf.mpg.de"
        client_name = "convinse"
        headers = {
            "accept": "application/json, text/javascript, */*; q=0.01",
            "accept-language": "en-US,en;q=0.8",
            "content-type": "application/json;charset=UTF-8",
            "sec-ch-ua": "\"Brave\";v=\"131\", \"Chromium\";v=\"131\", \"Not_A Brand\";v=\"24\"",
            "sec-ch-ua-mobile": "?0",
            "sec-ch-ua-platform": "\"Linux\"",
            "sec-fetch-dest": "empty",
            "sec-fetch-mode": "cors",
            "sec-fetch-site": "same-origin",
            "sec-gpc": "1",
            "x-requested-with": "XMLHttpRequest",
            "Referer": "https://convinse.mpi-inf.mpg.de/",
            "Referrer-Policy": "strict-origin-when-cross-origin"
        }
        super().__init__(base_url, client_name, headers)


class ExplaignnClient(APIClient):
    def __init__(self):
        base_url = "https://explaignn.mpi-inf.mpg.de"
        client_name = "explaignn"
        headers = {
            "accept": "application/json, text/javascript, */*; q=0.01",
            "accept-language": "en-US,en;q=0.8",
            "content-type": "application/json;charset=UTF-8",
            "sec-ch-ua": "\"Brave\";v=\"131\", \"Chromium\";v=\"131\", \"Not_A Brand\";v=\"24\"",
            "sec-ch-ua-mobile": "?0",
            "sec-ch-ua-platform": "\"Linux\"",
            "sec-fetch-dest": "empty",
            "sec-fetch-mode": "cors",
            "sec-fetch-site": "same-origin",
            "sec-gpc": "1",
            "x-requested-with": "XMLHttpRequest",
            "Referer": "https://explaignn.mpi-inf.mpg.de/",
            "Referrer-Policy": "strict-origin-when-cross-origin"
        }
        super().__init__(base_url, client_name, headers)



def extract_dialogues(input_file):
    """
    Extract dialogues from a JSON file.
    """
    try:
        with open(input_file, "r") as infile:
            data = json.load(infile)
        return [item["dialogue"] for item in data.get("data", [])]
    except Exception as e:
        print(f"An error occurred: {e}")
        return []

# pair of end point, dataset file name
kg_related_variables = {
    "yago": (
        "http://206.12.95.86:8892/sparql",
        "evaluation/data/yago_e11_20_5_original.json",
    ),
    "dblp": (
        "http://206.12.95.86:8894/sparql",
        "evaluation/data/dblp_e11_20_5_original.json",
    ),
    "dbpedia": (
        "http://206.12.95.86:8890/sparql",
        "evaluation/data/dbpedia_e11_20_5_original.json",
    ),
}

if __name__ == "__main__":
    kg_names = ["dblp", "yago", "dbpedia"]
    dialogue_mode = True

    for kg_name in kg_names:
        kg_endpoint, input_file = kg_related_variables[kg_name]
        # Convinse Client
        if not kg_name == "dblp":
            convinse_client = ConvinseClient()
            convinse_client.evaluate(kg_name, input_file, kg_endpoint, dialogue_mode, output_file=f"baseline/convinse_results_{kg_name}.json")

        # Explaignn Client
        explaignn_client = ExplaignnClient()
        explaignn_client.evaluate(kg_name, input_file, kg_endpoint, dialogue_mode, output_file=f"baseline/explaignn_results_{kg_name}.json")