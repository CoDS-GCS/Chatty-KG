import os
import requests
import csv
import json
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

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

def get_uri_label(kg_endpoint, uri):
    sparql_query = f"""
    SELECT ?label WHERE {{
        <{uri}> rdfs:label ?label .
        FILTER (lang(?label) = "en" || lang(?label) = "")
    }} LIMIT 1
    """

    response = evaluate_SPARQL_query(kg_endpoint, sparql_query)
    result_json = json.loads(response)

    bindings = result_json.get("results", {}).get("bindings", [])
    if bindings:
        return bindings[0]["label"]["value"]
    
    return None

def get_label(kg_endpoint, binding, kg_prefix=None):
    """
    Extract label for a binding value based on its type.
    """
    if binding["type"] == "uri":
        if kg_prefix and kg_prefix not in binding["value"]:
            return binding["value"]
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
            # return f"{binding['value']} ({binding['xml:lang']})"
            return f"{binding['value']}"
        return binding["value"]
    return ""

def process_sparql_results(kg_endpoint, json_data, kg_prefix=None):
    """
    Process SPARQL query JSON results and extract labels.
    """
    results = []
    if "boolean" in json_data:
        results = ["yes", "true", True] if json_data.get("boolean") == True else ["no", "false", False]
        return results
    for result in json_data.get("results", {}).get("bindings", []):
        # processed_result = {}
        processed_result = []
        for var, binding in result.items():
            # processed_result[var] = get_label(binding)
        #     processed_result.append(get_label(binding))
        # results.append(processed_result)
            results.append(get_label(kg_endpoint, binding, kg_prefix))
    return results


os.environ["OPENAI_API_KEY"] = ""
SPARQL_ENDPOINT = {
    "yago": "http://206.12.95.86:8892/sparql",
    "dblp": "http://206.12.95.86:8894/sparql",
    "dbpedia": "http://206.12.95.86:8890/sparql"
}

# pair of end point, dataset file name, ground truth, chattykg results, convinse results, explaignn results, gpt_results
kg_related_variables = {
    "yago": (
        SPARQL_ENDPOINT.get("yago"),
        "evaluation/data/yago_e11_20_5_original.json",
        "logs/chatbot/chattykgv2_golden/yago_e11_20_5_original.json",
        "logs/chatbot_v4/exp-yago-dialogue-new-data_20250407-010516.json",
        "baseline/convinse_results_yago.json",
        "baseline/explaignn_results_yago.json",
        "evaluation/llms/gpt_yago_output.json",
        "evaluation/llms/gemini_yago_output.json",
        "evaluation/llms/deepseek_yago_output.json"
    ),
    "dblp": (
        SPARQL_ENDPOINT.get("dblp"),
        "evaluation/data/dblp_e11_20_5_original.json",
        "logs/chatbot/chattykgv2_golden/dblp_e11_20_5_original.json",
        "logs/chatbot_v4/exp-dblp-dialogue-new-data_20250407-010954.json",
        "baseline/convinse_results_dblp.json",
        "baseline/explaignn_results_dblp.json",
        "evaluation/llms/gpt_dblp_output.json",
        "evaluation/llms/gemini_dblp_output.json",
        "evaluation/llms/deepseek_dblp_output.json"
    ),
    "dbpedia": (
        SPARQL_ENDPOINT.get("dbpedia"),
        "evaluation/data/dbpedia_e11_20_5_original.json",
        "logs/chatbot/chattykgv2_golden/dbpedia_e11_20_5_original.json",
        "logs/chatbot_v4/exp-dbpedia-dialogue-new-data_20250407-010000.json",
        "baseline/convinse_results_dbpedia.json",
        "baseline/explaignn_results_dbpedia.json",
        "evaluation/llms/gpt_dbpedia_output.json",
        "evaluation/llms/gemini_dbpedia_output.json",
        "evaluation/llms/deepseek_dbpedia_output.json"
    ),
}

def precision_at_1(predictions, ground_truths, verbose=True):
    correct_count = 0

    for i, (pred, truth_list) in enumerate(zip(predictions, ground_truths)):
        pred_normalized = str(pred).strip().lower()
        truth_set = set(str(item).strip().lower() for item in truth_list)
        is_correct = pred_normalized in truth_set

        if verbose:
            status = "✓" if is_correct else "✗"
            print(f"[{status}] Example {i}: Prediction = {pred} | Ground Truth = {truth_list}")
        if is_correct:
            correct_count += 1

    return correct_count / len(predictions)

def hit_at_k(predictions, ground_truths, k=5, verbose=False):
    hit_count = 0

    for i, (pred_list, truth_list) in enumerate(zip(predictions, ground_truths)):
        pred_set = set(str(p).strip().lower() for p in pred_list[:k])
        truth_set = set(str(t).strip().lower() for t in truth_list)
        is_hit = bool(pred_set & truth_set)
        if is_hit:
            hit_count += 1

        if verbose:
            status = "✓" if is_hit else "✗"
            print(f"[{status}] Example {i}: Top-{k} Predictions = {pred_list[:k]} | Ground Truth = {truth_list}")

    return hit_count / len(predictions)

def mean_reciprocal_rank(rankings, verbose=False):
    reciprocal_ranks = []

    for i, rank in enumerate(rankings):
        rr = 1 / rank if rank > 0 else 0
        reciprocal_ranks.append(rr)
        if verbose:
            status = "✓" if rank > 0 else "✗"
            print(f"[{status}] Example {i}: Rank = {rank}, Reciprocal Rank = {rr:.4f}")

    return sum(reciprocal_ranks) / len(reciprocal_ranks)


def compute_results_v2(predictions_top_1, predictions_top_5, rankings, ground_truths):
    p_at_1 = precision_at_1(predictions_top_1, ground_truths)
    mrr = mean_reciprocal_rank(rankings)
    hit_at_5_score = hit_at_k(predictions_top_5, ground_truths, k=5)
    return p_at_1, mrr, hit_at_5_score

def save_results(results, results_data, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    
    # Save as JSON
    json_path = os.path.join(output_dir, "evaluation_results.json")
    with open(json_path, "w") as json_file:
        json.dump(results, json_file, indent=4)
    logging.info(f"Results saved to {json_path}")

    json_path = os.path.join(output_dir, "evaluation_data.json")
    with open(json_path, "w") as json_file:
        json.dump(results_data, json_file, indent=4)
    logging.info(f"Results saved to {json_path}")

    # Save as CSV
    csv_path = os.path.join(output_dir, "evaluation_results.csv")
    with open(csv_path, "w", newline="") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(["Model", "P@1", "MRR", "Hit@5"])
        for model, metrics in results.items():
            writer.writerow([model, metrics["p1"], metrics["mrr"], metrics["hit@5"]])
    logging.info(f"Results saved to {csv_path}")


def evaluate_v2(kg_name, kg_endpoint, ground_results, chattykg_results, convinse_results, explaignn_results, gpt_results, gemini_results, deepseek_results, output_dir):
    pred1, pred5, rankings, ground_truths = None, None, None, None

    results = {
        "p1": None,
        "mrr": None,
        "p5": None
    }

    ground_truth_data = None
    with open(ground_results, "r") as f:
        ground_truth_data = json.load(f)
    ground_truth_qs = []
    for qs in ground_truth_data:
        for q,gt in zip(qs.get("dialogue"), qs.get("ground_truths")):
            ground_truth_qs.append({
                "question": q,
                "answer": process_sparql_results(kg_endpoint, gt, kg_name)
            })

    # evaluate chattykg
    chattykg_data = None
    with open(chattykg_results, "r") as f:
        chattykg_data = json.load(f)
    
    chattykg_qs = []
    for qs in chattykg_data.get("data").get("Output"):
        qans = qs.get("answer")
        if not qs.get("answer"):
            qans = [{}]
        chattykg_qs.append({
            "question": qs.get("question"),
            "answer": process_sparql_results(kg_endpoint, qans[0], kg_name)
        })

    # evaluate convinse
    convinse_data = None
    with open(convinse_results, "r") as f:
        convinse_data = json.load(f)
    convinse_qs = []
    for qs in convinse_data.get("questions"):
        qans = qs.get("answers")
        if not qans:
            qans = [{}]
        convinse_qs.append({
            "question": qs.get("question")[0].get("string"),
            "answer": process_sparql_results(kg_endpoint, qans[0], kg_name)
        })
    
    # evaluate explaignn
    explaignn_data = None
    with open(explaignn_results, "r") as f:
        explaignn_data = json.load(f)
    explaignn_qs = []
    for qs in explaignn_data.get("questions"):
        qans = qs.get("answers")
        if not qans:
            qans = [{}]
        explaignn_qs.append({
            "question": qs.get("question")[0].get("string"),
            "answer": process_sparql_results(kg_endpoint, qans[0], kg_name)
        })

    gpt_data = None
    with open(gpt_results, "r") as f:
        gpt_data = json.load(f)

    gpt_qs = []
    for qs in gpt_data:
        qans = qs.get("answers")
        if not qs.get("answers"):
            qans = [{}]
        gpt_qs.append({
            "question": qs.get("question"),
            "answer": process_sparql_results(kg_endpoint, qans[0], kg_name)
        })

    gemini_data = None
    with open(gemini_results, "r") as f:
        gemini_data = json.load(f)

    gemini_qs = []
    for qs in gemini_data:
        qans = qs.get("answers")
        if not qs.get("answers"):
            qans = [{}]
        gemini_qs.append({
            "question": qs.get("question"),
            "answer": process_sparql_results(kg_endpoint, qans[0], kg_name)
        })

    deepseek_data = None
    with open(deepseek_results, "r") as f:
        deepseek_data = json.load(f)

    deepseek_qs = []
    for qs in deepseek_data:
        qans = qs.get("answers")
        if not qs.get("answers"):
            qans = [{}]
        deepseek_qs.append({
            "question": qs.get("question"),
            "answer": process_sparql_results(kg_endpoint, qans[0], kg_name)
        })


    # Prepare evaluation inputs
    def prepare_evaluation_data(model_qs):
        predictions_top_1 = [entry["answer"][0] if entry["answer"] else None for entry in model_qs]
        predictions_top_5 = [entry["answer"][:5] for entry in model_qs]
        rankings = [
            next(
                (
                    i + 1
                    for i, ans in enumerate(entry["answer"])
                    if str(ans).strip().lower() in {str(gt).strip().lower() for gt in ground_truth}
                ),
                0
            )
            for entry, ground_truth in zip(model_qs, [gt["answer"] for gt in ground_truth_qs])
        ]
        # print(rankings)
        # rankings = [
        #     next((i + 1 for i, ans in enumerate(entry["answer"]) if ans in ground_truth), 0)
        #     for entry, ground_truth in zip(model_qs, [gt["answer"] for gt in ground_truth_qs])
        # ]
        return predictions_top_1, predictions_top_5, rankings

    # Compute results
    results = {}
    results_data = {"data":{}}
    for model_name, model_qs in [("chattykg", chattykg_qs), ("convinse", convinse_qs), ("explaignn", explaignn_qs), ("gpt", gpt_qs), ("gemini", gemini_qs), ("deepseek", deepseek_qs)]:
        gts = [gt["answer"] for gt in ground_truth_qs]
        pred1, pred5, rankings = prepare_evaluation_data(model_qs)
        p1, mrr, hit5 = compute_results_v2(pred1, pred5, rankings, gts)
        results[model_name] = {"p1": p1, "mrr": mrr, "hit@5": hit5}
        results_data["data"][model_name] = model_qs
        results_data["data"]["ground_truth"] = gts
    
    output_dir += f"/{kg_name}"
    save_results(results, results_data, output_dir)
    logging.info("Evaluation complete.")

if __name__ == "__main__":

    kg_names = ["dbpedia", "dblp", "yago"]
    # v3 with gpt-3.5 turbo for rephraser, 4 with gpt-4o
    output_dir = "evaluation_v4"
    for kg_name in kg_names:
        kg_endpoint, dataset_file, ground_results, chattykg_results, convinse_results, explaignn_results, gpt_results, gemini_results, deepseek_results = kg_related_variables[kg_name]
        evaluate_v2(kg_name, kg_endpoint, ground_results, chattykg_results, convinse_results, explaignn_results, gpt_results, gemini_results, deepseek_results, output_dir)