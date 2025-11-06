import re
import json
import csv
import pandas as pd


data_files = {
    "QALD9": "../../evaluation/qald9/qald-9-test-multilingual_1.json",
    "YAGO-S": "../../evaluation/yago/qald9_yago100.json",
    "DBLP-S": "../../evaluation/dblp/qald9_dblp100.json",
    "MAG-S": "../../evaluation/mag/qald9_ms100.json",
    "DBpedia-D": "../../chatbot/evaluation/data/dbpedia_e11_20_5_original.json",
    "YAGO-D": "../../chatbot/evaluation/data/yago_e11_20_5_original.json",
    "DBLP-D": "../../chatbot/evaluation/data/dblp_e11_20_5_original.json",
    "wikidata": "../../chatbot/evaluation/data/wikidata_subgraph_summarized_20_5_simplified.json"
}

def process_dataset(dataset_name, output_path):
    with open(data_files[dataset_name], 'r', encoding='utf-8') as f:
        data = json.load(f)

    results = []
    if dataset_name.lower() in ["qald9", "yago-s", "dblp-s", "mag-s"]:
        for q in data.get("questions", []):
            en_text = None
            for lang_q in q.get("question", []):
                if lang_q.get("language") == "en":
                    en_text = lang_q.get("string", "").strip()
                    break
            if en_text:
                results.append({"id": int(q.get("id")), "question": en_text, "sparql": q.get("query", "").get("sparql")})
    else:
        # Dialogue datasets
        qid = 0
        for obj in data["data"]:
            dataset_questions = obj["original"]
            dataset_queries = obj["queries"]
            for q, sp in zip(dataset_questions, dataset_queries):
                if q.strip():
                    results.append({"id": qid, "question": q.strip(), "sparql": sp.strip()})
                    qid += 1

    # Save to CSV
    df = pd.DataFrame(results)
    df = df.sort_values(by="id").reset_index(drop=True)  # ✅ sort by 'id'
    df.to_csv(output_path, index=False, quoting=csv.QUOTE_ALL)
    print(f"✅ Saved processed file to: {output_path}")

if __name__ == '__main__':
    for name, path in data_files.items():
        print(f"\n=== Processing {name} ===")
        process_dataset(name, f"{name}_question_triples.csv")

