import json
import csv
from collections import Counter, OrderedDict

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


QUESTION_CLASSES = {
    "WH": [
        "Who", "Whom", "What", "Which", "Where", "When",
        "From where", "Under which", "Through which", "In which", "To which", "In what"
    ],
    "Count": ["How many"],
    "How": ["How"],
    "Boolean": [
        "Is", "Are", "Was", "Did", "Does", "Can", "Has",
    ],
    "Imperative": ["List", "Give", "Show", "Return", "Name", "State"],
    "Noun": [
        "Babak", "Sean", "Darrin", "Rolf", "Butch", "Goto",
        "Queens", "Bullet", "Doe-slam", "Microsoft", "Forma",
        "Ordinal-measure", "Tunneling", "Simple", "Predicting",
        "Active perception", "A Dynamic", "The Transmission",
        "The concept", "A New", "A Theory", "The entertainment",
        "On the", "Crowds"
    ]
}


def read_dataset(file_path: str, dataset_name: str):
    """Read a dataset file and extract English questions with their IDs."""
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    questions = []

    # QALD-style format
    if dataset_name.lower() in ["qald9", "yago-s", "dblp-s", "mag-s"]:
        for q in data.get("questions", []):
            en_text = None
            for lang_q in q.get("question", []):
                if lang_q.get("language") == "en":
                    en_text = lang_q.get("string", "").strip()
                    break
            if en_text:
                questions.append({"id": q.get("id"), "question": en_text})
    else:
        # Dialogue datasets
        qid = 0
        for obj in data["data"]:
            dataset_questions = obj["original"]
            for q in dataset_questions:
                if q.strip():
                    questions.append({"id": qid, "question": q.strip()})
                    qid += 1

    return questions


def classify_question(full_text):
    """
    Return the class name for a question based on the start of the text.
    The text starts with any entry in the class list (case-insensitive).
    """
    text_lower = full_text.lower()
    for cls, words in QUESTION_CLASSES.items():
        for w in words:
            if text_lower.startswith(w.lower()):
                return cls
    return None  # unmatched


import csv
from collections import Counter, OrderedDict

def get_question_class_stats(questions, dataset_name):
    """Compute class, first word, and num_words for each question in a dataset."""
    id_to_word = {}
    rows = []
    all_words = []
    unmatched_words = set()

    # concise and readable
    two_word_starters = {"how many", "the", "a", "in", "from", "to", "on", "under", "through"}

    for item in questions:
        qid = item.get("id")
        text = item.get("question", "").strip()
        if not text:
            continue

        lower_text = text.lower()

        # Determine first_word or first_two_words
        if any(lower_text.startswith(prefix) for prefix in two_word_starters):
            first_word = " ".join(text.split()[:2])
        else:
            first_word = text.split()[0].capitalize()

        # Determine class by prefix
        question_class = classify_question(text)

        # Count words in question
        num_words = len(text.split())

        # Collect unmatched starters
        if not question_class:
            unmatched_words.add(first_word)

        # Store results
        id_to_word[qid] = first_word
        all_words.append(first_word)
        rows.append({
            "id": qid,
            "class": question_class if question_class else "Unmatched",
            "word": first_word,
            "num_words": num_words
        })

    # Aggregate and sort frequencies
    word_counts = Counter(all_words)
    word_counts_sorted = OrderedDict(
        sorted(word_counts.items(), key=lambda x: x[1], reverse=True)
    )

    id_to_word_sorted = OrderedDict(
        sorted(id_to_word.items(), key=lambda x: int(x[0]) if str(x[0]).isdigit() else x[0])
    )

    # Sort rows by numeric ID before writing CSV
    rows_sorted = sorted(
        rows,
        key=lambda x: int(x["id"]) if str(x["id"]).isdigit() else x["id"]
    )

    csv_path = f"{dataset_name}_question_stats.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["id", "class", "word", "num_words"])
        writer.writeheader()
        writer.writerows(rows_sorted)

    print(f"✅ Saved {csv_path} with {len(rows_sorted)} entries.")
    if unmatched_words:
        print(f"⚠️  Unmatched starters in {dataset_name}: {sorted(unmatched_words)}")

    return word_counts_sorted, id_to_word_sorted, unmatched_words



if __name__ == "__main__":
    global_word_counter = Counter()
    global_unmatched_words = set()
    dataset_id_to_word = {}
    dataset_id_to_class = {}

    for name, path in data_files.items():
        print(f"\n=== Processing {name} ===")
        questions = read_dataset(path, name)
        word_counts, id_to_word, unmatched = get_question_class_stats(questions, name)

        # Update global counters
        global_word_counter.update(word_counts)
        global_unmatched_words.update(unmatched)

        # Store per-dataset results
        dataset_id_to_word[name] = id_to_word
        # dataset_id_to_class[name] = id_to_class

