import json
import time
import re

from langchain_core.prompts.prompt import PromptTemplate

from kgqan.llms.llm_creation import get_llm, llm_type
from kgqan.llms.prompts import question_understanding_template_v3_cot


def is_camel_case(s):
    return bool(re.match(r'^[a-z]+(?:[A-Z][a-z]*)*$', s))

def camel_to_normal(s):
    return re.sub(r'([a-z])([A-Z])', r'\1 \2', s).lower()


def validate_output(llm_output, question):
    # 1. JSON needs to be valid
    # 2. for non-boolean questions a variable must exist
    try:
        json_output = json.loads(llm_output)
    except:
        print(f"Error: Generated invalid json: {llm_output}")
        return False
    # print("JSON OUTPUT: ", json_output)

    question_word = question.split()[0]
    is_boolean = question_word.lower() in ['is', 'are', 'does', 'was', 'did', 'has', 'can']
    variable_exist = False
    for triple in json_output["triples"]:
        # print(f'The length of triple {triple} is: {len(triple)}')
        if len(triple) != 3:
            return False
        for item in triple:
            try:
                if item.startswith("?"):
                    variable_exist = True
                    break
            except:
                print(f"Error: Item {item} is not a string")
    if not is_boolean and not variable_exist:
        return False
    else:
        return True

def remove_spaces_if_variable(input):
    if input.startswith("?"):
        input = input.replace("-", "_")
        input = input.replace(" ", "_")
        input = input.replace("'", "")
    if '\"' in input:
        input = input.replace('\"', '')
    return input

def clean_predicate(predicate):
    if '_' in predicate:
        predicate = predicate.replace('_', ' ')
    elif 'was' in predicate:
        predicate = predicate.replace('was', '')

    if is_camel_case(predicate):
        predicate = camel_to_normal(predicate)
        print("Fixed Predicate: " + predicate)
    return predicate

def prepare_output_list(llm_output):
    triple_list = list()
    try:
       json_output = json.loads(llm_output)
       for triple in json_output["triples"]:
           triple_list.append({"subject": remove_spaces_if_variable(triple[0]), "predicate": clean_predicate(triple[1]),
                               "object": remove_spaces_if_variable(triple[2])})
    except:
        print(f"Error: Generated invalid json: {llm_output}")

    return triple_list

def extract_json_from_output(llm_output):
    llm_output = llm_output.replace('\n', '')
    start_index = llm_output.index("{")
    end_index = llm_output.rindex("}")
    return llm_output[start_index:end_index + 1]

def get_understanding(question, json_logger):
    retry = 0
    template = question_understanding_template_v3_cot

    prompt = PromptTemplate(
        input_variables=["question"],
        template=template,
    )
    llm = get_llm()
    final_prompt = prompt.format(question=question)
    # print(final_prompt)
    chain = prompt | llm
    while retry < 3:
        if retry > 0:
            print("Retrying...")
        output = chain.invoke({"question": question})
        if llm_type == 'google':
            time.sleep(30)
        metadata = output.usage_metadata
        # print(metadata)
        if llm_type != 'vllm':
            output = output.content
        try:
            output = extract_json_from_output(output)
            if validate_output(output, question):
                break
        except:
            print("============Start Parsing Error")
            print(output)
            print("============End")

        retry += 1
    if json_logger is not None:
        json_logger.set("Understanding", output)
        json_logger.add_cost("Understanding" ,metadata)
    print("Understanding OUTPUT======")
    print(output)

    return prepare_output_list(output)

if __name__ == '__main__':
    yago_questions = [ "Where was IBM founded?",
                  "Who created English Wikipedia?",
                  "Which software has been published by Epic Games?",
                  "Give me all movies with Tom Cruise.",
                  "What is the profession of Frank Herbert?",
                  "Is the wife of President Obama called Michelle?"
    ]
    qald_questions = ["What is the revenue of IBM?",
                     "Who founded Intel?",
                     "In which city is the headquarter of Air China?",
                     "Show me hiking trails in the Grand Canyon where there's no danger of flash floods.",
                     "Who became president after JFK died?",
                      "Is Pamela Anderson a vegan?"]

    dblp_questions = ["Name the author of the paper of title MIMO Systems with Intentional Timing Offset",
                      "How many people authored A study of a divided learning method?",
                      "What was awarded to Stefano Ceri?",
                      "State the page numbers of Tracking Complete Deformable Objects with Finite Elements?",
                      "DOE-SLAM: Dynamic Object Enhanced Visual SLAM is published in which journal?"]

    qald_questions = ["What is the time zone of Salt Lake City?", "How many moons does Mars have?"]
    for question in qald_questions:
        get_understanding(question, None)
        # time.sleep(30)
