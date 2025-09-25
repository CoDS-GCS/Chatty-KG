import json
import time
import traceback
import torch
import re
import ast

from langchain_core.prompts.prompt import PromptTemplate

from chattykg.llms.llm_creation import get_llm, llm_type
from chattykg.llms.prompts import query_selection_template_v4, query_selection_template_v5, query_selection_template_v6

torch.set_default_tensor_type(torch.DoubleTensor)

def is_camel_case(s):
    return bool(re.match(r'^[a-z]+(?:[A-Z][a-z]*)*$', s))

def camel_to_normal(s):
    return re.sub(r'([a-z])([A-Z])', r'\1 \2', s).lower()

def extract_json_from_output(llm_output):
    if '### Instruction' in llm_output:
        index = llm_output.index('### Instruction')
        llm_output = llm_output[:index]
    elif '###' in llm_output:
        index = llm_output.index('###')
        llm_output = llm_output[:index]
    llm_output = llm_output.replace('\n', '')
    start_index = llm_output.index("{")
    end_index = llm_output.rindex("}")
    llm_output = llm_output[start_index:end_index+1]
    python_dict = ast.literal_eval(llm_output)
    llm_output = json.dumps(python_dict)
    return llm_output


def postprocess_result(text):
    if "Keywords: " in text:
        text = text.replace("Keywords: ", "")
    elif "-" in text and '\n' in text:
        start = text.index("-")
        text = text[start:]
        text = text.replace("- ", "")
        text = text.replace("\n", ",")
    return text


def get_name(predicate):
    pattern = r'[#/]([^/#]+)$'
    match = re.search(pattern, predicate)
    if match:
        name = match.group(1)
        # p2 = re.compile(r"([a-z0-9])([A-Z])")
        # name = p2.sub(r"\1 \2", name)
        if is_camel_case(name):
            name = camel_to_normal(name)
        return name
    else:
        return ""


def prepare_keywords_list(triples_list):
    predicate_to_query_id = {}
    output = ""
    for i, triples in enumerate(triples_list):
        for triple in triples:
            name = get_name(triple[1][0])
            if name != "" and not len(name) == 1:
                if name in predicate_to_query_id:
                    predicate_to_query_id[name].append(i)
                else:
                    predicate_to_query_id[name] = [i]
                    output += name + ', '
    return output[: len(output) - 2], predicate_to_query_id

def prepare_keywords_list_wikidata(triples_list, predicate_map):
    predicate_to_query_id = {}
    output = ""
    for i, triples in enumerate(triples_list):
        for triple in triples:
            # name = get_name(triple[1][0])
            name = predicate_map[triple[1][0]]
            if name != "" and not len(name) == 1:
                if name in predicate_to_query_id:
                    predicate_to_query_id[name].append(i)
                else:
                    predicate_to_query_id[name] = [i]
                    output += name + ', '
    return output[: len(output) - 2], predicate_to_query_id


def choose_question_from_keywords(question, query_list, triples_list, json_logger, predicate_map, kg_name):
    return_result = list()

    template = query_selection_template_v5
    retry = 0
    prompt = PromptTemplate(
        input_variables=["question", "predicate_list"],
        template=template,
    )
    if kg_name in ['wikidata']:
        predicate_list, predicate_to_query_id = prepare_keywords_list_wikidata(triples_list, predicate_map)
    else:
        predicate_list, predicate_to_query_id = prepare_keywords_list(triples_list)
    if len(predicate_list) == 0:
        return list()
    final_prompt = prompt.format(question=question, predicate_list=predicate_list)
    # print(final_prompt)
    llm = get_llm()
    chain = prompt | llm
    output, metadata = None, None
    while retry < 3:
        if retry > 0:
            print("Retrying...")
        try:
            output1 = chain.invoke({"question": question, "predicate_list": predicate_list})
            if llm_type == 'google':
                time.sleep(30)
            # print(metadata)
            if llm_type != 'vllm':
                metadata = output1.usage_metadata
                output1 = output1.content
            output1 = extract_json_from_output(output1)
            output = json.loads(output1)
            json_logger.set("Filtration", output)
            if metadata:
                json_logger.add_cost("Filtration", metadata)
            print("Filtration OUTPUT======")
            print(output)
            output = output["keywords"]
            output = postprocess_result(output)
            if len(output) > 0:
                break
        except Exception as e:
            traceback.print_exc()
            print("============Start Parsing Error")
            print(output1)
            print("============End")
        retry += 1


    # if output is None or len(output) == 0:
    #     output = predicate_list.split(',')

    keys = predicate_to_query_id.keys()
    lowerToKeyword = dict()
    for kw in keys:
        lowerToKeyword[kw.lower()] = kw
    # keys = {kw.lower() for kw in keys}
    for k in output:
        # If LLM hallucinated and return an object instead of strings
        if isinstance(k, dict):
            k = k["keyword"]
        k = k.strip()
        if "None" in k:
            continue
        elif k.lower() in lowerToKeyword:
            return_result.extend(predicate_to_query_id[lowerToKeyword[k.lower()]])
        else:
            print(f"Error {k} not in input list")
    return list(set(return_result))
