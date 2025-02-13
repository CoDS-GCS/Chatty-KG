import json
import time
import traceback
import torch
import re
import ast

from langchain_core.prompts.prompt import PromptTemplate

from kgqan.llms.llm_creation import get_llm, llm_type
from kgqan.llms.prompts import query_selection_template_v4

torch.set_default_tensor_type(torch.DoubleTensor)


def extract_json_from_output(llm_output):
    if '### Instruction' in llm_output:
        index = llm_output.index('### Instruction')
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
        return name
    else:
        return ""


def prepare_keywords_list(triples_list):
    predicate_to_query_id = {}
    output = ""
    for i, triples in enumerate(triples_list):
        for triple in triples:
            name = get_name(triple[1][0])
            if name != "":
                if name in predicate_to_query_id:
                    predicate_to_query_id[name].append(i)
                else:
                    predicate_to_query_id[name] = [i]
                    output += name + ', '
    return output[: len(output) - 2], predicate_to_query_id


def choose_question_from_keywords(question, query_list, triples_list, json_logger):
    return_result = list()

    template = query_selection_template_v4
    retry = 0
    prompt = PromptTemplate(
        input_variables=["question", "predicate_list"],
        template=template,
    )
    predicate_list, predicate_to_query_id = prepare_keywords_list(triples_list)
    if len(predicate_list) == 0:
        return list()
    final_prompt = prompt.format(question=question, predicate_list=predicate_list)
    print(final_prompt)
    llm = get_llm()
    chain = prompt | llm
    output = None
    while retry < 3:
        if retry > 0:
            print("Retrying...")
        try:
            output1 = chain.invoke({"question": question, "predicate_list": predicate_list})
            if llm_type == 'google':
                time.sleep(30)
            metadata = output1.usage_metadata
            # print(metadata)
            if llm_type != 'vllm':
                output1 = output1.content
            output1 = extract_json_from_output(output1)
            output = json.loads(output1)
            json_logger.set("Filtration", output)
            json_logger.add_cost("Filtration", metadata)
            print("Filtration OUTPUT======")
            print(output)
            output = output["keywords"]
        except Exception as e:
            traceback.print_exc()
            print("============Start Parsing Error")
            print(output1)
            print("============End")
        output = postprocess_result(output)
        if len(output) > 0:
            break
        retry += 1

    # if output is None or len(output) == 0:
    #     output = predicate_list.split(',')

    for k in output:
        k = k.strip()
        if "None" in k:
            continue
        elif k in predicate_to_query_id:
            return_result.extend(predicate_to_query_id[k])
        else:
            print(f"Error {k} not in input list")
    return list(set(return_result))
