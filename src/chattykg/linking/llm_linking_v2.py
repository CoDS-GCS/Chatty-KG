import json
import time
import traceback
import torch
import re

from langchain_core.prompts.prompt import PromptTemplate

from chattykg.llms.llm_creation import get_llm, llm_type
from chattykg.llms.prompts import vertex_linking_template_v2, vertex_linking_template_v3

torch.set_default_tensor_type(torch.DoubleTensor)

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
    return llm_output[start_index:end_index + 1]

def remove_unneeded_chars(text):
    text = text.strip().strip("\'").strip('\"')
    return text


def validate_vertex_linking(input_list, output):
    output = remove_unneeded_chars(output)
    if not (output and output in input_list):
        return False
    return True

def gets_indices_for_label(label, vertex_label_list):
    target_index = []
    if label in vertex_label_list:
        count = vertex_label_list.count(label)
        if count == 1:
            target_index.append(vertex_label_list.index(label))
        else:
            index = 0
            for item in vertex_label_list:
                if item == label:
                    target_index.append(index)
                index += 1
        return target_index
    else:
        return None

def get_name(uri):
    pattern = r'[#/]([^/#]+)$'
    match = re.search(pattern, uri)
    if match:
        name = match.group(1)
    else:
        name = ""
    p2 = re.compile(r"([a-z0-9])([A-Z])")
    name = p2.sub(r"\1 \2", name)
    name = name.replace("_", " ")
    return name

def extract_correct_uri(entity, candidate_indices, vertex_list):
    updated_list = list()
    new_names = list()
    for index in candidate_indices:
        uri = vertex_list[index]
        name =get_name(uri)
        updated_list.append(uri)
        new_names.append(name)

    template = vertex_linking_template_v3
    prompt = PromptTemplate(
        input_variables=["entity", "vertex_label_list"],
        template=template,
    )
    final_prompt = prompt.format(entity=entity, vertex_label_list=new_names)
    # print(final_prompt)
    llm, max_tokens = get_llm(True)
    chain = prompt | llm
    output1 = ""
    metadata = None
    try:
        output1 = chain.invoke({"entity": entity, "vertex_label_list": new_names})
        if llm_type == 'google':
            time.sleep(30)
        if llm_type != 'vllm':
            metadata = output1.usage_metadata
            output1 = output1.content
        output = extract_json_from_output(output1)
        # json_logger.set("Linking", output)
        # if metadata:
            # json_logger.add_cost("Linking", metadata)
        print("Linking OUTPUT======")
        print(output)
        output = json.loads(output)["value"]
        output = remove_unneeded_chars(output)
        if output in new_names:
            index = new_names.index(output)
            vertex = updated_list[index]
            return [vertex]
        else:
            return None
    except:
        traceback.print_exc()
        print("============Start Parsing Error")
        print(output1)
        print("============End")





def vertex_linking(entity, vertex_label_list, vertex_list, json_logger, kg):
    retry = 0
    template = vertex_linking_template_v3
    prompt = PromptTemplate(
        input_variables=["entity", "vertex_label_list"],
        template=template,
    )
    final_prompt = prompt.format(entity=entity, vertex_label_list=vertex_label_list)
    # print(final_prompt)
    llm, max_tokens = get_llm(True)
    chain = prompt | llm
    if max_tokens is not None:
        while len(final_prompt) > max_tokens:
            vertex_label_list = vertex_label_list[:-50]
            final_prompt = prompt.format(entity=entity, vertex_label_list=vertex_label_list)
    while retry < 3:
        if retry > 0:
            print("Retrying...")
        output1 = ""
        metadata = None
        try:
            output1 = chain.invoke({"entity": entity, "vertex_label_list": vertex_label_list})
            if llm_type == 'google':
                time.sleep(30)
            # print(output.content)
            # print(metadata)
            if llm_type != 'vllm':
                metadata = output1.usage_metadata
                output1 = output1.content
            output = extract_json_from_output(output1)
            json_logger.set("Linking", output)
            if metadata:
                json_logger.add_cost("Linking", metadata)
            print("Linking OUTPUT======")
            print(output)
            output = json.loads(output)["value"]
            output = remove_unneeded_chars(output)
            if output in vertex_label_list:
                break
        except:
            traceback.print_exc()
            print("============Start Parsing Error")
            print(output1)
            print("============End")
        retry += 1
    if kg not in ['microsoft_academic']:
        candidate_indices = gets_indices_for_label(output, vertex_label_list)
        if candidate_indices is None:
            return None, None
        elif len(candidate_indices) == 1:
            return [vertex_list[candidate_indices[0]]], [vertex_label_list[candidate_indices[0]]]
        else:
            vertex = extract_correct_uri(entity, candidate_indices, vertex_list)
            return vertex, [output]
    else:
        if output in vertex_label_list:
            vertex = vertex_label_list.index(output)
            vertex = vertex_list[vertex]
            return [vertex], [output]
        else:
            return None, None


if __name__ == '__main__':
    question_list = [
        "What is the revenue of IBM?",
        "In which city is the headquarter of Air China?",
        "Show me hiking trails in the Grand Canyon where there's no danger of flash floods.",
        "Who became president after JFK died?",
        "How many calories does a baguette have?"
    ]
