import json
import time
import traceback
import torch

from langchain_core.prompts.prompt import PromptTemplate

from kgqan.llms.llm_creation import get_llm, llm_type
from kgqan.llms.prompts import vertex_linking_template_v2

torch.set_default_tensor_type(torch.DoubleTensor)

def extract_json_from_output(llm_output):
    if '### Instruction' in llm_output:
        index = llm_output.index('### Instruction')
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


def vertex_linking(entity, vertex_label_list, json_logger):
    retry = 0
    template = vertex_linking_template_v2
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
        try:
            output1 = chain.invoke({"entity": entity, "vertex_label_list": vertex_label_list})
            if llm_type == 'google':
                time.sleep(30)
            # print(output.content)
            metadata = output1.usage_metadata
            # print(metadata)
            if llm_type != 'vllm':
                output1 = output1.content
            output = extract_json_from_output(output1)
            json_logger.set("Linking", output)
            json_logger.add_cost("Linking", metadata)
            print("Linking OUTPUT======")
            print(output)
            output = json.loads(output)["value"]
        except:
            traceback.print_exc()
            print("============Start Parsing Error")
            print(output1)
            print("============End")
        retry += 1
        output = remove_unneeded_chars(output)
        if output in vertex_label_list:
            break

    vertex = vertex_label_list.index(output) if output in vertex_label_list else None
    return vertex


if __name__ == '__main__':
    question_list = [
        "What is the revenue of IBM?",
        "In which city is the headquarter of Air China?",
        "Show me hiking trails in the Grand Canyon where there's no danger of flash floods.",
        "Who became president after JFK died?",
        "How many calories does a baguette have?"
    ]