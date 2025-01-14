from langchain_core.prompts.prompt import PromptTemplate
from langchain_openai import ChatOpenAI
import torch
import os

os.environ["OPENAI_API_KEY"] = ""
torch.set_default_tensor_type(torch.DoubleTensor)

model_to_token_length = {'gpt-3.5-turbo': 16000}

def remove_unneeded_chars(text):
    text = text.strip().strip("\'").strip('\"')
    return text

def get_openAI_llm():
    # Choose from [gpt-3.5-turbo, 'gpt-4', 'gpt-4-turbo', 'gpt-4o']
    model_name = "gpt-4o"
    llm = ChatOpenAI(model=model_name, temperature=0)
    max_tokens = model_to_token_length.get(llm.model_name, None)
    return llm, max_tokens


def validate_vertex_linking(input_list, output):
    output = remove_unneeded_chars(output)
    if not (output and output in input_list):
        return False
    return True

def vertex_linking(entity, vertex_label_list):

    template = """
Given an entity and a list of labels. Choose the most semantically similar label from the list.
Only Return the label.
    
Entity: {entity}
List of labels: {vertex_label_list}

Output:  
    """

    template = """
Given an entity and a list of labels, choose the label that is the most semantically similar to the entity.
1. If there is an exact match in the list, choose it immediately.
2. If there is no exact match, select the label that is closest in meaning.
Only Return the label.

 Entity: {entity}
List of labels: {vertex_label_list}

 Output:  
        """
    prompt = PromptTemplate(
        input_variables=["entity", "vertex_label_list"],
        template=template,
    )
    final_prompt = prompt.format(entity=entity, vertex_label_list=vertex_label_list)
    # print(final_prompt)
    llm, max_tokens = get_openAI_llm()
    chain = prompt | llm
    if max_tokens is not None:
        while len(final_prompt) > max_tokens:
            vertex_label_list = vertex_label_list[:-50]
            final_prompt = prompt.format(entity=entity, vertex_label_list=vertex_label_list)

    output = chain.invoke({"entity": entity, "vertex_label_list": vertex_label_list})
    print(output.content)
    output = output.content
    output = remove_unneeded_chars(output)
    return vertex_label_list.index(output) if output in vertex_label_list else None



if __name__ == '__main__':
    question_list = [
        "What is the revenue of IBM?",
        "In which city is the headquarter of Air China?",
        "Show me hiking trails in the Grand Canyon where there's no danger of flash floods.",
        "Who became president after JFK died?",
        "How many calories does a baguette have?"
    ]