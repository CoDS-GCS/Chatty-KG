from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts.prompt import PromptTemplate
from langchain_openai import ChatOpenAI
from pydantic import BaseModel
from typing import List
import torch
import os

os.environ["OPENAI_API_KEY"] = ""
torch.set_default_tensor_type(torch.DoubleTensor)


def post_process_relations(output, predicate_list):
    output = output.replace('```csv', '')
    output = output.replace('```', '')
    lines = output.split('\n')
    label_score_map = dict()
    for line in lines:
        try:
            if not line or line.lower() == 'label,score':
                continue
            if line.count(',') > 1:
                index = line.rindex(',')
                label = line[:index]
                score = float(line[index + 1:])
            else:
                label, score = line.split(',')
            label_score_map[label] = float(score)
        except:
            print(f"Error Handling line: {line}")
            continue
    scores = list()
    for predicate in predicate_list:
        try:
            if predicate in label_score_map:
                scores.append(label_score_map[predicate])
            else:
                scores.append(0)
        except:
            print(f"Error Handling predicate: {predicate}")
            continue
    return scores


def remove_unneeded_chars(text):
    text = text.strip().strip("\'").strip('\"')
    return text


def get_openAI_llm():
    # Choose from [gpt-3.5-turbo, 'gpt-4', 'gpt-4-turbo', 'gpt-4o']
    model_name = "gpt-4o"
    llm = ChatOpenAI(model=model_name, temperature=0)
    return llm


def relation_linking(relation, predicate_label_list):
    template_v1 = """
Given a relation and a list of labels, calculate the semantic similarity between the relation and each label. 
1. The score is from 0 to 1 where 1 is most similar and 0 is not similar
2. Only Return the list with scores in a csv format.
3. DO NOT return any explanations. 

Relation: {relation}
List of labels: {predicate_label_list}

 Output:  
        """
    template = """
Act as an embedding server. Your task is to calculate semantic similarity scores between a given relation and a list of labels, based on the conceptual embeddings of their meanings. 

1. Treat the relation and labels as vector representations in a high-dimensional semantic space.
2. Use cosine similarity to compute a score for each label relative to the relation.
3. Scores must be between 0 (completely dissimilar) and 1 (identical in meaning).
4. Ensure the comparison considers: Conceptual and contextual relationships, Morphological similarities (e.g., word stems or shared roots), and Case insensitivity.
5. Output only a CSV with two columns: Label and Score.
6. DO NOT return any explanations. 

Relation: {relation}
List of labels: {predicate_label_list}

Output: 
    """
    prompt = PromptTemplate(
        input_variables=["relation", "predicate_label_list"],
        template=template,
    )
    final_prompt = prompt.format(relation=relation, predicate_label_list=predicate_label_list)
    print(final_prompt)
    llm = get_openAI_llm()
    chain = prompt | llm
    output = chain.invoke({"relation": relation, "predicate_label_list": predicate_label_list})
    print(output.content)
    output = output.content
    scores = post_process_relations(output, predicate_label_list)
    return scores
    # output = output.split(',')
    # index_result = list()
    # for x in output:
    #     index_result.append(predicate_label_list.index(remove_unneeded_chars(x)))
    # return index_result


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
    # final_prompt = prompt.format(entity=entity, vertex_label_list=vertex_label_list)
    # print(final_prompt)
    llm = get_openAI_llm()
    chain = prompt | llm
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
    # for question in question_list:
    #     print(f"======={question}==========")
    #     extract_understanding_triples_from_question(question)