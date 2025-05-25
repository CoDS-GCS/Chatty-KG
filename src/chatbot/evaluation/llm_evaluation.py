from langchain_openai import ChatOpenAI
from langchain_community.llms import VLLMOpenAI
from langchain_core.prompts.prompt import PromptTemplate
from langchain.schema.output_parser import StrOutputParser
from langchain_google_vertexai import ChatVertexAI
from langchain_deepseek import ChatDeepSeek

import json
import os
import time

os.environ["OPENAI_API_KEY"] = ""
os.environ["DEEPSEEK_API_KEY"] = ""

def evaluate_dialogue(dialogue, llm, timeout):
    prompt = PromptTemplate.from_template(
"""
You are given a dialogue consisting of multiple questions. Return the answers to each question.
    
Return the answers in JSON format with this structure:
{{
  "id":  ["<answer1>", "<answer2>", ...],
  ...
}}

For Yes, No questions, return a list with a single item either `["Yes"]` or `["No"]` without any explanations. 
Each answer must be a list, even if it has only one item. Do not include any explanation.
    
Dialogue:
{questions}
    
Response:```json"""
    )

    # Combine the prompt + model + output parser
    chain = prompt | llm | StrOutputParser()

    final_prompt = prompt.format(questions=dialogue)
    # print(final_prompt)

    # Run the chain
    response = chain.invoke({"questions": dialogue})
    if timeout:
        time.sleep(30)
    #print(response)
    response = response.replace("```json", "").replace("```", "")
    try:
        answers_json = json.loads(response)
        return answers_json
    except json.JSONDecodeError:
        print("LLM returned invalid JSON:")
        print(response)
        return {}

def process_llm_output(start_id, original_questions, llm_answers):
    """
    Processes the LLM output to format it into the desired JSON structure.

    Args:
        original_id (str): The ID of the original data object.
        original_questions (list): A list of the original questions.
        llm_answers (dict): The dictionary of answers from the LLM.

    Returns:
        dict: The processed output in the desired JSON format, or None if llm_answers is None.
    """

    output = list()
    id = start_id
    for i, question in enumerate(original_questions):
        question_number = str(i + 1)
        llm_answer_list = llm_answers.get(question_number, [])
        if len(llm_answer_list) == 1:
            llm_answer_list = llm_answer_list[0].split(',')
        formatted_answer = {
            "head": {"link": [], "vars": ["variable"]},
            "results": {
                "distinct": False,
                "ordered": True,
                "bindings": [{"variable": {"type": "literal", "value": ans}} for ans in llm_answer_list],
            },
        }
        output_obj = {
            "id": id,
            "question": question,
            "answers": [formatted_answer],
        }
        output.append(output_obj)
        id += 1
    return output

def write_json_file(data, filename="output.json"):
    """
    Writes the given data to a JSON file.

    Args:
        data (list): The data to write to the JSON file.
        filename (str, optional): The name of the file to write to. Defaults to "output.json".
    """
    try:
        with open(filename, 'w') as outfile:
            json.dump(data, outfile, indent=2)
        print(f"Processed data written to {filename}")
    except Exception as e:
        print(f"Error writing to {filename}: {e}")

def get_llm(llm_name):
    if llm_name == 'gpt':
        llm = ChatOpenAI(model="gpt-4o", temperature=0)
    elif llm_name == 'gemini':
        llm = ChatVertexAI(model='gemini-2.0-flash',temperature=0,max_tokens=None, do_sample= True)
    elif llm_name == 'deepseek':
        llm = ChatDeepSeek(model="deepseek-chat", temperature=0, max_retries=2)
    elif llm_name == 'phi_local' or llm_name == 'qwen_instruct':
        llm = VLLMOpenAI(openai_api_key="EMPTY",openai_api_base="http://localhost:5000/v1", model_name=llm_name, model_kwargs={"stop": ["```"]}, temperature=0)
    return llm

if __name__ == '__main__':
    kgs = {"dbpedia": 'data/dbpedia_e11_20_5_original.json', "yago": 'data/yago_e11_20_5_original.json', "dblp": 'data/dblp_e11_20_5_original.json'}
    llms = ['gpt', 'gemini', 'deepseek', 'phi_local', 'qwen_instruct']
    for llm_name in llms:
        llm = get_llm(llm_name)
        for kg in kgs:
            file_name = kgs[kg]
            print(f"Start {llm_name} {kg}")
            output_file = f"llms/{llm_name}_{kg}_output.json"
            output_data = []

            with open(file_name, 'r') as f:
                data = json.load(f)
                output = list()

            count = 0
            id = 0
            for obj in data["data"]:
                original_questions = obj["dialogue"]
                dialogue_str = ""
                for index, q in enumerate(original_questions):
                    dialogue_str += f"{index+1}) {q}\n"

                llm_answers = evaluate_dialogue(dialogue_str, llm, llm_name=='gemini')
                processed_output = process_llm_output(id, original_questions, llm_answers)
                output_data.extend(processed_output)

                count += 1
                id += len(original_questions)

            write_json_file(output_data, output_file)
            print(f"End {llm_name} {kg}")


