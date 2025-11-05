from langchain_openai import ChatOpenAI
from langchain_community.llms import VLLMOpenAI
from langchain_core.prompts.prompt import PromptTemplate
from langchain.schema.output_parser import StrOutputParser
from langchain_google_vertexai import ChatVertexAI
from langchain_deepseek import ChatDeepSeek
from langchain.schema import HumanMessage, AIMessage, SystemMessage

import json
import os
import time
from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv())


def evaluate_dialogue_chat_interface(dialogue, llm, timeout):
    """
    Evaluate dialogue using a chat-based LLM interface, maintaining conversational context automatically.
    Assumes LLM supports history (ChatOpenAI, ChatVertexAI, etc.).

    Args:
        dialogue (list): List of questions representing a dialogue.
        llm: LangChain-compatible chat model.
        timeout (bool): Whether to apply a delay (e.g., for rate limits).

    Returns:
        dict: Mapping from question index to list of predicted answers.
    """

    chat_history = [
        SystemMessage(content="""
For each question, return the answers as a list of strings.
For Yes, No questions, return a list with a single item either `["Yes"]` or `["No"]`
Do not include explanations or extra text. Only return the JSON array.
""")
    ]

    answer = dict()

    for index, q in enumerate(dialogue):
        # print(chat_history)
        print(f"Question {index}: {q}\n")

        chat_history.append(HumanMessage(content=q))
        response = llm(chat_history)
        #response = llm.invoke(chat_history)

        if timeout:
            time.sleep(30)

        print(response.content)
        response_text = response.content.replace("```json", "").replace("```", "")

        try:
            answers_json = json.loads(response_text)
            if isinstance(answers_json, dict):
                results = list()
                for value in answers_json.values():
                    results.extend(value)
                answers_json = results
            answer[index] = answers_json
        except json.JSONDecodeError:
            print("LLM returned invalid JSON:")
            print(response_text)

        chat_history.append(AIMessage(content=response.content))

    return answer


# provide question and answer as context
def evaluate_dialogue_v3(dialogue, llm, timeout):
    prompt = PromptTemplate.from_template(
        """
        Given a question and a list of preceding questions in a dialogue with their answers, return the answers to the question in a list of strings.
        For Yes, No questions, return a list with a single item either `["Yes"]` or `["No"]` without any explanations.

        Previous questions with answers: {context}
        Question: {question} 

        Response:```json"""
    )
    # Combine the prompt + model + output parser
    chain = prompt | llm | StrOutputParser()
    answer = dict()
    context = list()
    for index, q in enumerate(dialogue):
        print(f"Question {index}: {q}\n")
        final_prompt = prompt.format(question=q, context=context)
        print(final_prompt)

        # Run the chain
        response = chain.invoke({"question": q, "context": context})
        if timeout:
            time.sleep(30)
        print(response)
        response = response.replace("```json", "").replace("```", "")
        try:
            answers_json = json.loads(response)
            if isinstance(answers_json, dict):
                results = list()
                for value in answers_json.values():
                    results.extend(value)
                answers_json = results
            answer[index] = answers_json
        except json.JSONDecodeError:
            print("LLM returned invalid JSON:")
            print(response)
        q_context = f"Q: {q}"
        a_context = f"A: {answer[index]}"
        context.append(q_context)
        context.append(a_context)
    return answer


# Provide only the questions as context
def evaluate_dialogue_v2(dialogue, llm, timeout):
    prompt = PromptTemplate.from_template(
        """
        Given a question and a list of preceding questions in a dialogue, return the answers to the question in a list of strings.
        For Yes, No questions, return a list with a single item either `["Yes"]` or `["No"]` without any explanations.

        Previous questions: {context}
        Question: {question} 

        Response:```json"""
    )
    # Combine the prompt + model + output parser
    chain = prompt | llm | StrOutputParser()
    answer = dict()
    context = list()
    for index, q in enumerate(dialogue):
        print(f"Question {index}: {q}\n")
        final_prompt = prompt.format(question=q, context=context)
        # print(final_prompt)

        # Run the chain
        response = chain.invoke({"question": q, "context": context})
        if timeout:
            time.sleep(30)
        print(response)
        response = response.replace("```json", "").replace("```", "")
        try:
            answers_json = json.loads(response)
            if isinstance(answers_json, dict):
                results = list()
                for value in answers_json.values():
                    results.extend(value)
                answers_json = results
            answer[index] = answers_json
        except json.JSONDecodeError:
            print("LLM returned invalid JSON:")
            print(response)
        context.append(q)
    return answer


# Provide only the questions
def evaluate_dialogue_v1(dialogue, llm, timeout):
    # prompt = PromptTemplate.from_template(
    #     """
    #     {question}
    #
    #     Return the answers in a list
    #     {{
    #      ["<answer1>", "<answer2>", ...],
    #     }}
    #
    #     In case of multiple answers, return them is a list
    #     For Yes, No questions, return a list with a single item either `["Yes"]` or `["No"]` without any explanations.
    #
    #     Response:```json"""
    # )

    prompt = PromptTemplate.from_template(
        """
        Given a question, return its answers in a list.
        For Yes, No questions, return a list with a single item either `["Yes"]` or `["No"]` without any explanations.

        Question: {question} 

        Response:```json"""
    )
    # Combine the prompt + model + output parser
    chain = prompt | llm | StrOutputParser()
    answer = dict()
    for index, q in enumerate(dialogue):
        print(f"Question {index}: {q}\n")
        final_prompt = prompt.format(question=q)
        # print(final_prompt)

        # Run the chain
        response = chain.invoke({"question": q})
        if timeout:
            time.sleep(30)
        print(response)
        response = response.replace("```json", "").replace("```", "")
        try:
            answers_json = json.loads(response)
            answer[index] = answers_json
        except json.JSONDecodeError:
            print("LLM returned invalid JSON:")
            print(response)
    return answer


def process_llm_output_v1(start_id, original_questions, llm_answers):
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
        # question_number = str(i + 1)
        llm_answer_list = llm_answers.get(i, [])
        if isinstance(llm_answer_list, int):
            llm_answer_list = [str(llm_answer_list)]
        if len(llm_answer_list) == 1:
            llm_answer_list = llm_answer_list[0].split(',') if llm_answer_list[0] is not None else []
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
        llm = ChatVertexAI(model='gemini-2.0-flash', temperature=0, max_tokens=None, do_sample=True)
    elif llm_name == 'deepseek':
        llm = ChatDeepSeek(model="deepseek-chat", temperature=0, max_retries=2)
    elif llm_name == 'phi_local' or llm_name == 'qwen_instruct':
        llm = ChatOpenAI(openai_api_key="EMPTY", openai_api_base="http://localhost:5000/v1", model_name=llm_name,
                         temperature=0)
    return llm


if __name__ == '__main__':
    kgs = {"dbpedia": '../../chatbot/evaluation/data/dbpedia_e11_20_5_original.json', "yago": '../../chatbot/evaluation/data/yago_e11_20_5_original.json',
           "dblp": '../../chatbot/evaluation/data/dblp_e11_20_5_original.json', "wikidata": '../../chatbot/evaluation/data/wikidata_subgraph_summarized_20_5_simplified.json'}
    # kgs = {"dblp": '../../chatbot/evaluation/data/dblp_e11_20_5_original.json'}
    llms = ['gpt', 'gemini', 'deepseek', 'phi_local', 'qwen_instruct']
    # llms = ['qwen_instruct']
    for llm_name in llms:
        llm = get_llm(llm_name)
        for kg in kgs:
            file_name = kgs[kg]
            print(f"Start {llm_name} {kg}")
            output_file = f"output/dialogue/{llm_name}_{kg}_output.json"
            output_data = []

            with open(file_name, 'r') as f:
                data = json.load(f)
                output = list()

            count = 0
            id = 0
            for obj in data["data"]:
                original_questions = obj["dialogue"]
                llm_answers = evaluate_dialogue_chat_interface(original_questions, llm, llm_name == 'gemini')
                processed_output = process_llm_output_v1(id, original_questions, llm_answers)
                output_data.extend(processed_output)

                count += 1
                id += len(original_questions)
                # break

            write_json_file(output_data, output_file)
            print(f"End {llm_name} {kg}")

