import os
import json
import time
import traceback
import csv

from termcolor import colored, cprint
from itertools import count
import xml.etree.ElementTree as Et
import sys

sys.path.append('..')
sys.path.append('../..')
from chattykg.json_logger import JsonLogger


from multi_agent.shared.state import AgentState
from multi_agent.utils.graph_builder import build_langgraph
from multi_agent.modules.State import State

file_dir = os.path.dirname(os.path.abspath(__file__))

file_name = os.path.join(file_dir, "../../evaluation/yago/qald9_yago100.json")

if __name__ == '__main__':
    root_element = Et.Element('dataset')
    root_element.set('id', 'dbpedia-test')
    timestr = time.strftime("%Y%m%d-%H%M%S")

    total_time = 0
    total_understanding_time = 0
    total_linking_time = 0
    total_execution_time = 0
    total_query_selection_time = 0
    total_query_execution_time = 0
    total_num_queries_executed = 0
    llm_name = "glm"

    # The main param:
    # max no of vertices and edges to annotate the PGP
    # max no of SPARQL queries to be generated from PGP
    max_Vs = 1
    max_Es = 21
    max_answers = 41
    limit_VQuery = 600
    limit_EQuery = 300
    json_logger = JsonLogger(log_file=f"output/{llm_name}/yago_chattykg_json_log.json")

    with open(file_name) as f:
        qald9_testset = json.load(f)
    dataset_id = qald9_testset['dataset']['id']

    qCount = count(1)
    chattykg_qald9 = {"dataset": {"id": "qald9_yago100"}, "questions": []}
    graph = build_langgraph()

    for i, question in enumerate(qald9_testset['questions']):
        qc = next(qCount)
        for language_variant_question in question['question']:
            if language_variant_question['language'] == 'en':
                question_text = language_variant_question['string'].strip()
                break

        text = colored(f"[PROCESSING: ] Question count: {qc}, ID {question['id']}  >>> {question_text}", 'blue',
                       attrs=['reverse', 'blink'])
        cprint(f"== {text}  ")

        st = time.time()
        try:
            kg_graph_state = State(
                knowledge_graph='yago',
                n_limit_VQuery=limit_VQuery,
                n_max_Vs=max_Vs,
                n_limit_EQuery=limit_EQuery,
                n_max_Es=max_Es,
                n_max_answers=max_answers,
                filtration_enabled=True,
                json_logger=json_logger,
            )

            state = AgentState(
                session_id=question['id'],
                question_id=question['id'],
                question=question_text,
                chat_history=[],
                system_mode="Standalone",
                kg_graph_state=kg_graph_state,
                query_done=False,
                qir_done=False,
                has_been_resolved=False,
                route="chat_agent",
                sparql_query=None,
                query_result=None,
                resolved_question=None,
                ambiguity_resolver_done=False,
                query_graph=None,
                matching_done=False,
                ambiguity_resolver_tries=0
            )

            raw_state = graph.invoke(state)
            final_state = AgentState(**dict(raw_state))

            answers = [
                answer.json() for answer in kg_graph_state.get_answers()
            ]

        except Exception as e:
            traceback.print_exc()
            continue

        all_bindings = list()
        for answer in answers:
            if answer['results'] and answer['results']['bindings']:
                all_bindings.extend(answer['results']['bindings'])

        try:
            if 'results' in question['answers'][0]:
                question['answers'][0]['results']['bindings'] = all_bindings.copy()
                all_bindings.clear()
        except:
            question['answers'] = []

        understanding_time = kg_graph_state.get_understanding_time()
        linking_time = kg_graph_state.get_linking_time()
        execution_time = kg_graph_state.get_execution_time()
        query_selection_time = kg_graph_state.get_query_selection_time()
        num_queries_executed = kg_graph_state.get_num_executed_queries()

        chattykg_qald9['questions'].append(question)

        et = time.time()
        total_time = total_time + (et - st)
        total_understanding_time = total_understanding_time + understanding_time
        total_linking_time = total_linking_time + linking_time
        if execution_time < 100:
            total_execution_time = total_execution_time + execution_time
            total_query_selection_time = total_query_selection_time + query_selection_time
            total_query_execution_time = total_query_execution_time + (execution_time - query_selection_time)
        total_num_queries_executed = total_num_queries_executed + num_queries_executed
        text = colored(f'[DONE!! in {et - st:.2f} SECs]', 'green', attrs=['bold', 'reverse', 'blink', 'dark'])
        cprint(f"== {text} ==")

        # break
    text1 = colored(f'total_time = [{total_time:.2f} sec]', 'yellow', attrs=['reverse', 'blink'])
    text2 = colored(f'avg time = [{total_time / qc:.2f} sec]', 'yellow', attrs=['reverse', 'blink'])
    cprint(f"== QALD 9 Statistics : {qc} questions, Total Time == {text1}, Average Time == {text2} ")
    cprint(f"== Understanding : {qc} questions, Total Time == {total_understanding_time}, Average Time == {(total_understanding_time / qc)*1000} ")
    cprint(f"== Linking : {qc} questions, Total Time == {total_linking_time}, Average Time == {(total_linking_time / qc)*1000} ")
    cprint(f"== Execution : {qc} questions, Total Time == {total_execution_time}, Average Time == {(total_execution_time / qc)*1000} ")
    cprint(f"== Query Selection : {qc} questions, Total Time == {total_query_selection_time}, Average Time == {(total_query_selection_time / qc) * 1000} ms")
    cprint(f"== Query Execution : {qc} questions, Total Time == {total_query_execution_time}, Average Time == {(total_query_execution_time / qc) * 1000} ms")
    cprint(f"== Queries Executed : {qc} questions, Total Number == {total_num_queries_executed}, Average Number == {(total_num_queries_executed / qc)}")
    response_time = [{"Question Understanding": (total_understanding_time / qc) * 1000,
                      "Linking": (total_linking_time / qc) * 1000,
                      "Execution": (total_execution_time / qc) * 1000,
                      "Query Selection": (total_query_selection_time / qc) * 1000,
                      "Query Execution": (total_query_execution_time / qc) * 1000,
                      "Number of queries": total_num_queries_executed / qc
                      }]

    with open(os.path.join(file_dir, f'output/{llm_name}/yago.json'), encoding='utf-8', mode='w') as rfobj:
        json.dump(chattykg_qald9, rfobj)
        rfobj.write('\n')

    field_names = response_time[0].keys()
    with open(os.path.join(file_dir, f'output/{llm_name}/yago_response_time_ms.csv'), mode='w', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=field_names)
        writer.writeheader()
        writer.writerows(response_time)
