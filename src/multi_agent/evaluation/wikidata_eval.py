import os
import json
import time
import traceback
import csv
import argparse
from termcolor import colored, cprint
from itertools import count
import xml.etree.ElementTree as Et
import numpy as np
import sys

sys.path.append('..')
sys.path.append('../..')

from chattykg.json_logger import JsonLogger
# from multi_agent.agents.translate_question import translate_question

from multi_agent.shared.state import AgentState
from multi_agent.utils.graph_builder import build_langgraph
from multi_agent.modules.State import State

file_dir = os.path.dirname(os.path.abspath(__file__))
# file_name = "../../evaluation/wikidata/annotated_wd_data_test_answerable.json"
file_name = "../../evaluation/wikidata/qald_9_plus_test_wikidata.json"

if __name__ == '__main__':
    root_element = Et.Element('dataset')
    root_element.set('id', 'dbpedia-test')
    root_element.append(Et.Comment('created by CoDS Lab'))

    timestr = time.strftime("%Y%m%d-%H%M%S")
    total_time = 0
    total_understanding_time = 0
    total_linking_time = 0
    total_execution_time = 0
    total_query_selection_time = 0
    total_query_execution_time = 0
    total_num_queries_executed = 0
    llm_name = "gpt-4o"

    parser = argparse.ArgumentParser()
    parser.add_argument("--filter", type=str, default="True", help="argument to enable filtration")
    args = parser.parse_args()
    filter = args.filter.lower() == 'true'
    json_logger = JsonLogger(log_file=f"output/{llm_name}/qald_chattykg_json_log.json")

    with open(file_name) as f:
        qald9_testset = json.load(f)
    # dataset_id = qald9_testset['dataset']['id']
    dataset_id = "qald_9_plus_test_wikidata"
    qCount = count(1)
    chattykg_qald9 = {"dataset": {"id": dataset_id}, "questions": []}
    graph = build_langgraph()

    for i, question in enumerate(qald9_testset['questions']):
        qc = next(qCount)
        # if qc == 2:
        #     break
        for lang_q in question['question']:
            if lang_q['language'] == 'en':
                question_text = lang_q['string'].strip()
                break
        # question_text = translate_question(question_text)
        text = colored(f"[PROCESSING: ] Question count: {qc}, ID {question['id']}  >>> {question_text}", 'blue', attrs=['reverse', 'blink'])
        cprint(f"== {text}  ")

        st = time.time()
        try:
            kg_graph_state = State(
                knowledge_graph='wikidata',
                n_limit_VQuery=600,
                n_max_Vs=1,
                n_limit_EQuery=25,
                n_max_Es=21,
                n_max_answers=41,
                filtration_enabled=True,
                json_logger=json_logger
            )

            state = AgentState(
                session_id=question['id'],
                question_id=question['id'],
                question=question_text,
                chat_history=[],
                system_mode="Standalone",
                answer_mode="Not Formulated",
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

            # Extract timing values from final_state (placeholders below)
            understanding_time = kg_graph_state.get_understanding_time()
            linking_time = kg_graph_state.get_linking_time()
            execution_time = kg_graph_state.get_execution_time()
            query_selection_time = kg_graph_state.get_query_selection_time()
            num_queries_executed = kg_graph_state.get_num_executed_queries()

        except Exception:
            traceback.print_exc()
            continue

        # try:
        #     if 'results' in question['answers'][0]:
        #         question['answers'][0]['results']['bindings'] = all_bindings.copy()
        #         all_bindings.clear()
        # except:
        #     question['answers'] = []

        chattykg_qald9['questions'].append(question)

        et = time.time()
        total_time += (et - st)
        total_understanding_time += understanding_time
        total_linking_time += linking_time
        if execution_time < 100:
            total_execution_time += execution_time
            total_query_selection_time += query_selection_time
            total_query_execution_time += (execution_time - query_selection_time)
        total_num_queries_executed += num_queries_executed

        text = colored(f'[DONE!! in {et - st:.2f} SECs]', 'green', attrs=['bold', 'reverse', 'blink', 'dark'])
        cprint(f"== {text} ==")

    text1 = colored(f'total_time = [{total_time:.2f} sec]', 'yellow', attrs=['reverse', 'blink'])
    text2 = colored(f'avg time = [{total_time / qc:.2f} sec]', 'yellow', attrs=['reverse', 'blink'])
    cprint(f"== QALD 9 Statistics : {qc} questions, Total Time == {text1}, Average Time == {text2} ")
    cprint(f"== Understanding : {qc} questions, Total Time == {total_understanding_time}, Average Time == {(total_understanding_time / qc)*1000} ms")
    cprint(f"== Linking : {qc} questions, Total Time == {total_linking_time}, Average Time == {(total_linking_time / qc)*1000} ms")
    cprint(f"== Execution : {qc} questions, Total Time == {total_execution_time}, Average Time == {(total_execution_time / qc)*1000} ms")
    cprint(f"== Query Selection : {qc} questions, Total Time == {total_query_selection_time}, Average Time == {(total_query_selection_time / qc)*1000} ms")
    cprint(f"== Query Execution : {qc} questions, Total Time == {total_query_execution_time}, Average Time == {(total_query_execution_time / qc)*1000} ms")
    cprint(f"== Queries Executed : {qc} questions, Total Number == {total_num_queries_executed}, Average Number == {(total_num_queries_executed / qc)}")

    response_time = [{
        "Question Understanding": (total_understanding_time / qc) * 1000,
        "Linking": (total_linking_time / qc) * 1000,
        "Execution": (total_execution_time / qc) * 1000,
        "Query Selection": (total_query_selection_time / qc) * 1000,
        "Query Execution": (total_query_execution_time / qc) * 1000,
        "Number of queries": total_num_queries_executed / qc
    }]

    with open(os.path.join(file_dir, f'output/{llm_name}/wikidata3.json'), encoding='utf-8', mode='w') as rfobj:
        json.dump(chattykg_qald9, rfobj)
        rfobj.write('\n')

    field_names = response_time[0].keys()
    with open(os.path.join(file_dir, f'output/{llm_name}/wikidata_response_time_ms.csv'), mode='w', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=field_names)
        writer.writeheader()
        writer.writerows(response_time)
