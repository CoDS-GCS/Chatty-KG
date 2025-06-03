from chatbot.prompts import CLASSIFY_QUESTION_PROMPT_2, CONDENSE_QUESTION_PROMPT_CUSTOM
from chattykg.question import Question
from .State import State
from . import utils
import json
from langchain_core.messages import AIMessage, HumanMessage
from chattykg.logger import logger
from termcolor import cprint
from chattykg.linking.llm_linking_v2 import vertex_linking
from chattykg.vertex import Vertex
from chattykg.filtration.llm_filtrationv2 import choose_question_from_keywords
import SPARQLBurger.SPARQLQueryBuilder as SparqlQB




# Input: Question
# Output: q_type: <non-self-contained, self-contained>
def classify_question(question, state: State):
    classify_question_chain = CLASSIFY_QUESTION_PROMPT_2 | state.get_chatbot_llm() | state.get_string_output_parser()
    q_type = classify_question_chain.invoke(
        {"question": question},
    )
    return q_type


# Input Question
# Output Rephrased question
# Requires handling of chat history
#  Session id should be unique per dialogue
def rephrase_question(session_id, question, state: State):
    rephrase_question_chain = CONDENSE_QUESTION_PROMPT_CUSTOM | state.get_chatbot_llm() | state.get_string_output_parser()
    chat_history = state.get_by_session_id(session_id)
    question = rephrase_question_chain.invoke({"question": question, "chat_history": chat_history})
    return question


# Input is question and question id.
#  Output is an object of type Question (chattykg.question)
def get_qir_from_question(question, question_id, state: State):
    current_question = Question(
        question_text=question, question_id=question_id, logger=logger, json_logger=state.get_json_logger()
    )
    state.set_question(current_question)
    state.detect_question_and_answer_type()
    return current_question


def perform_linking(state: State):
    query_graph = state.get_query_graph()
    for entity in query_graph:
        if utils.is_variable(entity):
            continue
        entity_query = utils.get_entity_query_for_kg(entity, state.get_knowledge_graph(), state.get_n_limit_VQuery())
        # entity_query = make_keyword_unordered_search_query_with_type(entity, limit=self.n_limit_VQuery)
        cprint(f"== SPARQL Q Find V: {entity_query}")

        try:
            uris, names = state.get_sparql_end_point().get_names_and_uris(entity_query)
        except:
            logger.log_error(
                f"Error at 'extract_possible_V_and_E' method with v_query value of {entity_query} "
            )
            # traceback.print_exc()
            continue
        if len(uris) == 0:
            continue
        elif len(uris) == 1:
            chosen_vertex_index = 0
            chosen_uri = uris
            chosen_label = names
        else:
            # chosen_vertex_index = vertex_linking(entity, names, uris, json_logger)
            chosen_uri, chosen_label = vertex_linking(entity, names, uris, state.get_json_logger(), state.get_knowledge_graph())
            # if chosen_vertex_index is None:
            if chosen_uri is None:
                continue
            # for i in chosen_vertex_index:
            #     print(uris[i])
            # chosen_vertex_index = chosen_vertex_index[0]

        # chosen_uri = [uris[chosen_vertex_index]]
        state.get_json_logger().set("Linking_vertex", chosen_uri)
        updated_vertex = Vertex(
            state.get_n_max_Vs(), chosen_uri, chosen_label, state.get_sparql_end_point(), state.get_n_limit_EQuery()
        )
        URIs_chosen = updated_vertex.get_vertex_uris()
        # URIs_chosen = remove_duplicates(URIs_sorted)[:self.n_max_Vs]
        query_graph.nodes[entity]["uris"].extend(URIs_chosen)
        query_graph.nodes[entity]["vertex"] = updated_vertex

    # Find E for all relations
    for source, destination, key, relation in query_graph.edges(
            data="relation", keys=True
    ):
        if not relation:
            continue
        source_URIs = query_graph.nodes[source]["uris"]
        destination_URIs = query_graph.nodes[destination]["uris"]
        uris, names = list(), list()
        if not utils.is_variable(source):
            for source_uri in source_URIs:
                # TODO what if source is associated with multiple vertices
                uris_source, names_source = query_graph.nodes[source][
                    "vertex"
                ].get_predicates()
                uris.extend(uris_source)
                names.extend(names_source)

        if not utils.is_variable(destination):
            for destination_uri in destination_URIs:
                # TODO what if source is associated with multiple vertices
                (
                    uris_destination,
                    names_destination,
                ) = query_graph.nodes[destination][
                    "vertex"
                ].get_predicates()
                uris.extend(uris_destination)
                names.extend(names_destination)

        URIs_chosen = utils.__get_chosen_URIs_for_relation(relation, uris, names, state.get_n_max_Es())
        # print("Edges CHosen")
        # print(URIs_chosen)
        query_graph[source][destination][key]["uris"].extend(
            URIs_chosen
        )
    else:
        logger.log_info(
            f"[GRAPH NODES WITH URIs:] {query_graph.nodes(data=True)}"
        )
        logger.log_info(
            f"[GRAPH EDGES WITH URIs:] {query_graph.edges(data=True)}"
        )


def query_selection_execution(state: State):
    generate_queries_new(state)
    evaluate_star_queries_predicate_based(state)


# Updates chat history after answering the question
def update_history(session_id, original_question, rephrased_question, state: State):
    chat_history = state.get_by_session_id(session_id)
    chat_history.add_messages([HumanMessage(original_question), AIMessage(rephrased_question)])


#------------------------------------------------------------------------ Helper functions
def generate_queries_new(state: State):
    possible_triples_for_all_relations = list()
    query_graph = state.get_query_graph()
    for source, destination, key, edge_info in query_graph.edges(
        data="uris", keys=True
    ):
        source_URIs = query_graph.nodes[source]["uris"]
        destination_URIs = query_graph.nodes[destination]["uris"]
        if not utils.is_variable(source):
            node1_uris = source_URIs
        elif source.startswith('?'):
            node1_uris = [source]
        else:
            node1_uris = ["?" + source]

        if not utils.is_variable(destination):
            node2_uris = destination_URIs
        elif destination.startswith('?'):
            node2_uris = [destination]
        else:
            node2_uris = ["?" + destination]

        possible_triples = utils.get_all_possible_triples_for_edge(
            edge_info, node1_uris, node2_uris
        )
        query_graph[source][destination][key][
            "possible_triples"
        ] = possible_triples
        if len(possible_triples) > 0:
            possible_triples_for_all_relations.append(possible_triples)
    if not utils.check_validity(possible_triples_for_all_relations):
        return
    possible_bgps = utils.get_possible_combinations(query_graph)
    for bgp in possible_bgps:
        score = utils.calculate_score(bgp)
        if len(bgp) == 0:
            continue

        query, node_uris, relation_uris, triples = generate_sparql_query_new(bgp, state)
        query = query.replace("\n", " ")
        state.add_possible_answer(query, score, node_uris, relation_uris, triples)
        # self.question.add_possible_answer(
        #     question=self.question.text, sparql=query, score=score, nodes=node_uris, edges=relation_uris,
        #     triples=triples
        #     )

def evaluate_star_queries_predicate_based(state: State):
    sparqls = list()
    sparqls_triples = list()
    for i, possible_answer in enumerate(
        state.get_answers()
            # self.question.possible_answers[: self._n_max_answers]
    ):
        sparqls.append(possible_answer.sparql)
        sparqls_triples.append(possible_answer.triples)
    if len(sparqls) == 0:
        return
    queries_indices = choose_question_from_keywords(state.get_question_text(), sparqls, sparqls_triples, state.get_json_logger())
    # self.query_selection_end = time.time()
    # self.num_queries_executed = len(queries_indices)
    for index in queries_indices:
        try:
            sparql_query = sparqls[index]
            state.append_sparql_query(sparql_query)
            #TODO: Orogat: Add the Query to the state object
            print(f"SPARQL Query: {sparql_query}")
            result = state.get_sparql_end_point().evaluate_SPARQL_query(sparql_query)
            v_result = json.loads(result)
            if "results" in v_result:
                v_result = utils.postprocess_answer_if_needed(v_result, state.get_target_variable())
                results=v_result["results"]
                vars=v_result["head"]["vars"]
                state.get_answers_at_index(index).update(
                    results=results, vars=vars
                )
                # self.question.possible_answers[index].update(
                #     results=v_result["results"], vars=v_result["head"]["vars"]
                # )
            else:
                state.get_answers_at_index(index).update(results=[], boolean=v_result["boolean"])
                # self.question.possible_answers[index].update(results=[], boolean=v_result["boolean"])
            answers = list()
            if "results" in v_result:
                target_variable = state.get_target_variable()
                target = target_variable[1:] if target_variable.startswith('?') else target_variable
                for binding in v_result["results"]["bindings"]:
                    answer = utils.extract_resource_name_from_uri(
                        binding[target]["value"]
                    )[0]
                    answers.append(answer)
                else:
                    if v_result["results"]["bindings"]:
                        logger.log_info(f"[POSSIBLE ANSWER {i}:] {answers}")
            else:
                answers.append(v_result["boolean"])
        except:
            print("Error while executing SPARQL query: ", sparqls[index])
            # traceback.print_exc()

def generate_sparql_query_new(star_query, state):
    triples = list()
    if state.get_answer_datatype() == "boolean":
        ask_triple = []
        node1_uris = []
        node2_uris = []
        relation_uris = []
        for n1_uri, predicate, n2_uri in star_query:
            # direction was decided in generation of BGP step
            #print(predicate)
            if utils.is_variable(n1_uri):
                uri1 = n1_uri
            else:
                uri1 = f"<{n1_uri}>"
            if utils.is_variable(n2_uri):
                uri2 = n2_uri
            elif 'http' not in n2_uri:
                uri2 = f"\"{n2_uri}\"@en"
            else:
                uri2 = f"<{n2_uri}>"
            if predicate[0] == '?p':
                p = predicate[0]
            else:
                p =  f'<{predicate[0]}>'

            ask_triple.append(f"{uri1} {p} {uri2}")
            # if predicate[1]:
            #     ask_triple.append(f"<{n2_uri}> <{predicate[0]}> <{n1_uri}>")
            # else:
            #     ask_triple.append(f"<{n1_uri}> <{predicate[0]}> <{n2_uri}>")

            node1_uris.append(n1_uri)
            node2_uris.append(n2_uri)
            relation_uris.append(predicate[0])
            triples.append([n1_uri, predicate, n2_uri])
        query = f"ASK {{ {' . '.join(ask_triple)} }}"
        # ask_query, node1_uris,node2_uris, relation_uris = self.generate_sparql_query(query)
        # ask_query = query.replace("\n", " ")
        # return query, node1_uris, node2_uris, relation_uris
        node_uris = list()
        node_uris.append(node1_uris)
        node_uris.append(node2_uris)
        return query, node_uris, relation_uris, triples
    else:
        select_query = SparqlQB.SPARQLSelectQuery()
        where_pattern = SparqlQB.SPARQLGraphPattern()
        node_uris = []
        relation_uris = []
        candidate_targets = []
        for n1_uri, predicate, n2_uri in star_query:
            triples.append([n1_uri, predicate, n2_uri])
            if utils.is_variable(n1_uri):
                uri1 = n1_uri
                candidate_targets.append(uri1)
            else:
                uri1 = f"<{n1_uri}>"
                node_uris.append(n1_uri)

            if utils.is_variable(n2_uri):
                uri2 = n2_uri
                candidate_targets.append(uri2)
            elif 'http' not in n2_uri:
                uri2 = f"\"{n2_uri}\"@en"
                node_uris.append(n2_uri)
            else:
                uri2 = f"<{n2_uri}>"
                node_uris.append(n2_uri)

            if predicate[0] == '?p':
                p = predicate[0]
            else:
                p =  f'<{predicate[0]}>'
                relation_uris.append(predicate[0])
            where_pattern.add_triples(
                triples=[SparqlQB.Triple(subject=uri1, predicate=p, object=uri2)]
            )

        candidate_targets.sort()
        target_variable = (
            candidate_targets[0] if len(candidate_targets) > 0 else "?var1"
        )
        # select_query.add_variables(variables=[self.target_variable, "?type"])
        select_query.add_variables(variables=[target_variable])
        # remove the question mark for further use
        target_variable = target_variable[1:]
        state.set_target_variable(target_variable)
        optional_pattern = SparqlQB.SPARQLGraphPattern(optional=True)
        optional_pattern.add_triples(
            triples=[
                SparqlQB.Triple(
                    subject="?var1",
                    predicate="<http://www.w3.org/1999/02/22-rdf-syntax-ns#type>",
                    object="?type",
                )
            ]
        )
        # where_pattern.add_nested_graph_pattern(optional_pattern)
        select_query.set_where_pattern(graph_pattern=where_pattern)
        return select_query.get_text(), node_uris, relation_uris, triples

