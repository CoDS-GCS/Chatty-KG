from typing import Any

from chattykg import sparqls
import chattykg.embeddings_client as w2v
import operator
from chattykg.nlp.utils import remove_duplicates
from itertools import product
import networkx as nx
from urllib.parse import urlparse
import datetime

import re
import os


def is_variable(label):
    return label.startswith('?')

def get_entity_query_for_kg(entity, knowledge_graph, n_limit_VQuery):
    if knowledge_graph in ["microsoft_academic", "bgee"]:
        entity_query = sparqls.make_Ms_academic_query(
            entity, limit=n_limit_VQuery
        )
    elif knowledge_graph in ['yago']:
        entity_query = sparqls.make_keyword_unordered_search_query_with_type_yago(
            entity, limit=n_limit_VQuery)
    elif knowledge_graph in ['wikidata']:
        entity_query = sparqls.make_entity_search_query_wikidata(
            entity, limit=n_limit_VQuery)
    else:
        entity_query = sparqls.make_keyword_unordered_search_query_with_type(
            entity, limit=n_limit_VQuery
        )
    return entity_query


def __get_chosen_URIs_for_relation(relation: str, uris: list, names: list, n_max_Es: int) -> list | tuple[
    list, list]:
    if not uris:
        return uris, []

    scores = __compute_semantic_similarity_between_single_word_and_word_list(
        relation, names
    )
    # (uri, vetrex, True) ===>  (uri, vertex, True, score)
    l1, l2, l3 = list(zip(*uris))
    URIs_with_scores = list(zip(l1, l2, l3, scores))
    URIs_with_scores.sort(key=operator.itemgetter(3), reverse=True)
    chosen = remove_duplicates(URIs_with_scores)[: n_max_Es]

    # Get names for the chosen URIs based on original index
    chosen_names = [names[l1.index(uri)] for uri, _, _, _ in chosen]

    return chosen, chosen_names
    # return remove_duplicates(URIs_with_scores)[:n_max_Es]


def __compute_semantic_similarity_between_single_word_and_word_list(
    word, word_list
):
    scores = list()
    score = 0.0
    for w in word_list:
        try:
            score = w2v.n_similarity(word.lower().split(), w.lower().split())
        except KeyError:
            score = 0.0
        finally:
            scores.append(score)
    else:
        return scores


def get_all_possible_triples_for_edge(edge, first_uris, second_uris):
    possible_triples = list()
    for predicate, vertex, orientation, score in edge:
        if orientation:
            # vertex is object
            object_uris = [vertex]
            if vertex in first_uris:
                subject_uris = second_uris
            elif vertex in second_uris:
                subject_uris = first_uris
        else:
            # vertex is subject
            subject_uris = [vertex]
            if vertex in first_uris:
                object_uris = second_uris
            elif vertex in second_uris:
                object_uris = first_uris
        triples = list(product(subject_uris, [(predicate, score)], object_uris))
        possible_triples.extend(triples)

    # This means that we have a triple of this structure (var1 ?p var2)
    if (
        len(edge) == 0
        and len(first_uris) > 0
        and len(second_uris)
        and is_variable(first_uris[0])
        and is_variable(second_uris[0])
    ):
        possible_triples.append((first_uris[0], ("?p", 0), second_uris[0]))
        possible_triples.append((second_uris[0], ("?p", 0), first_uris[0]))

    return possible_triples

def check_validity(triples):
    if (
        len(triples) == 1
        and len(triples[0]) == 2
        and is_variable(triples[0][0][0])
        and is_variable(triples[0][0][2])
        and is_variable(triples[0][1][0])
        and is_variable(triples[0][1][2])
    ):
        print("In condition")
        return False
    return True


def merge_tuples(product, tuple_list):
    final_list = list()
    for instance in product:
        for item in tuple_list:
            temp = instance
            temp = temp + (item,)
            final_list.append(temp)
    return final_list

def get_possible_combinations(query_graph):
    edges = list(nx.dfs_edges(query_graph))
    bgps = []
    handled_edges = 0
    for i in range(0, len(edges)):
        current_triples = query_graph[edges[i][0]][edges[i][1]][0][
            "possible_triples"
        ]
        if len(current_triples) == 0:
            continue

        if handled_edges == 0:
            bgps = query_graph[edges[i][0]][edges[i][1]][0][
                "possible_triples"
            ]
            handled_edges += 1
            continue

        connected_node = set(edges[i]).intersection(set(edges[i - 1]))
        if len(connected_node) == 0:
            continue
        connected_node = next(iter(connected_node))
        if is_variable(connected_node):
            if len(bgps) == 0:
                bgps = current_triples
            elif handled_edges == 1:
                bgps = product(bgps, current_triples)
                bgps = list(bgps)
            else:
                bgps = merge_tuples(bgps, current_triples)

        else:
            connected_node_vertices = query_graph.nodes[
                connected_node
            ]["uris"]
            updated_bgps = []
            for triple in current_triples:
                used_vertex_triple = (
                    triple[0] if triple[0] in connected_node_vertices else triple[2]
                )
                for bgp in bgps:
                    last_bgp_triple = bgp[len(bgp) - 1]
                    used_bgp_vertex = (
                        last_bgp_triple[0]
                        if last_bgp_triple[0] in connected_node_vertices
                        else last_bgp_triple[2]
                    )
                    if used_vertex_triple == used_bgp_vertex:
                        temp = bgp.append(triple)
                        updated_bgps.append(temp)
            bgps = updated_bgps
        handled_edges += 1

    if handled_edges == 1:
        bgps = list(product(bgps))
    return bgps


def calculate_score(star_query):
    score = 0.0
    score_count = 0
    for q in star_query:
        if len(q) == 2:
            score += q[1][2]
            score_count += 1
        elif len(q) == 3:
            score += q[1][1]
            score_count += 1
    return score / score_count if score_count > 0 else score


def extract_resource_name_from_uri(uri: str):
    resource_URI = uri
    uri_path = urlparse(resource_URI).path
    resource_name = os.path.basename(uri_path)
    resource_name = re.sub(r"(:|_|\(|\))", " ", resource_name)
    return resource_URI, resource_name

def postprocess_answer_if_needed(v_result, target_variable):
    target = target_variable[1: ] if target_variable.startswith('?') else target_variable
    for binding in v_result["results"]["bindings"]:
        if "datatype" in binding[target]:
            if "gYear" in binding[target]["datatype"]:
                if int(binding[target]["value"]) > 0:
                    obj = datetime.datetime.strptime(binding[target]["value"], "%Y")
                    binding[target]["value"] = str(obj.date())
    return v_result