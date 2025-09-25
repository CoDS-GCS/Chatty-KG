import json
import requests
from termcolor import cprint
import chattykg.sparqls as sparqls
from chattykg.sparql_end_points.EndPoint import EndPoint


class WikidataEndPoint(EndPoint):

    def evaluate_SPARQL_query(self, query: str):
        headers = {"Accept": "application/sparql-results+json"}
        query_response = requests.get(self.link, params={"query": query}, headers=headers)
        if query_response.status_code in [414]:
            return '{"head":{"vars":[]}, "results":{"bindings": []}, "status":414 }'
        return query_response.text

    def get_names_and_uris(self, entity_query):
        """
        Run the entity query and return (uris, names) for Wikidata queries.
        Expects the query to bind ?uri and ?label as in current code paths.
        """
        url = self.link
        headers = {"Accept": "application/sparql-results+json"}
        resp = requests.get(url, params={"query": entity_query}, headers=headers)
        resp.raise_for_status()
        entity_result = resp.json()
        bindings = entity_result.get("results", {}).get("bindings", [])
        uris = [b["uri"]["value"] for b in bindings if "uri" in b]
        names = [b["label"]["value"] for b in bindings if "label" in b]
        return uris, names

    def get_predicates_and_their_names_wikidata(self, subj=None, obj=None, nlimit: int = 100):
        if subj and obj:
            q = sparqls.sparql_query_to_get_predicates_when_subj_and_obj_are_known_wikidata(
                subj, obj, limit=nlimit
            )
        elif subj:
            q = sparqls.make_top_predicates_sbj_query_wikidata(subj, limit=nlimit)
        elif obj:
            q = sparqls.make_top_predicates_obj_query_wikidata(obj, limit=nlimit)
        else:
            raise Exception

        headers = {"Accept": "application/sparql-results+json"}
        cprint(f"== SPARQL Q Predicates (Wikidata): {q}")
        resp = requests.get(self.link, params={"query": q}, headers=headers)
        resp.raise_for_status()
        result = resp.json()
        bindings = result.get("results", {}).get("bindings", [])
        uris = [b.get("p", {}).get("value") for b in bindings if "p" in b]
        names = [b.get("propLabel", {}).get("value") for b in bindings if "propLabel" in b]
        return uris, names

    def get_predicates_and_their_names(self, subj=None, obj=None, nlimit: int = 100):
        uris, names = self.get_predicates_and_their_names_wikidata(subj, obj, nlimit)
        # Filter out noisy predicates, mirroring EndPoint filtering
        escaped_names = [
            "22-rdf-syntax-ns",
            "rdf-schema",
            "owl",
            "wiki Page External Link",
            "wiki Page ID",
            "wiki Page Revision ID",
            "is Primary Topic Of",
            "subject",
            "type",
            "prov",
            "wiki Page Disambiguates",
            "wiki Page Redirects",
            "primary Topic",
            "wiki Articles",
            "hypernym",
            "aliases",
            "was Derived From",
            "label",
            "see Also",
            "comment",
            "same As",
            "different From",
            "first",
            "has identifier",
            "wikipedia",
            "wikidata",
        ]
        filtered_uris, filtered_names = [], []
        for u, n in zip(uris, names):
            if n not in escaped_names:
                filtered_uris.append(u)
                filtered_names.append(n)
        return filtered_uris, filtered_names
