# filename: test_wikidata_query.py
import requests

# Wikidata SPARQL endpoint
ENDPOINT_URL = "https://query.wikidata.org/sparql"

# SPARQL query (fixed version)
# Using a real example: searching for "Albert Einstein"
QUERY = """
SELECT DISTINCT ?uri ?label WHERE {
  SERVICE wikibase:mwapi {
    bd:serviceParam wikibase:endpoint "www.wikidata.org";
                    wikibase:api "EntitySearch";
                    mwapi:search "Albert Einstein";
                    mwapi:language "en" .
    ?uri wikibase:apiOutputItem mwapi:item .
    ?label wikibase:apiOutputItem mwapi:label .
  }
}
LIMIT 10
"""

def run_query(endpoint_url: str, query: str):
    headers = {
        "Accept": "application/sparql-results+json"
    }
    response = requests.get(endpoint_url, params={"query": query}, headers=headers)
    if response.status_code != 200:
        raise Exception(f"Query failed: {response.status_code}\n{response.text}")
    return response.json()

if __name__ == "__main__":
    try:
        results = run_query(ENDPOINT_URL, QUERY)
        print("Results:")
        for row in results["results"]["bindings"]:
            uri = row["uri"]["value"]
            label = row["label"]["value"]
            print(f"{label}: {uri}")
    except Exception as e:
        print(f"Error running query: {e}")
