import pandas as pd
from pathlib import Path
from source.utils import get_root_dir
import requests
import json

def make_QID_batches(qids: pd.DataFrame, batch_size: int = 200):
    qid_col = qids["QID"]



# For each politician, get a list of all wikidata relations
def get_all_relations(qids: list) -> list:
    relations = []
    if type(qids) == str:
        qids = [qids]
    for qid in qids:
        # Fetch relations from Wikidata
        sparql_query = f"""
        SELECT ?property ?propertyLabel ?value ?valueLabel WHERE {{
          wd:{qid} ?p ?value .
          ?property wikibase:directClaim ?p .
          SERVICE wikibase:label {{ bd:serviceParam wikibase:language "[AUTO_LANGUAGE],en". }}
        }}
        """
        # Reusing code from fetch-wikipedia-links.py:

        # Send request to Wikidata SPARQL endpoint
        url = "https://query.wikidata.org/sparql"
        headers = {"User-Agent": "PoliticianFetcher/1.0 (example@example.com)"}
        print('Fetching data from Wikidata...')
        r = requests.get(url, params={"query": sparql_query, "format": "json"}, headers=headers)
        print('Data fetched successfully.\n')
        r.raise_for_status()

        # Parse results
        results = r.json()["results"]["bindings"]
        relations.append(results)
    return relations

def save_relations(relations, output_path: Path):
    """Save relations to a file"""
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(relations, f, ensure_ascii=False, indent=4)
    print(f'Relations saved to {output_path}')

def print_relation_set(relations):
    rel_set = set()
    for i, rel in enumerate(relations):
        print(f'Relations for politician {i+1}:')
        for item in rel:
            prop = item['propertyLabel']['value']
            link = item['property']['value']
            Pid = link.split('/')[-1]
            # value = item['valueLabel']['value']
            rel_set.add((prop, Pid))
    for prop, Pid in rel_set:
        print(f'{Pid} {prop}')
    print(f'Total unique relations: {len(rel_set)}')

if __name__ == "__main__":
    # Use the result of the original query for French politicians born after 1900:
    # politicians = pd.read_csv(get_root_dir() / "data" / "french_politicians.csv")

    # Example QIDs for testing
    qids = ["Q2105", "Q2038", "Q12940", "Q329"] # Chirac, Mitterrand, Chaban-Delmas, Sarkozy
    relations = get_all_relations(qids)

    print_relation_set(relations)

    # relations = get_all_relations(politicians)
    # save_relations(relations, get_root_dir() / "data")