import pandas as pd
from pathlib import Path
from source.utils import get_root_dir
import requests
import json
from tqdm import tqdm

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

def get_sample_relations():
    # Example QIDs for testing
    qids = ["Q2105", "Q2038", "Q12940", "Q329"]  # Chirac, Mitterrand, Chaban-Delmas, Sarkozy
    relations = get_all_relations(qids)
    print_relation_set(relations)

# - now that a list of useful relations has been established, get these relations for each politician

def make_QID_batches(politicians: pd.DataFrame, batch_size: int = 200, start: int = 0):
    qids = politicians["QID"].tolist()
    for i in range(start, len(qids), batch_size):
        yield i, qids[i:i + batch_size]

def load_useful_PIDs():
    # Load list of useful PIDs
    with open(get_root_dir() / 'data' / 'useful_relations.txt', 'r') as f:
        useful_Pids = [line.split(' ', maxsplit=1)[0] for line in f if line.strip()]
    return useful_Pids

def get_useful_relations(qids: list, useful_Pids: list, lang: str = "en") -> list:
    """
    Fetch relations for a list of persons (QIDs) with given properties (PIDs) from Wikidata.
    Returns QIDs and their labels (personLabel, propertyLabel, valueLabel).
    """
    qid_values = " ".join(f"wd:{qid}" for qid in qids)
    pid_values = " ".join(f"wdt:{pid}" for pid in useful_Pids)

    sparql_query = f"""
    SELECT ?person ?personLabel ?property ?propertyEntity ?propertyEntityLabel ?value ?valueLabel WHERE {{
      VALUES ?person {{ {qid_values} }}
      VALUES ?property {{ {pid_values} }}
      ?person ?property ?value .

      # Turn the direct-property IRI (prop/direct/Pxx) into the property entity IRI (entity/Pxx)
      BIND(IRI(CONCAT("http://www.wikidata.org/entity/", STRAFTER(STR(?property), "http://www.wikidata.org/prop/direct/"))) AS ?propertyEntity)

      SERVICE wikibase:label {{ bd:serviceParam wikibase:language "{lang}" . }}
    }}
    """

    url = "https://query.wikidata.org/sparql"
    headers = {"User-Agent": "PoliticianFetcher/1.0 (example@example.com)"}
    r = requests.get(url, params={"query": sparql_query, "format": "json"}, headers=headers)
    r.raise_for_status()

    bindings = r.json()["results"]["bindings"]
    results = []
    for b in bindings:
        person_qid = b["person"]["value"].split("/")[-1]
        person_label = b.get("personLabel", {}).get("value")
        # property comes back as http://www.wikidata.org/prop/direct/P19 -> extract 'P19'
        property_pid = b["property"]["value"].split("/")[-1]
        # the correct human-readable property label is supplied in propertyEntityLabel
        property_label = b.get("propertyEntityLabel", {}).get("value")
        # value might be a URI (entity) or a literal (string/date/number)
        if b["value"]["type"] == "uri":
            value = b["value"]["value"].split("/")[-1]
        else:
            value = b["value"]["value"]
        value_label = b.get("valueLabel", {}).get("value")

        results.append({
            "person": person_qid,
            "personLabel": person_label,
            "property": property_pid,
            "propertyLabel": property_label,
            "value": value,
            "valueLabel": value_label
        })

    return results

def test_relations():
    test_qids = ["Q2105", "Q2038", "Q12940", "Q329"]
    useful_Pids = load_useful_PIDs()
    results = get_useful_relations(test_qids, useful_Pids)
    # Save as JSONL
    with open(get_root_dir() / 'data' / f'politician_relations_test.jsonl', 'w', encoding='utf-8') as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + '\n')

def main():
    """
    For each politician in the list, get their useful relations and save them to a JSONL file.
    Takes about 25 mins on a Macbook M1.
    :return: nothing
    """
    politicians = pd.read_csv(get_root_dir() / 'data' / 'french_politicians_qid.csv')
    useful_Pids = load_useful_PIDs()
    for i, qids in tqdm(make_QID_batches(politicians)):
        results = get_useful_relations(qids, useful_Pids)
        # Save as JSONL
        with open(get_root_dir() / 'data' / 'relations' / f'politician_relations_{i}.jsonl', 'w',
                  encoding='utf-8') as f:
            for r in results:
                f.write(json.dumps(r, ensure_ascii=False) + '\n')


if __name__ == "__main__":
    # get_sample_relations()
    # test_relations()
    main()