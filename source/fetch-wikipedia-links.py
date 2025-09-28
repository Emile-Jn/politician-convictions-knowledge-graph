# 12/08/2025
# Code mostly from ChatGPT

import requests
import pandas as pd
from source.utils import get_root_dir

# SPARQL query to get all French politicians born after 1900
sparql_query = """
SELECT ?person ?personLabel ?birthDate ?frwiki
WHERE {
  ?person wdt:P31 wd:Q5;              # instance of human
          wdt:P106 wd:Q82955;         # occupation: politician
          wdt:P27 wd:Q142;            # nationality: France
          wdt:P569 ?birthDate.        # date of birth
  FILTER(?birthDate > "1900-01-01T00:00:00Z"^^xsd:dateTime)

  # Get the French Wikipedia link
  OPTIONAL {
    ?frwiki schema:about ?person ;
            schema:inLanguage "fr" ;
            schema:isPartOf <https://fr.wikipedia.org/> .
  }

  SERVICE wikibase:label {
    bd:serviceParam wikibase:language "en".
  }
}
ORDER BY ?birthDate
"""


def submit_query(sparql_query: str) -> dict:
    """
    Submit a SPARQL query to the Wikidata endpoint and return the results as a JSON dictionary.
    :param sparql_query: The SPARQL query string.
    :return: The results as a JSON dictionary.
    """
    # Send request to Wikidata SPARQL endpoint
    url = "https://query.wikidata.org/sparql"
    headers = {"User-Agent": "PoliticianFetcher/1.0 (example@example.com)"}
    print('Fetching data from Wikidata...')
    r = requests.get(url, params={"query": sparql_query, "format": "json"}, headers=headers)
    print('Data fetched successfully.\n')
    r.raise_for_status()

    # Parse results
    results = r.json()["results"]["bindings"]
    return results

def extract_name_year_link(results: dict) -> pd.DataFrame:
    """
    Extract names, birth years, and French Wikipedia links from the SPARQL results.
    :param results: The SPARQL results as a JSON dictionary.
    :return: A DataFrame with columns for Name, Birth Year, and French Wikipedia Link.
    """
    data = []
    for row in results:
        name = row["personLabel"]["value"]
        birth_year = row["birthDate"]["value"][:4]
        frwiki_link = row.get("frwiki", {}).get("value", "")
        data.append([name, birth_year, frwiki_link])

    # Make a DataFrame
    df = pd.DataFrame(data, columns=["Name", "Birth Year", "French Wikipedia Link"])
    return df

def extract_QID_name(results: dict) -> pd.DataFrame:
    """
    Extract QID and names from the SPARQL results.
    :param results: The SPARQL results as a JSON dictionary.
    :return: A DataFrame with columns for QID and Name.
    """
    data = []
    for row in results:
        qid = row["person"]["value"].split("/")[-1]
        name = row["personLabel"]["value"]
        data.append([qid, name])

    # Make a DataFrame
    df = pd.DataFrame(data, columns=["QID", "Name"])
    return df

def reduce_dataframe(df):
    # Make a reduced dataframe with only people who have a wikipedia page
    df_reduced = df[df["French Wikipedia Link"] != ""].reset_index(drop=True)
    print(f'Dataframe reduced to {len(df_reduced)} politicians with a Wikipedia page.')

    # Make sure the links are actual wikipedia links
    df_reduced = df_reduced[df_reduced["French Wikipedia Link"].str.contains("https://fr.wikipedia.org/wiki/", regex=False)].reset_index(drop=True)
    print(f'Dataframe filtered to {len(df_reduced)} to only include valid Wikipedia links.')

    # Save the reduced DataFrame to a new CSV
    df_reduced.to_csv("french_politicians_with_wikipedia.csv", index=False, encoding="utf-8")

if __name__ == "__main__":
    results = submit_query(sparql_query)
    # df = extract_name_year_link(results)
    # df.to_csv(get_root_dir() / "data" / "french_politicians.csv", index=False, encoding="utf-8")
    # reduce_dataframe(df)

    df_qid = extract_QID_name(results)
    df_qid.to_csv(get_root_dir() / "data" / "french_politicians_qid.csv", index=False, encoding="utf-8")
