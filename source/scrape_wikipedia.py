import re
import requests
from urllib.parse import unquote
import json
from tqdm import tqdm
import os
import pandas as pd
import time

# List of truncated keywords for regex, which may indicate a conviction of the person
# mentioned. This list is used to filter wikipedia pages, so that only those likely to
# contain information about convictions are kept. The keywords are in French, since the
# targeted people, and their respective Wikipedia pages, are French.

keywords = [
    r"condamn\w*",      # condamné, condamnation
    r"reconnu coupable",
    r"jug\w*",          # jugé, jugement
    r"inculp\w*",       # inculpé, inculpation
    r"relax\w*",        # relaxé, relaxation
    r"acquitt\w*",      # acquitté, acquittement
    r"poursuiv\w*",     # poursuivi
    r"mis en examen",
    r"peine\w*",        # peine, peines
    r"sanction\w*",     # sanction, sanctions
    r"amende\w*",       # amende, amendes
    r"prison\w*",       # prison, emprisonnement
    r"détention",
    r"réclusion",
    r"tribunal\w*",
    r"cour d’appel",
    r"juridiction\w*",
    r"procès",
    r"infraction\w*",
    r"délit\w*",
    r"crime\w*",
    r"accusation\w*",
    r"verdict",
    r"pénal\w*",
    r"corruption",
    r"fraud\w*",        # fraude, frauduleux
    r"abus de biens sociaux",
    r"trafic d’influence",
    r"détournement",
    r"malversation",
    r"escroquerie",
    r"recel",
    r"blanchiment",
]

# Compile the keywords into a regex pattern for matching
pattern = re.compile(r"|".join(keywords), re.IGNORECASE)

def filter_paragraphs(text):
    paragraphs = text.split("\n\n")  # Paragraphs are usually split by double newlines
    matches = [p for p in paragraphs if pattern.search(p)]
    return matches


def fetch_wikipedia_page(link):
    """Fetch raw wikitext from Wikipedia API."""
    title = link.split("/")[-1]  # Extract the title from the URL
    title = unquote(title)  # Decode URL-encoded characters (e.g., %C3%A9 to é)
    url = f"https://fr.wikipedia.org/w/api.php"
    params = {
        "action": "query",
        "format": "json",
        "prop": "extracts",
        "explaintext": True,
        "titles": title
    }
    r = requests.get(url, params=params)
    data = r.json()
    pages = data.get("query", {}).get("pages", {})
    if not pages:
        return None
    page = next(iter(pages.values()))
    return page.get("extract")

def scrape_and_save(url_list, output_file):
    """
    Iterate through a list of Wikipedia URLs, extract paragraphs containing conviction-related keywords,
    and save them to a JSON file in the format:
    { "Title1": ["paragraph1", "paragraph2", ...], "Title2": [...], ... }
    """
    results = {}
    pages_not_found = 0
    pages_without_keywords = 0

    for link in tqdm(url_list):
        print(f"Processing: {link}")
        text = fetch_wikipedia_page(link)
        if not text:
            pages_not_found += 1
            continue

        matches = filter_paragraphs(text)

        title = link.split("/")[-1]  # Wikipedia title
        # title = unquote(title)

        if matches:
            results[title] = matches
        else:
            pages_without_keywords += 1
        time.sleep(1)  # wait 1 second to avoid overloading the Wikipedia API

    # Save to JSON
    save_json_append(results, output_file)

    print(f'Pages not found: {pages_not_found}/{len(url_list)}')
    print(f'Pages without conviction keywords: {pages_without_keywords}/{len(url_list)}')
    print(f"Saved results to {output_file}")


def save_json_append(new_results, output_file):
    # Step 1: Load existing data if file exists
    if os.path.exists(output_file):
        with open(output_file, "r", encoding="utf-8") as f:
            try:
                existing_data = json.load(f)
            except json.JSONDecodeError:
                existing_data = {}
    else:
        existing_data = {}

    # Step 2: Merge
    existing_data.update(new_results)

    # Step 3: Save
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(existing_data, f, ensure_ascii=False, indent=2)

def batch_loop_csv(csv_file, batch_size=200):
    """
    Read a CSV file with Wikipedia links and process them in batches
    The CSV file should have a column named 'French Wikipedia Link'.
    """
    df = pd.read_csv(csv_file)
    links = df['French Wikipedia Link'].dropna().tolist()

    for i in range(0, len(links), batch_size):
        end = max(i + batch_size, len(links))
        scrape_and_save(links[i:end], "../data/convictions.json")

if __name__ == "__main__":
    #%% Test
    # test_link = "https://fr.wikipedia.org/wiki/Nicolas_Sarkozy"
    # test_text = fetch_wikipedia_page(test_link)

    #%% Example usage:
    # url_list = [
    #     "https://fr.wikipedia.org/wiki/Nicolas_Sarkozy",
    #     "https://fr.wikipedia.org/wiki/Jacques_Chirac"
    # ]
    #
    # scrape_and_save(url_list, "../data/convictions.json")
    batch_loop_csv("../data/french_politicians_with_wikipedia.csv")
