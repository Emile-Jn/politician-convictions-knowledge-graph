"""
this file uses the results from extraction_analysis.py to make a file with a binary
classification ground truth for each politician
"""

import pandas as pd
import json
from rapidfuzz import fuzz, process
from tqdm import tqdm

from source.utils import get_root_dir
from source.extraction_analysis import get_highest_probs

def clean_prompt_answers():
    """
    Remove artefacts like ' : ' from the prompt answers file.
    :return:
    """
    with open(get_root_dir() / 'data' / 'prompt_answers.txt', 'r', encoding='utf-8') as f:
        lines = f.readlines()
    cleaned_lines = [line.replace(' : ', ' ')
                         .replace(': ', ' ')
                         .replace(' — ', ' ')
                         .replace('’', '\'') for line in lines]
    with open(get_root_dir() / 'data' / 'prompt_answers_cleaned.txt', 'w', encoding='utf-8') as f:
        f.writelines(cleaned_lines)

def probs_to_booleans(highest_probs: dict, lower: float = 0.7, upper: float = 0.98) -> dict:
    return {
        name: True if prob >= upper else False if prob < lower else None
        for name, prob in highest_probs.items()
    }

def check_labels(data: dict):
    """
    Count how many labels are True, False, None
    :param data: {name: bool or None}
    :return:
    """
    true_count = sum(1 for v in data.values() if v is True)
    false_count = sum(1 for v in data.values() if v is False)
    none_count = sum(1 for v in data.values() if v is None)
    print(f'True: {true_count}, False: {false_count}, None: {none_count}')
    return true_count, false_count, none_count

# This function is only used because some names could not be matched between
# prompt_answers_cleaned and graph_data.parquet
def remove_Nones(data: dict) -> dict:
    return {name: label for name, label in data.items() if label is not None}

def get_name_link_dict():
    link_df = pd.read_csv(get_root_dir() / 'data' / 'french_politicians_with_wikipedia.csv')
    name_dict = {row['French Wikipedia Link'].split('/')[-1]: row['Name'] for _, row in link_df.iterrows()}
    return name_dict


def make_class_df():
    with open(get_root_dir() / 'data' / 'conviction_probabilities.json', 'r') as f:
        probs = json.load(f)
    # Get the highest probability of conviction for each politician
    highest_probs = get_highest_probs(probs)

    name_dict = get_name_link_dict()
    # Clean names to real title, e.g. 'Andr%C3%A9_Halbout' -> 'André Halbout'
    highest_probs = {name_dict[name] : prob for name, prob in highest_probs.items()}

    # Convert probabilities to booleans with thresholds
    data = probs_to_booleans(highest_probs)
    with open(get_root_dir() / 'data' / 'prompt_answers_cleaned.txt', 'r', encoding='utf-8') as f:
        prompt_answers = f.readlines()
    wrong_names = 0
    homonyms = 0
    missing_names = 0
    wrong_labels = 0
    for line in prompt_answers:
        parts = line.split(' ')
        name = ' '.join(parts[:-1]).strip()
        if name == "Zéna M'Déré": # Special case, apostrophe issue
            data['Zéna M’Déré'] = True
            continue
        answer = parts[-1].strip()
        # matching_names = [key for key, value in data.items() if (name in key or key in name) and value is None]
        matching_names = [
            key for key, value in data.items()
            if value is None and (fuzz.ratio(name, key) >= 86 or key in name or name in key)
        ]
        if len(matching_names) == 0:
            missing_names += 1
        elif len(matching_names) == 1:
            data[matching_names[0]] = {'OUI': True, 'NON': False}.get(answer, None)
        else:
            homonyms += 1
            print(f'HOMONYMS for {name}: {matching_names}')
    print(f'wrong names: {wrong_names}, wrong labels: {wrong_labels}')
    check_labels(data) # Count and print how many True, False, None
    data = remove_Nones(data) # Remove names with None labels
    return data

def inspect_ground_truth():
    with open(get_root_dir() / 'data' / 'ground_truth.json', 'r', encoding='utf-8') as f:
        class_df = json.load(f)
    for name, label in class_df.items():
        if label is None:
            print(f'{name}: {label}')

def main():
    class_df = make_class_df()
    with open(get_root_dir() / 'data' / 'ground_truth.json', 'w', encoding='utf-8') as f:
        json.dump(class_df, f, ensure_ascii=False, indent=4)

if __name__ == '__main__':
    # inspect_ground_truth() # Uncomment to print which names have None labels
    # clean_prompt_answers() # Uncomment to get a new clean prompt_answers_cleaned file
    main()