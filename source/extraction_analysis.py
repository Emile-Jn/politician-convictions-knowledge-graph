import json
import pandas as pd
from source.info_extraction_wikipedia import *
from urllib.parse import unquote
import os
from huggingface_hub import InferenceClient
from pathlib import Path
from source.utils import get_root_dir

def get_highest_probs(probs):
    highest_probs = {}
    empty_count = 0
    for name, p_list in probs.items():
        if p_list:  # Ensure the list is not empty
            highest_probs[name] = max(p_list)
        else:
            highest_probs[name] = None  # or some other placeholder for no data
            empty_count += 1
    print(f'empty probs: {empty_count}/{len(probs)}')
    return highest_probs

def quantiles(highest_probs, intervals=10):
    # Filter out None values
    filtered_probs = [prob for prob in highest_probs.values() if prob is not None]
    sorted_probs = sorted(filtered_probs)
    n = len(sorted_probs)
    quantile_values = []
    for i in range(intervals):
        section = [prob for prob in sorted_probs if (i / intervals) <= prob < ((i + 1) / intervals)]
        quantile_values.append(section)
    return quantile_values

def highest_paragraphs(paragraphs_dict: dict, probs_dict: dict) -> pd.DataFrame:
    """
    Return the paragraph with the highest conviction probability for each politician.
    :param paragraphs_dict: a dictionary of {name: [paragraphs]}
    :param probs_dict: a dictionary of {name: [probabilities]}
    :return:
    """
    highest_paragraphs_df = pd.DataFrame(columns=['politician', 'probability', 'paragraph'])
    for name, paragraphs in paragraphs_dict.items():
        probs = probs_dict.get(name, [])
        if paragraphs and probs:
            max_index = probs.index(max(probs))
            highest_paragraphs_df.loc[len(highest_paragraphs_df)] = {
                'politician': name,
                'probability': probs[max_index],
                'paragraph': paragraphs[max_index]
            }
        else:
            highest_paragraphs_df.loc[len(highest_paragraphs_df)] = {
                'politician': name,
                'probability': None,
                'paragraph': None
            }

    return highest_paragraphs_df

def make_ambiguous_sample(df, start=0.7, end=0.97):
    filtered_df = df[(df['probability'] >= start) & (df['probability'] <= end)]
    sample = filtered_df.sample(n=50, random_state=42)
    sample.sort_values(by='probability', ascending=False, inplace=True)
    sample_dict = {}
    for index, row in sample.iterrows():
        # print(f"Politician: {row['politician']}\nProbability: {row['probability']}\nParagraph: {row['paragraph']}\n")
        sample_dict[row['politician']] = {'probability': row['probability'], 'paragraph': row['paragraph']}
    with open(f'sample_{int(start*100)}_{int(end*100)}.json', 'w') as f:
        json.dump(sample_dict, f, ensure_ascii=False, indent=4)
def make_prompt_batches(highest_paragraphs_df: pd.DataFrame, batch_size: int = 20):
    """
    Make a batch of prompts for the LLM to process.
    :param highest_paragraphs_df: dataframe with the highest probability paragraph for
     each politician. columns: ['politician', 'probability', 'paragraph']
    :param batch_size: number of politicians in the batch
    :return:
    """
    prompts = []
    for i in range(0, len(highest_paragraphs_df), batch_size):
        batch = highest_paragraphs_df.iloc[i:i + batch_size]
        prompt = ("Pour chaque personne et chaque paragraphe ci-dessous, indique si le "
                  "paragraphe affirme que la personne en question (nommée juste avant le "
                  "paragraphe) a été condamnée par la justice, en écrivant le nom de la "
                  "personne, suivi de OUI ou NON. Ne dis rien d'autre. Les condamnations "
                  "d'autres personnes ne comptent pas.\n\n") # say if each politician was convicted or not
        for index, row in batch.iterrows():
            name = unquote(row['politician']).replace('_', ' ')
            prompt += f"Personne: {name}\nParagraphe: {row['paragraph']}\n\n"
        prompts.append(prompt)
    return prompts

def answer_prompt(prompt: str):
    """
    Use the LLM to answer the prompt.
    :param prompt: the prompt to answer
    :return: the LLM's answer
    """
    print("Initiating InferenceClient...")
    client = InferenceClient()
    print("Client initiated. Sending prompt to model...")
    completion = client.chat.completions.create(
        # model="HuggingFaceTB/SmolLM3-3B",
        model = "meta-llama/Llama-3.1-8B-Instruct",
        messages=[{"role": "user", "content": prompt}])
    print("Prompt answered.")
    return completion
    # print(completion.choices[0].message)

def write_to_txt(text: str, filename: str, folder: str = '.'):
    os.makedirs(folder, exist_ok=True)
    filepath = os.path.join(folder, f"{filename}.txt")
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(text)

if __name__ == '__main__':
    with open(get_root_dir() / 'data' / 'conviction_probabilities.json', 'r') as f:
        probs = json.load(f)
    with open(get_root_dir() / 'data' / 'convictions_split.json', 'r') as f:
        paragraphs = json.load(f)

    highest_paragraphs_df = highest_paragraphs(paragraphs, probs)
    range_70_98 = highest_paragraphs_df[(highest_paragraphs_df['probability'] >= 0.7) & (highest_paragraphs_df['probability'] < 0.98)]
    range_70_98.sort_values(by='probability', ascending=False, inplace=True)
    prompts = make_prompt_batches(range_70_98, batch_size=20)
    for i in range(len(prompts)):
        # Each prompt is made into a txt file and then copy-pasted into ChatGPT to get an answer
        write_to_txt(prompts[i], f'prompt_{i}', folder='prompts')

    # write_to_txt(prompts[0], 'prompt0', folder='prompts')
    # answer0 = answer_prompt(prompts[0])
    # print(answer0.choices[0].message)

    # with open('answer0.txt', 'w', encoding='utf-8') as f:
    #     f.write(answer0.choices[0].message.content)

    # make_ambiguous_sample(highest_paragraphs_df, start=0.5, end=0.7)
    # highest_probs = get_highest_probs(probs)

    # quantile_values = quantiles(highest_probs, intervals=10)
    # for q in quantile_values:
    #     print(len(q))

    # between_70_and_98 = 0
    # for i in range(100):
    #     slice = highest_paragraphs_df[(highest_paragraphs_df['probability'] >= i/100) & (highest_paragraphs_df['probability'] < (i+1)/100)]
    #     print(f'{i}-{i+1}%: {len(slice)}')
    #     if 70 <= i < 98:
    #         between_70_and_98 += len(slice)
    # print(f'Total between 70% and 98%: {between_70_and_98}')
