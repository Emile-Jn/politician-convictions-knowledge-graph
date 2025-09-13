import torch
from transformers import pipeline, AutoModelForSequenceClassification, AutoTokenizer
import json
from itertools import islice
from time import time
from urllib.parse import unquote

# global variable needed
if torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")


def convicted_yes_no(premise: str, name: str, nli_model, tokenizer) -> float:
    """
    Classify whether an extract states that the person was convicted of a crime.
    Returns 'Yes', 'No', or 'Unclear'.
    """

    # NLI setup: premise = extract, hypothesis = conviction statement
    hypothesis = f"La personne dont parle cet extrait, {name}, a été condamnée par la justice."

    # tokenize and run through model
    x = tokenizer.encode(premise, hypothesis, return_tensors="pt", truncation=True, max_length=512).to(device)
    with torch.no_grad():
        logits = nli_model(x)[0]

    # we throw away "neutral" (dim 1) and take the probability of
    # "entailment" (0) as the probability of the label being true
    entail_contradiction_logits = logits[:,::2]
    probs = entail_contradiction_logits.softmax(dim=1)
    prob_label_is_true = probs[:,0]
    # print(f'Type(prob_label_is_true) = {type(prob_label_is_true)}')
    # print(f'Type(prob_label_is_true)[0] = {type(prob_label_is_true)}')
    return prob_label_is_true[0].item() # probability p that the statement is true; 0 < p < 1

def chunk_paragraphs(extract: str) -> list[str]:
    """
    Split extract into paragraphs, while making sure that each paragraph is max 300 words long.
    If a paragraph is too long, break it down into smaller chunks, remove it from the list
    of paragraphs, and add the chunks.
    :param extract: A Wikipedia extract
    :return: a list of paragraphs
    """
    paragraphs = extract.split('\n')
    new_paragraphs = []
    paragraphs_to_remove = []
    for i in range(len(paragraphs)): # for each paragraph
        paragraph = paragraphs[i]
        if len(paragraph.split()) > 300: # if there are roughly more than 300 words
            sentences = paragraph.split('. ')
            sub_paragraphs = [sentences[0]]
            word_length = len(sentences[0].split())
            for sentence in sentences[1:]:
                sentence_length = len(sentence.split())
                if word_length + sentence_length > 300:  # sentence belongs in next sub-paragraph
                    sub_paragraphs.append(sentence)
                    word_length = sentence_length
                else:
                    sub_paragraphs[-1] += '. ' + sentence
                    word_length += sentence_length
            paragraphs_to_remove.append(i)
            new_paragraphs.extend(sub_paragraphs) # add sub-paragraphs
    # remove paragraphs
    for i in sorted(paragraphs_to_remove, reverse=True):
        paragraphs.pop(i)
    paragraphs.extend(new_paragraphs) # add new paragraphs
    return paragraphs

def remove_extract_title(extract: str) -> str:
    # Wikipedia sections follow the pattern "\n== Title ==\n Content..."
    return extract.split('\n', 2)[-1]

def process_politician(name: str, extracts: list[str], nli_model, tokenizer) -> list[float]:
    """
    For a politician, analyse each extract paragraph by paragraph to determine the
    probability of conviction.
    :param name: name of the politician
    :param extracts: list of extracts
    :return: list of probabilities that the person was convicted, for each paragraph
    """
    probs = []
    for extract in extracts:
        paragraphs = chunk_paragraphs(remove_extract_title(extract))
        for paragraph in paragraphs:
            prob = convicted_yes_no(paragraph, name, nli_model, tokenizer)
            probs.append(prob)
            if prob > 0.98: # if it is very likely that the person was convicted
                return probs # finish here and ignore all following paragraphs and extracts
    return probs


def process_all(data, batch_size=100, output_file="conviction_probabilities.json"):
    # load CamemBERT NLI model and tokenizer
    nli_model = AutoModelForSequenceClassification.from_pretrained("mtheo/camembert-base-xnli").to(device)
    tokenizer = AutoTokenizer.from_pretrained("mtheo/camembert-base-xnli")

    probs = {}

    # turn dict items into an iterator
    it = iter(data.items())
    batch_num = 0

    while True:
        start = time()
        # grab the next batch
        batch = list(islice(it, batch_size))
        if not batch:
            break  # no more data

        batch_num += 1
        print(f"Processing batch {batch_num} with {len(batch)} items...")

        for key, extracts in batch:
            name = unquote(key) # e.g. Andr%C3%A9_Halbout to André Halbout
            probs[key] = process_politician(unquote(key), extracts, nli_model, tokenizer)

        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(probs, f, ensure_ascii=False, indent=2)
        end = time()
        elapsed = end - start
        hours = elapsed // 3600
        minutes = (elapsed % 3600) // 60
        seconds = elapsed % 60
        print(f'done in {hours}:{minutes}:{seconds:.2f}, saved.')

    return probs

def test_nli_model():
    # 🔹 Example usage
    extracts = [
        "Jean Dupont a été reconnu coupable de fraude fiscale en 2018.",
        "Marie Martin a été accusée mais finalement acquittée.",
        "Pierre Durand est soupçonné d'avoir participé à un vol.",
        "Claire Bernard n'a jamais été impliquée dans une affaire judiciaire.",
        "François Lefevre a été condamné à une amende pour corruption en 2020."
    ]
    for text in extracts:
        print(text, "→", convicted_yes_no(text, ' '.join(text.split()[:2])))

def test_json_file():
    # read json file
    with open('../data/convictions.json', 'r', encoding='utf-8') as f:
        data = json.load(f)
    for key, value in list(data.items())[:5]:
        print(key)
        print(value[0][:15])

if __name__ == '__main__':
    # test_nli_model()
    # test_json_file()
    with open('../data/convictions.json', 'r', encoding='utf-8') as f:
        data = json.load(f)
    probs = process_all(data)
