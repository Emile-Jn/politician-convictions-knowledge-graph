from source.info_extraction_wikipedia import *

def split_paragraphs(data: dict) -> dict:
    """
    Split wikipedia extracts into paragraphs of max 300 words each.
    :param data: a dictionary of politicians and their wikipedia extracts
    :return: same, but with extracts all split into paragraphs
    """
    new_data = {}
    for name, extracts in data.items():
        new_extracts = []
        for extract in extracts:
            paragraphs = chunk_paragraphs(remove_extract_title(extract))
            new_extracts.extend(paragraphs)
        new_data[name] = new_extracts
    return new_data

if __name__ == "__main__":
    with open('../data/convictions.json', 'r') as f:
        data = json.load(f)
    new_data = split_paragraphs(data)
    with open('../data/convictions_split.json', 'w', encoding='utf-8') as f:
        json.dump(new_data, f, ensure_ascii=False, indent=4)
