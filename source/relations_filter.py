"""
This file is for filtering relations based on certain criteria
"""

from pathlib import Path
from source.utils import get_root_dir

def get_relations(file_path: str | Path):
    """load relations from the txt file"""
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    return lines

def remove_IDs(relations: list[str]):
    """Remove IDs from the relations"""
    cleaned_relations = []
    for line in relations:
        padded_line = line.strip() + ' ' # pad the line with spaces to avoid matching IDs at the start of a word
        if len(padded_line) > 25:
            print(padded_line)
        if not ' ID ' in padded_line: # if the word ID is not in the line
            cleaned_relations.append(line)
    return cleaned_relations

if __name__ == "__main__":
    relations = get_relations(get_root_dir() / "data" / "all_relations.txt")
    cleaned_relations = remove_IDs(relations)
    # Save cleaned relations to a new file
    with open(get_root_dir() / "data" / "filtered_relations.txt", 'w', encoding='utf-8') as f:
        f.writelines(cleaned_relations)