"""
This file is for converting the raw graph data from JSONL files into other formats: CSV, Parquet, pickle.
"""

import json
import pandas as pd
from pathlib import Path
from source.utils import get_root_dir
import networkx as nx
import pickle
from time import time
from tqdm import tqdm

def save_as_table(formats: list[str], relations_dir: Path, output_dir: Path):
    """
    Convert JSONL graph data to specified table formats (CSV, Parquet, pickle)
    :param formats: List of formats to save ('csv', 'parquet')
    :param relations_dir: Path to the directory containing JSONL files
    :param output_dir: Directory to save the output files
    """
    # Read all files in data / relations / in a loop
    # table = pd.DataFrame(columns=["person", "personLabel", "property", "propertyLabel", "value", "valueLabel"])
    # all_data = []
    frames = []
    for file_path in tqdm(relations_dir.glob("*.jsonl")):
        with open(file_path, 'r', encoding='utf-8') as f:
            df = pd.read_json(file_path, lines=True)
            """for line in f:
                data = json.loads(line)
                all_data.append(data)"""
    # table = pd.DataFrame(all_data)
            frames.append(df)
    table = pd.concat(frames, ignore_index=True)
    if 'csv' in formats:
        table.to_csv(output_dir / "graph_data.csv", index=False, encoding="utf-8")
    if 'parquet' in formats:
        table.to_parquet(output_dir / "graph_data.parquet", index=False)

def save_as_pickle(relations_dir: Path, output_dir: Path):
    pass # TODO: needed?

if __name__ == "__main__":
    start_time = time()
    relations_dir = get_root_dir() / "data" / "relations"
    output_dir = get_root_dir() / "data"
    save_as_table(['csv', 'parquet'], relations_dir, output_dir)
    end_time = time()
    print(f"Data saved in CSV and Parquet formats in {end_time - start_time:.2f} seconds.")
