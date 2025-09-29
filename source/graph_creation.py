"""
This file creates a graph in NetworkX from the extracted relations in JSONL files.
"""

import pandas as pd
from pathlib import Path
import networkx as nx
from source.utils import get_root_dir
from tqdm import tqdm
import json
import pickle

def match_names():
    # Read the Parquet file into a DataFrame
    parquet_file = get_root_dir() / "data" / "graph_data.parquet"  # change path if needed
    df = pd.read_parquet(parquet_file)

    # Write unique person labels to a text file
    unique_persons = df['personLabel'].unique()
    with open(get_root_dir() / 'data' / 'unique_persons.txt', 'w', encoding='utf-8') as f:
        for person in unique_persons:
            f.write(f"{person}\n")

    # Read the ground truth labels into a dictionary
    with open(get_root_dir() / 'data' / 'ground_truth.json', 'r') as f:  # change path if needed
        ground_truth = json.load(f)
    # Write ground truth names to a text file
    with open(get_root_dir() / 'data' / 'ground_truth_names.txt', 'w', encoding='utf-8') as f:
        for name in ground_truth.keys():
            f.write(f"{name}\n")

    mismatches = [name for name in ground_truth if name not in unique_persons]
    # Write mismatches to a text file
    with open(get_root_dir() / 'data' / 'mismatched_names.txt', 'w', encoding='utf-8') as f:
        for name in mismatches:
            f.write(f"{name}\n")


def graph_from_parquet(verbose: bool = False) -> nx.MultiDiGraph:
    """
    Create a directed multigraph from a Parquet file containing relations.
    Should take about 15 seconds (MacBook M1).
    :return: A NetworkX MultiDiGraph
    """
    # Read the Parquet file into a DataFrame
    parquet_file = get_root_dir() / "data" / "graph_data.parquet" # change path if needed
    df = pd.read_parquet(parquet_file)

    # Read the ground truth labels into a dictionary
    with open(get_root_dir() / 'data' / 'ground_truth.json', 'r') as f: # change path if needed
        ground_truth = json.load(f)

    # Initialize a directed multigraph
    G = nx.MultiDiGraph()

    # First, add all politicians as nodes
    unique_politicians = df[['person', 'personLabel']].drop_duplicates()
    print(f"Adding {len(unique_politicians)} unique politicians as nodes.")
    for _, row in unique_politicians.iterrows():
        person = row['person']
        personLabel = row['personLabel']
        if not G.has_node(person):
            G.add_node(person, label=personLabel, type='politician')

    # Add edges to the graph from the DataFrame
    for _, row in df.iterrows():
        person = row['person']
        property = row['property']
        propertyLabel = row['propertyLabel']
        value = row['value']
        valueLabel = row['valueLabel']

        # Add nodes with labels if they don't exist
        if not G.has_node(value):
            G.add_node(value, label=valueLabel, type='other')

        # Add an edge with property and propertyLabel as attributes
        G.add_edge(person, value, key=property, propertyLabel=propertyLabel)

    # Add ground truth labels to politician nodes
    name_not_found = 0
    for name, value in ground_truth.items():
        # Try matching by label instead of node ID
        found = False
        for node, data in G.nodes(data=True):
            if data.get('label') == name:
                G.nodes[node]['convicted'] = value
                found = True
                break
        if not found:
            name_not_found += 1
    print(f"Ground truth labels added. Names not found in graph: {name_not_found}/{len(ground_truth)}")

    # All politicians without a section mentioning conviction on their wikipedia page are
    # assumed to be not convicted (False)
    for node in G.nodes:
        if 'convicted' not in G.nodes[node]:
            G.nodes[node]['convicted'] = False
    if verbose:
        print(f"Graph created with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges.")
    return G

def check_node_types(G: nx.MultiDiGraph):
    types = {}
    for _, data in G.nodes(data=True):
        node_type = data.get('type', 'unknown')
        types[node_type] = types.get(node_type, 0) + 1
    print("Node types and their counts:")
    for node_type, count in types.items():
        print(f"{node_type}: {count}")

def save_graph_pickle(G: nx.MultiDiGraph, output_file: Path):
    with open(output_file, 'wb') as f:
        pickle.dump(G, f)
    print(f"Graph saved to {output_file}")

if __name__ == "__main__":
    G = graph_from_parquet(verbose=True)
    # match_names()
    # check_node_types(G)
    # Save the graph to a file

    output_file = get_root_dir() / "data" / "graph.gpickle"
    # save_graph_pickle(G, output_file)
