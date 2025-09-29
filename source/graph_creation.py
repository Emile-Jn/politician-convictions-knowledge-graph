"""
This file creates a graph in NetworkX from the extracted relations in JSONL files.
"""

import pandas as pd
from pathlib import Path
import networkx as nx
from source.utils import get_root_dir
from tqdm import tqdm

def graph_from_parquet(parquet_file: Path) -> nx.MultiDiGraph:
    """
    Create a directed multigraph from a Parquet file containing relations.
    :param parquet_file: Path to the Parquet file
    :return: A NetworkX MultiDiGraph
    """
    # Read the Parquet file into a DataFrame
    df = pd.read_parquet(parquet_file)

    # Initialize a directed multigraph
    G = nx.MultiDiGraph()

    # Add edges to the graph from the DataFrame
    for _, row in tqdm(df.iterrows()):
        person = row['person']
        personLabel = row['personLabel']
        property = row['property']
        propertyLabel = row['propertyLabel']
        value = row['value']
        valueLabel = row['valueLabel']

        # Add nodes with labels if they don't exist
        if not G.has_node(person):
            G.add_node(person, label=personLabel)
        if not G.has_node(value):
            G.add_node(value, label=valueLabel)

        # Add an edge with property and propertyLabel as attributes
        G.add_edge(person, value, key=property, propertyLabel=propertyLabel)

    return G

if __name__ == "__main__":
    parquet_file = get_root_dir() / "data" / "graph_data.parquet"
    G = graph_from_parquet(parquet_file)
    print(f"Graph created with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges.")
