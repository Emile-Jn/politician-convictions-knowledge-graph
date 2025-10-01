"""
This file splits a graph into training and test sets.
"""

import networkx as nx
import numpy as np
from sklearn.model_selection import train_test_split
import pickle
from source.utils import get_root_dir

def mark_train_test(G: nx.MultiDiGraph):
    # select politicians
    politician_nodes = [n for n, d in G.nodes(data=True) if d.get("type") == "politician"]
    # Extract labels (convicted True/False → 1/0)
    labels = np.array([int(G.nodes[n]["convicted"]) for n in politician_nodes])

    # stratified split to handle class imbalance
    train_nodes, temp_nodes, y_train, y_temp = train_test_split(
        politician_nodes,
        labels,
        test_size=0.30,
        stratify=labels,
        random_state=42,
    )

    val_nodes, test_nodes, y_val, y_test = train_test_split(
        temp_nodes,
        y_temp,
        test_size=0.50,  # half of the 30% → 15% each
        stratify=y_temp,
        random_state=42,
    )
    # Default split for all nodes
    nx.set_node_attributes(G, {n: "none" for n in G.nodes()}, "split")

    # Assign split information to politician nodes
    split_map = {n: "train" for n in train_nodes}
    split_map.update({n: "val" for n in val_nodes})
    split_map.update({n: "test" for n in test_nodes})
    nx.set_node_attributes(G, split_map, "split")

    # check class counts in each split
    def class_counts(nodes):
        vals, cnts = np.unique([G.nodes[n]["convicted"] for n in nodes], return_counts=True)
        return dict(zip(vals, cnts))

    print("Train class counts:", class_counts(train_nodes))
    print("Val   class counts:", class_counts(val_nodes))
    print("Test  class counts:", class_counts(test_nodes))

    # Count how many nodes in each split
    split_counts = {"train": 0, "val": 0, "test": 0, "none": 0}
    for _, data in G.nodes(data=True):
        split = data["split"]
        if split in split_counts:
            split_counts[split] += 1
        else:
            split_counts[split] = 1
    print("Node counts by split:", split_counts)

    return G

if __name__ == "__main__":
    # Get the graph without train/val/test split labels
    with open(get_root_dir() / 'data' / 'graph.gpickle', 'rb') as f:
        G = pickle.load(f)
    # Mark each politician node with 'train', 'val', or 'test'
    G = mark_train_test(G)
    # Save the updated graph
    with open(get_root_dir() / 'data' / 'graph_split.gpickle', 'wb') as f:
        pickle.dump(G, f)