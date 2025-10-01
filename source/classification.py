import networkx as nx
from node2vec import Node2Vec
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, classification_report
import pickle
from source.utils import get_root_dir
from tqdm import tqdm
import pandas as pd
from pykeen.pipeline import pipeline

# Load the graph with train/val/test split labels
with open(get_root_dir() / 'data' / 'graph_split.gpickle', 'rb') as f:
    G = pickle.load(f)
print(f'Number of nodes: {G.number_of_nodes()}')

# Make nodes into a list
node_list = list(G.nodes())

triples = []
for u, v, data in G.edges(data=True):
    relation = data.get("relation", "linked_to")
    triples.append((str(u), relation, str(v)))

triples_df = pd.DataFrame(triples, columns=["head", "relation", "tail"])

# Train TransE with PyKEEN
result = pipeline(
    training=triples_df,
    model="TransE",
    model_kwargs=dict(embedding_dim=64),
    training_kwargs=dict(num_epochs=20, batch_size=256),
    optimizer="Adam",
    optimizer_kwargs=dict(lr=0.001),
    random_seed=42,
)
"""
# Create Node2Vec embeddings
node2vec = Node2Vec(G, dimensions=64, walk_length=30, num_walks=200, workers=4, seed=42)
model = node2vec.fit(window=10, min_count=1, batch_words=4)
"""
# Build embedding matrix for politicians
X_train, y_train = [], []
X_val, y_val = [], []
X_test, y_test = [], []

for node, data in G.nodes(data=True):
    if data.get('type') == 'politician' and 'convicted' in data:
        emb = model.wv[str(node)]
        label = int(data['convicted'])
        split = data.get('split', 'none')
        if split == 'train':
            X_train.append(emb)
            y_train.append(label)
        elif split == 'val':
            X_val.append(emb)
            y_val.append(label)
        elif split == 'test':
            X_test.append(emb)
            y_test.append(label)

# Convert to numpy arrays
X_train = np.array(X_train)
y_train = np.array(y_train)
X_val = np.array(X_val)
y_val = np.array(y_val)
X_test = np.array(X_test)
y_test = np.array(y_test)

# Define classifiers
classifiers = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Random Forest": RandomForestClassifier(n_estimators=200, random_state=42),
    "Gradient Boosting": GradientBoostingClassifier(n_estimators=200, random_state=42)
}

# Train and evaluate
for name, clf in classifiers.items():
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"=== {name} ===")
    print(f"Test Accuracy: {acc:.4f}")
    print(classification_report(y_test, y_pred))
