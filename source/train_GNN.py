import torch
from torch_geometric.utils import from_networkx
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv
from torch_geometric.loader import NeighborLoader
import pickle
from source.utils import get_root_dir
from tqdm import tqdm

# Load the graph with train/val/test split labels
with open(get_root_dir() / 'data' / 'graph_split.gpickle', 'rb') as f:
    G = pickle.load(f)
print(f'Number of nodes: {G.number_of_nodes()}')

# Convert to PyTorch Geometric Data object
data = from_networkx(G)

# Make label tensor (only for politician nodes, others default -1)
labels = []
for n in G.nodes():
    if G.nodes[n]["type"] == "politician":
        labels.append(int(G.nodes[n]["convicted"]))
    else:
        labels.append(-1)  # unused
data.y = torch.tensor(labels, dtype=torch.long)

# Masks for train/val/test
data.train_mask = torch.tensor([G.nodes[n]['split'] == 'train' for n in G.nodes()])
data.val_mask   = torch.tensor([G.nodes[n]['split'] == 'val'   for n in G.nodes()])
data.test_mask  = torch.tensor([G.nodes[n]['split'] == 'test'  for n in G.nodes()])

device = torch.device("cuda" if torch.cuda.is_available()
                      # else "mps" if torch.backends.mps.is_available() # NotImplementedError: The operator 'aten::_convert_indices_from_coo_to_csr.out' is not currently implemented for the MPS device.
                      else "cpu")
print("Using device:", device)

# Learnable embedding per node instead of identity
embedding_dim = 64
data.x = nn.Embedding(data.num_nodes, embedding_dim).weight

class GraphSAGE(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super().__init__()
        self.conv1 = SAGEConv(in_channels, hidden_channels)
        self.conv2 = SAGEConv(hidden_channels, out_channels)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.conv2(x, edge_index)
        return x

model = GraphSAGE(data.num_features, 64, 2).to(device)
data = data.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)

# Neighbour sampling loader
train_loader = NeighborLoader(
    data,
    input_nodes=data.train_mask,
    num_neighbors=[10, 10],  # 10 neighbors per layer
    batch_size=512,
    shuffle=True
)

val_loader = NeighborLoader(
    data,
    input_nodes=data.val_mask,
    num_neighbors=[10, 10],
    batch_size=512,
    shuffle=False
)

test_loader = NeighborLoader(
    data,
    input_nodes=data.test_mask,
    num_neighbors=[10, 10],
    batch_size=512,
    shuffle=False
)

# Training function
def train():
    model.train()
    total_loss = 0
    for batch in train_loader:
        batch = batch.to(device)
        optimizer.zero_grad()
        out = model(batch.x, batch.edge_index)
        mask = batch.input_id  # only compute loss on sampled batch nodes
        loss = F.cross_entropy(out[mask], batch.y[mask])
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * mask.size(0)
    return total_loss / int(data.train_mask.sum())

# Evaluation
@torch.no_grad()
def evaluate(loader):
    model.eval()
    correct = 0
    total = 0
    for batch in loader:
        batch = batch.to(device)
        out = model(batch.x, batch.edge_index)
        pred = out.argmax(dim=1)
        mask = batch.input_id
        correct += (pred[mask] == batch.y[mask]).sum().item()
        total += mask.size(0)
    return correct / total

# Training loop
for epoch in range(1, 201):
    loss = train()
    if epoch % 20 == 0:
        train_acc = evaluate(train_loader)
        val_acc = evaluate(val_loader)
        test_acc = evaluate(test_loader)
        print(f"Epoch {epoch:03d}, Loss: {loss:.4f}, "
              f"Train: {train_acc:.4f}, Val: {val_acc:.4f}, Test: {test_acc:.4f}")
train_loop()

#%%

