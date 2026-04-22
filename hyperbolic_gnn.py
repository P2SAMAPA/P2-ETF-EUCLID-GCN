"""
Hyperbolic Graph Convolutional Network (HGCN) with temporal training.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing
import geoopt
import numpy as np

class HyperbolicGraphConv(MessagePassing):
    def __init__(self, in_dim, out_dim, manifold):
        super().__init__(aggr='add')
        self.manifold = manifold
        self.lin = nn.Linear(in_dim, out_dim)

    def forward(self, x, edge_index):
        x_tan = self.manifold.logmap0(x)
        out_tan = self.propagate(edge_index, x=x_tan)
        out_tan = self.lin(out_tan)
        return self.manifold.expmap0(out_tan)

    def message(self, x_j):
        return x_j

class TemporalHGCN(nn.Module):
    def __init__(self, node_feat_dim, hidden_dim, out_dim, num_layers, manifold):
        super().__init__()
        self.manifold = manifold
        self.embed = nn.Linear(node_feat_dim, hidden_dim)
        self.convs = nn.ModuleList()
        for _ in range(num_layers):
            self.convs.append(HyperbolicGraphConv(hidden_dim, hidden_dim, manifold))
        self.pred = nn.Linear(hidden_dim, out_dim)

    def forward(self, x_seq, edge_index):
        # x_seq: (batch, num_nodes, feat_dim)
        batch_size, num_nodes, feat_dim = x_seq.shape
        x_seq = x_seq.view(-1, feat_dim)
        x_hyp = self.manifold.expmap0(self.embed(x_seq))
        x_hyp = x_hyp.view(batch_size, num_nodes, -1)

        for conv in self.convs:
            # Apply conv to each graph in batch
            out = []
            for i in range(batch_size):
                out.append(conv(x_hyp[i], edge_index))
            x_hyp = torch.stack(out)
            x_hyp = self.manifold.expmap0(F.relu(self.manifold.logmap0(x_hyp)))

        # Predictions
        x_tan = self.manifold.logmap0(x_hyp)
        preds = self.pred(x_tan).squeeze(-1)  # (batch, num_nodes)
        return preds

class HyperbolicGNNPredictor:
    def __init__(self, in_dim, hidden_dim, out_dim=1, num_layers=2, lr=0.001, weight_decay=1e-4, seed=42):
        torch.manual_seed(seed)
        np.random.seed(seed)
        self.manifold = geoopt.manifolds.PoincareBall(c=1.0)
        self.model = TemporalHGCN(in_dim, hidden_dim, out_dim, num_layers, self.manifold)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        self.criterion = nn.MSELoss()

    def fit(self, graph_seq, targets, epochs=50, batch_size=64):
        features = torch.tensor(graph_seq["features_seq"])  # (days, num_nodes, feat_dim)
        edge_index = torch.tensor(graph_seq["edge_index"], dtype=torch.long)
        etf_indices = list(range(graph_seq["num_etfs"]))
        targets = torch.tensor(targets)[:, etf_indices]  # (days, num_etfs)

        dataset = torch.utils.data.TensorDataset(features, targets)
        loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

        self.model.train()
        for epoch in range(epochs):
            total_loss = 0.0
            for batch_feat, batch_y in loader:
                self.optimizer.zero_grad()
                preds = self.model(batch_feat, edge_index)  # (batch, num_nodes)
                preds = preds[:, etf_indices]
                loss = self.criterion(preds, batch_y)
                loss.backward()
                self.optimizer.step()
                total_loss += loss.item() * batch_feat.size(0)
            if (epoch+1) % 10 == 0:
                print(f"    Epoch {epoch+1}/{epochs} - Loss: {total_loss/len(dataset):.6f}")

    def predict(self, graph_snapshot):
        """Predict using the most recent snapshot."""
        self.model.eval()
        with torch.no_grad():
            features = torch.tensor(graph_snapshot["features_seq"][-1:])  # (1, num_nodes, feat_dim)
            edge_index = torch.tensor(graph_snapshot["edge_index"], dtype=torch.long)
            etf_indices = list(range(graph_snapshot["num_etfs"]))
            preds = self.model(features, edge_index)[0, etf_indices].numpy()
        return dict(zip(graph_snapshot["etf_tickers"], preds))
