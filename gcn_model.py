"""
Graph Convolutional Network (GCN) for ETF return prediction.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
import numpy as np

class TemporalGCN(nn.Module):
    def __init__(self, node_feat_dim, hidden_dim, out_dim, num_layers, dropout=0.1):
        super().__init__()
        self.embed = nn.Linear(node_feat_dim, hidden_dim)
        self.convs = nn.ModuleList()
        for _ in range(num_layers):
            self.convs.append(GCNConv(hidden_dim, hidden_dim))
        self.pred = nn.Linear(hidden_dim, out_dim)
        self.dropout = nn.Dropout(dropout)
        self.bn = nn.BatchNorm1d(hidden_dim)

    def forward(self, x_seq, edge_index):
        batch_size, num_nodes, feat_dim = x_seq.shape
        x_seq = x_seq.view(-1, feat_dim)
        x = self.embed(x_seq)
        x = x.view(batch_size, num_nodes, -1)

        for i, conv in enumerate(self.convs):
            out = []
            for j in range(batch_size):
                out.append(conv(x[j], edge_index))
            x = torch.stack(out)
            if i < len(self.convs) - 1:
                x = x.view(-1, x.size(-1))
                x = self.bn(x)
                x = x.view(batch_size, num_nodes, -1)
                x = F.relu(x)
                x = self.dropout(x)

        x = x.view(-1, x.size(-1))
        preds = self.pred(x).view(batch_size, num_nodes, -1).squeeze(-1)
        return preds

class GCNPredictor:
    def __init__(self, in_dim, hidden_dim, out_dim=1, num_layers=3, lr=0.001,
                 weight_decay=1e-4, dropout=0.1, seed=42):
        torch.manual_seed(seed)
        np.random.seed(seed)
        self.model = TemporalGCN(in_dim, hidden_dim, out_dim, num_layers, dropout)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        self.criterion = nn.MSELoss()

    def fit(self, graph_seq, targets, epochs=100, batch_size=64):
        features = torch.tensor(graph_seq["features_seq"])
        edge_index = torch.tensor(graph_seq["edge_index"], dtype=torch.long)
        etf_indices = list(range(graph_seq["num_etfs"]))
        targets = torch.tensor(targets)[:, etf_indices]

        dataset = torch.utils.data.TensorDataset(features, targets)
        loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

        self.model.train()
        for epoch in range(epochs):
            total_loss = 0.0
            for batch_feat, batch_y in loader:
                self.optimizer.zero_grad()
                preds = self.model(batch_feat, edge_index)
                preds = preds[:, etf_indices]
                loss = self.criterion(preds, batch_y)
                loss.backward()
                self.optimizer.step()
                total_loss += loss.item() * batch_feat.size(0)
            if (epoch+1) % 20 == 0:
                print(f"    Epoch {epoch+1}/{epochs} - Loss: {total_loss/len(dataset):.6f}")

    def predict(self, graph_snapshot, target_scaler):
        self.model.eval()
        with torch.no_grad():
            features = torch.tensor(graph_snapshot["features_seq"][-1:])
            edge_index = torch.tensor(graph_snapshot["edge_index"], dtype=torch.long)
            etf_indices = list(range(graph_snapshot["num_etfs"]))
            preds_scaled = self.model(features, edge_index)[0, etf_indices].numpy()
        # Inverse transform to original return scale
        preds = target_scaler.inverse_transform(preds_scaled.reshape(1, -1)).flatten()
        return dict(zip(graph_snapshot["etf_tickers"], preds))
