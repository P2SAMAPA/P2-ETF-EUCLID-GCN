"""
Hyperbolic Graph Convolutional Network (HGCN) in the Poincaré ball.
Uses geoopt for Riemannian optimization.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing
import geoopt
import numpy as np

class HyperbolicGraphConv(MessagePassing):
    def __init__(self, in_dim, out_dim, manifold, use_bias=True):
        super().__init__(aggr='add')
        self.manifold = manifold
        self.lin = nn.Linear(in_dim, out_dim, bias=False)
        if use_bias:
            self.bias = nn.Parameter(torch.zeros(out_dim))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        self.lin.reset_parameters()
        if self.bias is not None:
            nn.init.zeros_(self.bias)

    def forward(self, x, edge_index):
        # x is in hyperbolic space (Poincaré ball)
        x_tan = self.manifold.logmap0(x)
        # Propagate: simply sum neighbor features (no normalization)
        out_tan = self.propagate(edge_index, x=x_tan)
        out_tan = self.lin(out_tan)
        if self.bias is not None:
            out_tan = out_tan + self.bias
        out_hyp = self.manifold.expmap0(out_tan)
        return out_hyp

    def message(self, x_j):
        return x_j

class HGCN(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, num_layers, manifold, dropout=0.0):
        super().__init__()
        self.manifold = manifold
        self.convs = nn.ModuleList()
        self.convs.append(HyperbolicGraphConv(in_dim, hidden_dim, manifold))
        for _ in range(num_layers - 2):
            self.convs.append(HyperbolicGraphConv(hidden_dim, hidden_dim, manifold))
        self.convs.append(HyperbolicGraphConv(hidden_dim, out_dim, manifold))
        self.dropout = nn.Dropout(dropout)
        self.reset_parameters()

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()

    def forward(self, x, edge_index):
        for conv in self.convs[:-1]:
            x = conv(x, edge_index)
            x = self.manifold.expmap0(F.relu(self.manifold.logmap0(x)))
            x = self.dropout(x)
        x = self.convs[-1](x, edge_index)
        return x

class HyperbolicGNNPredictor:
    def __init__(self, in_dim, hidden_dim, out_dim=1, num_layers=2, lr=0.001, weight_decay=1e-4, seed=42):
        torch.manual_seed(seed)
        np.random.seed(seed)
        self.manifold = geoopt.manifolds.PoincareBall(c=1.0)
        self.model = HGCN(in_dim, hidden_dim, out_dim, num_layers, self.manifold)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        self.criterion = nn.MSELoss()

    def fit(self, graph_data, returns, epochs=100, batch_size=32):
        node_features = torch.tensor(graph_data["node_features"])
        edge_index = torch.tensor(graph_data["edge_index"], dtype=torch.long)
        etf_indices = list(range(graph_data["num_etfs"]))
        targets = torch.tensor(returns.iloc[-1].values, dtype=torch.float32).view(-1, 1)

        node_features_hyp = self.manifold.expmap0(node_features)

        self.model.train()
        for epoch in range(epochs):
            self.optimizer.zero_grad()
            embeddings = self.model(node_features_hyp, edge_index)
            preds = embeddings[etf_indices]
            loss = self.criterion(preds, targets)
            loss.backward()
            self.optimizer.step()
            if (epoch+1) % 20 == 0:
                print(f"    Epoch {epoch+1}/{epochs} - Loss: {loss.item():.6f}")

    def predict(self, graph_data):
        self.model.eval()
        with torch.no_grad():
            node_features = torch.tensor(graph_data["node_features"])
            edge_index = torch.tensor(graph_data["edge_index"], dtype=torch.long)
            etf_indices = list(range(graph_data["num_etfs"]))
            node_features_hyp = self.manifold.expmap0(node_features)
            embeddings = self.model(node_features_hyp, edge_index)
            preds = embeddings[etf_indices].squeeze().numpy()
        return dict(zip(graph_data["etf_tickers"], preds))
