"""
Graph Convolutional Network (GCN) for ETF return prediction.
v2.0 - True Spatial-Temporal processing with GRU.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
import numpy as np

class TemporalGCN(nn.Module):
    def __init__(self, node_feat_dim, hidden_dim, out_dim, num_layers, 
                 dropout=0.1, seq_len=10):
        super().__init__()
        self.seq_len = seq_len
        self.embed = nn.Linear(node_feat_dim, hidden_dim)
        
        # Spatial layers
        self.convs = nn.ModuleList()
        for _ in range(num_layers):
            self.convs.append(GCNConv(hidden_dim, hidden_dim))
            
        # LayerNorm replaces BatchNorm to prevent time-series data leakage
        self.ln = nn.LayerNorm(hidden_dim)
        
        # Temporal layer (Actually makes it "Temporal")
        self.gru = nn.GRU(hidden_dim, hidden_dim, batch_first=True)
        
        self.pred = nn.Linear(hidden_dim, out_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x_seq, edge_index):
        """
        x_seq: (Batch, Seq_len, Num_nodes, Feat_dim)
        edge_index: (2, Num_edges) - Shared static graph structure
        """
        B, L, N, F = x_seq.shape
        
        # -------------------------------------------------------------
        # 1. Spatial Feature Extraction (Fully Vectorized - No Python Loops)
        # -------------------------------------------------------------
        # Reshape to process all graphs in batch simultaneously
        x = x_seq.view(B * L * N, F)
        x = self.embed(x)
        
        # Dynamically expand edge_index for the batched graph
        # This allows GPU parallelization across the entire batch
        offset = torch.arange(B * L, device=edge_index.device).unsqueeze(1) * N
        edge_idx_exp = edge_index.unsqueeze(2).expand(-1, -1, B * L) + offset.transpose(0, 1)
        edge_idx_batch = edge_idx_exp.reshape(2, -1)
        
        for conv in self.convs:
            x_res = x
            x = conv(x, edge_idx_batch)
            x = self.ln(x)
            x = F.elu(x) # ELU generally outperforms ReLU in GCNs
            x = self.dropout(x)
            x = x + x_res  # Residual connection prevents over-smoothing
            
        # Reshape back to separate time steps
        x = x.view(B, L, N, -1)
        
        # -------------------------------------------------------------
        # 2. Temporal Feature Extraction (The missing piece)
        # -------------------------------------------------------------
        # Permute to (Batch, Num_nodes, Seq_len, Hidden)
        x = x.permute(0, 2, 1, 3)
        # Reshape to (Batch * Num_nodes, Seq_len, Hidden) for GRU
        x = x.reshape(B * N, L, -1)
        
        x, _ = self.gru(x)
        # Take the output of the last sequence step
        x = x[:, -1, :] 
        
        # -------------------------------------------------------------
        # 3. Prediction
        # -------------------------------------------------------------
        x = x.view(B, N, -1)
        preds = self.pred(x).squeeze(-1)
        return preds


class GCNPredictor:
    def __init__(self, in_dim, hidden_dim, out_dim=1, num_layers=3, lr=0.001,
                 weight_decay=1e-4, dropout=0.1, seed=42, seq_len=10):
        torch.manual_seed(seed)
        np.random.seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
            
        self.seq_len = seq_len
        self.model = TemporalGCN(in_dim, hidden_dim, out_dim, num_layers, dropout, seq_len)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        self.criterion = nn.MSELoss()

    def fit(self, graph_seq, targets, epochs=100, batch_size=64):
        """
        Note for trainer.py: graph_seq["features_seq"] must now be shaped
        as (Num_samples, Seq_len, Num_nodes, Feat_dim) instead of flat time steps.
        """
        features = torch.tensor(graph_seq["features_seq"], dtype=torch.float32)
        edge_index = torch.tensor(graph_seq["edge_index"], dtype=torch.long)
        etf_indices = list(range(graph_seq["num_etfs"]))
        targets = torch.tensor(targets, dtype=torch.float32)[:, etf_indices]

        dataset = torch.utils.data.TensorDataset(features, targets)
        # WARNING: Time-series MUST NOT be shuffled if using true temporal sequences!
        # If trainer.py passes sequential windows, drop shuffle=True.
        loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False)

        self.model.train()
        for epoch in range(epochs):
            total_loss = 0.0
            for batch_feat, batch_y in loader:
                self.optimizer.zero_grad()
                # Model now expects (Batch, Seq_len, Nodes, Feats)
                preds = self.model(batch_feat, edge_index)
                preds = preds[:, etf_indices]
                loss = self.criterion(preds, batch_y)
                loss.backward()
                
                # Gradient clipping to prevent temporal exploding gradients
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                
                self.optimizer.step()
                total_loss += loss.item() * batch_feat.size(0)
                
            if (epoch+1) % 20 == 0:
                print(f"    Epoch {epoch+1}/{epochs} - Loss: {total_loss/len(dataset):.6f}")

    def predict(self, graph_snapshot, target_scaler):
        self.model.eval()
        with torch.no_grad():
            # CRITICAL FIX: Must pass a sequence of length `seq_len`, not just `[-1:]`
            # graph_snapshot["features_seq"][-self.seq_len:] should be shape (seq_len, N, F)
            seq = graph_snapshot["features_seq"][-self.seq_len:]
            features = torch.tensor(seq, dtype=torch.float32).unsqueeze(0) # (1, L, N, F)
            
            edge_index = torch.tensor(graph_snapshot["edge_index"], dtype=torch.long)
            etf_indices = list(range(graph_snapshot["num_etfs"]))
            
            preds_scaled = self.model(features, edge_index)[0, etf_indices].numpy()
            
        preds = target_scaler.inverse_transform(preds_scaled.reshape(1, -1)).flatten()
        return dict(zip(graph_snapshot["etf_tickers"], preds))
