"""
Data loading and preprocessing for Hyperbolic GNN engine.
"""

import pandas as pd
import numpy as np
from huggingface_hub import hf_hub_download
from sklearn.preprocessing import StandardScaler
import config

def load_master_data() -> pd.DataFrame:
    print(f"Downloading {config.HF_DATA_FILE} from {config.HF_DATA_REPO}...")
    file_path = hf_hub_download(
        repo_id=config.HF_DATA_REPO,
        filename=config.HF_DATA_FILE,
        repo_type="dataset",
        token=config.HF_TOKEN,
        cache_dir="./hf_cache"
    )
    df = pd.read_parquet(file_path)
    if isinstance(df.index, pd.DatetimeIndex):
        df = df.reset_index().rename(columns={'index': 'Date'})
    df['Date'] = pd.to_datetime(df['Date'])
    return df

def prepare_returns_matrix(df_wide: pd.DataFrame, tickers: list) -> pd.DataFrame:
    """Prepare a wide-format DataFrame of log returns with Date index."""
    available_tickers = [t for t in tickers if t in df_wide.columns]
    df_long = pd.melt(
        df_wide, id_vars=['Date'], value_vars=available_tickers,
        var_name='ticker', value_name='price'
    )
    df_long = df_long.sort_values(['ticker', 'Date'])
    df_long['log_return'] = df_long.groupby('ticker')['price'].transform(
        lambda x: np.log(x / x.shift(1))
    )
    df_long = df_long.dropna(subset=['log_return'])
    return df_long.pivot(index='Date', columns='ticker', values='log_return')[available_tickers].dropna()

def prepare_macro_features(df_wide: pd.DataFrame) -> pd.DataFrame:
    """Extract macro columns and return as DataFrame with Date index."""
    macro_cols = [c for c in config.MACRO_COLS if c in df_wide.columns]
    macro_df = df_wide[['Date'] + macro_cols].copy()
    macro_df = macro_df.set_index('Date').ffill().dropna()
    return macro_df

def build_graph_sequence(returns: pd.DataFrame, macro: pd.DataFrame):
    """
    Build a sequence of graph snapshots (one per day).
    Returns:
        - features_seq: (num_days, num_nodes, EMBEDDING_DIM)
        - edge_index: (2, num_edges)
        - targets: (num_days, num_etfs)
        - etf_tickers: list
    """
    common_idx = returns.index.intersection(macro.index)
    returns = returns.loc[common_idx]
    macro = macro.loc[common_idx]

    etf_tickers = returns.columns.tolist()
    macro_cols = macro.columns.tolist()
    num_etfs = len(etf_tickers)
    num_macro = len(macro_cols)
    num_nodes = num_etfs + num_macro

    # Static edge index (fully connected bipartite)
    edge_list = []
    for i in range(num_etfs):
        for j in range(num_macro):
            edge_list.append([i, num_etfs + j])
            edge_list.append([num_etfs + j, i])
    edge_index = np.array(edge_list).T

    # Scale features globally
    all_etf_vals = returns.values.flatten()
    all_macro_vals = macro.values.flatten()
    etf_scaler = StandardScaler().fit(all_etf_vals.reshape(-1, 1))
    macro_scaler = StandardScaler().fit(all_macro_vals.reshape(-1, 1))

    features_seq = []
    targets = []

    for i in range(len(returns) - 1):
        # Build node features: each node gets a vector of length EMBEDDING_DIM
        node_feats = []
        for j, ticker in enumerate(etf_tickers):
            val = etf_scaler.transform([[returns.iloc[i][ticker]]])[0, 0]
            # Repeat to reach EMBEDDING_DIM
            feat = np.full(config.EMBEDDING_DIM, val, dtype=np.float32)
            node_feats.append(feat)
        for j, col in enumerate(macro_cols):
            val = macro_scaler.transform([[macro.iloc[i][col]]])[0, 0]
            feat = np.full(config.EMBEDDING_DIM, val, dtype=np.float32)
            node_feats.append(feat)
        features_seq.append(np.stack(node_feats))  # (num_nodes, EMBEDDING_DIM)
        targets.append(returns.iloc[i+1].values)

    features_seq = np.array(features_seq, dtype=np.float32)  # (days, num_nodes, EMBEDDING_DIM)
    targets = np.array(targets, dtype=np.float32)

    return {
        "features_seq": features_seq,
        "edge_index": edge_index,
        "targets": targets,
        "num_etfs": num_etfs,
        "num_macro": num_macro,
        "etf_tickers": etf_tickers,
        "macro_cols": macro_cols
    }
