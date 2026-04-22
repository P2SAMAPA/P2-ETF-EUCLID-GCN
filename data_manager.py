"""
Data loading and preprocessing for Euclidean GCN engine.
Builds richer node features and ETF-ETF edges.
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
    macro_cols = [c for c in config.MACRO_COLS if c in df_wide.columns]
    macro_df = df_wide[['Date'] + macro_cols].copy()
    macro_df = macro_df.set_index('Date').ffill().dropna()
    return macro_df

def build_graph_sequence(returns: pd.DataFrame, macro: pd.DataFrame):
    """
    Build a sequence of graph snapshots with rich node features.
    Node features for ETFs: last `ETF_WINDOW` returns.
    Node features for macro: last `MACRO_WINDOW` values.
    Edges: ETF-Macro (fully connected) + ETF-ETF (if correlation > threshold).
    """
    common_idx = returns.index.intersection(macro.index)
    returns = returns.loc[common_idx]
    macro = macro.loc[common_idx]

    etf_tickers = returns.columns.tolist()
    macro_cols = macro.columns.tolist()
    num_etfs = len(etf_tickers)
    num_macro = len(macro_cols)
    num_nodes = num_etfs + num_macro

    # --- Build ETF-ETF edges based on correlation (static, using training period) ---
    corr_matrix = returns.corr().values
    etf_edges = []
    for i in range(num_etfs):
        for j in range(i+1, num_etfs):
            if corr_matrix[i, j] > config.CORR_THRESHOLD:
                etf_edges.append([i, j])
                etf_edges.append([j, i])

    # --- ETF-Macro edges (fully connected bipartite) ---
    bipartite_edges = []
    for i in range(num_etfs):
        for j in range(num_macro):
            bipartite_edges.append([i, num_etfs + j])
            bipartite_edges.append([num_etfs + j, i])

    edge_index = np.array(bipartite_edges + etf_edges).T

    # --- Scale features globally ---
    all_etf_vals = returns.values.flatten()
    all_macro_vals = macro.values.flatten()
    etf_scaler = StandardScaler().fit(all_etf_vals.reshape(-1, 1))
    macro_scaler = StandardScaler().fit(all_macro_vals.reshape(-1, 1))

    features_seq = []
    targets = []

    # Precompute node features for each day
    for i in range(len(returns) - 1):
        node_feats = []
        # ETF nodes: last ETF_WINDOW returns (or padded)
        for ticker in etf_tickers:
            if i >= config.ETF_WINDOW - 1:
                feat = returns[ticker].iloc[i - config.ETF_WINDOW + 1 : i+1].values
            else:
                feat = returns[ticker].iloc[:i+1].values
                if len(feat) < config.ETF_WINDOW:
                    feat = np.pad(feat, (config.ETF_WINDOW - len(feat), 0), 'edge')
            feat = etf_scaler.transform(feat.reshape(-1, 1)).flatten()
            node_feats.append(feat.astype(np.float32))

        # Macro nodes: last MACRO_WINDOW values
        for col in macro_cols:
            if i >= config.MACRO_WINDOW - 1:
                feat = macro[col].iloc[i - config.MACRO_WINDOW + 1 : i+1].values
            else:
                feat = macro[col].iloc[:i+1].values
                if len(feat) < config.MACRO_WINDOW:
                    feat = np.pad(feat, (config.MACRO_WINDOW - len(feat), 0), 'edge')
            feat = macro_scaler.transform(feat.reshape(-1, 1)).flatten()
            node_feats.append(feat.astype(np.float32))

        features_seq.append(np.stack(node_feats))
        targets.append(returns.iloc[i+1].values)

    features_seq = np.array(features_seq, dtype=np.float32)
    targets = np.array(targets, dtype=np.float32)

    # Scale targets to zero mean unit variance per ETF (helps training)
    target_scaler = StandardScaler()
    targets = target_scaler.fit_transform(targets)

    return {
        "features_seq": features_seq,
        "edge_index": edge_index,
        "targets": targets,
        "target_scaler": target_scaler,  # for inverse transform if needed
        "num_etfs": num_etfs,
        "num_macro": num_macro,
        "etf_tickers": etf_tickers,
        "macro_cols": macro_cols
    }
