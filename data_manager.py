"""
Data loading and preprocessing for Hyperbolic GNN engine.
"""

import pandas as pd
import numpy as np
from huggingface_hub import hf_hub_download
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

def build_graph_data(returns: pd.DataFrame, macro: pd.DataFrame):
    """
    Build a bipartite graph: ETFs and macro nodes.
    Returns node features and edge indices.
    """
    etf_tickers = returns.columns.tolist()
    macro_cols = macro.columns.tolist()
    num_etfs = len(etf_tickers)
    num_macro = len(macro_cols)
    num_nodes = num_etfs + num_macro

    # Node features: last available values
    etf_features = returns.iloc[-1].values.reshape(-1, 1)
    macro_features = macro.iloc[-1].values.reshape(-1, 1)
    # Pad to same dimension if needed (use simple scaling)
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    all_features = np.concatenate([etf_features, macro_features])
    all_features = scaler.fit_transform(all_features)
    # Expand to embedding dimension by repeating (or use linear projection later)
    node_features = np.tile(all_features, (1, config.EMBEDDING_DIM // 1 + 1))[:, :config.EMBEDDING_DIM]

    # Build edges: fully connect ETFs to macro nodes (bipartite)
    edge_index = []
    for i in range(num_etfs):
        for j in range(num_macro):
            edge_index.append([i, num_etfs + j])
            edge_index.append([num_etfs + j, i])  # undirected
    edge_index = np.array(edge_index).T

    return {
        "node_features": node_features.astype(np.float32),
        "edge_index": edge_index,
        "num_etfs": num_etfs,
        "num_macro": num_macro,
        "etf_tickers": etf_tickers,
        "macro_cols": macro_cols
    }
