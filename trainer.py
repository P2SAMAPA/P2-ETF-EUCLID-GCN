"""
Main training script for Hyperbolic GNN engine.
"""

import json
import pandas as pd
import numpy as np

import config
import data_manager
from hyperbolic_gnn import HyperbolicGNNPredictor
import push_results

def run_hyperbolic_gnn():
    print(f"=== P2-ETF-HYPERBOLIC-GNN Run: {config.TODAY} ===")
    df_master = data_manager.load_master_data()
    df_master = df_master[df_master['Date'] >= "2008-01-01"]

    macro = data_manager.prepare_macro_features(df_master)

    all_results = {}
    top_picks = {}

    for universe_name, tickers in config.UNIVERSES.items():
        print(f"\n--- Processing Universe: {universe_name} ---")
        returns = data_manager.prepare_returns_matrix(df_master, tickers)
        if len(returns) < config.MIN_OBSERVATIONS:
            continue

        recent_returns = returns.iloc[-config.LOOKBACK_WINDOW:]
        recent_macro = macro.loc[recent_returns.index].dropna()
        common_idx = recent_returns.index.intersection(recent_macro.index)
        recent_returns = recent_returns.loc[common_idx]
        recent_macro = recent_macro.loc[common_idx]

        graph_data = data_manager.build_graph_data(recent_returns, recent_macro)

        predictor = HyperbolicGNNPredictor(
            in_dim=config.EMBEDDING_DIM,
            hidden_dim=config.HIDDEN_DIM,
            num_layers=config.NUM_LAYERS,
            lr=config.LEARNING_RATE,
            weight_decay=config.WEIGHT_DECAY,
            seed=config.RANDOM_SEED
        )

        print(f"  Training HGCN...")
        predictor.fit(graph_data, recent_returns, epochs=config.EPOCHS, batch_size=config.BATCH_SIZE)
        preds = predictor.predict(graph_data)

        sorted_preds = sorted(preds.items(), key=lambda x: x[1], reverse=True)
        top3 = [{"ticker": t, "predicted_return": float(v)} for t, v in sorted_preds[:3]]
        top_picks[universe_name] = top3[0] if top3 else None
        all_results[universe_name] = top3

    output_payload = {
        "run_date": config.TODAY,
        "config": {
            "lookback_window": config.LOOKBACK_WINDOW,
            "embedding_dim": config.EMBEDDING_DIM,
            "hidden_dim": config.HIDDEN_DIM,
            "num_layers": config.NUM_LAYERS,
            "epochs": config.EPOCHS
        },
        "daily_trading": {
            "universes": all_results,
            "top_picks": top_picks
        }
    }

    push_results.push_daily_result(output_payload)
    print("\n=== Run Complete ===")

if __name__ == "__main__":
    run_hyperbolic_gnn()
