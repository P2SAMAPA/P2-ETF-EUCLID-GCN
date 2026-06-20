"""
Main training script for P2-ETF-EUCLID-GCN engine.
"""

import json
import pandas as pd
import numpy as np

import config
import data_manager
from gcn_model import GCNPredictor
import push_results

def run_gcn():
    print(f"=== P2-ETF-EUCLID-GCN Run: {config.TODAY} ===")
    df_master = data_manager.load_master_data()
    df_master = df_master[df_master['Date'] >= config.TRAIN_START]

    macro = data_manager.prepare_macro_features(df_master)

    all_results = {}
    top_picks = {}

    # Default sequence length matching the model's default
    SEQ_LEN = getattr(config, 'SEQ_LEN', 10)

    for universe_name, tickers in config.UNIVERSES.items():
        print(f"\n--- Processing Universe: {universe_name} ---")
        returns = data_manager.prepare_returns_matrix(df_master, tickers)
        if len(returns) < config.MIN_OBSERVATIONS:
            continue

        full_returns = returns
        full_macro = macro.loc[full_returns.index].dropna()
        common_idx = full_returns.index.intersection(full_macro.index)
        full_returns = full_returns.loc[common_idx]
        full_macro = full_macro.loc[common_idx]

        graph_seq = data_manager.build_graph_sequence(full_returns, full_macro)

        raw_feat = graph_seq["features_seq"]
        raw_targets = graph_seq["targets"]

        # -----------------------------------------------------------------
        # CRITICAL FIX: Reshape flat time-series into sliding windows
        # Converts (Total_Days, Nodes, Features) -> (Samples, Seq_Len, Nodes, Features)
        # -----------------------------------------------------------------
        if raw_feat.ndim == 3:
            print(f"  Reshaping flat time-series into {SEQ_LEN}-day sliding windows for Temporal GCN...")
            T, N, F = raw_feat.shape
            if T <= SEQ_LEN:
                print(f"  Skipping {universe_name}: Not enough data ({T} days) for sequence length {SEQ_LEN}.")
                continue
            
            # Create sliding windows
            features_seq = np.array([raw_feat[i:i+SEQ_LEN] for i in range(T - SEQ_LEN + 1)])
            # The target corresponds to the day immediately following the end of the sequence
            targets_seq = raw_targets[SEQ_LEN - 1:] 
        else:
            # If data_manager already returns 4D sequences, use as-is
            features_seq = raw_feat
            targets_seq = raw_targets

        # Reconstruct the sequence dictionary with properly shaped data
        train_graph_seq = {
            "features_seq": features_seq,
            "targets": targets_seq,
            "edge_index": graph_seq["edge_index"],
            "num_etfs": graph_seq["num_etfs"],
            "etf_tickers": graph_seq["etf_tickers"],
            "target_scaler": graph_seq["target_scaler"]
        }

        predictor = GCNPredictor(
            in_dim=train_graph_seq["features_seq"].shape[-1],
            hidden_dim=config.HIDDEN_DIM,
            num_layers=config.NUM_LAYERS,
            lr=config.LEARNING_RATE,
            weight_decay=config.WEIGHT_DECAY,
            dropout=config.DROPOUT,
            seed=config.RANDOM_SEED,
            seq_len=SEQ_LEN  # Pass sequence length to the model
        )

        print(f"  Training on {len(train_graph_seq['features_seq'])} sequences...")
        predictor.fit(train_graph_seq, train_graph_seq["targets"],
                      epochs=config.EPOCHS, batch_size=config.BATCH_SIZE)

        # -----------------------------------------------------------------
        # CRITICAL FIX: Extract the exact final window for prediction
        # The new predict() method expects features_seq to be exactly (Seq_Len, Nodes, Feats)
        # -----------------------------------------------------------------
        predict_snapshot = {
            "features_seq": train_graph_seq["features_seq"][-1], 
            "edge_index": train_graph_seq["edge_index"],
            "num_etfs": train_graph_seq["num_etfs"],
            "etf_tickers": train_graph_seq["etf_tickers"]
        }

        preds = predictor.predict(predict_snapshot, train_graph_seq["target_scaler"])

        sorted_preds = sorted(preds.items(), key=lambda x: x[1], reverse=True)
        top3 = [{"ticker": t, "predicted_return": float(v)} for t, v in sorted_preds[:3]]
        top_picks[universe_name] = top3[0] if top3 else None
        all_results[universe_name] = top3

    output_payload = {
        "run_date": config.TODAY,
        "daily_trading": {
            "universes": all_results,
            "top_picks": top_picks
        }
    }

    push_results.push_daily_result(output_payload)
    print("\n=== Run Complete ===")

if __name__ == "__main__":
    run_gcn()
