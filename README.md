# P2-ETF-HYPERBOLIC-GNN

**Hyperbolic Graph Neural Networks for Hierarchical ETF‑Macro Modeling**

[![Daily Run](https://github.com/P2SAMAPA/P2-ETF-HYPERBOLIC-GNN/actions/workflows/daily_run.yml/badge.svg)](https://github.com/P2SAMAPA/P2-ETF-HYPERBOLIC-GNN/actions/workflows/daily_run.yml)
[![Hugging Face Dataset](https://img.shields.io/badge/🤗%20Dataset-p2--etf--hyperbolic--gnn--results-blue)](https://huggingface.co/datasets/P2SAMAPA/p2-etf-hyperbolic-gnn-results)

## Overview

`P2-ETF-HYPERBOLIC-GNN` constructs a bipartite graph of ETFs and macro factors, then trains a **Hyperbolic Graph Convolutional Network (HGCN)** in the Poincaré ball to learn hierarchical embeddings. These embeddings are used to predict next‑day ETF returns, ranking them per universe.

## Methodology

- **Graph Construction**: Nodes = ETFs + macro features; edges connect all ETFs to all macro nodes.
- **Hyperbolic Embeddings**: HGCN layers map node features into the Poincaré ball, capturing hierarchical relationships.
- **Prediction**: Final layer outputs a scalar predicted return per ETF.
