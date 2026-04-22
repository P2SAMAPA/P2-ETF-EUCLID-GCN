# P2-ETF-EUCLID-GCN

**Euclidean Graph Convolutional Network for ETF‑Macro Relational Learning**

[![Daily Run](https://github.com/P2SAMAPA/P2-ETF-EUCLID-GCN/actions/workflows/daily_run.yml/badge.svg)](https://github.com/P2SAMAPA/P2-ETF-EUCLID-GCN/actions/workflows/daily_run.yml)
[![Hugging Face Dataset](https://img.shields.io/badge/🤗%20Dataset-p2--etf--euclid--gcn--results-blue)](https://huggingface.co/datasets/P2SAMAPA/p2-etf-euclid-gcn-results)

## Overview

`P2-ETF-EUCLID-GCN` constructs a bipartite graph connecting ETFs and macro factors, then trains a **Graph Convolutional Network (GCN)** on the full history from 2008 to present. The model learns relational embeddings that capture how macro conditions influence ETF returns, producing daily predictions for next‑day returns.

## Methodology

- **Graph Construction**: Nodes = ETFs + macro factors; edges connect all ETFs to all macro nodes.
- **Temporal GCN**: Multiple GCN layers process daily snapshots; batch normalization and dropout for regularization.
- **Prediction**: Final linear layer outputs a scalar predicted return per ETF.
