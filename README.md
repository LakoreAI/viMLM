# ViMLM - Vietnamese Masked Language Model

A BERT-based Masked Language Model pre-trained from scratch on Vietnamese text, built with pure PyTorch.

## Overview

ViMLM pre-trains a BERT encoder on Vietnamese corpora using the Masked Language Modeling (MLM) objective. The goal is to produce contextual word representations that capture Vietnamese morphology, tonal patterns, and syntax — suitable for fine-tuning on downstream NLP tasks.

## Why from scratch?

Existing multilingual models (mBERT, XLM-R) under-represent Vietnamese. Training on a dedicated Vietnamese corpus yields richer, domain-specific representations.

## Project Structure

- `src/`: Python package containing the model components, pipelines, datasets, and callbacks.
- `notebooks/`: Contains the standalone pretraining notebook.
- `config/`: Pretraining configuration files.
- `data/`: Raw corpora for training and evaluation.

## Standalone Notebook (Google Colab / Kaggle)

For quick testing or training in cloud environments with zero local setup:
- [viMLM_pretraining.ipynb](notebooks/viMLM_pretraining.ipynb)

This notebook includes all modules (model, dataset, training logic) inline. You can open and execute it directly in Google Colab using a GPU instance.

## Getting Started Locally

### 1. Environment Setup

We recommend using the fast `uv` package manager for virtual environment setup and dependency installation:

```bash
# Create python virtual environment
python3 -m venv .venv

# Install dependencies
uv pip install --python .venv "torch>=2.4.1" huggingface_hub==0.35.3 PyYAML==6.0.2 transformers==4.53.2 wandb==0.28.0
```

### 2. Running Training

To run the Masked Language Model pretraining locally:

```bash
# Start pretraining with config parameters
.venv/bin/python -m src
```

## References

- Devlin et al. (2018) — BERT: Pre-training of Deep Bidirectional Transformers
- Liu et al. (2019) — RoBERTa: A Robustly Optimized BERT Pretraining Approach
- Nguyen et al. (2020) — PhoBERT: Pre-trained language models for Vietnamese