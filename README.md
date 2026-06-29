🇻🇳 ViMLM — Vietnamese Masked Language Model

A BERT-based Masked Language Model pre-trained from scratch on Vietnamese text, built with pure PyTorch.


## Overview

ViMLM pre-trains a BERT encoder on Vietnamese corpora using the Masked Language Modeling (MLM) objective. The goal is to produce contextual word representations that capture Vietnamese morphology, tonal patterns, and syntax — suitable for fine-tuning on downstream NLP tasks.


## Why from scratch?
Existing multilingual models (mBERT, XLM-R) under-represent Vietnamese. Training on a dedicated Vietnamese corpus yields richer, domain-specific representations.


## References

- Devlin et al. (2018) — BERT: Pre-training of Deep Bidirectional Transformers
- Liu et al. (2019) — RoBERTa: A Robustly Optimized BERT Pretraining Approach
- Nguyen et al. (2020) — PhoBERT: Pre-trained language models for Vietnamese