# TODO

- [x] Implement model and pipeline optimizations:
  - [x] Implement weight tying between token embeddings and the output MLM projection head (Press & Wolf, 2017)
  - [x] Vectorize the dynamic masking in `BertDataCollator` to eliminate CPU/Python token-level loops
  - [x] Implement Rotary Position Embeddings (RoPE) (Su et al., 2021) for improved relative positional representation
  - [x] Integrate PyTorch Automatic Mixed Precision (AMP) (FP16/BF16) into the training pipeline
  - [x] Support Cosine Decay learning rate scheduler with warmup
- [x] Logging & Tracking:
  - [x] Add Weights & Biases (wandb) logging using the predefined `WandbCallback` for train and eval steps
- [x] Tokenizer Pipeline:
  - [x] Implement pipeline in `src/pipe/build_tokenizer.py` to train and serialize custom tokenizers (BPE, WordPiece, Unigram, Char)