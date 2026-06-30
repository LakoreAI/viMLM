# NOTE


## 30/06/2026

Proposed Improvements for viMLM Pretraining Pipeline
This document proposes several modern architectural and training optimizations for viMLM based on current best practices in deep learning and NLP.

1. Weight Tying (Weight Sharing)
Description: Share the weight matrix between the input token embedding layer (BertEmbeddings.token_emb.weight) and the final output classification layer in the MLM projection head (self.mlm_head[-1].weight).
Why: The input embeddings map tokens to dense vectors, while the output MLM head projects dense vectors back to vocabulary probabilities. Using the same weights (transposed) regularizes the representations, reduces overfitting, and decreases the number of trainable parameters by $V \times H$ (where $V$ is vocabulary size and $H$ is hidden dimension). For a 30k vocabulary and 768 hidden dimension, this saves 23 million parameters!
PyTorch Code:
python
self.mlm_head[-1].weight = self.embeddings.token_emb.weight
References:
Press & Wolf (2017): "Using the Output Embedding to Improve Language Models" (EACL).
Devlin et al. (2018): "BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding" (arXiv:1810.04805).
2. Vectorized Dynamic Masking in Dataloader
Description: Refactor BertDataCollator to perform vectorized batch masking using PyTorch tensor operations (torch.bernoulli, torch.where, tensor indexing) instead of iterating token-by-token in a python loop.
Why: Pure Python loops over each token in each batch sequence represent a massive CPU bottleneck that prevents maximum GPU utilization (keeps step-per-second throughput low).
PyTorch Code:
python
# Instead of token-by-token loop:
probability_matrix = torch.full(input_ids.shape, self.mlm_probability)
# Set special tokens mask probability to 0.0...
masked_indices = torch.bernoulli(probability_matrix).bool()
labels[~masked_indices] = -100 # only calculate loss on masked tokens
References:
Hugging Face DataCollatorForLanguageModeling: Industry standard vectorized MLM masking implementation.
Liu et al. (2019): "RoBERTa: A Robustly Optimized BERT Pretraining Approach" (arXiv:1907.11692).
3. Rotary Position Embeddings (RoPE)
Description: Replace static absolute position embeddings with Rotary Position Embeddings (RoPE). RoPE encodes relative positions by multiplying key and query representations by a rotation matrix.
Why: Absolute position embeddings do not generalize well to sequences longer than max_seq_len seen during training. RoPE enables the model to extrapolate to longer context lengths and captures relative distance relationships naturally.
References:
Su et al. (2021): "RoFormers: Enhanced Transformer with Rotary Position Embedding" (arXiv:2104.09864).
Used in modern state-of-the-art models like LLaMA, Mistral, and Qwen.
4. Automatic Mixed Precision (AMP)
Description: Integrate PyTorch Automatic Mixed Precision (torch.cuda.amp.autocast and torch.cuda.amp.GradScaler) in the training loop.
Why: Modern GPUs support FP16/BF16 tensor cores. Using mixed precision training speeds up steps-per-second by 2x to 3x, halves memory (allowing larger batch sizes), and maintains numerical stability via dynamic loss scaling.
PyTorch Code:
python
scaler = torch.cuda.amp.GradScaler()
with torch.cuda.amp.autocast():
    out = model(input_ids, segment_ids, mask=attn_mask, mlm_labels=mlm_labels)
    loss = out["loss"]
scaler.scale(loss).backward()
scaler.step(optimizer)
scaler.update()
References:
Micikevicius et al. (2017): "Mixed Precision Training" (ICLR).
5. Configurable Cosine Decay Learning Rate Scheduler
Description: Replace the linear warmup & decay scheduler with a Cosine Annealing learning rate schedule with warmup.
Why: Cosine decay allows the learning rate to decrease slowly at the beginning, accelerate the decrease in the middle, and flatten out towards the end. It is empirically shown to yield lower final validation loss and better downstream generalization compared to linear schedules.
References:
Loshchilov & Hutter (2016): "SGDR: Stochastic Gradient Descent with Warm Restarts" (arXiv:1608.03983).
Used in GPT-3, RoBERTa, and LLaMA pretraining runs.