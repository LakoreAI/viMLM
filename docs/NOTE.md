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

6. U-Net Style Bottlenecking Encoder Layers (UNetEncoderLayer)
Description: Introduce a U-Net style architecture where the hidden dimensions of intermediate layers shrink in the middle of the encoder stack (e.g., from 256 down to 168) and expand back to the base size, saving parameter counts in between.
Why: Reduces the overall model parameters (saving ~811K parameters in a 4-layer config) while keeping the same vocabulary representation size at the input and output boundaries.
Note: Implemented as an additional class (`UNetEncoderLayer` in `src/models/layers/encoder.py`), preserving the original `EncoderLayer` class for modularity.


## Suggested Improvements — References

### 1. Config-Driven Hyperparameters
Currently `learning_rate`, `weight_decay`, and `warmup_ratio` in the YAML are not wired to the optimizer.
Standard practice: read all training hyperparameters from a single config object so experiments are fully reproducible from the config file alone.
Reference:
- HuggingFace TrainingArguments: https://huggingface.co/docs/transformers/main_classes/trainer#transformers.TrainingArguments

### 2. Model Checkpointing
`save_best`, `save_last`, `save_steps` are parsed from YAML but never acted on.
Reference:
- PyTorch `torch.save` / `torch.load` documentation: https://pytorch.org/docs/stable/generated/torch.save.html
- HuggingFace Trainer checkpoint pattern: https://huggingface.co/docs/transformers/main_classes/trainer#checkpoints

### 3. Mid-Epoch Evaluation (eval_steps)
Evaluating only once per epoch is insufficient for large pretraining runs.
Reference:
- Liu et al. (2019): "RoBERTa: A Robustly Optimized BERT Pretraining Approach" (arXiv:1907.11692)
  - Section 4.1 discusses the importance of frequent evaluation checkpoints during pretraining.

### 4. head_dim Assertion in MultiHeadSelfAttention
Without `assert hidden_size % num_heads == 0`, mismatched configs cause silent integer truncation.
Reference:
- Vaswani et al. (2017): "Attention Is All You Need" (arXiv:1706.03762)
  - Requires d_model divisible by h (num_heads); this is a foundational constraint.

### 5. W&B `reinit` Deprecation Fix
`reinit=True` is deprecated since wandb>=0.28.
Reference:
- W&B Python SDK `wandb.init()` migration guide: https://docs.wandb.ai/ref/python/init/
  - Use `reinit="finish_previous"` to finish the current run and start a new one.

### 6. BPE Regex Caching
`BPETokenizer._tokenize_word()` recompiles a regex per merge rule per word, which is O(merges × words) at inference.
Reference:
- HuggingFace `tokenizers` library (Rust-backed): https://github.com/huggingface/tokenizers
  - Compiles merge patterns once during `from_pretrained()` / `train()`.
- Sennrich et al. (2016): "Neural Machine Translation of Rare Words with Subword Units" (arXiv:1508.07909)
  - Original BPE paper; standard implementation pre-sorts and pre-compiles merge rules.

### 7. Random Replacement Token Range Hardening
The current `torch.randint(len(self.special_ids), self.vocab_size, ...)` relies on specials being at IDs [0, N).
Reference:
- Devlin et al. (2018): "BERT: Pre-training of Deep Bidirectional Transformers" (arXiv:1810.04805)
  - Section 3.1 specifies the 80/10/10 masking rule; random tokens must be drawn from the full non-special vocabulary.
- HuggingFace `DataCollatorForLanguageModeling`: https://github.com/huggingface/transformers/blob/main/src/transformers/data/data_collator.py
  - Uses `special_tokens_mask` to exclude specials from random candidates.


## Architectural Improvements — References

### A. SwiGLU Feed-Forward Network
Current FFN: `Linear(d, ff_dim) → GELU → Linear(ff_dim, d)`
SwiGLU variant: `(Linear(d, ff_dim) ⊙ SiLU(Linear(d, ff_dim))) → Linear(ff_dim, d)`
Two gates (W and V) replace the single projection. Standard practice is to scale ff_dim by 2/3 to keep parameter count equal.
Why: The gating mechanism allows the network to selectively suppress irrelevant activations, consistently improving perplexity on language tasks.
Used in: LLaMA (Touvron et al., 2023), PaLM (Chowdhery et al., 2022), Gemma (Google, 2024).
Reference:
- Shazeer (2020): "GLU Variants Improve Transformer" (arXiv:2002.05202) — https://arxiv.org/abs/2002.05202

### B. Flash Attention (via PyTorch 2.0+ SDPA)
The current implementation manually computes attention as: `softmax((Q @ K.T) / sqrt(d)) @ V`
This materializes the full (B, H, L, L) attention matrix in HBM memory — O(L²) cost.
PyTorch 2.0+ `F.scaled_dot_product_attention(Q, K, V, attn_mask=mask, dropout_p=p)` uses the FlashAttention kernel internally: tiles the computation to stay in SRAM, O(L) memory, 2–4× faster wall-clock time.
Migration is a one-line change in `attention.py` forward().
Reference:
- Dao et al. (2022): "FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness" (arXiv:2205.14135) — https://arxiv.org/abs/2205.14135
- PyTorch docs: https://pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention.html

### C. Whole Word Masking (WWM) — Critical for Vietnamese
Standard token-level MLM masks subword tokens independently. For Vietnamese, which writes multi-syllable compound words with spaces (e.g., "học sinh" = student, "Việt Nam" = Vietnam), a BPE/WordPiece tokenizer may split words into multiple tokens. Masking one subword while leaving its neighbor visible makes the task trivially easy.
WWM masks all subword tokens that originated from the same surface word simultaneously.
Implementation: after encoding a sentence, build a `word_ids` list (one int per token indicating which word it came from), then extend the per-word masking decision to all its tokens.
Reference:
- Cui et al. (2021): "Pre-Training with Whole Word Masking for Chinese BERT" (arXiv:1906.08101) — https://arxiv.org/abs/1906.08101
  - Directly analogous to Vietnamese: character-segmented script where standard MLM leaks word-level information.
- Devlin et al. (2018): BERT appendix discusses WWM as a post-hoc improvement — https://arxiv.org/abs/1810.04805

### D. Cross-Layer Parameter Sharing (ALBERT)
Instead of N independent EncoderLayer instances, instantiate one and call it N times:
```python
layer = EncoderLayer(hidden_size, ff_dim, num_heads, dropout)
for _ in range(num_layers):
    x = layer(x, mask)
```
Effect: reduces trainable parameters by ~N× with minimal loss in representational power (the shared weights learn to be universally useful across all depths).
Enables training with hidden_size = 768 at the parameter cost of hidden_size = 256.
Reference:
- Lan et al. (2019): "ALBERT: A Lite BERT for Self-supervised Learning of Language Representations" (arXiv:1909.11942) — https://arxiv.org/abs/1909.11942

### E. Stochastic Depth / LayerDrop
During training, each encoder layer is randomly dropped (identity skip applied) with probability `p`:
```python
if self.training and torch.rand(1).item() < self.drop_prob:
    return x  # skip this layer entirely
```
`p` is typically linearly increased from 0 (first layer) to `max_drop` (last layer) — earlier layers are more important.
Serves as both a regularizer and a speed-up (fewer FLOPs per step on average).
At inference: all layers are used; the residual is scaled by `(1 - p)` to maintain expected activations.
Reference:
- Fan et al. (2019): "Reducing Transformer Depth on Demand with Structured Dropout" (arXiv:1909.11556) — https://arxiv.org/abs/1909.11556
- Huang et al. (2016): "Deep Networks with Stochastic Depth" (arXiv:1603.09382) — original stochastic depth paper for ResNets


## Research-Backed Improvements (2024) — References

### F. ModernBERT (2024) — Lessons for this Codebase
ModernBERT (Werner et al., Dec 2024) is the most comprehensive recent study on modernizing BERT-style encoder pretraining. Key takeaways directly applicable here:

| Change | Original BERT | ModernBERT |
|---|---|---|
| Masking ratio | 15% | **30%** |
| Positional encoding | Absolute | **RoPE** ✅ (already done) |
| Activation | GELU | **GeGLU / SwiGLU** |
| Bias in linear layers | Yes | **No** (except output) |
| Padding strategy | Zero-pad to max_len | **Unpadding (sequence packing)** |
| Attention | Standard O(L²) | **Flash Attention** |
| Context length | 512 tokens | 8192 tokens |

Reference:
- Werner et al. (2024): "Smarter, Better, Faster, Longer: A Modern Bidirectional Encoder for Fast, Memory Efficient, and Long Context Finetuning and Inference" (arXiv:2412.13663) — https://arxiv.org/abs/2412.13663
- HuggingFace model card: https://huggingface.co/answerdotai/ModernBERT-base

### G. Increased Masking Ratio (30%)
The original 15% rate was chosen for BERT but provides high gradient variance (85% of tokens produce no signal per step).
Raising to 30% gives the model 2× more learning signal per forward pass, reducing the number of steps needed to converge.
Where to change: `mlm_probability` in `training_config.yml` and the `BertDataCollator` constructor call in `__main__.py`.
Reference:
- Werner et al. (2024): ModernBERT (arXiv:2412.13663)

### H. Linear Bias Removal
Removing `bias=True` from all `nn.Linear` layers in attention and FFN is a modern default (GPT-3, LLaMA, ModernBERT).
The bias term is redundant when LayerNorm follows every sublayer (LayerNorm has its own learnable offset β).
Removes a small but non-trivial number of parameters at zero quality cost.
Reference:
- Werner et al. (2024): ModernBERT — Section 3.2 "Architecture"
- Su et al. (2023): LLaMA architecture choices — https://arxiv.org/abs/2302.13971

### I. Vietnamese Word Segmentation (VnCoreNLP)
Vietnamese is morpho-syllabic: each written syllable is a semantic unit, but ~85% of word types are multi-syllabic.
Without pre-segmentation, a BPE tokenizer trained on raw Vietnamese text will over-split words, producing poor subword boundaries.
PhoBERT's solution: run RDRSegmenter (VnCoreNLP) first, which joins syllables into underscore-linked words, then apply BPE on the segmented output.
Example: "học sinh trường đại học" → "học_sinh trường đại_học" → BPE subwords.
Reference:
- Nguyen & Tuan Nguyen (2020): "PhoBERT: Pre-trained language models for Vietnamese" (arXiv:2003.00744) — https://arxiv.org/abs/2003.00744
- VnCoreNLP toolkit: https://github.com/vncorenlp/VnCoreNLP

### J. Vietnamese Evaluation Benchmarks
MLM loss alone is not a reliable indicator of downstream NLP quality.
Standard practice is to evaluate on task-specific benchmarks:
- **VN-MTEB**: Vietnamese Massive Text Embedding Benchmark — measures retrieval, classification, clustering, STS
  - https://huggingface.co/spaces/mteb/leaderboard
- **VLSP**: Vietnam Language and Speech Processing shared tasks — NER, POS tagging, dependency parsing
  - http://vlsp.org.vn/
- **ViMMRC**: Vietnamese Machine Reading Comprehension — multiple-choice reading comprehension
Reference:
- NLP Progress Vietnamese: http://nlpprogress.com/vietnamese/

### K. Dynamic Sequence Packing
Current training pads every sequence to `max_seq_len = 128`, meaning short sentences waste ~40–60% of GPU FLOPs on padding tokens.
Sequence packing concatenates multiple samples (with a separator) into a single `max_seq_len` context, using a block-diagonal attention mask to prevent cross-sample attention.
Implementation note: requires attention mask changes — works natively with `F.scaled_dot_product_attention(attn_mask=...)`.
Benchmark: ModernBERT reports this as "unpadding" and cites it as one of the primary throughput gains.
Reference:
- HuggingFace `DataCollatorWithFlattening`: https://huggingface.co/docs/transformers/main/en/main_classes/data_collator
- Zeng et al. (2023): "FlexPacking: Flexible Sequence Packing for Efficient LLM Training"

### L. Grouped Query Attention (GQA)
Standard MHA has H Q-heads, H K-heads, and H V-heads. Each head has `d_k = hidden_size / H` dimensions.
GQA reduces K/V to `G` groups (G < H): each group of `H/G` Q-heads shares one K-head and one V-head.
This cuts the KV projection parameters by `H/G` and reduces memory bandwidth — critical at long sequence lengths.
Special case G=1 is Multi-Query Attention (MQA), used in Falcon and PaLM.
Reference:
- Ainslie et al. (2023): "GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints" (arXiv:2305.13245) — https://arxiv.org/abs/2305.13245
- Shazeer (2019): "Fast Transformer Decoding: One Write-Head is All You Need" (MQA original) — https://arxiv.org/abs/1911.02150


## Practical Implementation References

### M. ELECTRA-style Replaced Token Detection (RTD)
The key insight: MLM trains on only ~15–30% of tokens per step. RTD trains on 100% of tokens every step, making it 4× more sample-efficient.

Architecture (two models trained jointly):
- **Generator** (small, e.g., 1/3 the hidden size): a standard MLM that fills in `[MASK]` tokens with plausible replacements
- **Discriminator** (your main model): binary classifier on every token — is this the original token or a generated replacement?

Only the discriminator is kept after pretraining. The generator is discarded.

Loss: `L = L_MLM_generator + λ · L_RTD_discriminator` (λ ≈ 50)

Vietnamese prior art on HuggingFace:
- `NlpHUST/vi-electra-small` — https://huggingface.co/NlpHUST/vi-electra-small
- `aiface/velectra-base_v2` — https://huggingface.co/aiface/velectra-base_v2

Reference:
- Clark et al. (2020): "ELECTRA: Pre-training Text Encoders as Discriminators Rather Than Generators" (arXiv:2003.10555) — https://arxiv.org/abs/2003.10555

### N. Gradient Checkpointing
How to enable in this codebase (two options):

Option 1 — model-level (simplest, if using HuggingFace PreTrainedModel base):
```python
model.gradient_checkpointing_enable()
```

Option 2 — manual wrapping inside `BertEncoder.forward()`:
```python
from torch.utils.checkpoint import checkpoint
for layer in self.layers:
    x = checkpoint(layer, x, mask, use_reentrant=False)
```
Note: use `use_reentrant=False` (PyTorch ≥ 2.0) to avoid issues with dropout RNG state.

Memory saved: 50–80% of activation memory. Training time cost: ~20–30% extra (recomputes forward once per layer in backward).

Reference:
- PyTorch `torch.utils.checkpoint` — https://pytorch.org/docs/stable/checkpoint.html
- Chen et al. (2016): "Training Deep Nets with Sublinear Memory Cost" (arXiv:1604.06174)

### O. `torch.compile()` (PyTorch 2.0+)
One line in `__main__.py` after `model.to(device)`:
```python
model = torch.compile(model)
```
Internally uses TorchInductor + Triton to fuse kernels (e.g., attention + dropout + layernorm fused into one GPU kernel).
Typical speedup: 1.5–2× on A100/H100, less on older GPUs. Use `mode="reduce-overhead"` for small models or `mode="max-autotune"` for maximum throughput.
Note: first batch is slow (compilation); subsequent batches are fast.

Reference:
- PyTorch 2.0 — https://pytorch.org/get-started/pytorch-2.0/

### P. Text Deduplication
Duplicate text is the #1 source of unintentional memorisation and wasted compute in pretraining.
Two practical approaches:

**Exact deduplication** (fast, good for sentence-level):
```python
seen = set()
deduped = [s for s in sentences if s not in seen and not seen.add(s)]
```

**Near-duplicate deduplication** (MinHash LSH, catches paraphrases):
```python
from datasketch import MinHash, MinHashLSH
# pip install datasketch
```
Recommended: exact dedup first, then MinHash with Jaccard threshold = 0.8 on character n-grams.

Reference:
- Lee et al. (2022): "Deduplicating Training Data Makes Language Models Better" (arXiv:2107.06499) — https://arxiv.org/abs/2107.06499

### Q. Vietnamese Pretraining Data Sources
The current config uses an English HuggingFace dataset. Replace with:

| Dataset | Size | Notes |
|---|---|---|
| BKAINewsCorpus | 53 GB / 32M articles | Clean Vietnamese news, ideal for pretraining |
| OSCAR vi | ~8 GB | Filtered CommonCrawl Vietnamese |
| Wikipedia vi | ~1 GB | High quality, encyclopedic |
| VietMix | Small | Vietnamese–English code-mixed text |

Load with HuggingFace `datasets`:
```python
from datasets import load_dataset
ds = load_dataset("bkai-foundation-models/BKAINewsCorpus", split="train")
```

Links:
- https://huggingface.co/datasets/bkai-foundation-models/BKAINewsCorpus
- https://huggingface.co/datasets/oscar-corpus/OSCAR-2301 (language=`vi`)
- https://huggingface.co/datasets/wikimedia/wikipedia (language=`vi`)

### R. Curriculum Learning by Sequence Length
Easiest implementation: sort the sentence list by token count before creating the dataset.
```python
sentences.sort(key=lambda s: len(tokenizer.encode(s)))
```
More advanced: use a `ScheduledSampler` that gradually increases the max allowed length each epoch.

Why this helps:
1. Early batches have very little padding → higher GPU utilisation immediately
2. The model learns simple patterns first, then builds to complex ones
3. Prevents the model getting stuck on long, noisy examples at initialisation

Reference:
- Platanios et al. (2019): "Competence-based Curriculum Learning for Neural Machine Translation" (arXiv:1903.09848)
- Bengio et al. (2009): "Curriculum Learning" (original paper)

### S. Gradient Norm Logging
Grad clipping already happens in `training_pipe.py`. The return value of `clip_grad_norm_` is the raw norm before clipping — log it:
```python
grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
# then pass grad_norm to on_step_end callback
```
Spiking norms = unstable training. Flat norms at `max_norm` = clipping too aggressively (LR may be too high).

Reference:
- Pascanu et al. (2013): "On the difficulty of training recurrent neural networks" — introduced gradient clipping

### T. Seed Everything
```python
import random, numpy as np, torch

def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
```
Call this at the very top of `main()`, before any data loading or model init, using `cfg.seed` (already parsed from YAML as `seed: 18210`).

Reference:
- HuggingFace `transformers.set_seed()` — https://huggingface.co/docs/transformers/main_classes/utilities


## Maximum Impact References

### U. Whole Word Masking + `word_segmenter.py` Integration
`word_segmenter.py` uses `underthesea.word_tokenize(text, format="text")` which returns a space-joined string of underscore-linked words (e.g., `"học_sinh trường đại_học"`).

To use it in `BertDataCollator`:
1. Call `segment_words(sentence)` to get the segmented string
2. Tokenize the segmented string; record which token index belongs to which word (`word_ids`)
3. In `torch_mask_tokens()`, select words to mask (not individual tokens), then mark all tokens of that word as masked

```python
# Pseudocode for word_ids construction
segmented = segment_words(sentence)          # "học_sinh trường đại_học"
tokens = tokenizer.encode(segmented)         # [101, 123, 456, 789, 102]
# word_ids: [None, 0, 1, 2, None] — None for [CLS]/[SEP]
```

Reference:
- Cui et al. (2021): "Pre-Training with Whole Word Masking for Chinese BERT" (arXiv:1906.08101)
- `underthesea` docs: https://underthesea.readthedocs.io/

### V. SpanBERT Span Masking
Instead of choosing individual tokens to mask, sample a *span length* `l` from a geometric distribution `Geo(p=0.2)` clamped to [1, 10], then mask `l` consecutive tokens starting at a random position.

Geometric sampling in Python:
```python
import numpy as np
span_len = min(np.random.geometric(p=0.2), 10)
start = random.randint(0, seq_len - span_len)
mask[start:start + span_len] = True
```

Also add the **Span Boundary Objective (SBO)**: predict the masked span tokens using only the two boundary tokens on each side. This teaches the model to encode span content into boundary representations — critical for NER and QA tasks.

Reference:
- Joshi et al. (2020): "SpanBERT: Improving Pre-training by Representing and Predicting Spans" (arXiv:1907.10529) — https://arxiv.org/abs/1907.10529

### W. 8-bit AdamW (bitsandbytes)
AdamW stores two fp32 accumulators per parameter (`m` and `v`). For a model with 10M parameters that's 80 MB just for optimizer states.
8-bit Adam quantises these accumulators into unsigned int8 with per-block dynamic scaling, recovering fp32-level precision during the update step.

```python
# pip install bitsandbytes
import bitsandbytes as bnb
optimizer = bnb.optim.AdamW8bit(
    model.parameters(), lr=lr, weight_decay=weight_decay
)
```

Memory saved: 75% of optimizer state RAM. Quality impact: negligible on models < 1B parameters.

Reference:
- Dettmers et al. (2022): "8-bit Optimizers via Block-wise Quantization" (arXiv:2110.02861) — https://arxiv.org/abs/2110.02861
- bitsandbytes library: https://github.com/TimDettmers/bitsandbytes

### X. BF16 vs FP16 for AMP
| Property | FP16 | BF16 |
|---|---|---|
| Exponent bits | 5 | 8 (same as FP32) |
| Mantissa bits | 10 | 7 |
| Max value | ~65,504 | ~3.4 × 10³⁸ |
| Overflow risk | High | Near zero |
| Loss scaling needed | Yes | No |
| Hardware support | All modern GPUs | A100, RTX 30xx+, TPU |

BF16 is strictly better for training stability. Only use FP16 if your GPU doesn't support BF16 (e.g., V100, RTX 20xx).

```python
# In training_pipe.py
with torch.cuda.amp.autocast(dtype=torch.bfloat16):
    outputs = model(...)
```

Note: when using BF16, you can remove the `GradScaler` entirely — it's only needed for FP16's narrow dynamic range.

### Y. DeBERTa / ViDeBERTa Disentangled Attention
Standard attention computes: `Attn(Q, K) = softmax(QKᵀ / √d)`
where Q and K encode both *content* and *absolute position*.

DeBERTa disentangles them:
```
Attn = c2c + c2p + p2c
```
- `c2c`: content-to-content (standard attention)
- `c2p`: content-to-position (each word attends to relative positions)
- `p2c`: position-to-content (each position attends to surrounding content)

Why it's better: the model can independently learn *what* to attend to (content) and *where* (position), rather than conflating the two in a single embedding.

ViDeBERTa results on Vietnamese benchmarks:
- POS tagging: 94.7% F1 (+0.3 vs PhoBERT)
- NER: 91.6% F1 (+1.2 vs PhoBERT)
- QA (UIT-ViQuAD): 78.9 EM (+2.1 vs PhoBERT)

Reference:
- He et al. (2021): "DeBERTa: Decoding-enhanced BERT with Disentangled Attention" (arXiv:2006.03654) — https://arxiv.org/abs/2006.03654
- Tran et al. (2023): "ViDeBERTa: A powerful pre-trained language model for Vietnamese" — https://aclanthology.org/2023.findings-emnlp.293/

### Z. Knowledge Distillation from PhoBERT
Distillation transfers the *soft probability distributions* from a large teacher to a small student, giving the student access to the teacher's uncertainty and generalisation patterns.

Loss function:
```
L_total = α · L_CE(student_logits, hard_labels)
        + (1-α) · KL(softmax(student/T), softmax(teacher/T))
```
- `T` = temperature (typically 4–8); softens the probability distributions
- `α` = 0.5 is a common starting point
- Teacher: `vinai/phobert-base-v2` (loaded frozen, no gradient)
- Student: your model

```python
from transformers import AutoModel
teacher = AutoModel.from_pretrained("vinai/phobert-base-v2").eval()
for p in teacher.parameters():
    p.requires_grad = False
```

Reference:
- Hinton et al. (2015): "Distilling the Knowledge in a Neural Network" (arXiv:1503.02531) — https://arxiv.org/abs/1503.02531
- PhoBERT-base-v2: https://huggingface.co/vinai/phobert-base-v2

### Summary: Recommended Execution Order
```
1.  seed_everything()                    done
2.  wire lr/wd/warmup from cfg           done
3.  BF16 AMP                             trivial — 1 line
4.  torch.compile()                      trivial — 1 line
5.  gradient checkpointing               trivial — 1 line
6.  WWM via word_segmenter.py            segmenter built — wired in
7.  8-bit AdamW                          1 import change
8.  Switch to Vietnamese corpus          config change
9.  SpanBERT span masking                low complexity
10. ELECTRA RTD                          medium — new training loop
11. DeBERTa attention                    medium — new attention class
12. Knowledge distillation               medium — new pipe
13. DDP multi-GPU                        medium — launcher change
```

## Word Segmentation vs. Token-Level Masking for Vietnamese MLM

### Overview
In Vietnamese, spaces are used to separate syllables rather than words. Around 85% of Vietnamese words are multi-syllabic compounds consisting of multiple space-separated syllables (e.g., "học sinh" = student, "đại học" = university). 

### The Problem: Token-Level Masking
If pretraining is performed without word segmentation:
1. The subword tokenizer treats spaces as boundaries, split-encoding multi-syllable compounds into separate tokens (e.g., "học" and "sinh").
2. During training, if "sinh" is masked but "học" is left visible, the model can trivially predict "sinh" based on local spelling collocation.
3. This creates a low pretraining loss but prevents the model from learning deeper syntactic and semantic representations of the overall sentence structure.

### The Solution: Word Segmentation & Whole Word Masking (WWM)
Applying word segmentation (e.g., converting "học sinh" to "học_sinh") and masking all subwords of a word together (WWM) fixes this issue:
1. The tokenizer views "học_sinh" as a single compound entity.
2. If selected for masking, all subwords of the compound are masked together.
3. This forces the model to rely on global sentence-level semantics to predict the missing unit, leading to much richer contextualized representations.
4. Extrinsic benchmarks (such as PhoBERT and ViDeBERTa) demonstrate that word segmentation prior to pretraining yields significantly higher downstream transfer accuracy (F1-score) on classification and sequence labeling tasks.

---

## 3-Year SOTA Plan (2026-2029) to Make viMLM the Best Vietnamese NLU Model

Based on industry trends, encoder-only models are optimized using the following architectural and pretraining paradigm shifts to outperform older structures:

### 1. ModernBERT Architecture Upgrades
*   **GeGLU Activations:** Replace standard GELU feedforward layers with gated GeGLU/SwiGLU variants. To keep parameters constant, scale the intermediate dimension `ff_dim` down by approximately 2/3.
*   **Remove Linear Layer Biases:** Strip the `bias=True` parameter from all key, query, value, and output linear projections. Biases are redundant when LayerNorm/RMSNorm immediately follows, and removing them increases hardware throughput.
*   **Unpadding / Dynamic Sequence Packing:** Do not pad sequences to `max_seq_len` inside training batches. Instead, concatenate all training sentences into a single continuous stream separated by special tokens, and pass a block-diagonal attention mask to prevent attention cross-contamination. This increases training throughput by up to 2x.
*   **8k Context Windows:** Train using Rotary Position Embeddings (RoPE) and dynamically extend sequence context lengths up to 8,192 tokens using sequence length curriculum training.

### 2. Modern Pretraining Objectives
*   **Replaced Token Detection (RTD):** Transition training from standard MLM to an ELECTRA-style RTD task where a small generator model masks tokens and the target discriminator model (viMLM) predicts whether each token was replaced or is original. This trains on 100% of tokens in every step rather than 15-30% in MLM, improving sample efficiency by 4x.
*   **Instruction-Tuning during Pretraining:** Mix structured instruction-following datasets directly into the pretraining corpus, training the model to follow prompts zero-shot directly using its pretraining mask projection head.

### 3. Data Engineering & Curation
*   **MinHash LSH Deduplication:** Apply MinHash LSH deduplication on character n-grams to remove low-value, duplicate crawled content.
*   **Curated Academic / News Corpus:** Prioritize Bkainews, Wikipedia, and OSCAR-vi over general crawled web text.