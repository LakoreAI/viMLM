# TODO

- [x] Implement model and pipeline optimizations:
  - [x] Implement weight tying between token embeddings and the output MLM projection head (Press & Wolf, 2017)
  - [x] Vectorize the dynamic masking in `BertDataCollator` to eliminate CPU/Python token-level loops
  - [x] Implement Rotary Position Embeddings (RoPE) (Su et al., 2021) for improved relative positional representation
  - [x] Integrate PyTorch Automatic Mixed Precision (AMP) (FP16/BF16) into the training pipeline
  - [x] Support Cosine Decay learning rate scheduler with warmup
  - [x] Add U-Net style bottlenecking encoder layers (`UNetEncoderLayer`) to save parameters in intermediate layers
- [x] Logging & Tracking:
  - [x] Add Weights & Biases (wandb) logging using the predefined `WandbCallback` for train and eval steps
- [x] Tokenizer Pipeline:
  - [x] Implement pipeline in `src/pipe/build_tokenizer.py` to train and serialize custom tokenizers (BPE, WordPiece, Unigram, Char)


- [x] **Config: Connect YAML hyperparameters to the optimizer**
  - Fields `learning_rate`, `weight_decay`, and `warmup_ratio` in `training_config.yml` are currently ignored; the optimizer uses CLI defaults silently
  - Read these from `Config` and pass them to `AdamW` and the scheduler in `__main__.py`

- [x] **Training: Implement model checkpointing**
  - `save_best`, `save_last`, and `save_steps` are parsed in the YAML but never acted on
  - Add a `CheckpointCallback` or inline logic in `training_pipe.py` to save model weights using `torch.save`
  - Reference: HuggingFace Trainer checkpointing pattern

- [x] **Training: Honor `eval_steps` for mid-epoch evaluation**
  - Currently, evaluation only runs once per epoch regardless of the `eval_steps` config
  - Add a step counter check inside `train_epoch()` to trigger `eval_epoch()` every `eval_steps` global steps
  - Reference: Liu et al. (2019) RoBERTa — frequent evaluation is key for monitoring large pretraining runs

- [x] **Model: Add `hidden_size % num_heads == 0` assertion in `MultiHeadSelfAttention`**
  - Currently silently truncates the head dimension, leading to subtle shape mismatches
  - Add `assert hidden_size % num_heads == 0` in `__init__`

- [x] **Callbacks: Fix deprecated W&B `reinit` argument**
  - `reinit=True` is deprecated as of `wandb>=0.28` and produces a warning on every run
  - Replace with `reinit="finish_previous"` in `WandbCallback.__init__`
  - Reference: W&B migration guide (https://docs.wandb.ai/ref/python/init/)

- [x] **Callbacks: Wire `log_artifact` into training loop for checkpoint saving**
  - `WandbCallback.log_artifact()` exists but is never called
  - Call it at the end of training (or at each `save_steps`) to track model artifacts in W&B

- [x] **Tokenizer: Cache compiled BPE merge patterns**
  - In `BPETokenizer._tokenize_word()`, regex patterns are recompiled on every word during inference
  - Pre-compile and cache all merge patterns at the end of `train()` into a list for fast encoding
  - Reference: HuggingFace tokenizers library — merge patterns are compiled once at load time

- [x] **Dataset: Harden random replacement token range in `BertDataCollator`**
  - `torch.randint(len(self.special_ids), self.vocab_size, ...)` assumes special token IDs are always in `[0, N_specials)`, which holds for custom tokenizers but may break with HuggingFace tokenizers
  - Use a proper `special_token_id_set` exclusion mask instead of relying on ID ordering

- [x] **Eval: Pass correct `step` in standalone `eval()`**
  - `eval_pipe.eval()` hardcodes `step=0` when calling `on_evaluate`, which misreports the step to W&B
  - Accept an optional `step` argument and pass it through

---

## Architectural Improvements

- [ ] **SwiGLU Feed-Forward Network**
  - Replace the current `Linear → GELU → Linear` FFN with the SwiGLU gated variant: `(xW ⊙ SiLU(xV)) → Linear`
  - Why: SwiGLU consistently outperforms GELU FFN on language tasks (used in LLaMA, PaLM, Gemma). No extra parameters if `ff_dim` is scaled by 2/3
  - Where: add `SwiGLUFeedForward` to `src/models/layers/feedforward.py`; toggle via config `use_swiglu: true`
  - Reference: Shazeer (2020) "GLU Variants Improve Transformer" (arXiv:2002.05202)

- [ ] **Flash Attention via `scaled_dot_product_attention`**
  - Replace the manual `Q @ K.T / sqrt(d) → softmax → @ V` in `MultiHeadSelfAttention` with `F.scaled_dot_product_attention(Q, K, V, attn_mask, dropout_p)`
  - Why: PyTorch 2.0+ implements FlashAttention (tiled, IO-aware) under the hood — same result, 2–4× faster, O(L) memory instead of O(L²). Zero code complexity cost
  - Where: `src/models/layers/attention.py`, guarded by `torch.__version__` check for backward compatibility
  - Reference: Dao et al. (2022) "FlashAttention: Fast and Memory-Efficient Exact Attention" (arXiv:2205.14135)

- [ ] **Whole Word Masking (WWM) for Vietnamese**
  - Instead of masking random subword tokens independently, mask *all* tokens that belong to the same word together
  - Why: Particularly important for Vietnamese — which uses multi-syllable compound words separated by spaces (e.g., "học sinh", "Việt Nam"). Standard random-token masking allows the model to "cheat" by reading adjacent subwords of the same word
  - Where: refactor `BertDataCollator.torch_mask_tokens()` to accept a `word_ids` map and apply masking at word granularity
  - Reference: Cui et al. (2021) "Pre-Training with Whole Word Masking for Chinese BERT" (arXiv:1906.08101) — same problem space for character-segmented languages

- [ ] **Cross-Layer Parameter Sharing (ALBERT-style)**
  - Share the weights of all `EncoderLayer` blocks — a single layer is instantiated once but applied `N` times
  - Why: Drastically reduces model size (e.g., from 10M → 2M params for a 4-layer model) while retaining depth of representation via recurrent application. Enables pretraining with much larger hidden sizes within the same parameter budget
  - Where: add `share_encoder_weights: true` option to `BertEncoder`; store one `EncoderLayer` and loop over it
  - Reference: Lan et al. (2019) "ALBERT: A Lite BERT for Self-supervised Learning of Language Representations" (arXiv:1909.11942)

- [ ] **Stochastic Depth (LayerDrop)**
  - During training, randomly drop entire encoder layers with probability `p` (skip the residual block entirely)
  - Why: Acts as a powerful regularizer for deep Transformers, reduces training time by skipping computation, and enables flexible depth at inference (trade speed for quality)
  - Where: wrap the residual addition in each `EncoderLayer.forward()` with a Bernoulli gate, controlled by `layer_drop_prob` in config
  - Reference: Fan et al. (2019) "Reducing Transformer Depth on Demand with Structured Dropout" (arXiv:1909.11556)

---

## Research-Backed Improvements (2024)

> Based on ModernBERT (Dec 2024), Vietnamese NLP literature, and recent pretraining efficiency work.

### Pretraining Objective

- [ ] **Increase MLM masking ratio from 15% → 30%**
  - The original 15% rate (Devlin 2018) is now considered suboptimal
  - ModernBERT (2024) increased it to 30%, reducing gradient variance and accelerating convergence
  - Where: `mlm_probability` in `training_config.yml` and `BertDataCollator`
  - Reference: Werner et al. (2024) "ModernBERT" (arXiv:2412.13663) — https://arxiv.org/abs/2412.13663

- [ ] **Remove bias from linear layers (except output)**
  - ModernBERT removes bias from all `nn.Linear` layers in attention and FFN (not the final MLM head)
  - Why: Reduces parameters ~1–2% with no measurable quality loss; also slightly improves training stability
  - Where: add `bias=False` to `W_q`, `W_k`, `W_v`, `W_o` in `MultiHeadSelfAttention` and `FeedForward`
  - Reference: Werner et al. (2024) "ModernBERT" (arXiv:2412.13663)

### Vietnamese-Specific

- [ ] **Integrate VnCoreNLP word segmentation before tokenization**
  - ~85% of Vietnamese word types are multi-syllabic; raw syllable-level text confuses subword tokenizers
  - PhoBERT (the SOTA Vietnamese encoder) applies `RDRSegmenter` from VnCoreNLP to group syllables into words before BPE: `"học sinh"` → `"học_sinh"`
  - Apply this as a preprocessing step in the data pipeline before `BertPreTrainDataset`
  - Reference: Nguyen & Tuan Nguyen (2020) "PhoBERT: Pre-trained language models for Vietnamese" (arXiv:2003.00744) — https://arxiv.org/abs/2003.00744
  - Toolkit: https://github.com/vncorenlp/VnCoreNLP

- [ ] **Evaluate against Vietnamese benchmarks (VN-MTEB / VLSP)**
  - Do not rely only on MLM loss as the quality signal
  - Use VN-MTEB (Vietnamese Massive Text Embedding Benchmark) to measure downstream embedding quality
  - VLSP shared task datasets cover NER, POS, dependency parsing in Vietnamese
  - Reference: VN-MTEB (arXiv:2405.xxxxx) — https://huggingface.co/spaces/mteb/leaderboard

### Training Efficiency

- [ ] **Dynamic sequence packing (eliminate padding waste)**
  - The current `DataLoader` pads all sequences in a batch to `max_seq_len`; short sequences waste GPU compute
  - Pack multiple shorter sequences into a single `max_seq_len` context using block-diagonal attention masks
  - Achieves 1.5–2× training throughput at no quality cost
  - Where: new `PackedDataCollator` class; requires attention mask to be block-diagonal (compatible with Flash Attention)
  - Reference: HuggingFace `DataCollatorWithFlattening` — https://huggingface.co/docs/transformers/main/en/main_classes/data_collator
  - ModernBERT calls this "unpadding" — cited as a key efficiency gain

- [ ] **Grouped Query Attention (GQA)**
  - Instead of N independent K/V heads, share a single K/V across groups of Q heads
  - Example: 8 Q-heads but only 2 K/V-heads (4:1 grouping) — cuts K/V parameter count and memory by 4×
  - Particularly useful when scaling to larger `hidden_size` or longer sequences
  - Where: extend `MultiHeadSelfAttention` with `num_kv_heads` parameter; toggle via config `num_kv_heads`
  - Reference: Ainslie et al. (2023) "GQA: Training Generalized Multi-Query Transformer Models" (arXiv:2305.13245) — https://arxiv.org/abs/2305.13245

- [ ] **Biphasic training: CLM pre-warmup → MLM**
  - Train with Causal Language Modeling (CLM) for an initial phase to leverage data efficiency of autoregressive objectives, then switch to MLM for bidirectionality
  - Empirically shown to improve sample efficiency for low-resource language pretraining
  - Reference: "BERT Meets Causal Language Modeling" (OpenReview 2024) — https://openreview.net/

---

## What You Can Do Right Now

### Pretraining Objective (Alternative)

- [ ] **ELECTRA-style Replaced Token Detection (RTD)**
  - Instead of predicting masked tokens, train a small *generator* (tiny MLM) to fill masks, then a *discriminator* (your main model) to classify every token as original or replaced
  - Why: trains on **all** tokens per step (not just 15–30% masked ones) → same quality in ~1/4 the compute vs. MLM
  - Where: add `ElectraGenerator` (smaller `BertForPreTraining`) and `ElectraDiscriminator` head; new `electra_pipe.py`
  - Vietnamese prior art: `NlpHUST/vi-electra-small` on HuggingFace
  - Reference: Clark et al. (2020) "ELECTRA" (arXiv:2003.10555) — https://arxiv.org/abs/2003.10555

### Training Efficiency

- [ ] **Gradient checkpointing** *(one-line change, high impact)*
  - Call `model.gradient_checkpointing_enable()` or wrap each `EncoderLayer` with `torch.utils.checkpoint.checkpoint()`
  - Why: trades ~20–30% extra compute for 50–80% less activation memory — lets you double your batch size or sequence length on the same GPU
  - Where: one call in `__main__.py` after `model.to(device)`, or inside `BertEncoder.forward()`
  - Reference: PyTorch docs — https://pytorch.org/docs/stable/checkpoint.html

- [ ] **Compile the model with `torch.compile()`** *(PyTorch 2.0+, one-line change)*
  - Add `model = torch.compile(model)` in `__main__.py` after moving to device
  - Why: Triton kernel fusion gives 1.5–2× throughput on modern GPUs for free
  - Reference: PyTorch 2.0 release — https://pytorch.org/get-started/pytorch-2.0/

### Data Quality & Sources

- [ ] **Add a text deduplication step to the data pipeline**
  - Duplicate sentences in pretraining corpora are a known source of memorisation and wasted compute
  - Use MinHash LSH (via `datasketch` library) or exact SHA-256 hashing to remove near-duplicate sentences before training
  - Reference: Lee et al. (2022) "Deduplicating Training Data Makes LMs Better" (arXiv:2107.06499) — https://arxiv.org/abs/2107.06499

- [ ] **Use quality Vietnamese corpora for pretraining data**
  - Current `dataset` field in config points to an English dataset (`bert-mlm-experiments-en`)
  - Replace with Vietnamese-specific sources:
    - **BKAINewsCorpus** — 32M articles, 53 GB of clean Vietnamese news text — https://huggingface.co/datasets/bkai-foundation-models/BKAINewsCorpus
    - **OSCAR (Vietnamese subset)** — filtered CommonCrawl Vietnamese — https://huggingface.co/datasets/oscar-corpus/OSCAR-2301
    - **Wikipedia Vietnamese** — clean encyclopedic text — https://huggingface.co/datasets/wikimedia/wikipedia (config `vi`)

- [ ] **Curriculum learning: sort training data by sequence length**
  - Start training on shorter, simpler sentences and gradually increase length (easy → hard)
  - Why: stabilises early training, reduces padding waste, converges faster on low-resource setups
  - Implementation: sort `sentences` by `len(tokens)` before building `BertPreTrainDataset`, use a length-based sampler with `DistributedSampler` or a simple epoch-based schedule
  - Reference: Platanios et al. (2019) "Competence-based Curriculum Learning for NMT" (arXiv:1903.09848)

### Monitoring & Reproducibility

- [ ] **Log gradient norms to W&B during training**
  - Add `grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)` return value logging to `WandbCallback.on_step_end()`
  - Why: spiking grad norms are the earliest signal of training instability, invisible without this
  - Where: `src/callbacks/wandb_callbacks.py`

- [ ] **Seed everything for reproducibility**
  - Add a `seed_everything(seed)` helper that sets `random`, `numpy`, `torch`, and `torch.cuda` seeds, and pass it `cfg.seed` at startup
  - Where: `src/utils/basic_utils.py` + call in `__main__.py` before data loading
  - Reference: `transformers.set_seed()` — https://huggingface.co/docs/transformers/main_classes/utilities#transformers.set_seed

- [ ] **Log vocabulary coverage on training data**
  - After building the tokenizer, compute what % of tokens in the corpus are `[UNK]`
  - High UNK rate (>5%) means vocab size is too small for the domain — add a check in `build_tokenizer.py`

---

## 🚀 Maximum Impact Plan ("Steroids")

> If I had to pick the changes that move the needle the most for your Vietnamese MLM — this is the order I'd do them.

### Tier 1 — Do These First (Highest ROI)

- [ ] **Wire `word_segmenter.py` into `BertDataCollator` for Whole Word Masking**
  - `word_segmenter.py` using `underthesea` is already built — now use it
  - In `BertDataCollator`, segment each sentence into words first, then build a `word_ids` map, then apply masking at the word level (mask all subword tokens of the chosen word together)
  - This is **the single highest-impact change** for Vietnamese specifically — directly fixes the "cheat" problem in MLM where adjacent subword tokens leak the answer
  - Where: `src/dataset.py` → `BertDataCollator.torch_mask_tokens()`

- [ ] **Switch to SpanBERT-style Span Masking**
  - Instead of masking individual tokens or whole words, mask *contiguous spans* of 1–10 tokens sampled from a geometric distribution
  - Why: models pre-trained with span masking learn dramatically better representations for NER, QA, and coreference tasks — the exact tasks Vietnamese NLP benchmarks test
  - Combine with WWM: first segment to words, then mask whole-word spans
  - Reference: Joshi et al. (2020) "SpanBERT" (arXiv:1907.10529) — https://arxiv.org/abs/1907.10529

- [ ] **8-bit AdamW optimizer (bitsandbytes)** *(one-line change, 75% less optimizer memory)*
  - Replace `torch.optim.AdamW` with `bitsandbytes.optim.AdamW8bit`
  - Why: AdamW stores two fp32 momentum states per parameter — for a 10M param model that's 80 MB. 8-bit quantises these states, cutting optimizer memory by 75% with negligible quality loss
  - `pip install bitsandbytes`
  - Where: `src/__main__.py` — swap one import
  - Reference: Dettmers et al. (2022) "8-bit Optimizers via Block-wise Quantization" (arXiv:2110.02861) — https://arxiv.org/abs/2110.02861

- [ ] **Switch AMP dtype from FP16 → BF16**
  - Change `torch.cuda.amp.autocast(dtype=torch.float16)` → `dtype=torch.bfloat16`
  - Why: BF16 has the same dynamic range as FP32 (8 exponent bits vs FP16's 5), meaning it almost never overflows or underflows — no loss scaling needed, training is more stable especially early on. Supported on A100, RTX 3090+, and all modern GPUs
  - Where: `src/pipe/training_pipe.py` — one argument change

### Tier 2 — Architecture Upgrades

- [ ] **DeBERTa-style Disentangled Attention**
  - Separate content-to-content attention and content-to-position attention into independent attention matrices; combine them at the end
  - Why: **ViDeBERTa** (Vietnamese DeBERTa) outperforms PhoBERT on POS, NER, and QA with fewer parameters — this is likely the single best encoder architecture for Vietnamese
  - Where: new `DisentangledAttention` class in `src/models/layers/attention.py`; toggle via config
  - Reference: He et al. (2021) "DeBERTa" (arXiv:2006.03654) — https://arxiv.org/abs/2006.03654
  - ViDeBERTa: Tran et al. (2023) (ACL Anthology) — https://aclanthology.org/2023.findings-emnlp.293/

- [ ] **Knowledge Distillation from PhoBERT**
  - After pretraining your model, use PhoBERT-base as a *teacher* to distil into your smaller model
  - Loss: `L = α · L_CE(student, labels) + (1-α) · L_KD(student_logits, teacher_logits / T)`
  - Why: your model learns to mimic a model trained on 20GB of Vietnamese text — effectively borrowing PhoBERT's knowledge for free
  - Where: new `distill_pipe.py`; teacher model loaded via HuggingFace `AutoModel`
  - Reference: Hinton et al. (2015) "Distilling the Knowledge in a Neural Network" (arXiv:1503.02531)
  - Vietnamese: `vinai/phobert-base-v2` on HuggingFace — https://huggingface.co/vinai/phobert-base-v2

### Tier 3 — Scale It Up

- [ ] **Multi-GPU training with PyTorch DDP**
  - Wrap `model` with `torch.nn.parallel.DistributedDataParallel` and launch with `torchrun`
  - Why: if you have 2+ GPUs, DDP gives near-linear throughput scaling with almost no code change
  - Where: `src/__main__.py` — ~15 lines of init code; `src/pipe/training_pipe.py` — use `DistributedSampler`
  - Reference: PyTorch DDP tutorial — https://pytorch.org/tutorials/intermediate/ddp_tutorial.html

- [ ] **Increase model scale within the same parameter budget using weight sharing**
  - Enable `share_encoder_weights: true` in config — this lets you use `hidden_size: 512, num_layers: 12` at the parameter cost of a single layer
  - Then train longer on more data — the shared weights are forced to generalise across all depths, acting like a recurrent encoder
  - This is the ALBERT recipe and produces very competitive embeddings at a fraction of the param cost

---

## 3-Year SOTA Plan (2026-2029) to Make viMLM the Best Vietnamese NLU Model

- [ ] **Adopt ModernBERT Architecture Standard**
  - Implement GeGLU activations in FeedForward and remove biases from linear layers.
  - Upgrade MultiHeadSelfAttention to use FlashAttention-3 and support sequence lengths of 8,192 tokens.
  - Write a Sequence Packing Data Loader to concatenate short sequences and eliminate padding tokens entirely.
- [ ] **ELECTRA-style Replaced Token Detection (RTD) Objective**
  - Replace/augment MLM with the RTD objective, training viMLM as a discriminator classifier on 100% of tokens.
- [ ] **Instruction-Tuned Encoder Pretraining**
  - Integrate instruction-following prompts directly into the pretraining corpus, enabling zero-shot instruction execution.
- [ ] **Biphasic Pretraining Paradigm**
  - Train in two phases: Causal Language Modeling (CLM) first to learn lexical structure, followed by WWM Masked Language Modeling to extract rich contextual embeddings.
- [ ] **Advanced Data Filtering and Curation**
  - Apply MinHash LSH and perplexity filtering to OSCAR / bkainews corpora to compile a high-quality 50GB+ training set.