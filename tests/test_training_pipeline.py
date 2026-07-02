"""
Integration checks for the training-loop plumbing added alongside the
model-level regression tests: text dedup, gradient checkpointing, grad_norm
reaching callbacks, and a full train() smoke run under BF16 autocast
(no GradScaler).

Run: .venv/bin/python tests/test_training_pipeline.py
"""

import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch
from torch.utils.data import DataLoader

from src.models.bert import BertForPreTraining
from src.utils.data_utils import load_sentences_from_file
from src.pipe.training_pipe import train, train_epoch


def make_cfg(**overrides):
    base = dict(
        pad_token_id=0,
        vocab_size=48,
        max_seq_len=16,
        hidden_size=16,
        num_layers=4,
        num_heads=2,
        ff_dim=32,
        dropout=0.2,
        use_rope=False,
        use_unet_shrink=False,
        unet_bottleneck_ratio=0.5,
        layer_hidden_sizes=None,
        use_unet_skip=False,
        use_gradient_checkpointing=False,
    )
    base.update(overrides)
    return SimpleNamespace(**base)


def test_dedup():
    with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False) as f:
        f.write("hello world\nfoo bar\nhello world\nfoo bar\nbaz\n")
        path = f.name

    sentences = load_sentences_from_file(path)
    assert sentences == ["hello world", "foo bar", "baz"], sentences
    print("[OK] load_sentences_from_file drops exact duplicates, preserves order")


def fake_batch(cfg, batch=2, seq_len=10):
    g = torch.Generator().manual_seed(7)
    input_ids = torch.randint(5, cfg.vocab_size, (batch, seq_len), generator=g)
    segment_ids = torch.zeros(batch, seq_len, dtype=torch.long)
    attn_mask = torch.ones(batch, seq_len, dtype=torch.long)
    mlm_labels = input_ids.clone()
    mlm_labels[:, 3:6] = torch.randint(5, cfg.vocab_size, (batch, 3), generator=g)
    ignore = torch.ones(batch, seq_len, dtype=torch.bool)
    ignore[:, 3:6] = False
    mlm_labels[ignore] = -100
    return input_ids, segment_ids, attn_mask, mlm_labels


def test_gradient_checkpointing_matches_eager():
    device = torch.device("cpu")

    torch.manual_seed(42)
    cfg_ckpt = make_cfg(use_gradient_checkpointing=True)
    model_ckpt = BertForPreTraining(cfg_ckpt).to(device)

    torch.manual_seed(42)
    cfg_plain = make_cfg(use_gradient_checkpointing=False)
    model_plain = BertForPreTraining(cfg_plain).to(device)

    input_ids, segment_ids, attn_mask, mlm_labels = fake_batch(cfg_ckpt)

    model_ckpt.train()
    model_plain.train()

    torch.manual_seed(123)
    out_ckpt = model_ckpt(input_ids, segment_ids, mask=attn_mask, mlm_labels=mlm_labels)
    out_ckpt["loss"].backward()

    torch.manual_seed(123)
    out_plain = model_plain(input_ids, segment_ids, mask=attn_mask, mlm_labels=mlm_labels)
    out_plain["loss"].backward()

    assert torch.allclose(out_ckpt["loss"], out_plain["loss"], atol=1e-6), (
        f"checkpointed vs eager loss mismatch: {out_ckpt['loss'].item()} vs {out_plain['loss'].item()}"
    )
    for (n1, p1), (n2, p2) in zip(
        model_ckpt.named_parameters(), model_plain.named_parameters()
    ):
        assert n1 == n2
        if p1.grad is not None:
            assert torch.allclose(p1.grad, p2.grad, atol=1e-5), f"grad mismatch at {n1}"

    print("[OK] gradient checkpointing reproduces eager forward/backward exactly (RNG-preserved recompute)")


def test_grad_norm_reaches_callbacks():
    device = torch.device("cpu")
    cfg = make_cfg()
    torch.manual_seed(0)
    model = BertForPreTraining(cfg).to(device)

    batch = fake_batch(cfg)
    loader = [batch] * 3  # train_epoch just needs something iterable of 4-tuples

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

    recorded = []

    class RecorderCallback:
        def on_step_end(self, trainer, step, loss, lr, grad_norm=None):
            recorded.append(grad_norm)

    train_epoch(
        model,
        loader,
        optimizer,
        scheduler=None,
        device=device,
        callbacks=[RecorderCallback()],
    )

    assert len(recorded) == 3, f"expected 3 recorded steps, got {len(recorded)}"
    for gn in recorded:
        assert gn is not None, "grad_norm was not passed to on_step_end"
        assert gn == gn and gn >= 0, f"grad_norm not finite/non-negative: {gn}"

    print(f"[OK] grad_norm reaches callbacks.on_step_end: {recorded}")


def test_full_train_smoke():
    device = torch.device("cpu")
    cfg = make_cfg(use_unet_shrink=True, use_unet_skip=True, use_gradient_checkpointing=True)
    torch.manual_seed(0)
    model = BertForPreTraining(cfg).to(device)

    batch = fake_batch(cfg)
    train_loader = [batch] * 4
    eval_loader = [batch] * 2

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

    # train() no longer takes/builds a GradScaler -- this call would raise
    # TypeError if that plumbing were still expected.
    train(
        model=model,
        train_loader=train_loader,
        optimizer=optimizer,
        device=device,
        epochs=1,
        eval_loader=eval_loader,
        callbacks=None,
        eval_steps=2,
    )
    print("[OK] full train() smoke run (BF16 autocast, no GradScaler, checkpointing+skip on) completes")


if __name__ == "__main__":
    test_dedup()
    test_gradient_checkpointing_matches_eager()
    test_grad_norm_reaches_callbacks()
    test_full_train_smoke()
    print("\nAll training-pipeline checks passed.")
