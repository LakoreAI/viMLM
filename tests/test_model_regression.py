"""
Golden-output regression test for BertForPreTraining.

Purpose: catch *unintended* changes to the model's forward pass when you
refactor internals (attention, encoder, embeddings, etc). It builds a few
small fixed-config models, seeds the RNG so weight init is reproducible,
runs a fixed input through them in eval() mode (no dropout -> fully
deterministic), and compares mlm_logits/loss against saved golden tensors
in tests/golden/.

If a change is *intentional* (new architecture, changed default, etc),
regenerate the golden files:

    .venv/bin/python tests/test_model_regression.py --update-golden

Run normally:

    .venv/bin/python tests/test_model_regression.py
"""

import argparse
import sys
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch

from src.models.bert import BertForPreTraining

GOLDEN_DIR = ROOT / "tests" / "golden"

BASE_CFG = dict(
    pad_token_id=0,
    vocab_size=64,
    max_seq_len=16,
    hidden_size=32,
    num_layers=4,
    num_heads=4,
    ff_dim=64,
    dropout=0.1,
    use_rope=False,
    use_unet_shrink=False,
    unet_bottleneck_ratio=0.5,
    layer_hidden_sizes=None,
    use_unet_skip=False,
    use_gradient_checkpointing=False,
    ffn_type="gelu",
)

SCENARIOS = {
    "baseline": {},
    "rope": {"use_rope": True},
    "unet_shrink": {"use_unet_shrink": True},
    "unet_shrink_skip": {"use_unet_shrink": True, "use_unet_skip": True},
    "swiglu": {"ffn_type": "swiglu"},
    "geglu": {"ffn_type": "geglu"},
}


def make_cfg(overrides):
    cfg = dict(BASE_CFG)
    cfg.update(overrides)
    return SimpleNamespace(**cfg)


def fixed_batch(vocab_size, batch=2, seq_len=8):
    g = torch.Generator().manual_seed(2024)
    input_ids = torch.randint(5, vocab_size, (batch, seq_len), generator=g)
    segment_ids = torch.zeros(batch, seq_len, dtype=torch.long)
    mlm_labels = input_ids.clone()
    # Pretend a handful of positions were masked; leave the rest as -100.
    mlm_labels[:, 2:5] = torch.randint(5, vocab_size, (batch, 3), generator=g)
    ignore = torch.ones(batch, seq_len, dtype=torch.bool)
    ignore[:, 2:5] = False
    mlm_labels[ignore] = -100
    return input_ids, segment_ids, mlm_labels


def run_scenario(name, overrides):
    torch.manual_seed(1234)  # deterministic weight init
    cfg = make_cfg(overrides)
    model = BertForPreTraining(cfg)
    model.eval()  # disables dropout -> deterministic forward

    input_ids, segment_ids, mlm_labels = fixed_batch(cfg.vocab_size)

    with torch.no_grad():
        out = model(input_ids, segment_ids, mlm_labels=mlm_labels)

    return {"mlm_logits": out["mlm_logits"], "loss": out["loss"]}


def compare_or_update(name, result, update_golden, atol=1e-5, rtol=1e-4):
    path = GOLDEN_DIR / f"{name}.pt"

    if update_golden or not path.exists():
        GOLDEN_DIR.mkdir(parents=True, exist_ok=True)
        torch.save(result, path)
        print(f"[WROTE] {name}: golden file (re)generated at {path}")
        return

    golden = torch.load(path)

    assert result["mlm_logits"].shape == golden["mlm_logits"].shape, (
        f"{name}: mlm_logits shape changed: "
        f"{golden['mlm_logits'].shape} -> {result['mlm_logits'].shape}"
    )
    logits_match = torch.allclose(
        result["mlm_logits"], golden["mlm_logits"], atol=atol, rtol=rtol
    )
    loss_match = torch.allclose(result["loss"], golden["loss"], atol=atol, rtol=rtol)

    if not (logits_match and loss_match):
        max_diff = (result["mlm_logits"] - golden["mlm_logits"]).abs().max().item()
        raise AssertionError(
            f"{name}: forward pass output changed vs. golden reference "
            f"(max logit diff={max_diff:.6g}, loss {golden['loss'].item():.6f} "
            f"-> {result['loss'].item():.6f}).\n"
            f"If this change was intentional, regenerate with:\n"
            f"  .venv/bin/python tests/test_model_regression.py --update-golden"
        )

    print(f"[OK] {name}: matches golden reference (loss={result['loss'].item():.6f})")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--update-golden",
        action="store_true",
        help="(Re)write golden reference files instead of comparing against them.",
    )
    args = parser.parse_args()

    for name, overrides in SCENARIOS.items():
        result = run_scenario(name, overrides)
        compare_or_update(name, result, update_golden=args.update_golden)

    print("\nAll regression checks passed." if not args.update_golden else "\nGolden files updated.")


if __name__ == "__main__":
    main()
