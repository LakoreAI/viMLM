"""
Correctness checks for the SwiGLU / GeGLU FeedForward variants and their
config-driven wiring (model.ffn_type: "gelu" | "swiglu" | "geglu").

Run: .venv/bin/python tests/test_ffn_variants.py
"""

import sys
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch

from src.models.layers.feedforward import FeedForward
from src.models.bert import BertForPreTraining


def count_params(module):
    return sum(p.numel() for p in module.parameters())


def test_shapes_and_gradients():
    for ffn_type in ("gelu", "swiglu", "geglu"):
        ff = FeedForward(hidden_size=16, ff_dim=64, dropout=0.0, ffn_type=ffn_type)
        x = torch.randn(2, 5, 16, requires_grad=True)
        out = ff(x)
        assert out.shape == (2, 5, 16), f"{ffn_type}: bad output shape {out.shape}"
        out.sum().backward()
        assert x.grad is not None and torch.isfinite(x.grad).all(), f"{ffn_type}: bad input grad"
        for name, p in ff.named_parameters():
            assert p.grad is not None, f"{ffn_type}: {name} got no gradient"
            assert torch.isfinite(p.grad).all(), f"{ffn_type}: {name} has non-finite grad"
        print(f"[OK] {ffn_type}: forward/backward shapes and gradients are correct")


def test_gated_variants_are_distinct():
    torch.manual_seed(0)
    ff_swiglu = FeedForward(16, 64, dropout=0.0, ffn_type="swiglu")
    torch.manual_seed(0)
    ff_geglu = FeedForward(16, 64, dropout=0.0, ffn_type="geglu")

    x = torch.randn(2, 5, 16)
    with torch.no_grad():
        out_swiglu = ff_swiglu(x)
        out_geglu = ff_geglu(x)
    assert not torch.allclose(out_swiglu, out_geglu), (
        "SwiGLU (SiLU gate) and GeGLU (GELU gate) produced identical output -- "
        "activation function isn't actually different"
    )
    print("[OK] swiglu and geglu produce different outputs (SiLU vs GELU gate confirmed wired)")


def test_param_parity_with_gelu_baseline():
    # Gated variants use a ~2/3-scaled gate/up dim specifically so total
    # params stay in the same ballpark as the plain GELU FFN (which has a
    # single up-projection instead of two).
    gelu_params = count_params(FeedForward(64, 256, dropout=0.0, ffn_type="gelu"))
    swiglu_params = count_params(FeedForward(64, 256, dropout=0.0, ffn_type="swiglu"))
    ratio = swiglu_params / gelu_params
    assert 0.85 <= ratio <= 1.15, (
        f"swiglu param count ({swiglu_params}) is not within 15% of gelu baseline "
        f"({gelu_params}), ratio={ratio:.3f} -- the 2/3 scaling may be broken"
    )
    print(f"[OK] swiglu param count ({swiglu_params}) is within 15% of gelu baseline ({gelu_params})")


def test_invalid_ffn_type_rejected():
    try:
        FeedForward(16, 64, ffn_type="not_a_real_activation")
    except AssertionError:
        print("[OK] invalid ffn_type raises AssertionError")
        return
    raise AssertionError("FeedForward accepted an invalid ffn_type without error")


def test_config_driven_toggle_end_to_end():
    base = dict(
        pad_token_id=0,
        vocab_size=48,
        max_seq_len=16,
        hidden_size=16,
        num_layers=4,
        num_heads=2,
        ff_dim=32,
        dropout=0.0,
        use_rope=False,
        use_unet_shrink=False,
        unet_bottleneck_ratio=0.5,
        layer_hidden_sizes=None,
        use_unet_skip=False,
        use_gradient_checkpointing=False,
    )

    input_ids = torch.randint(5, 48, (2, 6))
    segment_ids = torch.zeros(2, 6, dtype=torch.long)
    labels = input_ids.clone()
    labels[:, :2] = -100

    outputs = {}
    for ffn_type in ("gelu", "swiglu", "geglu"):
        torch.manual_seed(1)
        cfg = SimpleNamespace(**base, ffn_type=ffn_type)
        model = BertForPreTraining(cfg)
        model.eval()
        with torch.no_grad():
            out = model(input_ids, segment_ids, mlm_labels=labels)
        assert torch.isfinite(out["loss"]), f"{ffn_type}: non-finite loss"
        outputs[ffn_type] = out["mlm_logits"]
        # Sanity: every UNetEncoderLayer-free EncoderLayer picked up the type
        for layer in model.encoder.layers:
            assert layer.ff.ffn_type == ffn_type

    assert not torch.allclose(outputs["gelu"], outputs["swiglu"])
    assert not torch.allclose(outputs["swiglu"], outputs["geglu"])
    print("[OK] cfg.ffn_type drives BertForPreTraining end-to-end and changes output per variant")

    # Default (no ffn_type in cfg at all) must still behave like "gelu".
    cfg_no_key = SimpleNamespace(**base)
    torch.manual_seed(1)
    model_default = BertForPreTraining(cfg_no_key)
    model_default.eval()
    with torch.no_grad():
        out_default = model_default(input_ids, segment_ids, mlm_labels=labels)
    assert torch.allclose(out_default["mlm_logits"], outputs["gelu"]), (
        "omitting ffn_type from cfg should default to plain GELU FFN"
    )
    print("[OK] omitting ffn_type from config defaults to gelu (backward compatible)")


if __name__ == "__main__":
    test_shapes_and_gradients()
    test_gated_variants_are_distinct()
    test_param_parity_with_gelu_baseline()
    test_invalid_ffn_type_rejected()
    test_config_driven_toggle_end_to_end()
    print("\nAll FFN variant checks passed.")
