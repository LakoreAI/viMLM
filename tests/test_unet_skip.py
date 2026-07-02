import sys
from types import SimpleNamespace

sys.path.insert(0, "/Users/minhld/workspace/projects/viMLM")

import torch
from src.models.bert import BertEncoder, BertForPreTraining


def make_cfg(**overrides):
    base = dict(
        pad_token_id=0,
        vocab_size=50,
        max_seq_len=16,
        hidden_size=24,
        num_layers=4,
        num_heads=2,
        ff_dim=48,
        dropout=0.0,
        use_rope=False,
        use_unet_shrink=False,
        unet_bottleneck_ratio=0.5,
        layer_hidden_sizes=None,
        use_unet_skip=False,
    )
    base.update(overrides)
    return SimpleNamespace(**base)


def check_forward_backward(name, encoder, hidden_size, batch=2, seq_len=6):
    x = torch.randn(batch, seq_len, hidden_size, requires_grad=True)
    out = encoder(x)
    assert out.shape == (batch, seq_len, hidden_size), f"{name}: bad output shape {out.shape}"
    loss = out.sum()
    loss.backward()
    assert x.grad is not None and torch.isfinite(x.grad).all(), f"{name}: bad input grad"
    for pname, p in encoder.named_parameters():
        if p.requires_grad:
            assert p.grad is not None, f"{name}: {pname} got no gradient"
            assert torch.isfinite(p.grad).all(), f"{name}: {pname} has non-finite grad"
    print(f"[OK] {name}: layer_sizes={encoder.layer_sizes} pair_of={encoder.pair_of}")


# 1) Baseline: no shrink, no skip -> plain EncoderLayer stack, unaffected
enc1 = BertEncoder(hidden_size=24, ff_dim=48, num_heads=2, num_layers=4, use_unet_shrink=False, use_unet_skip=False)
check_forward_backward("baseline (no shrink)", enc1, 24)
assert enc1.use_skip is False and enc1.pair_of == {}

# 2) Shrink on, skip off -> old behavior preserved (no skip_fuse modules)
enc2 = BertEncoder(hidden_size=24, ff_dim=48, num_heads=2, num_layers=4, use_unet_shrink=True, unet_bottleneck_ratio=0.5, use_unet_skip=False)
check_forward_backward("shrink, skip=False", enc2, 24)
assert enc2.use_skip is False and enc2.pair_of == {}
for layer in enc2.layers:
    assert not hasattr(layer, "skip_fuse"), "skip_fuse should not exist when use_unet_skip=False"

# 3) Shrink on, skip on, even num_layers -> pairing should mirror indices, no self-pair
enc3 = BertEncoder(hidden_size=24, ff_dim=48, num_heads=2, num_layers=4, use_unet_shrink=True, unet_bottleneck_ratio=0.5, use_unet_skip=True)
check_forward_backward("shrink, skip=True, num_layers=4", enc3, 24)
assert enc3.use_skip is True
assert enc3.pair_of == {3: 0, 2: 1}, f"unexpected pairing {enc3.pair_of}"
assert enc3.layers[3].use_skip and enc3.layers[2].use_skip
assert not enc3.layers[0].use_skip and not enc3.layers[1].use_skip

# 4) Shrink on, skip on, odd num_layers -> middle bottleneck layer has no pair
enc4 = BertEncoder(hidden_size=24, ff_dim=48, num_heads=2, num_layers=5, use_unet_shrink=True, unet_bottleneck_ratio=0.5, use_unet_skip=True)
check_forward_backward("shrink, skip=True, num_layers=5", enc4, 24)
assert 2 not in enc4.pair_of and 2 not in enc4.pair_of.values(), "middle bottleneck layer should have no skip pair"

# 5) Custom symmetric layer_hidden_sizes with skip
enc5 = BertEncoder(hidden_size=24, ff_dim=48, num_heads=2, num_layers=4, layer_hidden_sizes=[24, 12, 12, 24], use_unet_skip=True)
check_forward_backward("custom symmetric sizes, skip=True", enc5, 24)
assert enc5.pair_of == {3: 0, 2: 1}

# 6) Custom asymmetric layer_hidden_sizes -> no valid pairs, should degrade gracefully (no skip applied)
enc6 = BertEncoder(hidden_size=24, ff_dim=48, num_heads=2, num_layers=3, layer_hidden_sizes=[24, 16, 20], use_unet_skip=True)
check_forward_backward("custom asymmetric sizes, skip=True", enc6, 24)
assert enc6.pair_of == {}, f"asymmetric sizes should produce no pairs, got {enc6.pair_of}"

# 7) Verify a skip connection changes the output vs. no skip (i.e. it's actually wired, not a no-op)
torch.manual_seed(0)
enc_skip = BertEncoder(hidden_size=24, ff_dim=48, num_heads=2, num_layers=4, use_unet_shrink=True, use_unet_skip=True)
torch.manual_seed(0)
enc_noskip = BertEncoder(hidden_size=24, ff_dim=48, num_heads=2, num_layers=4, use_unet_shrink=True, use_unet_skip=False)
torch.manual_seed(1)
x = torch.randn(2, 6, 24)
with torch.no_grad():
    out_skip = enc_skip(x.clone())
    out_noskip = enc_noskip(x.clone())
assert not torch.allclose(out_skip, out_noskip), "skip connection had no effect on output"
print("[OK] skip connection measurably changes encoder output vs. shrink-only baseline")

# 8) End-to-end BertForPreTraining forward/backward with skip enabled, driven purely by cfg (config-driven toggle)
cfg_on = make_cfg(use_unet_shrink=True, use_unet_skip=True, num_layers=4, hidden_size=24, num_heads=2)
model_on = BertForPreTraining(cfg_on)
input_ids = torch.randint(5, cfg_on.vocab_size, (2, 6))
segment_ids = torch.zeros(2, 6, dtype=torch.long)
labels = input_ids.clone()
labels[:, :2] = -100
out = model_on(input_ids, segment_ids, mlm_labels=labels)
out["loss"].backward()
assert model_on.encoder.use_skip is True
assert torch.isfinite(out["loss"]).all()
print("[OK] BertForPreTraining end-to-end with cfg.use_unet_skip=True")

cfg_off = make_cfg(use_unet_shrink=True, use_unet_skip=False, num_layers=4, hidden_size=24, num_heads=2)
model_off = BertForPreTraining(cfg_off)
assert model_off.encoder.use_skip is False
print("[OK] BertForPreTraining end-to-end with cfg.use_unet_skip=False (default) -> skip disabled")

print("\nAll tests passed.")
