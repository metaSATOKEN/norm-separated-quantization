# Copyright 2026 Kentaro Sato
# Licensed under the Apache License, Version 2.0

# ============================================================
# PoC: Gemma 4 E2B Outlier Ratio Measurement
# ============================================================
# Motivation:
#   Grok reports Gemma 4 has KV cache issues (memory explosion,
#   quant quality degradation). Does it also hit our 8x outlier
#   threshold -- the condition under which naive4 catastrophically
#   fails on Qwen2-7B?
#
# Known baselines from our prior experiments:
#   Qwen2-7B    : outlier ratio 8.6x  -> naive4 breaks (0/26 NIAH)
#   Qwen2.5-14B : outlier ratio 3.5x  -> naive4 works  (26/26 NIAH)
#   Pythia-6.9B : outlier ratio 4.6x  -> naive4 +22.56 PPL
#   Mistral-7B  : outlier ratio 3.1x  -> naive4 +0.10 PPL
#
# Target: Gemma 4 E2B-it
#   Special architecture: SWA + Full attention hybrid, Shared KV layers
#   head_dim 256 (2-4x larger than typical)
#
# This script only MEASURES outlier ratios. No quantization applied.
# Runs on M1 Mac (CPU or MPS). ~5GB VRAM/RAM for FP16.
# ============================================================

import gc
import json
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from datetime import datetime

# ── Device selection (M1-friendly) ─────────────────────────────────────────

if torch.cuda.is_available():
    device = "cuda"
    dtype = torch.float16
elif torch.backends.mps.is_available():
    device = "mps"
    dtype = torch.float16
else:
    device = "cpu"
    dtype = torch.float32

print(f"Device: {device}, dtype: {dtype}")

# ── Model ──────────────────────────────────────────────────────────────────

MODEL_ID = "google/gemma-4-E2B-it"

print(f"\nLoading {MODEL_ID} ...")
tok = AutoTokenizer.from_pretrained(MODEL_ID)
mdl = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    torch_dtype=dtype,
    device_map=device if device != "cpu" else None,
    use_safetensors=True,
)
mdl.eval()

# Log architecture details
cfg = mdl.config
print(f"\n--- Config ---")
print(f"  num_hidden_layers    : {getattr(cfg, 'num_hidden_layers', '?')}")
print(f"  num_attention_heads  : {getattr(cfg, 'num_attention_heads', '?')}")
print(f"  num_key_value_heads  : {getattr(cfg, 'num_key_value_heads', '?')}")
print(f"  head_dim             : {getattr(cfg, 'head_dim', '?')}")
print(f"  hidden_size          : {getattr(cfg, 'hidden_size', '?')}")
print(f"  sliding_window       : {getattr(cfg, 'sliding_window', '?')}")
print(f"  num_kv_shared_layers : {getattr(cfg, 'num_kv_shared_layers', 'N/A')}")
print(f"  max_position         : {getattr(cfg, 'max_position_embeddings', '?')}")

# ── Probe text (~512 tokens, mix of styles for realistic K/V) ──────────────

PROBE = """
The scientific method represents a fundamental approach to understanding
the natural world. It involves the systematic observation of phenomena,
the formulation of hypotheses, and the rigorous testing of those hypotheses
through experimentation. Transformer architectures have revolutionized
natural language processing by enabling models to attend to arbitrary
positions within a sequence, unlike the sequential nature of recurrent
networks. The key innovation is the self-attention mechanism, which allows
each token to interact with every other token. Quantization of neural
network activations to low bit widths such as INT4 can dramatically reduce
memory and compute, but introduces approximation error. Per-row symmetric
quantization uses the absolute maximum of each row as the scale, mapping
values to 16 integer levels in the case of INT4. When certain channels
contain outlier values many times larger than the typical magnitude,
the per-row scale becomes dominated by those outliers, and the remaining
dimensions suffer severe precision loss. This phenomenon was first
characterized by SmoothQuant and LLM.int8 in the weight and activation
quantization literature. For key-value caches specifically, the
distribution of outliers across channels and tokens determines whether
naive quantization schemes remain viable or collapse catastrophically.
The Mona Lisa painted by Leonardo da Vinci in the early sixteenth century
hangs in the Louvre Museum. The speed of light in vacuum is approximately
299792458 meters per second. Photosynthesis converts carbon dioxide and
water into glucose and oxygen using light energy captured by chlorophyll.
"""

ids = tok.encode(PROBE, return_tensors="pt").to(device)
print(f"\nProbe tokens: {ids.shape[1]}")

# ── Forward pass with KV cache retrieval ──────────────────────────────────

def get_kv(past, li):
    if hasattr(past, 'layers'):
        return past.layers[li].keys, past.layers[li].values
    return past[li][0], past[li][1]

def n_cache_layers(past):
    return len(past.layers) if hasattr(past, 'layers') else len(past)

with torch.inference_mode():
    out = mdl(ids, use_cache=True)
    past = out.past_key_values

n_layers = n_cache_layers(past)
print(f"\nKV cache layers: {n_layers}")

# ── Outlier ratio measurement ──────────────────────────────────────────────
#
# Definition: for each (layer, head), flatten tensor to [tokens, head_dim].
#   For each column c: absmax_c = |x[:, c]|.max()
#   outlier_ratio = absmax_c.max() / absmax_c.mean()
#
# Interpretation:
#   ratio ~1-2x  : uniform channels, naive quantization safe
#   ratio 3-5x   : moderate outliers, some degradation expected
#   ratio >8x    : severe outliers, naive4 likely catastrophic
# ──────────────────────────────────────────────────────────────────────────

def outlier_stats(x):
    """x: [tokens, head_dim]"""
    x = x.float().abs()
    col_absmax = x.amax(dim=0)
    mean_absmax = col_absmax.mean().item()
    max_absmax = col_absmax.amax().item()
    if mean_absmax < 1e-12:
        return float('nan'), max_absmax, mean_absmax
    return max_absmax / mean_absmax, max_absmax, mean_absmax

results = {"layers": []}

print(f"\n{'='*70}")
print(f"{'Layer':>5} | {'K head avg':>11} | {'K max head':>11} | {'V head avg':>11} | {'V max head':>11}")
print(f"{'-'*70}")

for li in range(n_layers):
    k, v = get_kv(past, li)
    # Shape usually [batch=1, num_kv_heads, tokens, head_dim]
    k = k[0]  # [heads, tokens, head_dim]
    v = v[0]

    k_ratios, v_ratios = [], []
    for h in range(k.shape[0]):
        r_k, mx_k, me_k = outlier_stats(k[h])
        r_v, mx_v, me_v = outlier_stats(v[h])
        k_ratios.append(r_k)
        v_ratios.append(r_v)

    k_avg = float(np.nanmean(k_ratios))
    k_max = float(np.nanmax(k_ratios))
    v_avg = float(np.nanmean(v_ratios))
    v_max = float(np.nanmax(v_ratios))

    print(f"{li:>5} | {k_avg:>11.2f} | {k_max:>11.2f} | {v_avg:>11.2f} | {v_max:>11.2f}")

    results["layers"].append({
        "layer": li,
        "k_ratio_avg": k_avg, "k_ratio_max": k_max,
        "v_ratio_avg": v_avg, "v_ratio_max": v_max,
        "k_shape": list(k.shape), "v_shape": list(v.shape),
    })

# ── Global summary ─────────────────────────────────────────────────────────

all_k_avg = [r["k_ratio_avg"] for r in results["layers"]]
all_k_max = [r["k_ratio_max"] for r in results["layers"]]
all_v_avg = [r["v_ratio_avg"] for r in results["layers"]]
all_v_max = [r["v_ratio_max"] for r in results["layers"]]

print(f"\n{'='*70}")
print(f"GLOBAL SUMMARY (Gemma 4 E2B-it)")
print(f"{'='*70}")
print(f"  K outlier ratio avg (layer-avg heads) : {np.mean(all_k_avg):.2f}x")
print(f"  K outlier ratio max (worst head)      : {np.max(all_k_max):.2f}x")
print(f"  V outlier ratio avg (layer-avg heads) : {np.mean(all_v_avg):.2f}x")
print(f"  V outlier ratio max (worst head)      : {np.max(all_v_max):.2f}x")

worst_k_layer = int(np.argmax(all_k_max))
worst_v_layer = int(np.argmax(all_v_max))
print(f"\n  Worst K outlier layer: {worst_k_layer} ({all_k_max[worst_k_layer]:.2f}x)")
print(f"  Worst V outlier layer: {worst_v_layer} ({all_v_max[worst_v_layer]:.2f}x)")

# ── Classification vs our known thresholds ────────────────────────────────

overall_max = max(np.max(all_k_max), np.max(all_v_max))
print(f"\n  Overall worst outlier ratio: {overall_max:.2f}x")
print(f"\n  Comparison with known baselines:")
print(f"    Qwen2-7B    : 8.6x -> naive4 BREAKS      (0/26 NIAH)")
print(f"    Pythia-6.9B : 4.6x -> naive4 +22 PPL     (survives)")
print(f"    Qwen2.5-14B : 3.5x -> naive4 fine        (26/26 NIAH)")
print(f"    Mistral-7B  : 3.1x -> naive4 +0.10 PPL   (fine)")
print(f"\n  Gemma 4 E2B : {overall_max:.2f}x -> prediction:")
if overall_max > 8.0:
    print(f"    >>> HIGH RISK: naive4 likely catastrophic, nsep+pchan4 critical")
elif overall_max > 5.0:
    print(f"    >>> MODERATE: naive4 will degrade significantly, nsep+pchan4 helpful")
else:
    print(f"    >>> LOW RISK: naive4 likely survives, similar to Qwen2.5-14B / Mistral")

# ── Save ───────────────────────────────────────────────────────────────────

results["model"] = MODEL_ID
results["timestamp"] = datetime.now().isoformat()
results["probe_tokens"] = ids.shape[1]
results["summary"] = {
    "k_avg_of_layer_means": float(np.mean(all_k_avg)),
    "k_worst_head_in_any_layer": float(np.max(all_k_max)),
    "v_avg_of_layer_means": float(np.mean(all_v_avg)),
    "v_worst_head_in_any_layer": float(np.max(all_v_max)),
    "worst_k_layer": worst_k_layer,
    "worst_v_layer": worst_v_layer,
    "overall_max_ratio": float(overall_max),
}

out_path = "../results/poc_gemma4_outlier.json"
try:
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved -> {out_path}")
except Exception as e:
    print(f"\n(save failed: {e}) -- printing JSON:")
    print(json.dumps(results["summary"], indent=2))

del mdl, past, out
gc.collect()
if device == "cuda":
    torch.cuda.empty_cache()
