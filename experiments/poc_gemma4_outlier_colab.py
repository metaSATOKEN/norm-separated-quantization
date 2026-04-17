# Copyright 2026 Kentaro Sato
# Licensed under the Apache License, Version 2.0

# ============================================================
# PoC: Gemma 4 E2B Outlier Ratio Measurement  (Colab version)
# ============================================================
# Copy-paste each CELL into a separate Colab cell.
# T4 (free) is fine for E2B (~5GB FP16).
#
# Known baselines from our prior experiments:
#   Qwen2-7B    : 8.6x -> naive4 BREAKS      (0/26 NIAH)
#   Pythia-6.9B : 4.6x -> naive4 +22 PPL
#   Qwen2.5-14B : 3.5x -> naive4 fine        (26/26 NIAH)
#   Mistral-7B  : 3.1x -> naive4 +0.10 PPL
# ============================================================


# =============================================================
# === CELL 1 === install + HF login
# =============================================================

!pip install -q -U transformers accelerate hf_transfer
import os
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"

# ─────────────────────────────────────────────────────────────
# >>>>>  PASTE YOUR HF TOKEN BELOW (Read-only is enough)  <<<<<
# Create one at: https://huggingface.co/settings/tokens
# Gemma 4 is Apache 2.0 but HF still requires auth for download.
# ─────────────────────────────────────────────────────────────

HF_TOKEN = "hf_XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX"  # <-- paste your token here

# ─────────────────────────────────────────────────────────────

from huggingface_hub import login
login(token=HF_TOKEN)
print("HF login OK")

# Sanity check GPU
import torch
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name()}")
    print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory/1e9:.1f} GB")


# =============================================================
# === CELL 2 === load model + measure outlier ratios
# =============================================================

import gc, json, numpy as np, torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from datetime import datetime

device = "cuda"
dtype  = torch.float16

MODEL_ID = "google/gemma-4-E2B-it"

print(f"\nLoading {MODEL_ID} ...")
tok = AutoTokenizer.from_pretrained(MODEL_ID)
mdl = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    torch_dtype=dtype,
    device_map="auto",
    use_safetensors=True,
)
mdl.eval()

# Log architecture
cfg = mdl.config
print(f"\n--- Config ---")
for attr in ["num_hidden_layers", "num_attention_heads", "num_key_value_heads",
             "head_dim", "hidden_size", "sliding_window", "num_kv_shared_layers",
             "max_position_embeddings"]:
    print(f"  {attr:<22}: {getattr(cfg, attr, 'N/A')}")

# ── Probe text (~512 tokens, mixed content) ──────────────────
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

# ── Forward pass to get KV cache ─────────────────────────────
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


# =============================================================
# === CELL 3 === outlier stats + summary
# =============================================================

def outlier_stats(x):
    """x: [tokens, head_dim] -> (ratio, max_absmax, mean_absmax)"""
    x = x.float().abs()
    col_absmax = x.amax(dim=0)
    m = col_absmax.mean().item()
    mx = col_absmax.amax().item()
    return (float('nan') if m < 1e-12 else mx / m, mx, m)

results = {"layers": []}

print(f"\n{'='*70}")
print(f"{'Layer':>5} | {'K head avg':>11} | {'K max head':>11} | {'V head avg':>11} | {'V max head':>11}")
print(f"{'-'*70}")

for li in range(n_layers):
    k, v = get_kv(past, li)
    k = k[0]  # [heads, tokens, head_dim]
    v = v[0]

    k_ratios, v_ratios = [], []
    for h in range(k.shape[0]):
        rk, _, _ = outlier_stats(k[h])
        rv, _, _ = outlier_stats(v[h])
        k_ratios.append(rk); v_ratios.append(rv)

    k_avg = float(np.nanmean(k_ratios)); k_max = float(np.nanmax(k_ratios))
    v_avg = float(np.nanmean(v_ratios)); v_max = float(np.nanmax(v_ratios))

    print(f"{li:>5} | {k_avg:>11.2f} | {k_max:>11.2f} | {v_avg:>11.2f} | {v_max:>11.2f}")

    results["layers"].append({
        "layer": li,
        "k_ratio_avg": k_avg, "k_ratio_max": k_max,
        "v_ratio_avg": v_avg, "v_ratio_max": v_max,
        "k_shape": list(k.shape), "v_shape": list(v.shape),
    })

# ── Summary ──────────────────────────────────────────────────
all_k_avg = [r["k_ratio_avg"] for r in results["layers"]]
all_k_max = [r["k_ratio_max"] for r in results["layers"]]
all_v_avg = [r["v_ratio_avg"] for r in results["layers"]]
all_v_max = [r["v_ratio_max"] for r in results["layers"]]

print(f"\n{'='*70}")
print(f"GLOBAL SUMMARY (Gemma 4 E2B-it)")
print(f"{'='*70}")
print(f"  K ratio avg (layer-avg heads): {np.mean(all_k_avg):.2f}x")
print(f"  K ratio max (worst head)     : {np.max(all_k_max):.2f}x")
print(f"  V ratio avg (layer-avg heads): {np.mean(all_v_avg):.2f}x")
print(f"  V ratio max (worst head)     : {np.max(all_v_max):.2f}x")

worst_k_layer = int(np.argmax(all_k_max))
worst_v_layer = int(np.argmax(all_v_max))
print(f"\n  Worst K layer: {worst_k_layer} ({all_k_max[worst_k_layer]:.2f}x)")
print(f"  Worst V layer: {worst_v_layer} ({all_v_max[worst_v_layer]:.2f}x)")

overall = max(np.max(all_k_max), np.max(all_v_max))
print(f"\n  Overall worst outlier ratio: {overall:.2f}x")
print(f"\n  Baselines from our prior experiments:")
print(f"    Qwen2-7B    : 8.6x -> naive4 BREAKS      (0/26 NIAH)")
print(f"    Pythia-6.9B : 4.6x -> naive4 +22 PPL")
print(f"    Qwen2.5-14B : 3.5x -> naive4 fine        (26/26 NIAH)")
print(f"    Mistral-7B  : 3.1x -> naive4 +0.10 PPL")
print(f"\n  Gemma 4 E2B : {overall:.2f}x -> prediction:")
if overall > 8.0:
    print(f"    >>> HIGH RISK : naive4 likely catastrophic, nsep+pchan4 critical")
elif overall > 5.0:
    print(f"    >>> MODERATE  : naive4 will degrade, nsep+pchan4 helpful")
else:
    print(f"    >>> LOW RISK  : naive4 likely survives")

# ── JSON output ─────────────────────────────────────────────
results["model"] = MODEL_ID
results["timestamp"] = datetime.now().isoformat()
results["probe_tokens"] = ids.shape[1]
results["summary"] = {
    "k_avg": float(np.mean(all_k_avg)),
    "k_max": float(np.max(all_k_max)),
    "v_avg": float(np.mean(all_v_avg)),
    "v_max": float(np.max(all_v_max)),
    "worst_k_layer": worst_k_layer,
    "worst_v_layer": worst_v_layer,
    "overall_max": float(overall),
}

print(f"\n{'='*70}")
print("JSON:")
print(f"{'='*70}")
print(json.dumps(results["summary"], indent=2))

# ── cleanup ──
del mdl, past, out
gc.collect()
torch.cuda.empty_cache()
