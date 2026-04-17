# Copyright 2026 Kentaro Sato
# Licensed under the Apache License, Version 2.0

# ============================================================
# K vs V Outlier Asymmetry Hypothesis Test
# ============================================================
# Hypothesis (from Gemma 4 26B-A4B surprise result):
#   K outlier > 8x is the catastrophic condition.
#   V outlier > 8x alone is tolerated.
#
# Evidence so far:
#   Gemma 4 26B-A4B: K=6.82, V=10.22 -> NIAH 21/21 safe
#   Gemma 4 E2B:     K=5.56, V=7.48  -> NIAH 16/16 safe
#
# Test: measure K_max and V_max separately on:
#   Qwen2-7B    (overall 8.6x, known catastrophic)
#   Mistral-7B  (overall 3.1x, known safe)
#   Qwen2.5-14B (overall 3.5x, known safe)
#
# If Qwen2-7B K_max >= 8x, hypothesis confirmed.
# If Qwen2-7B K_max < 8x, hypothesis refuted and we need more work.
# ============================================================


# =============================================================
# === CELL B1 === Free memory + login + common helpers
# =============================================================

try:
    del mdl, tok, proc
except NameError:
    pass
import gc, torch
gc.collect()
if torch.cuda.is_available():
    torch.cuda.empty_cache()

import numpy as np, json
from datetime import datetime
from transformers import AutoModelForCausalLM, AutoTokenizer

# ─────────────────────────────────────────────────────────────
# >>>>>  PASTE YOUR HF TOKEN (Mistral needs gated access)  <<<<
HF_TOKEN = "hf_XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX"
# ─────────────────────────────────────────────────────────────
from huggingface_hub import login
login(token=HF_TOKEN)

# Common probe text (same across models for consistency)
PROBE = """
The scientific method represents a fundamental approach to understanding the natural
world through systematic observation and rigorous testing. Transformer architectures
have revolutionized natural language processing by enabling models to attend to arbitrary
positions within a sequence. Quantization of neural network activations to low bit
widths such as INT4 can dramatically reduce memory and compute, but introduces
approximation error. Per-row symmetric quantization uses the absolute maximum of each
row as the scale, mapping values to 16 integer levels. When certain channels contain
outlier values many times larger than the typical magnitude, the per-row scale becomes
dominated by those outliers. This phenomenon was characterized by SmoothQuant and
LLM.int8. For key-value caches specifically, the distribution of outliers across channels
determines whether naive quantization schemes remain viable or collapse. The Mona Lisa
painted by Leonardo da Vinci hangs in the Louvre. Photosynthesis converts carbon dioxide
and water into glucose using light energy captured by chlorophyll.
"""

def get_kv(past, li):
    if hasattr(past, 'layers'):
        return past.layers[li].keys, past.layers[li].values
    return past[li][0], past[li][1]

def n_cache_layers(past):
    return len(past.layers) if hasattr(past, 'layers') else len(past)

def outlier_stats(x):
    x = x.float().abs()
    col_absmax = x.amax(dim=0)
    m = col_absmax.mean().item()
    mx = col_absmax.amax().item()
    return (float('nan') if m < 1e-12 else mx / m, mx, m)

def measure_model(hf_id, label):
    """Load model, measure K/V outlier ratios separately, return summary."""
    print(f"\n{'='*60}")
    print(f"Loading {label} ({hf_id}) ...")
    print(f"{'='*60}")

    tok = AutoTokenizer.from_pretrained(hf_id)
    mdl = AutoModelForCausalLM.from_pretrained(
        hf_id,
        torch_dtype=torch.float16,
        device_map="auto",
        use_safetensors=True,
    )
    mdl.eval()

    ids = tok.encode(PROBE, return_tensors="pt").to("cuda")
    print(f"Probe tokens: {ids.shape[1]}")

    with torch.inference_mode():
        out = mdl(input_ids=ids, use_cache=True)
        past = out.past_key_values

    n_layers = n_cache_layers(past)
    print(f"KV cache layers: {n_layers}")

    # Per-layer K and V stats (over all heads)
    per_layer = []
    for li in range(n_layers):
        k, v = get_kv(past, li)
        k = k[0]  # [heads, tokens, head_dim]
        v = v[0]
        k_ratios = [outlier_stats(k[h])[0] for h in range(k.shape[0])]
        v_ratios = [outlier_stats(v[h])[0] for h in range(v.shape[0])]
        per_layer.append({
            "layer": li,
            "k_avg": float(np.nanmean(k_ratios)),
            "k_max": float(np.nanmax(k_ratios)),
            "v_avg": float(np.nanmean(v_ratios)),
            "v_max": float(np.nanmax(v_ratios)),
        })

    # Overall aggregates
    all_k = [r["k_max"] for r in per_layer]
    all_v = [r["v_max"] for r in per_layer]
    k_max_overall = float(np.max(all_k))
    v_max_overall = float(np.max(all_v))
    worst_k = int(np.argmax(all_k))
    worst_v = int(np.argmax(all_v))

    # Print compact summary
    print(f"\n{label} breakdown:")
    print(f"  K_max (any layer, any head): {k_max_overall:.2f}x  (worst layer: {worst_k})")
    print(f"  V_max (any layer, any head): {v_max_overall:.2f}x  (worst layer: {worst_v})")
    print(f"  max(K, V)                  : {max(k_max_overall, v_max_overall):.2f}x")

    # Top-3 layers for K and for V
    k_sorted = sorted(per_layer, key=lambda r: -r["k_max"])[:3]
    v_sorted = sorted(per_layer, key=lambda r: -r["v_max"])[:3]
    print(f"  Top-3 K layers: ", [(r["layer"], f"{r['k_max']:.2f}") for r in k_sorted])
    print(f"  Top-3 V layers: ", [(r["layer"], f"{r['v_max']:.2f}") for r in v_sorted])

    result = {
        "model": hf_id,
        "label": label,
        "n_cache_layers": n_layers,
        "k_max": k_max_overall,
        "v_max": v_max_overall,
        "worst_k_layer": worst_k,
        "worst_v_layer": worst_v,
        "per_layer": per_layer,
    }

    # cleanup
    del mdl, tok, out, past
    gc.collect(); torch.cuda.empty_cache()
    return result


# =============================================================
# === CELL B2 === Qwen2-7B (THE critical test case)
# =============================================================

qwen2_result = measure_model("Qwen/Qwen2-7B", "Qwen2-7B")

# Immediate hypothesis test
k_max = qwen2_result["k_max"]
print(f"\n{'='*60}")
print(f"HYPOTHESIS TEST on Qwen2-7B")
print(f"{'='*60}")
print(f"  Qwen2-7B K_max = {k_max:.2f}x")
if k_max >= 8.0:
    print(f"  >>> K >= 8x CONFIRMED")
    print(f"  >>> K-dominance hypothesis SUPPORTED")
    print(f"  >>> Qwen2-7B catastrophic failure = K outliers > 8x")
else:
    print(f"  >>> K < 8x (hypothesis challenged)")
    print(f"  >>> Need deeper analysis: how does 8.6x overall split into K/V?")


# =============================================================
# === CELL B3 === Mistral-7B (known safe, sanity check)
# =============================================================

mistral_result = measure_model("mistralai/Mistral-7B-v0.1", "Mistral-7B")


# =============================================================
# === CELL B4 === Qwen2.5-14B (known safe, larger)
# =============================================================

qwen25_result = measure_model("Qwen/Qwen2.5-14B", "Qwen2.5-14B")


# =============================================================
# === CELL B5 === Comparison table + decision
# =============================================================

# Known results from prior experiments
known = [
    {"label": "Gemma 4 E2B-it",     "k_max": 5.56,  "v_max": 7.48,
     "niah": "16/16", "note": "safe, Phase 1+2 verified"},
    {"label": "Gemma 4 26B-A4B-it", "k_max": 6.82,  "v_max": 10.22,
     "niah": "21/21", "note": "safe despite V>8x"},
]

measured = [
    {"label": qwen2_result["label"], "k_max": qwen2_result["k_max"],
     "v_max": qwen2_result["v_max"], "niah": "0/26",
     "note": "catastrophic (prior)"},
    {"label": mistral_result["label"], "k_max": mistral_result["k_max"],
     "v_max": mistral_result["v_max"], "niah": "15/15",
     "note": "safe (prior)"},
    {"label": qwen25_result["label"], "k_max": qwen25_result["k_max"],
     "v_max": qwen25_result["v_max"], "niah": "26/26",
     "note": "safe (prior)"},
]

all_data = known + measured

print(f"\n{'='*80}")
print(f"K vs V OUTLIER ASYMMETRY ANALYSIS")
print(f"{'='*80}")
print(f"  {'Model':<22} | {'K_max':>7} | {'V_max':>7} | {'NIAH':>10} | Note")
print(f"  {'-'*22} + {'-'*7} + {'-'*7} + {'-'*10} + {'-'*30}")
for d in all_data:
    flag_k = "!!" if d["k_max"] >= 8.0 else "  "
    flag_v = "!!" if d["v_max"] >= 8.0 else "  "
    print(f"  {d['label']:<22} | {d['k_max']:>5.2f}{flag_k} | {d['v_max']:>5.2f}{flag_v} | {d['niah']:>10} | {d['note']}")

print(f"\n  !! = outlier >= 8x threshold")

# Decision logic
qwen2_k = qwen2_result["k_max"]
qwen2_v = qwen2_result["v_max"]
mistral_k = mistral_result["k_max"]
qwen25_k = qwen25_result["k_max"]

print(f"\n{'─'*80}")
print(f"HYPOTHESIS EVALUATION")
print(f"{'─'*80}")
if qwen2_k >= 8.0:
    print(f"  ✓ Qwen2-7B K_max = {qwen2_k:.2f}x (>= 8x)")
    if qwen2_v < 8.0:
        print(f"  ✓ Qwen2-7B V_max = {qwen2_v:.2f}x (< 8x)")
        print(f"  >>> Pattern: K high + V low -> catastrophic (Qwen2-7B)")
    else:
        print(f"  - Qwen2-7B V_max = {qwen2_v:.2f}x (>= 8x, both high)")
        print(f"  >>> Pattern: K high + V high -> catastrophic")

    if mistral_k < 8.0 and qwen25_k < 8.0:
        print(f"  ✓ Safe models have K_max < 8x")
        print(f"\n  Gemma 4 26B-A4B: K={all_data[1]['k_max']:.2f}, V={all_data[1]['v_max']:.2f}")
        print(f"    Pattern: K low + V high -> safe")
        print(f"\n  >>> K-DOMINANCE HYPOTHESIS CONFIRMED")
        print(f"      Catastrophic failure requires K_max >= 8x.")
        print(f"      V_max alone (up to 10.22x observed) is tolerated.")
else:
    print(f"  ✗ Qwen2-7B K_max = {qwen2_k:.2f}x (< 8x)")
    print(f"    Qwen2-7B V_max = {qwen2_v:.2f}x")
    print(f"  >>> K-dominance hypothesis REFUTED by Qwen2-7B data.")
    print(f"  >>> Need to investigate: concentrated vs distributed outliers,")
    print(f"      head dim, KV head count, or other factors.")

# Save
output = {
    "experiment": "k_vs_v_outlier_asymmetry",
    "timestamp": datetime.now().isoformat(),
    "hypothesis": "K_max >= 8x is the catastrophic condition; V_max alone is tolerated",
    "measurements": {
        "qwen2_7b": qwen2_result,
        "mistral_7b": mistral_result,
        "qwen2_5_14b": qwen25_result,
    },
    "known_prior": known,
    "summary_table": all_data,
}

print(f"\n{'='*80}")
print(f"FINAL JSON")
print(f"{'='*80}")
# Compact: don't include per_layer in printed JSON
compact = {k: v for k, v in output.items()}
for m in compact["measurements"]:
    if "per_layer" in compact["measurements"][m]:
        compact["measurements"][m] = {
            k: v for k, v in compact["measurements"][m].items() if k != "per_layer"
        }
print(json.dumps(compact, indent=2, ensure_ascii=False))
