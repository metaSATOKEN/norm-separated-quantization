# Copyright 2026 Kentaro Sato
# Licensed under the Apache License, Version 2.0

# ============================================================
# Llama Family K/V Outlier Measurement (Colab, 95GB VRAM)
# ============================================================
# Test the K/V asymmetry hypothesis on Llama models.
#
# Hypothesis to test:
#   Does any Llama model exhibit the Qwen2-7B pathology
#   (K_max > 15x concentrated at Layer 0)?
#
# Current 6-model sample:
#   Qwen2-7B        K=17.23 @ L0  -> catastrophic
#   Qwen2.5-14B     K=10.65 @ L30 -> safe
#   Mistral-7B      K=6.17,  V=16.47 -> safe
#   Gemma 4 E2B     K=5.56          -> safe
#   Gemma 4 26B-A4B K=6.82,  V=10.22 -> safe
#   Gemma 4 31B     K=8.92,  V=39.63 -> safe
#
# Test models (add to sample):
#   L1: Llama-3.1-8B-Instruct  (primary, ~16GB)
#   L2: Llama-3.2-3B-Instruct  (small sanity, ~6GB)
#
# Runtime: ~20 min total (DL + measurement)
# ============================================================


# =============================================================
# === CELL L1 === Free + install + login
# =============================================================

try:
    del mdl, tok
except NameError:
    pass
import gc, torch
gc.collect()
if torch.cuda.is_available():
    torch.cuda.empty_cache()

!pip install -q -U transformers accelerate hf_transfer
import os
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"

# ─────────────────────────────────────────────────────────────
# >>>>>  PASTE YOUR HF TOKEN (Llama is gated!)  <<<<<
# Steps:
#   1. Go to https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct
#   2. Click "Access request" and fill Meta's form
#   3. Wait for approval (hours to 1 day)
#   4. Create read token at https://huggingface.co/settings/tokens
# ─────────────────────────────────────────────────────────────
HF_TOKEN = "hf_XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX"
# ─────────────────────────────────────────────────────────────

from huggingface_hub import login
login(token=HF_TOKEN)
print("HF login OK")
print(f"VRAM free: {torch.cuda.mem_get_info()[0]/1e9:.1f} GB")


# =============================================================
# === CELL L2 === Common helpers
# =============================================================

import numpy as np, json
from datetime import datetime
from transformers import AutoModelForCausalLM, AutoTokenizer

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
    print(f"\n{'='*60}")
    print(f"Loading {label} ({hf_id}) ...")
    print(f"{'='*60}")
    tok = AutoTokenizer.from_pretrained(hf_id)
    mdl = AutoModelForCausalLM.from_pretrained(
        hf_id, torch_dtype=torch.float16,
        device_map="auto", use_safetensors=True,
    )
    mdl.eval()

    cfg = mdl.config
    print(f"  class: {type(mdl).__name__}")
    for k in ["num_hidden_layers", "num_attention_heads",
              "num_key_value_heads", "head_dim", "hidden_size",
              "vocab_size"]:
        print(f"  {k:<25}: {getattr(cfg, k, 'N/A')}")

    ids = tok.encode(PROBE, return_tensors="pt").to("cuda")
    print(f"\n  Probe tokens: {ids.shape[1]}")

    with torch.inference_mode():
        out = mdl(input_ids=ids, use_cache=True)
        past = out.past_key_values
    n_layers = n_cache_layers(past)
    print(f"  KV cache layers: {n_layers}")

    per_layer = []
    for li in range(n_layers):
        k, v = get_kv(past, li)
        k = k[0]; v = v[0]
        k_ratios = [outlier_stats(k[h])[0] for h in range(k.shape[0])]
        v_ratios = [outlier_stats(v[h])[0] for h in range(v.shape[0])]
        per_layer.append({
            "layer": li,
            "k_avg": float(np.nanmean(k_ratios)),
            "k_max": float(np.nanmax(k_ratios)),
            "v_avg": float(np.nanmean(v_ratios)),
            "v_max": float(np.nanmax(v_ratios)),
        })

    all_k = [r["k_max"] for r in per_layer]
    all_v = [r["v_max"] for r in per_layer]
    k_max = float(np.max(all_k))
    v_max = float(np.max(all_v))
    worst_k = int(np.argmax(all_k))
    worst_v = int(np.argmax(all_v))
    layer_0_k = per_layer[0]["k_max"]

    print(f"\n  --- {label} Summary ---")
    print(f"  K max          : {k_max:.2f}x @ Layer {worst_k}")
    print(f"  V max          : {v_max:.2f}x @ Layer {worst_v}")
    print(f"  Layer 0 K      : {layer_0_k:.2f}x")

    k_sorted = sorted(per_layer, key=lambda r: -r["k_max"])[:3]
    v_sorted = sorted(per_layer, key=lambda r: -r["v_max"])[:3]
    k_top3 = [(r['layer'], round(r['k_max'], 2)) for r in k_sorted]
    v_top3 = [(r['layer'], round(r['v_max'], 2)) for r in v_sorted]
    print(f"  Top-3 K layers : {k_top3}")
    print(f"  Top-3 V layers : {v_top3}")

    # Pathology check
    print(f"\n  Pathology test (K@L0 >= 15x):")
    if layer_0_k >= 15.0:
        print(f"    >>> PATHOLOGY DETECTED (2nd catastrophic candidate)")
    elif layer_0_k >= 10.0:
        print(f"    >>> elevated Layer 0 K (moderate)")
    else:
        print(f"    >>> Layer 0 K normal ({layer_0_k:.2f}x)")

    result = {
        "model": hf_id, "label": label,
        "n_cache_layers": n_layers,
        "k_max": k_max, "v_max": v_max,
        "worst_k_layer": worst_k, "worst_v_layer": worst_v,
        "layer_0_k": layer_0_k,
        "top3_k_layers": k_top3,
        "top3_v_layers": v_top3,
        "per_layer": per_layer,
    }

    del mdl, tok, out, past
    gc.collect(); torch.cuda.empty_cache()
    return result


# =============================================================
# === CELL L3 === Llama-3.1-8B-Instruct (primary test)
# =============================================================

llama_8b = measure_model(
    "meta-llama/Llama-3.1-8B-Instruct",
    "Llama-3.1-8B-Instruct"
)


# =============================================================
# === CELL L4 === Llama-3.2-3B-Instruct (sanity, smaller)
# =============================================================

llama_3b = measure_model(
    "meta-llama/Llama-3.2-3B-Instruct",
    "Llama-3.2-3B-Instruct"
)


# =============================================================
# === CELL L5 === Full 8-model summary
# =============================================================

# Prior 9 models (6 measured NIAH + 3 predicted from Phi/DeepSeek)
prior = [
    {"label": "Qwen2-7B",                 "k_max": 17.23, "v_max": 6.09,  "layer_0_k": 17.23, "niah": "0/26",  "status": "catastrophic"},
    {"label": "Qwen2.5-14B",              "k_max": 10.65, "v_max": 7.71,  "layer_0_k": None,  "niah": "26/26", "status": "safe"},
    {"label": "Mistral-7B",               "k_max": 6.17,  "v_max": 16.47, "layer_0_k": None,  "niah": "15/15", "status": "safe"},
    {"label": "Gemma 4 E2B-it",           "k_max": 5.56,  "v_max": 7.48,  "layer_0_k": None,  "niah": "16/16", "status": "safe"},
    {"label": "Gemma 4 26B-A4B",          "k_max": 6.82,  "v_max": 10.22, "layer_0_k": None,  "niah": "21/21", "status": "safe"},
    {"label": "Gemma 4 31B-it",           "k_max": 8.92,  "v_max": 39.63, "layer_0_k": 6.59,  "niah": "21/21", "status": "safe"},
    {"label": "Phi-3-mini-4k-instruct",   "k_max": 5.85,  "v_max": 24.37, "layer_0_k": 4.33,  "niah": "TBD",   "status": "safe (pred)"},
    {"label": "Phi-3-medium-4k-instruct", "k_max": 5.67,  "v_max": 34.99, "layer_0_k": 2.82,  "niah": "TBD",   "status": "safe (pred)"},
    {"label": "DeepSeek-LLM-7B-Chat",     "k_max": 9.07,  "v_max": 10.04, "layer_0_k": 7.45,  "niah": "TBD",   "status": "safe (pred)"},
]

# New measurements
new_measured = [
    {"label": llama_8b["label"], "k_max": llama_8b["k_max"],
     "v_max": llama_8b["v_max"], "layer_0_k": llama_8b["layer_0_k"],
     "niah": "TBD", "status": "TBD"},
    {"label": llama_3b["label"], "k_max": llama_3b["k_max"],
     "v_max": llama_3b["v_max"], "layer_0_k": llama_3b["layer_0_k"],
     "niah": "TBD", "status": "TBD"},
]

all_models = prior + new_measured

print("\n" + "="*95)
print("COMPLETE 11-MODEL K vs V TABLE")
print("="*95)
print(f"  {'Model':<25} | {'K_max':>6} | {'V_max':>6} | {'L0 K':>6} | {'NIAH':>7} | Status")
print(f"  {'-'*25} + {'-'*6} + {'-'*6} + {'-'*6} + {'-'*7} + {'-'*15}")
for d in all_models:
    fk = "!!" if d["k_max"] >= 8 else "  "
    fv = "!!" if d["v_max"] >= 8 else "  "
    l0 = f"{d['layer_0_k']:.2f}" if d["layer_0_k"] is not None else "N/A"
    print(f"  {d['label']:<25} | {d['k_max']:>4.2f}{fk} | {d['v_max']:>4.2f}{fv} | {l0:>6} | {d['niah']:>7} | {d['status']}")

# Llama-specific observations
print(f"\n{'─'*85}")
print(f"LLAMA OBSERVATIONS:")
print(f"{'─'*85}")

for llama in [llama_8b, llama_3b]:
    print(f"\n  {llama['label']}:")
    print(f"    K_max: {llama['k_max']:.2f}x @ Layer {llama['worst_k_layer']}")
    print(f"    V_max: {llama['v_max']:.2f}x @ Layer {llama['worst_v_layer']}")
    print(f"    Layer 0 K: {llama['layer_0_k']:.2f}x")

    if llama["layer_0_k"] >= 15.0:
        print(f"    >>> PATHOLOGY: Layer 0 K >= 15x")
        print(f"    >>> Would be the 2nd catastrophic case after Qwen2-7B")
        print(f"    >>> Recommend running NIAH to confirm")
    elif llama["k_max"] >= 10.0:
        print(f"    >>> Moderate K outlier regime")
    else:
        print(f"    >>> Low K outlier regime (safe predicted)")

# Final JSON
final = {
    "experiment": "poc_llama_kv_outlier",
    "timestamp": datetime.now().isoformat(),
    "measurements": {
        "llama_3_1_8b": llama_8b,
        "llama_3_2_3b": llama_3b,
    },
    "complete_table_8_models": all_models,
}
# Compact for display (exclude per_layer)
compact = {k: v for k, v in final.items()}
for m in list(compact["measurements"].values()):
    if "per_layer" in m: del m["per_layer"]

print(f"\n{'='*85}")
print(f"FINAL JSON (compact)")
print(f"{'='*85}")
print(json.dumps(compact, indent=2, ensure_ascii=False))
