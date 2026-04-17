# Copyright 2026 Kentaro Sato
# Licensed under the Apache License, Version 2.0

# ============================================================
# Phi-3 + DeepSeek K/V Outlier Measurement (Colab, 95GB VRAM)
# ============================================================
# Waiting for Meta Llama approval -- meanwhile test non-gated
# models for architectural diversity.
#
# Test models:
#   P1: microsoft/Phi-3-mini-4k-instruct   (3.8B, MS recipe)
#   P2: microsoft/Phi-3-medium-4k-instruct (14B,  MS recipe)
#   P3: deepseek-ai/deepseek-llm-7b-chat   (7B,   Chinese lab)
#
# Expands our 6-model sample to 9 models across vendors:
#   Anthropic-free, Google (Gemma), Alibaba (Qwen), Mistral AI,
#   Microsoft (Phi), DeepSeek.
#
# All non-gated. Runtime ~25 min (DL dominates).
# ============================================================


# =============================================================
# === CELL P1 === Free + install + login
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
# >>>>>  PASTE YOUR HF TOKEN (all non-gated, read-only OK)  <<<<
HF_TOKEN = "hf_XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX"
# ─────────────────────────────────────────────────────────────
from huggingface_hub import login
login(token=HF_TOKEN)
print("HF login OK")
print(f"VRAM free: {torch.cuda.mem_get_info()[0]/1e9:.1f} GB")


# =============================================================
# === CELL P2 === Common helpers
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

def measure_model(hf_id, label, trust_remote_code=False):
    print(f"\n{'='*60}")
    print(f"Loading {label} ({hf_id}) ...")
    print(f"{'='*60}")

    tok_kwargs = {"trust_remote_code": trust_remote_code}
    mdl_kwargs = {
        "torch_dtype": torch.float16,
        "device_map": "auto",
        "use_safetensors": True,
        "trust_remote_code": trust_remote_code,
    }
    tok = AutoTokenizer.from_pretrained(hf_id, **tok_kwargs)
    mdl = AutoModelForCausalLM.from_pretrained(hf_id, **mdl_kwargs)
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

    print(f"\n  Pathology test (K@L0 >= 15x):")
    if layer_0_k >= 15.0:
        print(f"    >>> PATHOLOGY DETECTED (2nd catastrophic candidate!)")
    elif layer_0_k >= 10.0:
        print(f"    >>> elevated L0 K ({layer_0_k:.2f}x), moderate risk")
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
# === CELL P3 === Phi-3-mini-4k-instruct (3.8B)
# =============================================================

phi3_mini = measure_model(
    "microsoft/Phi-3-mini-4k-instruct",
    "Phi-3-mini-4k-instruct",
    trust_remote_code=False,  # Use transformers native impl (remote code outdated)
)


# =============================================================
# === CELL P4 === Phi-3-medium-4k-instruct (14B)
# =============================================================

phi3_medium = measure_model(
    "microsoft/Phi-3-medium-4k-instruct",
    "Phi-3-medium-4k-instruct",
    trust_remote_code=False,  # Use transformers native impl
)


# =============================================================
# === CELL P5 === DeepSeek-LLM-7B-Chat
# =============================================================

deepseek_7b = measure_model(
    "deepseek-ai/deepseek-llm-7b-chat",
    "DeepSeek-LLM-7B-Chat",
    trust_remote_code=False,
)


# =============================================================
# === CELL P6 === 9-model complete summary
# =============================================================

prior = [
    {"label": "Qwen2-7B",          "k_max": 17.23, "v_max": 6.09,  "layer_0_k": 17.23, "niah": "0/26",  "status": "catastrophic"},
    {"label": "Qwen2.5-14B",       "k_max": 10.65, "v_max": 7.71,  "layer_0_k": None,  "niah": "26/26", "status": "safe"},
    {"label": "Mistral-7B",        "k_max": 6.17,  "v_max": 16.47, "layer_0_k": None,  "niah": "15/15", "status": "safe"},
    {"label": "Gemma 4 E2B-it",    "k_max": 5.56,  "v_max": 7.48,  "layer_0_k": None,  "niah": "16/16", "status": "safe"},
    {"label": "Gemma 4 26B-A4B",   "k_max": 6.82,  "v_max": 10.22, "layer_0_k": None,  "niah": "21/21", "status": "safe"},
    {"label": "Gemma 4 31B-it",    "k_max": 8.92,  "v_max": 39.63, "layer_0_k": 6.59,  "niah": "21/21", "status": "safe"},
]

new_measured = [
    {"label": phi3_mini["label"],   "k_max": phi3_mini["k_max"],
     "v_max": phi3_mini["v_max"],   "layer_0_k": phi3_mini["layer_0_k"],
     "niah": "TBD", "status": "TBD"},
    {"label": phi3_medium["label"], "k_max": phi3_medium["k_max"],
     "v_max": phi3_medium["v_max"], "layer_0_k": phi3_medium["layer_0_k"],
     "niah": "TBD", "status": "TBD"},
    {"label": deepseek_7b["label"], "k_max": deepseek_7b["k_max"],
     "v_max": deepseek_7b["v_max"], "layer_0_k": deepseek_7b["layer_0_k"],
     "niah": "TBD", "status": "TBD"},
]

all_models = prior + new_measured

print("\n" + "="*95)
print("COMPLETE 9-MODEL K vs V TABLE (before Llama)")
print("="*95)
print(f"  {'Model':<30} | {'K_max':>6} | {'V_max':>6} | {'L0 K':>6} | {'NIAH':>7} | Status")
print(f"  {'-'*30} + {'-'*6} + {'-'*6} + {'-'*6} + {'-'*7} + {'-'*15}")
for d in all_models:
    fk = "!!" if d["k_max"] >= 8 else "  "
    fv = "!!" if d["v_max"] >= 8 else "  "
    l0 = f"{d['layer_0_k']:.2f}" if d["layer_0_k"] is not None else "N/A"
    print(f"  {d['label']:<30} | {d['k_max']:>4.2f}{fk} | {d['v_max']:>4.2f}{fv} | {l0:>6} | {d['niah']:>7} | {d['status']}")

# Pathology assessment on new measurements
print(f"\n{'─'*95}")
print(f"NEW MEASUREMENT OBSERVATIONS:")
print(f"{'─'*95}")
for result in [phi3_mini, phi3_medium, deepseek_7b]:
    print(f"\n  {result['label']}:")
    print(f"    K_max: {result['k_max']:.2f}x @ Layer {result['worst_k_layer']}")
    print(f"    V_max: {result['v_max']:.2f}x @ Layer {result['worst_v_layer']}")
    print(f"    Layer 0 K: {result['layer_0_k']:.2f}x")
    if result["layer_0_k"] >= 15.0:
        print(f"    >>> PATHOLOGY -- recommend NIAH test to confirm")
    elif result["k_max"] >= 10.0:
        print(f"    >>> Moderate K regime")
    else:
        print(f"    >>> Low K regime (safe predicted)")

final = {
    "experiment": "poc_phi3_deepseek_kv_outlier",
    "timestamp": datetime.now().isoformat(),
    "measurements": {
        "phi_3_mini_4k_instruct": phi3_mini,
        "phi_3_medium_4k_instruct": phi3_medium,
        "deepseek_llm_7b_chat": deepseek_7b,
    },
    "complete_9_model_table": all_models,
}
compact = {k: v for k, v in final.items()}
for m in list(compact["measurements"].values()):
    if "per_layer" in m: del m["per_layer"]

print(f"\n{'='*95}")
print(f"FINAL JSON (compact)")
print(f"{'='*95}")
print(json.dumps(compact, indent=2, ensure_ascii=False))
