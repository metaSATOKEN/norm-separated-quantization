# Copyright 2026 Kentaro Sato
# Licensed under the Apache License, Version 2.0

# ============================================================
# Catastrophic Pathology Hunt: Looking for N > 1
# ============================================================
# Motivation:
#   Reviewer 2 (and the author) want to know: is the Layer-0 K
#   pathology unique to Qwen2-7B, or a recurring pattern in some
#   model family? Finding even one more catastrophic case would
#   transform the paper from "high-quality case study" to
#   "characterization of a recurring failure mode".
#
# Strategy:
#   Screen models from Chinese labs of the Qwen2-era (2023-2024)
#   and from other non-Meta/Google/MS recipes, because all of the
#   Meta/Google/MS models tested so far are clean.
#
# Measurement only (Phase 1 style): no NIAH per screen, only K/V
# outlier ratios. If K_max >= 12x at Layer 0 shows up in any model,
# we will follow up with targeted NIAH in a second session.
#
# Runtime: ~1 hour for all 11 candidates. Stop early if needed.
# ============================================================


# =============================================================
# === CELL H1 === Setup + login + common helpers
# =============================================================

try:
    del mdl, tok
except NameError:
    pass
import gc, torch
gc.collect()
if torch.cuda.is_available():
    torch.cuda.empty_cache()

!pip install -q -U transformers accelerate hf_transfer sentencepiece
!pip install -q transformers_stream_generator tiktoken einops bitsandbytes
import os
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"

# ─────────────────────────────────────────────────────────────
# >>>>>  PASTE YOUR HF TOKEN  <<<<<
HF_TOKEN = "hf_XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX"
# ─────────────────────────────────────────────────────────────
from huggingface_hub import login
login(token=HF_TOKEN)

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
    try:
        tok = AutoTokenizer.from_pretrained(hf_id, trust_remote_code=trust_remote_code)
        mdl = AutoModelForCausalLM.from_pretrained(
            hf_id, torch_dtype=torch.float16, device_map="auto",
            use_safetensors=True, trust_remote_code=trust_remote_code,
        )
        mdl.eval()
    except Exception as e:
        print(f"  LOAD FAILED: {type(e).__name__}: {str(e)[:200]}")
        return None

    cfg = mdl.config
    print(f"  class: {type(mdl).__name__}")
    for k in ["num_hidden_layers", "num_attention_heads",
              "num_key_value_heads", "head_dim", "hidden_size"]:
        print(f"  {k:<25}: {getattr(cfg, k, 'N/A')}")

    try:
        ids = tok.encode(PROBE, return_tensors="pt").to("cuda")
        print(f"\n  Probe tokens: {ids.shape[1]}")
        with torch.inference_mode():
            out = mdl(input_ids=ids, use_cache=True)
            past = out.past_key_values
        n_layers = n_cache_layers(past)
        print(f"  KV cache layers: {n_layers}")
    except Exception as e:
        print(f"  FORWARD FAILED: {type(e).__name__}: {str(e)[:200]}")
        del mdl, tok
        gc.collect(); torch.cuda.empty_cache()
        return None

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

    k_sorted = sorted(per_layer, key=lambda r: -r["k_max"])[:3]
    v_sorted = sorted(per_layer, key=lambda r: -r["v_max"])[:3]
    k_top3 = [(r['layer'], round(r['k_max'], 2)) for r in k_sorted]
    v_top3 = [(r['layer'], round(r['v_max'], 2)) for r in v_sorted]

    print(f"\n  --- {label} Summary ---")
    print(f"  K max          : {k_max:.2f}x @ Layer {worst_k}")
    print(f"  V max          : {v_max:.2f}x @ Layer {worst_v}")
    print(f"  Layer 0 K      : {layer_0_k:.2f}x")
    print(f"  Top-3 K layers : {k_top3}")
    print(f"  Top-3 V layers : {v_top3}")

    # Hunt verdict
    print(f"\n  Hunt verdict:")
    if layer_0_k >= 15.0:
        print(f"    🎯🎯🎯 PATHOLOGICAL MATCH! K @ L0 = {layer_0_k:.2f}x (>= 15x)")
        print(f"    >>> Candidate catastrophic case. Run NIAH to confirm.")
        verdict = "PATHOLOGICAL"
    elif layer_0_k >= 10.0:
        print(f"    ⚠️  ELEVATED L0 K = {layer_0_k:.2f}x (10-15x window)")
        print(f"    >>> Worth testing NIAH to refine threshold")
        verdict = "ELEVATED"
    elif k_max >= 15.0:
        print(f"    ⚠️  K max {k_max:.2f}x at Layer {worst_k} (not L0)")
        print(f"    >>> Late-layer high K; likely safe but worth NIAH check")
        verdict = "LATE_HIGH_K"
    else:
        print(f"    ✓ Clean profile (L0 K = {layer_0_k:.2f}x, max = {k_max:.2f}x)")
        verdict = "CLEAN"

    result = {
        "model": hf_id, "label": label,
        "n_cache_layers": n_layers,
        "k_max": k_max, "v_max": v_max,
        "worst_k_layer": worst_k, "worst_v_layer": worst_v,
        "layer_0_k": layer_0_k,
        "top3_k_layers": k_top3,
        "top3_v_layers": v_top3,
        "verdict": verdict,
    }

    del mdl, tok, out, past
    gc.collect(); torch.cuda.empty_cache()
    return result


# =============================================================
# === CELL H2 === Tier S: Qwen1.5-7B (bridge gen)
# =============================================================
# Qwen family timeline:
#   Qwen-7B    (2023-08): old remote-code, incompatible with
#                         current transformers -- SKIPPED.
#   Qwen1.5-7B (2024-02): standard transformers class. Sits
#                         between old recipe and Qwen2-7B
#                         pathology. The critical test case:
#                         if Qwen1.5 is pathological too, the
#                         pattern is a family trait, not a
#                         one-off regression.
#   Qwen2-7B   (2024-06): KNOWN pathological (K=17.23 @ L0).
#   Qwen2.5-14B(2024-10): already measured clean.

results = {}

results["qwen1_5_7b"] = measure_model(
    "Qwen/Qwen1.5-7B", "Qwen1.5-7B",
    trust_remote_code=False
)


# =============================================================
# === CELL H3 === Qwen1.5-14B
# =============================================================

results["qwen1_5_14b"] = measure_model(
    "Qwen/Qwen1.5-14B", "Qwen1.5-14B",
    trust_remote_code=False
)


# =============================================================
# === CELL H4 === Tier A: Yi-6B (01.ai)
# =============================================================

results["yi_6b"] = measure_model(
    "01-ai/Yi-6B", "Yi-6B",
    trust_remote_code=False  # Yi uses standard Llama-like config
)


# =============================================================
# === CELL H5 === Baichuan2-7B-Base
# =============================================================

results["baichuan2_7b"] = measure_model(
    "baichuan-inc/Baichuan2-7B-Base", "Baichuan2-7B-Base",
    trust_remote_code=True
)


# =============================================================
# === CELL H6 === InternLM2-7B
# =============================================================

results["internlm2_7b"] = measure_model(
    "internlm/internlm2-7b", "InternLM2-7B",
    trust_remote_code=True
)


# =============================================================
# === CELL H7 === ChatGLM3-6B
# =============================================================

results["chatglm3_6b"] = measure_model(
    "THUDM/chatglm3-6b", "ChatGLM3-6B",
    trust_remote_code=True
)


# =============================================================
# === CELL H8 === Tier B: GPT-J-6B (older, head_dim=256)
# =============================================================

results["gpt_j_6b"] = measure_model(
    "EleutherAI/gpt-j-6b", "GPT-J-6B",
    trust_remote_code=False
)


# =============================================================
# === CELL H9 === Salesforce/xgen-7b
# =============================================================

results["xgen_7b"] = measure_model(
    "Salesforce/xgen-7b-8k-base", "XGen-7B-8k",
    trust_remote_code=True
)


# =============================================================
# === CELL H10 === Summary of the hunt
# =============================================================

print("\n" + "="*90)
print("CATASTROPHIC PATHOLOGY HUNT RESULTS")
print("="*90)
print(f"  {'Model':<30} | {'K_max':>7} | {'V_max':>7} | {'L0 K':>6} | Verdict")
print(f"  {'-'*30} + {'-'*7} + {'-'*7} + {'-'*6} + {'-'*15}")

# Prior known data
prior = [
    ("Qwen2-7B (known pathology)", 17.23, 6.09, 17.23, "PATHOLOGICAL (prior)"),
]

for label, km, vm, l0, verdict in prior:
    print(f"  {label:<30} | {km:>7.2f} | {vm:>7.2f} | {l0:>6.2f} | {verdict}")

for key, r in results.items():
    if r is None:
        print(f"  {key:<30} | {'LOAD':>7} | {'FAIL':>7} | {'---':>6} | load/forward failed")
        continue
    flag = ""
    if r["verdict"] == "PATHOLOGICAL": flag = "🎯"
    elif r["verdict"] == "ELEVATED": flag = "⚠️ "
    elif r["verdict"] == "LATE_HIGH_K": flag = "⚠️ "
    print(f"  {r['label']:<30} | {r['k_max']:>7.2f} | {r['v_max']:>7.2f} | {r['layer_0_k']:>6.2f} | {flag}{r['verdict']}")

# Count pathological
pathological = [r for r in results.values() if r and r["verdict"] == "PATHOLOGICAL"]
elevated = [r for r in results.values() if r and r["verdict"] == "ELEVATED"]
print(f"\n  PATHOLOGICAL matches (L0 K >= 15x): {len(pathological)}")
print(f"  ELEVATED candidates (L0 K 10-15x) : {len(elevated)}")

if pathological:
    print(f"\n  🎯 GENERALIZATION CONFIRMED: found additional pathological model(s)!")
    for p in pathological:
        print(f"     - {p['label']}: K @ L0 = {p['layer_0_k']:.2f}x")
elif elevated:
    print(f"\n  ⚠️  Candidate zone: elevated but not unambiguously pathological")
else:
    print(f"\n  ✗ No new pathological cases found. Qwen2-7B remains unique in this sample.")
    print(f"     This strengthens the 'candidate signature' framing as appropriately humble.")

final = {
    "experiment": "poc_catastrophic_hunt",
    "timestamp": datetime.now().isoformat(),
    "motivation": "Reviewer 2 + self-asked question: is the K@L0>=15x pathology unique to Qwen2-7B (n=1) or a recurring pattern? Screened 8 additional models from Chinese labs (Qwen initial gen, Yi, Baichuan, InternLM, ChatGLM) and other non-Meta/Google/MS sources.",
    "measurements": {k: v for k, v in results.items() if v is not None},
    "failed_to_load": [k for k, v in results.items() if v is None],
    "pathological_count": len(pathological),
    "elevated_count": len(elevated),
}
print(f"\n{'='*90}")
print("FINAL JSON")
print(f"{'='*90}")
print(json.dumps(final, indent=2, ensure_ascii=False))
