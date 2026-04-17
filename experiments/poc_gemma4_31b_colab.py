# Copyright 2026 Kentaro Sato
# Licensed under the Apache License, Version 2.0

# ============================================================
# Gemma 4 31B-it Full Evaluation (Colab, 95GB VRAM)
# ============================================================
# Gemma 4 31B is the largest dense variant, and attracts the most
# Reddit complaints about KV cache behavior.
#
# Core questions:
#   Q1: Does 31B exhibit K_max >= 15x at Layer 0? (pathological
#       pattern seen only in Qwen2-7B so far)
#   Q2: Is Gemma 4 family uniformly robust, or does 31B break?
#
# Data points so far:
#   - Gemma 4 E2B   (2B dense) : K=5.56, V=7.48  safe 16/16
#   - Gemma 4 26B-A4B (MoE)    : K=6.82, V=10.22 safe 21/21
#   - Qwen2-7B                 : K=17.23 @ L0    catastrophic 0/26
#   - Qwen2.5-14B              : K=10.65 @ L30   safe 26/26
#   - Mistral-7B               : V=16.47 @ L0    safe 15/15
#
# Runtime:
#   DL ~20 min (62GB), load ~3 min, Phase 1+2 ~15 min
#   Total ~40 min
# ============================================================


# =============================================================
# === CELL C1 === Free memory + login
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
# >>>>>  PASTE YOUR HF TOKEN (new one, read-only)  <<<<<
HF_TOKEN = "hf_XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX"
# ─────────────────────────────────────────────────────────────

from huggingface_hub import login
login(token=HF_TOKEN)
print("HF login OK")
print(f"VRAM free: {torch.cuda.mem_get_info()[0]/1e9:.1f} GB")


# =============================================================
# === CELL C2 === Download and load 31B-it
# =============================================================

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_ID = "google/gemma-4-31B-it"

print(f"Loading {MODEL_ID} ... (DL ~20 min first time)")
tok = AutoTokenizer.from_pretrained(MODEL_ID)
mdl = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.float16,
    device_map="auto",
    use_safetensors=True,
    low_cpu_mem_usage=True,
)
mdl.eval()

# Architecture info
cfg = mdl.config
txt_cfg = getattr(cfg, "text_config", cfg)
print(f"\n--- Config ---")
print(f"  class: {type(mdl).__name__}")
for k in ["num_hidden_layers", "num_attention_heads", "num_key_value_heads",
         "head_dim", "hidden_size", "vocab_size", "sliding_window",
         "num_kv_shared_layers", "tie_word_embeddings"]:
    print(f"  text.{k:<25}: {getattr(txt_cfg, k, 'N/A')}")

print(f"\nVRAM used: {torch.cuda.memory_allocated()/1e9:.1f} GB")


# =============================================================
# === CELL C3 === Phase 1: K/V outlier ratio measurement
# =============================================================

import numpy as np, json
from datetime import datetime

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
and water into glucose using light energy captured by chlorophyll. Mixture-of-experts
architectures route tokens to specialized sub-networks based on learned gating functions.
"""

ids = tok.encode(PROBE, return_tensors="pt").to("cuda")
print(f"Probe tokens: {ids.shape[1]}")

def get_kv(past, li):
    if hasattr(past, 'layers'):
        return past.layers[li].keys, past.layers[li].values
    return past[li][0], past[li][1]

def n_cache_layers(past):
    return len(past.layers) if hasattr(past, 'layers') else len(past)

with torch.inference_mode():
    out = mdl(input_ids=ids, use_cache=True)
    past = out.past_key_values

n_layers = n_cache_layers(past)
print(f"KV cache layers: {n_layers}")

def outlier_stats(x):
    x = x.float().abs()
    col_absmax = x.amax(dim=0)
    m = col_absmax.mean().item()
    mx = col_absmax.amax().item()
    return (float('nan') if m < 1e-12 else mx / m, mx, m)

phase1 = {"layers": []}
print(f"\n{'Layer':>5} | {'K head avg':>11} | {'K max':>11} | {'V head avg':>11} | {'V max':>11}")
print("-" * 65)
for li in range(n_layers):
    k, v = get_kv(past, li)
    k = k[0]; v = v[0]
    k_ratios = [outlier_stats(k[h])[0] for h in range(k.shape[0])]
    v_ratios = [outlier_stats(v[h])[0] for h in range(v.shape[0])]
    ka = float(np.nanmean(k_ratios)); km = float(np.nanmax(k_ratios))
    va = float(np.nanmean(v_ratios)); vm = float(np.nanmax(v_ratios))
    print(f"{li:>5} | {ka:>11.2f} | {km:>11.2f} | {va:>11.2f} | {vm:>11.2f}")
    phase1["layers"].append({"layer": li, "k_avg": ka, "k_max": km,
                             "v_avg": va, "v_max": vm})

all_k = [r["k_max"] for r in phase1["layers"]]
all_v = [r["v_max"] for r in phase1["layers"]]
k_overall = float(np.max(all_k))
v_overall = float(np.max(all_v))
worst_k = int(np.argmax(all_k))
worst_v = int(np.argmax(all_v))

print(f"\n{'='*60}")
print(f"  K max : {k_overall:.2f}x @ Layer {worst_k}")
print(f"  V max : {v_overall:.2f}x @ Layer {worst_v}")

# Layer 0 K check (THE critical pattern)
layer_0_k = phase1["layers"][0]["k_max"]
print(f"\n  Layer 0 K_max: {layer_0_k:.2f}x")
print(f"\n  Pathological pattern test:")
if layer_0_k >= 15.0:
    print(f"    >>> LAYER 0 K >= 15x !! Pathological pattern DETECTED")
    print(f"    >>> Predicting catastrophic naive4 failure")
    print(f"    >>> This would be the 2nd catastrophic case in our sample")
elif layer_0_k >= 10.0:
    print(f"    >>> Layer 0 K = {layer_0_k:.2f}x (elevated but below threshold)")
    print(f"    >>> Expecting moderate degradation at worst")
else:
    print(f"    >>> Layer 0 K = {layer_0_k:.2f}x (normal)")
    print(f"    >>> Predicting safe, consistent with other Gemma 4 variants")

# Top-3 critical layers
k_sorted = sorted(phase1["layers"], key=lambda r: -r["k_max"])[:3]
v_sorted = sorted(phase1["layers"], key=lambda r: -r["v_max"])[:3]
k_top3_str = [(r['layer'], round(r['k_max'], 2)) for r in k_sorted]
v_top3_str = [(r['layer'], round(r['v_max'], 2)) for r in v_sorted]
print(f"\n  Top-3 K layers: {k_top3_str}")
print(f"  Top-3 V layers: {v_top3_str}")

del out, past; gc.collect(); torch.cuda.empty_cache()


# =============================================================
# === CELL C4 === Phase 2: NIAH + PPL (full pipeline)
# =============================================================

import torch.nn.functional as F

def qa_perrow(x, b):
    x = x.float(); qm = 2**(b-1)-1
    s = x.abs().amax(dim=-1, keepdim=True).clamp(min=1e-12) / qm
    return ((x/s).round().clamp(-qm, qm)) * s

def qa_perchan(x, b):
    x = x.float(); qm = 2**(b-1)-1
    s = x.abs().amax(dim=0, keepdim=True).clamp(min=1e-12) / qm
    return ((x/s).round().clamp(-qm, qm)) * s

def apply_method(x, name):
    if name == "naive4":
        return qa_perrow(x, 4)
    if name == "nsep+pchan4":
        x = x.float()
        n = x.norm(dim=-1, keepdim=True).clamp(min=1e-12)
        d = x / n
        dq = qa_perchan(d, 4)
        dq = dq / dq.norm(dim=-1, keepdim=True).clamp(min=1e-8)
        return n * dq

def set_kv(past, li, k, v):
    if hasattr(past, 'layers'):
        past.layers[li].keys = k; past.layers[li].values = v
    else:
        past[li] = (k, v)

def compress_cache(past, method):
    for li in range(n_cache_layers(past)):
        ok, ov = get_kv(past, li)
        ok = ok.clone(); ov = ov.clone()
        nk = torch.zeros_like(ok); nv = torch.zeros_like(ov)
        for h in range(ok.shape[1]):
            nk[0, h] = apply_method(ok[0, h], method).to(ok.dtype)
            nv[0, h] = apply_method(ov[0, h], method).to(ov.dtype)
        set_kv(past, li, nk, nv)

def chat_encode(messages, add_gen=True):
    r = tok.apply_chat_template(
        messages, add_generation_prompt=add_gen, return_tensors="pt"
    )
    if hasattr(r, "input_ids"): r = r.input_ids
    elif isinstance(r, dict):   r = r["input_ids"]
    return r.to("cuda")

# ── NIAH setup ───────────────────────────────────────────────
FILLER = [
    "The weather in coastal regions is influenced by ocean currents and prevailing winds.",
    "Ancient civilizations developed sophisticated agricultural techniques and crop rotation.",
    "Mathematics has evolved from simple counting to abstract algebra over centuries.",
    "Music theory encompasses harmony, melody, rhythm, and distinct cultural traditions.",
    "Volcanic activity shapes landscapes through destructive and creative geological forces.",
    "Printing technology revolutionized information spread from Gutenberg to digital publishing.",
    "Marine ecosystems support biodiversity with coral reefs hosting thousands of species.",
    "Aerodynamics governs aircraft and automobile design through air flow understanding.",
    "Traditional medicine contributed to modern pharmacology via indigenous plant knowledge.",
    "Roman architecture influences modern design through the arch and concrete innovations.",
    "Quantum mechanics describes matter at atomic scales via wave-particle duality.",
    "The human brain contains 86 billion neurons inspiring modern AI research.",
    "Plate tectonics explains lithospheric movement and continental drift acceptance.",
    "The steam engine powered the Industrial Revolution transforming European economies.",
    "Photosynthesis converts sunlight to chemical energy sustaining most life on Earth.",
    "The periodic table organizes elements by atomic number as Mendeleev predicted.",
    "Shakespeare wrote 37 plays and 154 sonnets while inventing 1700 English words.",
    "General relativity describes gravity as spacetime curvature predicting black holes.",
    "DNA carries genetic instructions determined as a double helix by Watson and Crick.",
    "The Renaissance marked cultural rebirth transforming art, science, and philosophy.",
]
NEEDLES = [
    {"text": "SECRET FACT ALPHA: The laboratory password is CRIMSON TIGER 9981.", "key": "CRIMSON TIGER 9981"},
    {"text": "SECRET FACT BETA: The project codename is SILVER DOLPHIN 2247.",    "key": "SILVER DOLPHIN 2247"},
    {"text": "SECRET FACT GAMMA: The vault combination is GOLDEN EAGLE 5563.",   "key": "GOLDEN EAGLE 5563"},
    {"text": "SECRET FACT DELTA: The access phrase is PURPLE FALCON 3318.",      "key": "PURPLE FALCON 3318"},
    {"text": "SECRET FACT EPSILON: The encryption key is AZURE PANTHER 7704.",   "key": "AZURE PANTHER 7704"},
]

def build_msg(nf, nn):
    paras = [FILLER[i % len(FILLER)] for i in range(nf)]
    used = NEEDLES[:nn]
    pos = [int(len(paras) * (i+1) / (nn+1)) for i in range(nn)]
    for off, (p, n) in enumerate(zip(pos, used)):
        paras.insert(p + off, n["text"])
    hay = " ".join(paras)
    greek = ["ALPHA","BETA","GAMMA","DELTA","EPSILON"]
    qs = [f"What is SECRET FACT {greek[i]}?" for i in range(nn)]
    content = (f"Read the following text carefully, then answer every question.\n\n"
               f"TEXT:\n{hay}\n\nQUESTIONS:\n" + "\n".join(qs) +
               "\n\nProvide each answer on a separate line.")
    return [{"role": "user", "content": content}], [n["key"] for n in used]

def check(resp, keys):
    r = resp.lower()
    return [k for k in keys if all(p in r for p in k.lower().split())]

def run_niah(prompt_ids, method, max_new=200):
    with torch.inference_mode():
        out = mdl(input_ids=prompt_ids, use_cache=True)
        past = out.past_key_values
        if method != "baseline":
            compress_cache(past, method)
        gen = []
        logits = out.logits[:, -1:]
        for _ in range(max_new):
            nt = logits.argmax(dim=-1)
            tid = nt[0, 0].item()
            gen.append(tid)
            if tid == tok.eos_token_id: break
            out = mdl(input_ids=nt, past_key_values=past, use_cache=True)
            past = out.past_key_values
            logits = out.logits[:, -1:]
    del out, past; gc.collect(); torch.cuda.empty_cache()
    return tok.decode(gen, skip_special_tokens=True).strip()

CONFIGS = [(10, 3), (20, 3), (30, 5), (50, 5), (80, 5)]
METHODS = ["baseline", "naive4", "nsep+pchan4"]

print("="*60)
print("NIAH on Gemma 4 31B-it")
print("="*60)
niah_results = []
for nf, nn in CONFIGS:
    msgs, keys = build_msg(nf, nn)
    pids = chat_encode(msgs)
    ntok = pids.shape[-1]
    row = {"n_filler": nf, "n_needles": nn, "tokens": ntok, "methods": {}}
    print(f"\n  {nf}p / {nn}n, {ntok} tokens:")
    for m in METHODS:
        resp = run_niah(pids, m)
        found = check(resp, keys)
        row["methods"][m] = {"found": len(found), "total": nn, "resp": resp[:120]}
        print(f"    {m:>14}: {len(found)}/{nn}  \"{resp[:70]}\"")
    niah_results.append(row)

print(f"\n{'─'*60}\nNIAH total:")
for m in METHODS:
    f = sum(r["methods"][m]["found"] for r in niah_results)
    t = sum(r["n_needles"] for r in niah_results)
    print(f"  {m:>14}: {f}/{t}")

# ── Final JSON ───────────────────────────────────────────────
final = {
    "experiment": "poc_gemma4_31b_main_eval",
    "model": MODEL_ID,
    "timestamp": datetime.now().isoformat(),
    "outlier": {
        "k_max_overall": k_overall,
        "v_max_overall": v_overall,
        "worst_k_layer": worst_k,
        "worst_v_layer": worst_v,
        "layer_0_k": layer_0_k,
        "n_cache_layers": n_layers,
        "top3_k": [(r['layer'], r['k_max']) for r in k_sorted],
        "top3_v": [(r['layer'], r['v_max']) for r in v_sorted],
    },
    "niah": niah_results,
}
print("\n" + "="*60)
print("FINAL JSON")
print("="*60)
print(json.dumps(final, indent=2, ensure_ascii=False))
