# Copyright 2026 Kentaro Sato
# Licensed under the Apache License, Version 2.0

# ============================================================
# Gemma 4 26B-A4B Full Evaluation (Colab, 95GB VRAM)
# ============================================================
# This is THE experiment the paper needs:
#   - Reddit / r/LocalLLaMA reports KV cache issues on 26B+
#   - 26B-A4B is MoE (4B active, 26B total)
#   - MoE + Shared KV + SWA = likely high outlier regime
#
# Prediction (if Reddit reports are accurate):
#   - Outlier V_max > 8x (threshold crossed)
#   - naive4 breaks multi-needle retrieval
#   - nsep+pchan4 recovers
#
# Runtime:
#   - DL: ~15-20 min (~52GB)
#   - Load: ~2-3 min
#   - Phase 1 (outlier): ~2 min
#   - Phase 2 (NIAH+PPL): ~10-15 min
#   Total: ~30-40 min
# ============================================================


# =============================================================
# === CELL A1 === Free memory + install + HF login
# =============================================================

# Free existing 2B model if loaded
try:
    del mdl
    del tok
    del proc
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
# === CELL A2 === Download and load 26B-A4B
# =============================================================

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_ID = "google/gemma-4-26B-A4B-it"

print(f"Loading {MODEL_ID} ...")
tok = AutoTokenizer.from_pretrained(MODEL_ID)
mdl = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.float16,
    device_map="auto",
    use_safetensors=True,
)
mdl.eval()

# Architecture info
cfg = mdl.config
print(f"\n--- Config ---")
print(f"  type: {type(cfg).__name__}")

# Multimodal config? check text_config
txt_cfg = getattr(cfg, "text_config", cfg)
for k in ["num_hidden_layers", "num_attention_heads", "num_key_value_heads",
         "head_dim", "hidden_size", "vocab_size", "sliding_window",
         "num_kv_shared_layers", "num_experts", "num_experts_per_tok",
         "tie_word_embeddings"]:
    v = getattr(txt_cfg, k, "N/A")
    print(f"  text.{k:<25}: {v}")

print(f"\nVRAM used: {torch.cuda.memory_allocated()/1e9:.1f} GB")


# =============================================================
# === CELL A3 === Phase 1: Outlier ratio measurement
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
overall = max(np.max(all_k), np.max(all_v))
print(f"\n{'='*60}")
print(f"  K max (any layer, any head): {np.max(all_k):.2f}x")
print(f"  V max (any layer, any head): {np.max(all_v):.2f}x")
print(f"  Overall: {overall:.2f}x")
print(f"  Worst K layer: {int(np.argmax(all_k))}")
print(f"  Worst V layer: {int(np.argmax(all_v))}")
print(f"\n  Prior Gemma 4 E2B-it V_max: 7.48x (below threshold)")
print(f"  Qwen2-7B V_max: 8.6x (catastrophic)")
print(f"\n  Prediction:")
if overall > 8.0:
    print(f"    >>> {overall:.1f}x > 8x -- naive4 likely CATASTROPHIC on 26B-A4B")
elif overall > 5.0:
    print(f"    >>> {overall:.1f}x MODERATE zone")
else:
    print(f"    >>> {overall:.1f}x LOW risk")

del out, past; gc.collect(); torch.cuda.empty_cache()


# =============================================================
# === CELL A4 === Phase 2: NIAH + PPL
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
    ids = prompt_ids
    with torch.inference_mode():
        out = mdl(input_ids=ids, use_cache=True)
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
print("NIAH on 26B-A4B")
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

# ── PPL ──────────────────────────────────────────────────────
CHATS = [
    [{"role":"user","content":"Explain photosynthesis in two sentences."},
     {"role":"assistant","content":
      "Photosynthesis is the process by which plants, algae, and some bacteria convert "
      "light energy into chemical energy stored in glucose. They use carbon dioxide and "
      "water as inputs, releasing oxygen as a byproduct that supports most life on Earth."}],
    [{"role":"user","content":"What is the significance of the Turing Test?"},
     {"role":"assistant","content":
      "The Turing Test, proposed by Alan Turing in 1950, evaluates whether a machine can "
      "exhibit intelligent behavior indistinguishable from a human during a text-based "
      "conversation. It shaped decades of AI philosophy and remains a cultural benchmark."}],
    [{"role":"user","content":"Describe plate tectonics briefly."},
     {"role":"assistant","content":
      "Plate tectonics is the theory that Earth's outer shell is divided into large plates "
      "that slowly move over the underlying mantle. Their interactions cause earthquakes, "
      "volcanic activity, mountain building, and the gradual reshaping of continents."}],
    [{"role":"user","content":"Summarize the history of AI in three sentences."},
     {"role":"assistant","content":
      "Artificial intelligence research began at the 1956 Dartmouth workshop, with early "
      "programs demonstrating symbolic reasoning and game playing. Enthusiasm alternated "
      "with AI winters as funding and expectations oscillated. The 2010s deep learning "
      "revolution, powered by neural networks and massive data, transformed AI from a "
      "research curiosity into mainstream technology."}],
    [{"role":"user","content":"What causes tides on Earth?"},
     {"role":"assistant","content":
      "Tides are primarily caused by the gravitational pull of the Moon on Earth's oceans, "
      "creating bulges of water on the side facing the Moon and the opposite side. The Sun "
      "also contributes, and when Sun, Earth, and Moon align, tides are larger (spring tides)."}],
]

def ppl_on_chat(chat, method):
    full = chat_encode(chat, add_gen=False)
    user = chat_encode(chat[:1], add_gen=True)
    split = min(user.shape[-1], full.shape[-1] - 2)
    ctx = full[:, :split]
    tgt = full[:, split:]
    if tgt.shape[-1] < 2: return float("nan"), 0
    with torch.inference_mode():
        out_c = mdl(input_ids=ctx, use_cache=True)
        past = out_c.past_key_values
        first = out_c.logits[:, -1:, :]
        if method != "baseline":
            compress_cache(past, method)
        out_t = mdl(input_ids=tgt, past_key_values=past, use_cache=False)
        rest = out_t.logits[:, :-1, :]
        allp = torch.cat([first, rest], dim=1)
        lp = F.log_softmax(allp[0].float(), dim=-1)
        nll = -lp.gather(-1, tgt[0].unsqueeze(-1)).squeeze(-1)
    loss = nll.mean().item()
    nt = tgt.shape[-1]
    del out_c, out_t, past; gc.collect(); torch.cuda.empty_cache()
    return loss, nt

print("\n" + "="*60)
print("PPL on 5 chat exchanges")
print("="*60)
print(f"  {'Method':>14} | {'NLL':>8} | {'PPL':>10} | {'tokens':>8}")
print("-"*50)
ppl_results = {}
for m in METHODS:
    w_nll = 0.0; tt = 0
    for c in CHATS:
        l, n = ppl_on_chat(c, m)
        if n > 0 and not np.isnan(l):
            w_nll += l * n; tt += n
    mn = w_nll / max(tt, 1)
    ppl = float(np.exp(mn))
    ppl_results[m] = {"nll": mn, "ppl": ppl, "tokens": tt}
    print(f"  {m:>14} | {mn:>8.3f} | {ppl:>10.2f} | {tt:>8}")

base_ppl = ppl_results["baseline"]["ppl"]
for m in ["naive4", "nsep+pchan4"]:
    d = ppl_results[m]["ppl"] - base_ppl
    ppl_results[m]["delta"] = d
    print(f"  ΔPPL {m:>10}: {'+' if d>=0 else ''}{d:.3f}")

# ── Final ────────────────────────────────────────────────────
final = {
    "experiment": "poc_gemma4_26b_a4b_main_eval",
    "model": MODEL_ID,
    "timestamp": datetime.now().isoformat(),
    "outlier": {
        "k_max_overall": float(np.max(all_k)),
        "v_max_overall": float(np.max(all_v)),
        "overall_max": float(overall),
        "worst_k_layer": int(np.argmax(all_k)),
        "worst_v_layer": int(np.argmax(all_v)),
        "n_cache_layers": n_layers,
    },
    "niah": niah_results,
    "ppl_chat": ppl_results,
}
print("\n" + "="*60)
print("FINAL JSON")
print("="*60)
print(json.dumps(final, indent=2, ensure_ascii=False))
