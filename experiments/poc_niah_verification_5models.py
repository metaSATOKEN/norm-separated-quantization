# Copyright 2026 Kentaro Sato
# Licensed under the Apache License, Version 2.0

# ============================================================
# NIAH Verification for 5 Predicted-Safe Models
# ============================================================
# Verify our "safe (predicted)" classifications with actual
# multi-needle retrieval.
#
# Target models (all predicted safe from K/V asymmetry study):
#   - Llama-3.1-8B-Instruct  (Meta, GQA 4:1)
#   - Llama-3.2-3B-Instruct  (Meta, GQA 3:1)
#   - Phi-3-mini-4k-instruct (Microsoft, MHA)
#   - Phi-3-medium-4k-instruct (Microsoft, GQA 4:1)
#   - DeepSeek-LLM-7B-Chat   (DeepSeek, MHA, closest to Qwen2-7B)
#
# Goal: convert "TBD" to measured NIAH scores in paper Appendix H.
#
# Expected result: all 21/21 across baseline / naive4 / nsep+pchan4.
# DeepSeek (L0 K=7.45, elevated) is the most interesting to watch.
#
# Runtime: ~60 min total (5 models, each ~10-15 min)
# ============================================================


# =============================================================
# === CELL N1 === Setup + install + login + common helpers
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
# >>>>>  PASTE YOUR HF TOKEN  <<<<<
HF_TOKEN = "hf_XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX"
# ─────────────────────────────────────────────────────────────
from huggingface_hub import login
login(token=HF_TOKEN)

import json, numpy as np
import torch.nn.functional as F
from datetime import datetime
from transformers import AutoModelForCausalLM, AutoTokenizer

# ── Quantization primitives ─────────────────────────────────
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

# ── KV cache helpers ────────────────────────────────────────
def get_kv(past, li):
    if hasattr(past, 'layers'):
        return past.layers[li].keys, past.layers[li].values
    return past[li][0], past[li][1]

def set_kv(past, li, k, v):
    if hasattr(past, 'layers'):
        past.layers[li].keys = k; past.layers[li].values = v
    else:
        past[li] = (k, v)

def n_cache_layers(past):
    return len(past.layers) if hasattr(past, 'layers') else len(past)

def compress_cache(past, method):
    for li in range(n_cache_layers(past)):
        ok, ov = get_kv(past, li)
        ok = ok.clone(); ov = ov.clone()
        nk = torch.zeros_like(ok); nv = torch.zeros_like(ov)
        for h in range(ok.shape[1]):
            nk[0, h] = apply_method(ok[0, h], method).to(ok.dtype)
            nv[0, h] = apply_method(ov[0, h], method).to(ov.dtype)
        set_kv(past, li, nk, nv)

# ── Robust chat encode ──────────────────────────────────────
def chat_encode(tok, messages, add_gen=True):
    r = tok.apply_chat_template(
        messages, add_generation_prompt=add_gen, return_tensors="pt"
    )
    if hasattr(r, "input_ids"): r = r.input_ids
    elif isinstance(r, dict):   r = r["input_ids"]
    return r.to("cuda")

# ── NIAH content ────────────────────────────────────────────
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

# ── NIAH driver ─────────────────────────────────────────────
CONFIGS = [(10, 3), (20, 3), (30, 5), (50, 5), (80, 5)]
METHODS = ["baseline", "naive4", "nsep+pchan4"]

def run_niah_suite(mdl, tok, label):
    print(f"\n{'='*60}")
    print(f"NIAH on {label}")
    print(f"{'='*60}")

    def test_once(prompt_ids, method, max_new=200):
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

    results = []
    for nf, nn in CONFIGS:
        msgs, keys = build_msg(nf, nn)
        pids = chat_encode(tok, msgs)
        ntok = pids.shape[-1]
        row = {"n_filler": nf, "n_needles": nn, "tokens": ntok, "methods": {}}
        print(f"\n  {nf}p / {nn}n, {ntok} tokens:")
        for m in METHODS:
            resp = test_once(pids, m)
            found = check(resp, keys)
            row["methods"][m] = {"found": len(found), "total": nn, "resp": resp[:100]}
            print(f"    {m:>14}: {len(found)}/{nn}  \"{resp[:60]}\"")
        results.append(row)

    print(f"\n{'─'*60}\n{label} NIAH total:")
    totals = {}
    for m in METHODS:
        f = sum(r["methods"][m]["found"] for r in results)
        t = sum(r["n_needles"] for r in results)
        totals[m] = f"{f}/{t}"
        print(f"  {m:>14}: {f}/{t}")
    return {"label": label, "configs": results, "totals": totals}

def load_and_niah(hf_id, label):
    print(f"\n{'#'*60}")
    print(f"# Loading {label} ({hf_id})")
    print(f"{'#'*60}")
    tok = AutoTokenizer.from_pretrained(hf_id)
    mdl = AutoModelForCausalLM.from_pretrained(
        hf_id, torch_dtype=torch.float16, device_map="auto",
        use_safetensors=True,
    )
    mdl.eval()
    result = run_niah_suite(mdl, tok, label)
    del mdl, tok
    gc.collect(); torch.cuda.empty_cache()
    return result

print("Setup complete. VRAM free:",
      f"{torch.cuda.mem_get_info()[0]/1e9:.1f} GB")


# =============================================================
# === CELL N2 === Llama-3.1-8B-Instruct NIAH
# =============================================================

llama_8b_niah = load_and_niah(
    "meta-llama/Llama-3.1-8B-Instruct",
    "Llama-3.1-8B-Instruct"
)


# =============================================================
# === CELL N3 === Llama-3.2-3B-Instruct NIAH
# =============================================================

llama_3b_niah = load_and_niah(
    "meta-llama/Llama-3.2-3B-Instruct",
    "Llama-3.2-3B-Instruct"
)


# =============================================================
# === CELL N4 === Phi-3-mini-4k-instruct NIAH
# =============================================================

phi3_mini_niah = load_and_niah(
    "microsoft/Phi-3-mini-4k-instruct",
    "Phi-3-mini-4k-instruct"
)


# =============================================================
# === CELL N5 === Phi-3-medium-4k-instruct NIAH
# =============================================================

phi3_medium_niah = load_and_niah(
    "microsoft/Phi-3-medium-4k-instruct",
    "Phi-3-medium-4k-instruct"
)


# =============================================================
# === CELL N6 === DeepSeek-LLM-7B-Chat NIAH
# =============================================================

deepseek_niah = load_and_niah(
    "deepseek-ai/deepseek-llm-7b-chat",
    "DeepSeek-LLM-7B-Chat"
)


# =============================================================
# === CELL N7 === Final 11-model measured summary
# =============================================================

verified_results = {
    "Llama-3.1-8B-Instruct":    llama_8b_niah["totals"],
    "Llama-3.2-3B-Instruct":    llama_3b_niah["totals"],
    "Phi-3-mini-4k-instruct":   phi3_mini_niah["totals"],
    "Phi-3-medium-4k-instruct": phi3_medium_niah["totals"],
    "DeepSeek-LLM-7B-Chat":     deepseek_niah["totals"],
}

print("\n" + "="*75)
print("FULL 11-MODEL NIAH VERIFIED SUMMARY")
print("="*75)
print(f"  {'Model':<30} | {'baseline':>10} | {'naive4':>10} | {'nsep+pchan4':>12}")
print(f"  {'-'*30} + {'-'*10} + {'-'*10} + {'-'*12}")

# Prior (already measured)
prior_niah = [
    ("Qwen2-7B",        "15/15",  "0/15",  "15/15"),   # single-needle measured
    ("Qwen2-7B (multi)","26/26",  "0/26",  "26/26"),   # multi-needle measured
    ("Qwen2.5-14B",     "26/26",  "26/26", "26/26"),
    ("Mistral-7B",      "15/15",  "15/15", "15/15"),
    ("Gemma 4 E2B-it",  "16/16",  "16/16", "16/16"),
    ("Gemma 4 26B-A4B", "21/21",  "21/21", "21/21"),
    ("Gemma 4 31B-it",  "21/21",  "21/21", "21/21"),
]
for m, b, n4, nsep in prior_niah:
    print(f"  {m:<30} | {b:>10} | {n4:>10} | {nsep:>12}")
for m, tot in verified_results.items():
    print(f"  {m:<30} | {tot['baseline']:>10} | {tot['naive4']:>10} | {tot['nsep+pchan4']:>12}")

print(f"\n{'='*75}")
print("FINAL JSON")
print(f"{'='*75}")
final = {
    "experiment": "poc_niah_verification_5models",
    "timestamp": datetime.now().isoformat(),
    "newly_measured": {
        "llama_3_1_8b":    llama_8b_niah,
        "llama_3_2_3b":    llama_3b_niah,
        "phi_3_mini":      phi3_mini_niah,
        "phi_3_medium":    phi3_medium_niah,
        "deepseek_7b":     deepseek_niah,
    },
    "all_measured_summary": {
        # Prior 6-model NIAH
        "Qwen2-7B": {"baseline": "15/15", "naive4": "0/15",  "nsep+pchan4": "15/15"},
        "Qwen2.5-14B": {"baseline": "26/26", "naive4": "26/26", "nsep+pchan4": "26/26"},
        "Mistral-7B": {"baseline": "15/15", "naive4": "15/15", "nsep+pchan4": "15/15"},
        "Gemma 4 E2B-it": {"baseline": "16/16", "naive4": "16/16", "nsep+pchan4": "16/16"},
        "Gemma 4 26B-A4B": {"baseline": "21/21", "naive4": "21/21", "nsep+pchan4": "21/21"},
        "Gemma 4 31B-it": {"baseline": "21/21", "naive4": "21/21", "nsep+pchan4": "21/21"},
        # Newly verified
        **{k: v for k, v in verified_results.items()},
    },
}
print(json.dumps(final, indent=2, ensure_ascii=False))
