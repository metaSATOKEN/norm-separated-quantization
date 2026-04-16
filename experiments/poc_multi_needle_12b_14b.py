# Copyright 2026 Kentaro Sato
# Licensed under the Apache License, Version 2.0

# ============================================================
# Multi-Needle-in-Haystack: Pythia-12B + Qwen2.5-14B
# ============================================================
# Completing the cross-model matrix:
#   Qwen2-7B (8.6x outlier): 0/26 naive4, 26/26 nsep -- DONE
#   Pythia-6.9B (4.6x):      0/21 baseline -- DONE (model limitation)
#   Mistral-7B (3.1x):       7/26 baseline -- DONE (inconsistent)
#   Pythia-12B (4.6x):       ??? -- base model, likely same as 6.9B
#   Qwen2.5-14B (3.5x):      ??? -- KEY: strong model + low outlier
# ============================================================

# === CELL 1 ===
!pip install -q transformers accelerate hf_transfer
import os; os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"

import gc, json, numpy as np, torch, torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from datetime import datetime

device = "cuda"
print(f"GPU: {torch.cuda.get_device_name()}")

# === CELL 2 ===

def qa_perrow(x, b):
    x = x.float(); qm = 2**(b-1)-1
    s = x.abs().amax(dim=-1, keepdim=True).clamp(min=1e-12) / qm
    return ((x/s).round().clamp(-qm, qm)) * s

def qa_perchan(x, b):
    x = x.float(); qm = 2**(b-1)-1
    s = x.abs().amax(dim=0, keepdim=True).clamp(min=1e-12) / qm
    return ((x/s).round().clamp(-qm, qm)) * s

def apply_method(x, name):
    if name == "naive4": return qa_perrow(x, 4)
    if name == "nsep+pchan4":
        x = x.float()
        n = x.norm(dim=-1, keepdim=True).clamp(min=1e-12)
        d = x / n
        dq = qa_perchan(d, 4)
        dq = dq / dq.norm(dim=-1, keepdim=True).clamp(min=1e-8)
        return n * dq

def get_kv(past, li):
    if hasattr(past, 'layers'): return past.layers[li].keys, past.layers[li].values
    return past[li][0], past[li][1]

def set_kv(past, li, k, v):
    if hasattr(past, 'layers'):
        past.layers[li].keys = k; past.layers[li].values = v

def n_cache_layers(past):
    return len(past.layers) if hasattr(past, 'layers') else len(past)

def compress_cache(past, mn):
    for li in range(n_cache_layers(past)):
        ok, ov = get_kv(past, li)
        ok, ov = ok.clone(), ov.clone()
        nk, nv = torch.zeros_like(ok), torch.zeros_like(ov)
        for h in range(ok.shape[1]):
            nk[0,h] = apply_method(ok[0,h], mn).to(ok.dtype)
            nv[0,h] = apply_method(ov[0,h], mn).to(ov.dtype)
        set_kv(past, li, nk, nv)

# ── Haystack ───────────────────────────────────────────────────────────────

FILLER = [
    "The weather in coastal regions is influenced by ocean currents and prevailing winds. Maritime climates tend to have moderate temperatures year-round.",
    "Ancient civilizations developed sophisticated agricultural techniques. Irrigation systems and crop rotation were among the key innovations.",
    "Mathematics has evolved from simple counting to abstract algebra. Each era built upon prior discoveries.",
    "Music theory encompasses harmony, melody, and rhythm. Different cultures developed distinct traditions.",
    "Volcanic activity shapes landscapes through destructive and creative forces. Mineral-rich soils become fertile land.",
    "Printing technology revolutionized information spread. From Gutenberg to digital publishing, knowledge was democratized.",
    "Marine ecosystems support incredible biodiversity. Coral reefs host thousands of interdependent species.",
    "Aerodynamics governs aircraft and automobile design. Air flow understanding enables efficient structures.",
    "Traditional medicine contributed to modern pharmacology. Many drugs derive from indigenous plant knowledge.",
    "Roman architecture influences modern building design. The arch and concrete enabled unprecedented structures.",
    "Quantum mechanics describes matter at atomic scales. Wave-particle duality challenged classical physics.",
    "The human brain contains 86 billion neurons. Neural network understanding inspired AI research.",
    "Plate tectonics explains lithospheric plate movement. Continental drift took decades to gain acceptance.",
    "The steam engine powered the Industrial Revolution. Manufacturing transformed European economies.",
    "Photosynthesis converts sunlight to chemical energy. Plants and algae sustain nearly all life on Earth.",
    "The periodic table organizes elements by atomic number. Mendeleev predicted undiscovered elements.",
    "Shakespeare wrote 37 plays and 154 sonnets. He invented over 1700 English words.",
    "General relativity describes gravity as spacetime curvature. Einstein predicted black holes and gravitational waves.",
    "DNA carries genetic instructions for all organisms. Watson and Crick determined the double helix in 1953.",
    "The Renaissance marked cultural rebirth in Europe. Art, science, and philosophy were transformed.",
    "Glaciers cover 10 percent of Earth's land. They contain 69 percent of fresh water as ice.",
    "The Silk Road connected civilizations for centuries. Trade spread goods, ideas, and technologies.",
    "Cryptography evolved from simple ciphers to complex algorithms. Modern encryption secures global transactions.",
    "The Amazon rainforest produces 20 percent of world oxygen. Deforestation threatens biodiversity.",
    "Nuclear fusion powers the sun at 15 million degrees Celsius. Hydrogen converts to helium in the core.",
]

NEEDLES = [
    {"text": "SECRET FACT ALPHA: The laboratory password is CRIMSON TIGER 9981.", "key": "CRIMSON TIGER 9981"},
    {"text": "SECRET FACT BETA: The project codename is SILVER DOLPHIN 2247.", "key": "SILVER DOLPHIN 2247"},
    {"text": "SECRET FACT GAMMA: The vault combination is GOLDEN EAGLE 5563.", "key": "GOLDEN EAGLE 5563"},
    {"text": "SECRET FACT DELTA: The access phrase is PURPLE FALCON 3318.", "key": "PURPLE FALCON 3318"},
    {"text": "SECRET FACT EPSILON: The encryption key is AZURE PANTHER 7704.", "key": "AZURE PANTHER 7704"},
]

def build_multi_needle(n_filler, n_needles):
    paras = [FILLER[i % len(FILLER)] for i in range(n_filler)]
    used = NEEDLES[:n_needles]
    positions = [int(len(paras) * (i+1) / (n_needles+1)) for i in range(n_needles)]
    for offset, (pos, needle) in enumerate(zip(positions, used)):
        paras.insert(pos + offset, needle["text"])
    haystack = " ".join(paras)
    questions = []
    for i, needle in enumerate(used):
        greek = ["ALPHA","BETA","GAMMA","DELTA","EPSILON"][i]
        questions.append(f"What is SECRET FACT {greek}?")
    prompt = haystack + "\n\nAnswer the following questions based on the text above:\n"
    for q in questions:
        prompt += f"Q: {q}\nA: "
    return prompt, [n["key"] for n in used]

def check_needles(response, keys):
    r = response.lower()
    return [k for k in keys if all(p in r for p in k.lower().split())]

def test_retrieval(model, tok, prompt, method, max_new=150):
    ids = tok.encode(prompt, return_tensors="pt").to(device)
    max_pos = getattr(model.config, "max_position_embeddings",
              getattr(model.config, "n_positions", 2048))
    if ids.shape[1] > max_pos - max_new:
        ids = ids[:, -(max_pos - max_new):]
    with torch.inference_mode():
        out = model(ids, use_cache=True)
        past = out.past_key_values
        if method != "baseline": compress_cache(past, method)
        generated = []
        logits = out.logits[:, -1:]
        for _ in range(max_new):
            nt = logits.argmax(dim=-1)
            generated.append(nt[0,0].item())
            if nt[0,0].item() == tok.eos_token_id: break
            out = model(nt, past_key_values=past, use_cache=True)
            past = out.past_key_values; logits = out.logits[:, -1:]
    text = tok.decode(generated, skip_special_tokens=True).strip()
    del out, past; gc.collect(); torch.cuda.empty_cache()
    return text

# ── Run ────────────────────────────────────────────────────────────────────

MODELS = [
    ("Pythia-12B",   "EleutherAI/pythia-12b", torch.float16),
    ("Qwen2.5-14B", "Qwen/Qwen2.5-14B",      torch.float16),
]

METHODS = ["baseline", "naive4", "nsep+pchan4"]
CONFIGS = [(10, 3), (20, 3), (30, 5), (50, 5), (80, 5), (120, 5)]

all_results = {}

for mname, hf_id, dtype in MODELS:
    print(f"\n{'='*60}")
    print(f"  {mname} -- Multi-Needle")
    print(f"{'='*60}")

    tok = AutoTokenizer.from_pretrained(hf_id)
    mdl = AutoModelForCausalLM.from_pretrained(
        hf_id, torch_dtype=dtype, device_map="auto", use_safetensors=True
    )
    mdl.eval()
    if tok.pad_token is None: tok.pad_token = tok.eos_token or "<|endoftext|>"
    max_pos = getattr(mdl.config, "max_position_embeddings",
              getattr(mdl.config, "n_positions", 2048))

    model_results = []
    for n_filler, n_needles in CONFIGS:
        prompt, keys = build_multi_needle(n_filler, n_needles)
        n_tok = len(tok.encode(prompt))
        if n_tok > max_pos - 150:
            print(f"\n  Skip: {n_filler}p/{n_needles}n ({n_tok}t > max {max_pos})")
            continue

        row = {"n_filler": n_filler, "n_needles": n_needles, "tokens": n_tok, "methods": {}}
        print(f"\n  {n_filler}p/{n_needles}n, {n_tok}t:")
        for method in METHODS:
            resp = test_retrieval(mdl, tok, prompt, method)
            found = check_needles(resp, keys)
            row["methods"][method] = {"found": len(found), "total": n_needles, "resp": resp[:80]}
            print(f"    {method:>14}: {len(found)}/{n_needles}  \"{resp[:50]}\"")
        model_results.append(row)

    # Summary
    print(f"\n  {mname} Total:")
    for m in METHODS:
        f = sum(r["methods"][m]["found"] for r in model_results)
        t = sum(r["n_needles"] for r in model_results)
        print(f"    {m:>14}: {f}/{t}")

    all_results[mname] = model_results
    del mdl, tok; gc.collect(); torch.cuda.empty_cache()

# ── Grand Summary (all 5 models) ──
print(f"\n{'='*60}")
print("COMPLETE MULTI-NEEDLE MATRIX (5 models)")
print(f"{'='*60}")
print(f"  {'Model':<15} {'outlier':>8} {'baseline':>10} {'naive4':>10} {'nsep+pc':>10}")
print(f"  {'-'*55}")

prior = [
    ("Qwen2-7B",    8.6, "26/26", "0/26",  "26/26"),
    ("Pythia-6.9B", 4.6, "0/21",  "0/21",  "0/21"),
    ("Mistral-7B",  3.1, "7/26",  "8/26",  "5/26"),
]
for name, ratio, b, n, ns in prior:
    print(f"  {name:<15} {ratio:>7.1f}x {b:>10} {n:>10} {ns:>10}")

for mname, results in all_results.items():
    f_b = sum(r["methods"]["baseline"]["found"] for r in results)
    f_n = sum(r["methods"]["naive4"]["found"] for r in results)
    f_ns = sum(r["methods"]["nsep+pchan4"]["found"] for r in results)
    t = sum(r["n_needles"] for r in results)
    ratio = "4.6" if "Pythia" in mname else "3.5"
    print(f"  {mname:<15} {ratio:>7}x {f_b}/{t:>2}{'':<6} {f_n}/{t:>2}{'':<6} {f_ns}/{t:>2}")

print(f"\n{'='*60}")
print("JSON OUTPUT:")
print(f"{'='*60}")
output = {"experiment": "poc_multi_needle_12b_14b", "timestamp": datetime.now().isoformat(), "results": all_results}
print(json.dumps(output, indent=2, ensure_ascii=False))
