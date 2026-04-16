# Copyright 2026 Kentaro Sato
# Licensed under the Apache License, Version 2.0

# ============================================================
# Needle-in-Haystack: Pythia-6.9B + Mistral-7B
# ============================================================
# Qwen2-7B: naive4 0/15, nsep+pchan4 15/15
# Now test: does the same pattern hold on other 7B models?
#   Pythia-6.9B: moderate naive4 failure (DPPL +22)
#   Mistral-7B: naive4 works fine (DPPL +0.10)
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
    "The weather in coastal regions is influenced by ocean currents and prevailing winds. Maritime climates tend to have moderate temperatures year-round, with cool summers and mild winters.",
    "Ancient civilizations developed sophisticated agricultural techniques to sustain growing populations. Irrigation systems, crop rotation, and selective breeding were among the key innovations.",
    "The study of mathematics has evolved over millennia, from simple counting systems to abstract algebra and topology. Each era built upon the discoveries of its predecessors.",
    "Music theory encompasses the rules and practices of harmony, melody, and rhythm. Different cultures have developed distinct musical traditions across the world.",
    "Volcanic activity shapes landscapes through both destructive and creative forces. Eruptions can devastate entire regions, but mineral-rich soils often become highly fertile.",
    "The development of printing technology revolutionized the spread of information. From Gutenberg's movable type to modern digital publishing, each advance democratized knowledge.",
    "Marine ecosystems support an incredible diversity of life forms. Coral reefs host thousands of species in complex interdependent relationships.",
    "The principles of aerodynamics govern the design of aircraft and automobiles. Understanding air flow has enabled engineers to create efficient structures.",
    "Traditional medicine practices have contributed to modern pharmacology. Many drugs were originally derived from plants used by indigenous peoples.",
    "The architecture of ancient Rome continues to influence building design. Innovations like the arch and concrete enabled structures of unprecedented scale.",
]

NEEDLE = "The secret code for the experiment is: BLUE ELEPHANT 7742."
EXPECTED = "BLUE ELEPHANT 7742"

def build_haystack(n_para, pos_ratio):
    paras = [FILLER[i % len(FILLER)] for i in range(n_para)]
    idx = int(len(paras) * pos_ratio)
    idx = max(0, min(idx, len(paras)))
    paras.insert(idx, NEEDLE)
    return " ".join(paras) + "\n\nQuestion: What is the secret code for the experiment?\nAnswer:"

def check_found(response):
    r = response.lower()
    if "blue elephant" in r and "7742" in r: return True, "exact"
    if "blue elephant" in r or "7742" in r: return True, "partial"
    return False, "missed"

def test_retrieval(model, tok, prompt, method, max_new=30):
    ids = tok.encode(prompt, return_tensors="pt").to(device)
    max_pos = getattr(model.config, "max_position_embeddings",
              getattr(model.config, "n_positions", 2048))
    if ids.shape[1] > max_pos - max_new:
        ids = ids[:, -(max_pos - max_new):]

    with torch.inference_mode():
        out = model(ids, use_cache=True)
        past = out.past_key_values
        if method != "baseline":
            compress_cache(past, method)
        generated = []
        logits = out.logits[:, -1:]
        for _ in range(max_new):
            nt = logits.argmax(dim=-1)
            generated.append(nt[0, 0].item())
            if nt[0, 0].item() == tok.eos_token_id: break
            out = model(nt, past_key_values=past, use_cache=True)
            past = out.past_key_values
            logits = out.logits[:, -1:]
    text = tok.decode(generated, skip_special_tokens=True).strip()
    del out, past; gc.collect(); torch.cuda.empty_cache()
    return text

# ── Run ────────────────────────────────────────────────────────────────────

MODELS = [
    ("Pythia-6.9B", "EleutherAI/pythia-6.9b", torch.float16),
    ("Mistral-7B", "mistralai/Mistral-7B-v0.1", torch.float16),
]

METHODS = ["baseline", "naive4", "nsep+pchan4"]
SIZES = [5, 10, 20]
POSITIONS = [0.0, 0.25, 0.5, 0.75, 1.0]

all_results = {}

for mname, hf_id, dtype in MODELS:
    print(f"\n{'='*60}")
    print(f"  {mname} -- Needle in Haystack")
    print(f"{'='*60}")

    tok = AutoTokenizer.from_pretrained(hf_id)
    mdl = AutoModelForCausalLM.from_pretrained(
        hf_id, torch_dtype=dtype, device_map="auto", use_safetensors=True
    )
    mdl.eval()
    if tok.pad_token is None: tok.pad_token = tok.eos_token or "<|endoftext|>"

    model_results = []
    for n_para in SIZES:
        for pos in POSITIONS:
            prompt = build_haystack(n_para, pos)
            n_tokens = len(tok.encode(prompt))
            row = {"n_para": n_para, "pos": pos, "tokens": n_tokens, "methods": {}}

            print(f"\n  p={n_para}, pos={pos:.0%}, tok={n_tokens}")
            for method in METHODS:
                resp = test_retrieval(mdl, tok, prompt, method)
                found, match = check_found(resp)
                row["methods"][method] = {"resp": resp[:80], "found": found, "match": match}
                status = "FOUND" if found else "MISS "
                print(f"    {method:>14}: [{status}] \"{resp[:50]}\"")

            model_results.append(row)

    # Summary
    print(f"\n  {mname} Retrieval Rate:")
    for m in METHODS:
        found = sum(1 for r in model_results if r["methods"][m]["found"])
        total = len(model_results)
        print(f"    {m:>14}: {found}/{total} ({found/total:.0%})")

    all_results[mname] = model_results
    del mdl, tok; gc.collect(); torch.cuda.empty_cache()

# ── Grand Summary ──────────────────────────────────────────────────────────

print(f"\n{'='*60}")
print("GRAND SUMMARY: Needle Retrieval Rate")
print(f"{'='*60}")

# Include Qwen2-7B from previous run
print(f"  {'Model':<15} {'baseline':>10} {'naive4':>10} {'nsep+pc':>10}")
print(f"  {'-'*45}")
print(f"  {'Qwen2-7B':<15} {'15/15':>10} {'0/15':>10} {'15/15':>10}  (from previous)")
for mname, results in all_results.items():
    scores = {}
    for m in METHODS:
        scores[m] = sum(1 for r in results if r["methods"][m]["found"])
    total = len(results)
    print(f"  {mname:<15} {scores['baseline']:>3}/{total:>2}{'':<5} {scores['naive4']:>3}/{total:>2}{'':<5} {scores['nsep+pchan4']:>3}/{total:>2}")

print(f"\n{'='*60}")
print("JSON OUTPUT:")
print(f"{'='*60}")
output = {
    "experiment": "poc_needle_pythia_mistral",
    "timestamp": datetime.now().isoformat(),
    "results": {k: [{"n_para": r["n_para"], "pos": r["pos"], "tokens": r["tokens"],
                      "baseline": r["methods"]["baseline"]["found"],
                      "naive4": r["methods"]["naive4"]["found"],
                      "nsep_pchan4": r["methods"]["nsep+pchan4"]["found"]}
                     for r in v] for k, v in all_results.items()},
}
print(json.dumps(output, indent=2, ensure_ascii=False))
