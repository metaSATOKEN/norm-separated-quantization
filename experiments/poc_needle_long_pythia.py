# Copyright 2026 Kentaro Sato
# Licensed under the Apache License, Version 2.0

# ============================================================
# Needle-in-Haystack: Pythia-6.9B Long Context (up to 2048t)
# ============================================================
# Short haystack (587t): naive4 15/15 -- too easy
# Question: does naive4 start losing needles at longer contexts
# where DPPL error accumulates?
#
# Haystack sizes: 5/10/20/30/40/50 paragraphs
#   ~185 / ~310 / ~587 / ~870 / ~1150 / ~1430 tokens
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

# ── Extended filler (more unique paragraphs to avoid repetition) ───────────

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
    "Quantum mechanics describes the behavior of matter at atomic and subatomic scales. The wave-particle duality of light challenged classical physics fundamentally.",
    "The human brain contains approximately 86 billion neurons connected by trillions of synapses. Understanding neural networks has inspired artificial intelligence research.",
    "Plate tectonics explains the movement of Earth's lithospheric plates. Continental drift was first proposed by Alfred Wegener in 1912 but took decades to gain acceptance.",
    "The invention of the steam engine powered the Industrial Revolution. Factories transformed economies from agricultural to manufacturing-based systems across Europe.",
    "Photosynthesis converts sunlight into chemical energy, sustaining nearly all life on Earth. Plants, algae, and cyanobacteria are the primary photosynthetic organisms.",
    "The periodic table organizes chemical elements by atomic number and electron configuration. Mendeleev's original table predicted elements that were later discovered.",
    "Shakespeare's works include 37 plays, 154 sonnets, and several longer poems. His influence on the English language includes the invention of over 1700 words.",
    "The theory of general relativity describes gravity as the curvature of spacetime. Einstein's field equations predict phenomena such as black holes and gravitational waves.",
    "DNA carries the genetic instructions for the development and function of all known organisms. The double helix structure was determined by Watson and Crick in 1953.",
    "The Renaissance marked a period of cultural and intellectual rebirth in Europe. Advances in art, science, and philosophy transformed Western civilization profoundly.",
    "Glaciers cover about 10 percent of Earth's land surface and contain roughly 69 percent of the world's fresh water supply stored in ice.",
    "The Silk Road connected civilizations across Asia, Europe, and Africa for centuries. Trade along these routes spread goods, ideas, religions, and technologies.",
    "Cryptography has evolved from simple substitution ciphers to complex mathematical algorithms. Modern encryption secures financial transactions and communications worldwide.",
    "The Amazon rainforest produces approximately 20 percent of the world's oxygen. Deforestation threatens biodiversity and contributes to climate change globally.",
    "Nuclear fusion powers the sun, converting hydrogen into helium at temperatures exceeding 15 million degrees Celsius in its core.",
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
        if method != "baseline": compress_cache(past, method)
        generated = []
        logits = out.logits[:, -1:]
        for _ in range(max_new):
            nt = logits.argmax(dim=-1)
            generated.append(nt[0, 0].item())
            if nt[0, 0].item() == tok.eos_token_id: break
            out = model(nt, past_key_values=past, use_cache=True)
            past = out.past_key_values; logits = out.logits[:, -1:]
    text = tok.decode(generated, skip_special_tokens=True).strip()
    del out, past; gc.collect(); torch.cuda.empty_cache()
    return text

# ── Run ────────────────────────────────────────────────────────────────────

print("Loading Pythia-6.9B...")
tok = AutoTokenizer.from_pretrained("EleutherAI/pythia-6.9b")
mdl = AutoModelForCausalLM.from_pretrained(
    "EleutherAI/pythia-6.9b", torch_dtype=torch.float16, device_map="auto", use_safetensors=True
)
mdl.eval()
if tok.pad_token is None: tok.pad_token = tok.eos_token

METHODS = ["baseline", "naive4", "nsep+pchan4"]
SIZES = [5, 10, 20, 30, 40, 50]
POSITIONS = [0.0, 0.25, 0.5, 0.75, 1.0]

results = []

for n_para in SIZES:
    # Check token count first
    sample_prompt = build_haystack(n_para, 0.5)
    n_tokens = len(tok.encode(sample_prompt))
    if n_tokens > 2000:
        print(f"\n  Skipping p={n_para} (tokens={n_tokens} > 2000, near max_pos limit)")
        continue

    for pos in POSITIONS:
        prompt = build_haystack(n_para, pos)
        n_tok = len(tok.encode(prompt))
        row = {"n_para": n_para, "pos": pos, "tokens": n_tok, "methods": {}}

        print(f"\n  p={n_para}, pos={pos:.0%}, tok={n_tok}")
        for method in METHODS:
            resp = test_retrieval(mdl, tok, prompt, method)
            found, match = check_found(resp)
            row["methods"][method] = {"resp": resp[:80], "found": found, "match": match}
            status = "FOUND" if found else "MISS "
            print(f"    {method:>14}: [{status}] \"{resp[:60]}\"")

        results.append(row)

del mdl, tok; gc.collect(); torch.cuda.empty_cache()

# ── Summary by haystack size ───────────────────────────────────────────────

print(f"\n{'='*60}")
print("SUMMARY: Pythia-6.9B Needle Retrieval by Context Length")
print(f"{'='*60}")

print(f"\n  {'Size':>6} {'Tokens':>7} {'baseline':>10} {'naive4':>10} {'nsep+pc':>10}")
print(f"  {'-'*50}")

sizes_seen = sorted(set(r["n_para"] for r in results))
for sz in sizes_seen:
    sz_results = [r for r in results if r["n_para"] == sz]
    tok_avg = int(np.mean([r["tokens"] for r in sz_results]))
    for m in METHODS:
        pass
    b_found = sum(1 for r in sz_results if r["methods"]["baseline"]["found"])
    n_found = sum(1 for r in sz_results if r["methods"]["naive4"]["found"])
    ns_found = sum(1 for r in sz_results if r["methods"]["nsep+pchan4"]["found"])
    total = len(sz_results)
    print(f"  p={sz:<4} ~{tok_avg:>5}t {b_found}/{total:>2}{'':<6} {n_found}/{total:>2}{'':<6} {ns_found}/{total:>2}")

# Overall
print(f"\n  Overall:")
for m in METHODS:
    found = sum(1 for r in results if r["methods"][m]["found"])
    total = len(results)
    print(f"    {m:>14}: {found}/{total} ({found/total:.0%})")

print(f"\n{'='*60}")
print("JSON OUTPUT:")
print(f"{'='*60}")
output = {
    "experiment": "poc_needle_long_pythia",
    "timestamp": datetime.now().isoformat(),
    "model": "Pythia-6.9B",
    "results": [{"n_para": r["n_para"], "pos": r["pos"], "tokens": r["tokens"],
                 "baseline": r["methods"]["baseline"]["found"],
                 "naive4": r["methods"]["naive4"]["found"],
                 "nsep_pchan4": r["methods"]["nsep+pchan4"]["found"]}
                for r in results],
}
print(json.dumps(output, indent=2, ensure_ascii=False))
