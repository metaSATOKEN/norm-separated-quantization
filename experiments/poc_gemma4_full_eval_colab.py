# Copyright 2026 Kentaro Sato
# Licensed under the Apache License, Version 2.0

# ============================================================
# PoC: Gemma 4 E2B Full Evaluation (Colab continuation)
# ============================================================
# Runs after poc_gemma4_outlier_colab.py in the SAME Colab session.
# Reuses mdl, tok already loaded in memory.
#
# Experiments:
#   A (CELL 4): Perplexity on WikiText-2
#               baseline / naive4 / nsep+pchan4
#   B (CELL 5): Multi-Needle-in-Haystack
#               Gemma 4 is instruction-tuned, can do multi-needle QA
#
# Prediction (from outlier_max = 7.48x measurement):
#   naive4 : severe degradation expected (worse than Pythia-6.9B +22 PPL)
#   nsep+pchan4 : should fully recover
# ============================================================


# =============================================================
# === CELL 4 === Perplexity on WikiText-2
# =============================================================

import gc, json, numpy as np, torch
import torch.nn.functional as F
from datetime import datetime

# ── Ensure model + tokenizer are loaded ──────────────────────
# If this cell is run in a fresh Colab session, reload from HF cache.
try:
    _ = mdl  # noqa: F821
    _ = tok  # noqa: F821
    print("Reusing existing mdl + tok from session")
except NameError:
    print("mdl/tok not found -- reloading (cached, should be fast)")
    from transformers import AutoModelForCausalLM, AutoTokenizer
    MODEL_ID = "google/gemma-4-E2B-it"
    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    mdl = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.float16,
        device_map="auto",
        use_safetensors=True,
    )
    mdl.eval()
    print("Reload done")

# Install datasets if missing
try:
    from datasets import load_dataset
except ImportError:
    import subprocess
    subprocess.run(["pip", "install", "-q", "datasets"], check=True)
    from datasets import load_dataset

# ── Quantization primitives ──────────────────────────────────
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
    raise ValueError(name)

# ── KV cache helpers ─────────────────────────────────────────
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

# ── Perplexity measurement ───────────────────────────────────
# Protocol: sliding window with stride, teacher-forced NLL,
# compress cache once per window, measure generation NLL.
# ────────────────────────────────────────────────────────────
def ppl_on_window(mdl, tok, text, method, ctx_len=1024, stride=512,
                  device="cuda", max_windows=10):
    """Fast batched teacher-forced PPL with compressed KV cache."""
    ids = tok.encode(text, return_tensors="pt").to(device)
    T = ids.shape[1]

    total_nll = 0.0
    total_tokens = 0
    windows_done = 0

    with torch.inference_mode():
        for begin in range(0, T - ctx_len, stride):
            if windows_done >= max_windows:
                break

            window = ids[:, begin:begin + ctx_len]
            split = ctx_len // 2
            context = window[:, :split]
            targets = window[:, split:]

            # 1. Build cache on context (one forward)
            out_ctx = mdl(context, use_cache=True)
            past = out_ctx.past_key_values
            first_logits = out_ctx.logits[:, -1:, :]  # predicts targets[0]

            # 2. Compress cache (or not for baseline)
            if method != "baseline":
                compress_cache(past, method)

            # 3. Batched forward on all target tokens (one forward)
            out_tgt = mdl(targets, past_key_values=past, use_cache=False)
            # logits[:, t-1] predicts targets[t] for t=1..len-1
            rest_logits = out_tgt.logits[:, :-1, :]

            all_logits = torch.cat([first_logits, rest_logits], dim=1)
            # shape: [1, len(targets), vocab]

            log_probs = F.log_softmax(all_logits[0].float(), dim=-1)
            nll = -log_probs.gather(-1, targets[0].unsqueeze(-1)).squeeze(-1)
            total_nll += nll.sum().item()
            total_tokens += targets.shape[1]

            windows_done += 1
            del out_ctx, out_tgt, past, first_logits, rest_logits, all_logits, log_probs, nll
            gc.collect(); torch.cuda.empty_cache()

    mean_nll = total_nll / max(total_tokens, 1)
    ppl = float(np.exp(mean_nll))
    return ppl, total_tokens

# ── Load WikiText-2 ──────────────────────────────────────────
print("Loading WikiText-2 test split ...")
ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
wtext = "\n\n".join([x for x in ds["text"] if x.strip()])
print(f"Text length: {len(wtext)} chars")

# Limit to keep runtime reasonable (~10 windows)
n_tok_approx = len(wtext) // 4
if n_tok_approx > 12000:
    wtext = wtext[:48000]  # ~12000 tokens -> ~20 windows at stride 512
print(f"Using first {len(wtext)} chars for PPL eval")

# ── Run three methods ────────────────────────────────────────
print(f"\n{'='*60}")
print(f"Gemma 4 E2B-it -- WikiText-2 PPL comparison")
print(f"{'='*60}")
print(f"  {'Method':>14} | {'PPL':>10} | {'ΔPPL':>10} | {'tokens':>8}")
print(f"  {'-'*52}")

ppl_results = {}
baseline_ppl = None
for method in ["baseline", "naive4", "nsep+pchan4"]:
    print(f"  Running {method} ...", flush=True)
    ppl, nt = ppl_on_window(mdl, tok, wtext, method,
                            ctx_len=1024, stride=512, max_windows=10)
    ppl_results[method] = {"ppl": ppl, "tokens": nt}
    if method == "baseline":
        baseline_ppl = ppl
        dppl_str = "-"
    else:
        dppl = ppl - baseline_ppl
        ppl_results[method]["delta"] = dppl
        dppl_str = f"+{dppl:.2f}"
    print(f"  {method:>14} | {ppl:>10.3f} | {dppl_str:>10} | {nt:>8}")

print(f"\nPPL Results (Gemma 4 E2B-it, ctx=1024, stride=512):")
print(json.dumps(ppl_results, indent=2))

# Quick interpretation
print(f"\n{'─'*60}")
print(f"Interpretation:")
naive_d = ppl_results["naive4"].get("delta", 0)
nsep_d = ppl_results["nsep+pchan4"].get("delta", 0)
if naive_d > 10 and nsep_d < 3:
    print(f"  ✓ Pattern CONFIRMED: naive4 degrades severely (+{naive_d:.1f}),")
    print(f"    nsep+pchan4 recovers (+{nsep_d:.2f}).")
elif naive_d < 2:
    print(f"  ✗ Pattern NOT TRIGGERED: naive4 already safe (+{naive_d:.2f}).")
    print(f"    Outlier 7.48x not enough by itself.")
else:
    print(f"  ~ MIXED: naive4 +{naive_d:.2f}, nsep+pchan4 +{nsep_d:.2f}")


# =============================================================
# === CELL 5 === Multi-Needle-in-Haystack
# =============================================================

# ── Haystack (same as prior experiments for consistency) ─────
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
    for i in range(n_needles):
        greek = ["ALPHA", "BETA", "GAMMA", "DELTA", "EPSILON"][i]
        questions.append(f"What is SECRET FACT {greek}?")
    prompt = haystack + "\n\nAnswer the following questions based on the text above:\n"
    for q in questions:
        prompt += f"Q: {q}\nA: "
    return prompt, [n["key"] for n in used]

def check_needles(response, keys):
    r = response.lower()
    return [k for k in keys if all(p in r for p in k.lower().split())]

def test_retrieval(mdl, tok, prompt, method, max_new=150, device="cuda"):
    ids = tok.encode(prompt, return_tensors="pt").to(device)
    max_pos = getattr(mdl.config, "max_position_embeddings",
              getattr(mdl.config, "n_positions", 2048)) or 2048
    if ids.shape[1] > max_pos - max_new:
        ids = ids[:, -(max_pos - max_new):]

    with torch.inference_mode():
        out = mdl(ids, use_cache=True)
        past = out.past_key_values
        if method != "baseline":
            compress_cache(past, method)

        generated = []
        logits = out.logits[:, -1:]
        for _ in range(max_new):
            nt = logits.argmax(dim=-1)
            generated.append(nt[0, 0].item())
            if nt[0, 0].item() == tok.eos_token_id:
                break
            out = mdl(nt, past_key_values=past, use_cache=True)
            past = out.past_key_values
            logits = out.logits[:, -1:]

    text = tok.decode(generated, skip_special_tokens=True).strip()
    del out, past
    gc.collect(); torch.cuda.empty_cache()
    return text

# ── Run matrix ───────────────────────────────────────────────
CONFIGS = [(10, 3), (20, 3), (30, 5), (50, 5), (80, 5)]
METHODS = ["baseline", "naive4", "nsep+pchan4"]

niah_results = []
print(f"\n{'='*60}")
print(f"Gemma 4 E2B-it -- Multi-Needle-in-Haystack")
print(f"{'='*60}")

for n_filler, n_needles in CONFIGS:
    prompt, keys = build_multi_needle(n_filler, n_needles)
    n_tok = len(tok.encode(prompt))
    print(f"\n  {n_filler} filler / {n_needles} needles, {n_tok} tokens:")
    row = {"n_filler": n_filler, "n_needles": n_needles, "tokens": n_tok, "methods": {}}
    for method in METHODS:
        resp = test_retrieval(mdl, tok, prompt, method)
        found = check_needles(resp, keys)
        row["methods"][method] = {
            "found": len(found), "total": n_needles, "resp_head": resp[:80]
        }
        print(f"    {method:>14}: {len(found)}/{n_needles}  \"{resp[:50]}\"")
    niah_results.append(row)

# ── NIAH summary ─────────────────────────────────────────────
print(f"\n{'='*60}")
print(f"NIAH Total (Gemma 4 E2B-it):")
print(f"{'='*60}")
for m in METHODS:
    f = sum(r["methods"][m]["found"] for r in niah_results)
    t = sum(r["n_needles"] for r in niah_results)
    print(f"    {m:>14}: {f}/{t}")

# ── Final JSON combining A + B ───────────────────────────────
final = {
    "experiment": "poc_gemma4_full_eval",
    "model": "google/gemma-4-E2B-it",
    "timestamp": datetime.now().isoformat(),
    "outlier_summary": {
        "k_max": 5.56, "v_max": 7.48, "overall_max": 7.48,
        "worst_layer": 9, "kv_cache_layers": 15
    },
    "ppl_wikitext2": ppl_results,
    "multi_needle": niah_results,
}

print(f"\n{'='*60}")
print("FINAL JSON (Gemma 4 E2B full eval):")
print(f"{'='*60}")
print(json.dumps(final, indent=2, ensure_ascii=False))
