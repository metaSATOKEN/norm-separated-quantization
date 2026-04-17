# Copyright 2026 Kentaro Sato
# Licensed under the Apache License, Version 2.0

# ============================================================
# Gemma 4 E2B Main Evaluation (after v3 diagnostic PASSED)
# ============================================================
# v3 confirmed:
#   - tok.apply_chat_template works (proc variant is buggy)
#   - 2B can do single-needle retrieval with chat template
#   - PPL requires chat-formatted input (not raw text)
#
# This script:
#   E1. Multi-needle NIAH on 2B (baseline / naive4 / nsep+pchan4)
#   E2. PPL on chat-formatted exchanges
#
# Architecture for the paper:
#   text_config.num_hidden_layers  = 35
#   actual kv_cache layers         = 15   <- 20 layers share KV
#   num_key_value_heads            = 1    (MQA)
#   head_dim                       = 256
#   vocab_size                     = 262144
#   outlier V_max (layer 9)        = 7.48x
# ============================================================

import gc, json, numpy as np, torch
import torch.nn.functional as F
from datetime import datetime

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

# ── Robust chat encode (use tokenizer, skip broken processor) ─
def chat_encode(messages, add_gen=True):
    r = tok.apply_chat_template(
        messages, add_generation_prompt=add_gen, return_tensors="pt"
    )
    if hasattr(r, "input_ids"):
        r = r.input_ids
    elif isinstance(r, dict):
        r = r["input_ids"]
    return r.to("cuda")


# =============================================================
# === E1 === Multi-Needle NIAH with chat template
# =============================================================

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

def build_needle_message(n_filler, n_needles):
    paras = [FILLER[i % len(FILLER)] for i in range(n_filler)]
    used = NEEDLES[:n_needles]
    positions = [int(len(paras) * (i+1) / (n_needles+1)) for i in range(n_needles)]
    for offset, (pos, needle) in enumerate(zip(positions, used)):
        paras.insert(pos + offset, needle["text"])
    haystack = " ".join(paras)

    questions = []
    for i in range(n_needles):
        greek = ["ALPHA","BETA","GAMMA","DELTA","EPSILON"][i]
        questions.append(f"What is SECRET FACT {greek}?")

    content = (f"Read the following text carefully, then answer every question.\n\n"
               f"TEXT:\n{haystack}\n\nQUESTIONS:\n" +
               "\n".join(questions) +
               "\n\nProvide each answer on a separate line.")
    return [{"role": "user", "content": content}], [n["key"] for n in used]

def check_found(response, keys):
    r = response.lower()
    return [k for k in keys if all(p in r for p in k.lower().split())]

def test_niah(prompt_ids, method, max_new=200):
    ids = prompt_ids
    max_pos = 8192  # Gemma 4 supports 128K but keep eval window manageable
    if ids.shape[1] > max_pos - max_new:
        ids = ids[:, -(max_pos - max_new):]
    with torch.inference_mode():
        out = mdl(input_ids=ids, use_cache=True)
        past = out.past_key_values
        if method != "baseline":
            compress_cache(past, method)
        generated = []
        logits = out.logits[:, -1:]
        for _ in range(max_new):
            nt = logits.argmax(dim=-1)
            tid = nt[0, 0].item()
            generated.append(tid)
            if tid == tok.eos_token_id:
                break
            out = mdl(input_ids=nt, past_key_values=past, use_cache=True)
            past = out.past_key_values
            logits = out.logits[:, -1:]
    txt = tok.decode(generated, skip_special_tokens=True).strip()
    del out, past; gc.collect(); torch.cuda.empty_cache()
    return txt

CONFIGS = [(10, 3), (20, 3), (30, 5), (50, 5)]
METHODS = ["baseline", "naive4", "nsep+pchan4"]

niah_results = []
print("="*60)
print("E1: Multi-Needle NIAH (Gemma 4 E2B-it, chat template)")
print("="*60)

for n_filler, n_needles in CONFIGS:
    msgs, keys = build_needle_message(n_filler, n_needles)
    prompt_ids = chat_encode(msgs, add_gen=True)
    n_tok = prompt_ids.shape[-1]
    row = {"n_filler": n_filler, "n_needles": n_needles, "tokens": n_tok, "methods": {}}
    print(f"\n  {n_filler}p / {n_needles}n, {n_tok} tokens:")
    for method in METHODS:
        resp = test_niah(prompt_ids, method)
        found = check_found(resp, keys)
        row["methods"][method] = {"found": len(found), "total": n_needles,
                                  "resp_head": resp[:120]}
        print(f"    {method:>14}: {len(found)}/{n_needles}  \"{resp[:80]}\"")
    niah_results.append(row)

print(f"\n{'─'*60}\nNIAH total:")
for m in METHODS:
    f = sum(r["methods"][m]["found"] for r in niah_results)
    t = sum(r["n_needles"] for r in niah_results)
    print(f"  {m:>14}: {f}/{t}")


# =============================================================
# === E2 === PPL on chat-formatted exchanges
# =============================================================

CHATS = [
    [
        {"role": "user", "content": "Explain photosynthesis in two sentences."},
        {"role": "assistant", "content":
         "Photosynthesis is the process by which plants, algae, and some bacteria convert "
         "light energy into chemical energy stored in glucose. They use carbon dioxide and "
         "water as inputs, releasing oxygen as a byproduct that supports most life on Earth."}
    ],
    [
        {"role": "user", "content": "What is the significance of the Turing Test?"},
        {"role": "assistant", "content":
         "The Turing Test, proposed by Alan Turing in 1950, evaluates whether a machine can "
         "exhibit intelligent behavior indistinguishable from a human during a text-based "
         "conversation. It shaped decades of AI philosophy and remains a cultural benchmark."}
    ],
    [
        {"role": "user", "content": "Describe plate tectonics briefly."},
        {"role": "assistant", "content":
         "Plate tectonics is the theory that Earth's outer shell is divided into large plates "
         "that slowly move over the underlying mantle. Their interactions cause earthquakes, "
         "volcanic activity, mountain building, and the gradual reshaping of continents."}
    ],
]

def ppl_on_chat(chat, method):
    full_ids = chat_encode(chat, add_gen=False)
    # Find boundary between user prompt and assistant reply for cache split
    user_only = chat_encode(chat[:1], add_gen=True)
    split = user_only.shape[-1]
    if split >= full_ids.shape[-1] - 2:
        split = full_ids.shape[-1] // 2

    context = full_ids[:, :split]
    targets = full_ids[:, split:]
    if targets.shape[-1] < 2:
        return float("nan"), 0

    with torch.inference_mode():
        out_c = mdl(input_ids=context, use_cache=True)
        past = out_c.past_key_values
        first_logits = out_c.logits[:, -1:, :]

        if method != "baseline":
            compress_cache(past, method)

        out_t = mdl(input_ids=targets, past_key_values=past, use_cache=False)
        rest_logits = out_t.logits[:, :-1, :]
        all_logits = torch.cat([first_logits, rest_logits], dim=1)

        log_probs = F.log_softmax(all_logits[0].float(), dim=-1)
        nll = -log_probs.gather(-1, targets[0].unsqueeze(-1)).squeeze(-1)
    loss = nll.mean().item()
    nt = targets.shape[-1]
    del out_c, out_t, past
    gc.collect(); torch.cuda.empty_cache()
    return loss, nt

print("\n" + "="*60)
print("E2: PPL on chat-formatted exchanges")
print("="*60)
print(f"  {'Method':>14} | {'mean NLL':>10} | {'PPL':>10} | {'tokens':>8}")
print(f"  {'-'*52}")

ppl_results = {}
for method in METHODS:
    total_loss = 0.0
    total_nll_weighted = 0.0
    total_tokens = 0
    for chat in CHATS:
        loss, nt = ppl_on_chat(chat, method)
        if nt > 0 and not np.isnan(loss):
            total_nll_weighted += loss * nt
            total_tokens += nt
    mean_nll = total_nll_weighted / max(total_tokens, 1)
    ppl = float(np.exp(mean_nll))
    ppl_results[method] = {"nll": mean_nll, "ppl": ppl, "tokens": total_tokens}
    print(f"  {method:>14} | {mean_nll:>10.3f} | {ppl:>10.2f} | {total_tokens:>8}")

baseline_ppl = ppl_results["baseline"]["ppl"]
for m in ["naive4", "nsep+pchan4"]:
    d = ppl_results[m]["ppl"] - baseline_ppl
    ppl_results[m]["delta"] = d
    print(f"  ΔPPL {m:>8}: {'+' if d>=0 else ''}{d:.3f}")


# =============================================================
# === Final report ===
# =============================================================

final = {
    "experiment": "poc_gemma4_main_eval",
    "model": "google/gemma-4-E2B-it",
    "timestamp": datetime.now().isoformat(),
    "architecture": {
        "num_hidden_layers": 35,
        "kv_cache_layers_exposed": 15,
        "shared_kv_layers": 20,
        "num_key_value_heads": 1,
        "head_dim": 256,
        "vocab_size": 262144,
    },
    "outlier_phase1": {
        "k_max_head": 5.56, "v_max_head": 7.48,
        "overall_max": 7.48, "worst_layer": 9
    },
    "niah_multi": niah_results,
    "ppl_chat": ppl_results,
}

print("\n" + "="*60)
print("FINAL JSON")
print("="*60)
print(json.dumps(final, indent=2, ensure_ascii=False))
