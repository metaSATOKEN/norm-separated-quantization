# Copyright 2026 Kentaro Sato
# Licensed under the Apache License, Version 2.0

# ============================================================
# PoC: Gemma 4 Diagnostic (5-minute sanity check)
# ============================================================
# Goal: Determine whether our protocol issues are fixable
# BEFORE investing in downloading 26B-A4B.
#
# Tests:
#   D1: Does Gemma 4 generate coherent text AT ALL?
#       (chat template + greedy generation, no cache quant)
#   D2: Is baseline PPL reasonable with labels=ids protocol?
#       (one-shot forward, no context/target split)
#   D3: Can the model answer a single needle question
#       with proper chat template?
#
# Decision:
#   ALL PASS -> fix protocol, scale to 26B-A4B
#   ANY FAIL -> abandon Gemma 4 full eval, keep Phase 1 only
# ============================================================


# =============================================================
# === CELL D === Run after CELL 1-3 (mdl, tok already loaded)
# =============================================================

import torch
import torch.nn.functional as F
import numpy as np

assert 'mdl' in dir() and 'tok' in dir(), "mdl/tok not loaded"

device = "cuda"
print(f"{'='*60}\nGEMMA 4 E2B DIAGNOSTIC\n{'='*60}")

# ── D1: Coherent generation test with chat template ──────────
print(f"\n--- D1: Generation with chat template ---")
messages = [
    {"role": "user", "content": "What is the capital of France? Answer in one word."}
]

# Try chat template
try:
    chat_ids = tok.apply_chat_template(
        messages, add_generation_prompt=True, return_tensors="pt"
    ).to(device)
    print(f"  Chat template applied. Input tokens: {chat_ids.shape[1]}")
    print(f"  Decoded input:\n    {tok.decode(chat_ids[0])}")

    with torch.inference_mode():
        out = mdl.generate(
            chat_ids,
            max_new_tokens=20,
            do_sample=False,
            pad_token_id=tok.eos_token_id or 0,
        )
    generated = out[0, chat_ids.shape[1]:]
    response = tok.decode(generated, skip_special_tokens=True).strip()
    print(f"  Response: '{response}'")

    d1_pass = "paris" in response.lower()
    print(f"  D1: {'PASS ✓' if d1_pass else 'FAIL ✗'} (expected 'Paris' in response)")
except Exception as e:
    print(f"  ERROR: {e}")
    d1_pass = False

# ── D2: Simple PPL with labels ───────────────────────────────
print(f"\n--- D2: One-shot PPL (labels=ids protocol) ---")

# Short text to keep it fast
SHORT_TEXT = (
    "The history of artificial intelligence dates back to ancient times, "
    "with myths and stories of artificial beings endowed with intelligence or "
    "consciousness by master craftsmen. The field of AI research was founded "
    "at a workshop held on the campus of Dartmouth College in 1956. "
    "Since then, AI has experienced several cycles of optimism followed by "
    "disappointment and the loss of funding, known as AI winters, and then "
    "renewed interest and funding. Modern AI techniques have become an "
    "essential part of the technology industry. Machine learning algorithms "
    "now power everything from search engines to medical diagnostics, and "
    "large language models have demonstrated remarkable capabilities in "
    "natural language understanding and generation."
)

ids = tok.encode(SHORT_TEXT, return_tensors="pt").to(device)
print(f"  Input tokens: {ids.shape[1]}")

try:
    with torch.inference_mode():
        out = mdl(ids, labels=ids)
    loss = out.loss.item()
    ppl = float(np.exp(loss))
    print(f"  NLL per token: {loss:.3f}")
    print(f"  PPL: {ppl:.2f}")

    d2_pass = ppl < 200
    verdict = "PASS ✓ (reasonable PPL)" if d2_pass else f"FAIL ✗ (PPL too high)"
    print(f"  D2: {verdict}")
    print(f"       Reference: GPT-2 ~30, Mistral-7B ~6, coherent 2B ~10-50")
except Exception as e:
    print(f"  ERROR: {e}")
    d2_pass = False

# ── D3: Single needle with chat template ─────────────────────
print(f"\n--- D3: Single-needle QA with chat template ---")

haystack = (
    "The weather in coastal regions is influenced by ocean currents. "
    "Ancient civilizations developed agriculture. "
    "SECRET FACT: The laboratory password is CRIMSON TIGER 9981. "
    "Mathematics has evolved from counting to algebra. "
    "Music theory encompasses harmony and rhythm. "
    "Volcanic activity shapes landscapes. "
)
question = (
    f"Read the following text and answer the question.\n\n"
    f"TEXT:\n{haystack}\n\n"
    f"QUESTION: What is the laboratory password mentioned in the SECRET FACT?\n"
    f"Answer with just the password."
)

messages = [{"role": "user", "content": question}]

try:
    chat_ids = tok.apply_chat_template(
        messages, add_generation_prompt=True, return_tensors="pt"
    ).to(device)
    print(f"  Input tokens: {chat_ids.shape[1]}")

    with torch.inference_mode():
        out = mdl.generate(
            chat_ids,
            max_new_tokens=30,
            do_sample=False,
            pad_token_id=tok.eos_token_id or 0,
        )
    generated = out[0, chat_ids.shape[1]:]
    response = tok.decode(generated, skip_special_tokens=True).strip()
    print(f"  Response: '{response}'")

    d3_pass = "crimson" in response.lower() and "tiger" in response.lower() and "9981" in response
    print(f"  D3: {'PASS ✓' if d3_pass else 'FAIL ✗'} (expected 'CRIMSON TIGER 9981')")
except Exception as e:
    print(f"  ERROR: {e}")
    d3_pass = False

# ── Decision matrix ──────────────────────────────────────────
print(f"\n{'='*60}")
print(f"DIAGNOSTIC SUMMARY")
print(f"{'='*60}")
print(f"  D1 (coherent gen)    : {'✓' if d1_pass else '✗'}")
print(f"  D2 (reasonable PPL)  : {'✓' if d2_pass else '✗'}")
print(f"  D3 (needle retrieval): {'✓' if d3_pass else '✗'}")

all_pass = d1_pass and d2_pass and d3_pass
any_fail = not (d1_pass and d2_pass and d3_pass)

print(f"\n{'─'*60}")
if all_pass:
    print(f"DECISION: ALL PASS ✓✓✓")
    print(f"  -> Protocol fix works on 2B. Scale up to 26B-A4B for")
    print(f"     the real experiment (multi-needle + PPL).")
elif d1_pass and d2_pass and not d3_pass:
    print(f"DECISION: PROTOCOL OK, MODEL CAPACITY INSUFFICIENT")
    print(f"  -> 2B can generate and has reasonable PPL, but can't")
    print(f"     retrieve needle even at FP16 baseline. Scale to 26B-A4B")
    print(f"     with confidence the protocol is correct.")
elif not d1_pass and not d2_pass:
    print(f"DECISION: FUNDAMENTAL PROTOCOL ISSUES")
    print(f"  -> Gemma 4 doesn't work with our infrastructure.")
    print(f"     Abandon full eval. Keep Phase 1 outlier finding only.")
else:
    print(f"DECISION: PARTIAL - investigate specific failure")
    print(f"  -> Debug the failed test before scaling.")
