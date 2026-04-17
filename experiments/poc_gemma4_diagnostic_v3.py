# Copyright 2026 Kentaro Sato
# Licensed under the Apache License, Version 2.0

# ============================================================
# Gemma 4 Diagnostic v3 -- BatchFeature + model bypass
# ============================================================
# v2 findings:
#   - Manual PPL 63292 even when forward logits have one extreme
#     token per position. Interpretation: IT model genuinely bad
#     at raw text continuation. Must use chat format for PPL.
#   - AutoProcessor.apply_chat_template returns BatchFeature,
#     not dict or tensor. Need .input_ids access.
#
# Fixes:
#   - Robust extraction: tensor | dict | BatchFeature | obj.input_ids
#   - Try mdl.model (bypass multimodal wrapper) for cleaner forward
#   - PPL on chat-formatted text
# ============================================================

import torch
import torch.nn.functional as F
import numpy as np

print("="*60)
print("GEMMA 4 E2B DIAGNOSTIC v3")
print("="*60)

from transformers import AutoProcessor
proc = AutoProcessor.from_pretrained("google/gemma-4-E2B-it")
print(f"  processor: {type(proc).__name__}")

# ── Robust chat encode ───────────────────────────────────────
def to_input_ids(x):
    """Extract a 2D LongTensor from whatever apply_chat_template returned."""
    if isinstance(x, torch.Tensor):
        return x
    if hasattr(x, "input_ids"):        # BatchFeature / BatchEncoding
        return x.input_ids
    if hasattr(x, "get") and callable(x.get):
        r = x.get("input_ids")
        if r is not None:
            return r
    if isinstance(x, dict):
        return x["input_ids"]
    raise TypeError(f"Unknown chat output type: {type(x).__name__}")

def chat_encode(messages):
    try:
        r = proc.apply_chat_template(
            messages, add_generation_prompt=True,
            return_tensors="pt", tokenize=True
        )
        return to_input_ids(r).to("cuda")
    except Exception as e:
        print(f"    (processor route failed: {e}, trying tokenizer)")
        r = tok.apply_chat_template(
            messages, add_generation_prompt=True, return_tensors="pt"
        )
        return to_input_ids(r).to("cuda")

# ── D1v3: Generation ─────────────────────────────────────────
print("\n--- D1v3: Generation with chat template ---")
msgs = [{"role": "user", "content": "What is the capital of France? One word."}]
try:
    ids = chat_encode(msgs)
    print(f"  Shape: {tuple(ids.shape)}")
    print(f"  Decoded (head): {tok.decode(ids[0])[:200]!r}")
    with torch.inference_mode():
        out = mdl.generate(
            ids, max_new_tokens=20, do_sample=False,
            pad_token_id=tok.eos_token_id or 0,
        )
    resp = tok.decode(out[0, ids.shape[-1]:], skip_special_tokens=True).strip()
    print(f"  Response: {resp!r}")
    d1_pass = "paris" in resp.lower()
    print(f"  D1v3: {'PASS ✓' if d1_pass else 'FAIL ✗'}")
except Exception as e:
    import traceback; traceback.print_exc()
    d1_pass = False

# ── D2v3: PPL on CHAT-FORMATTED text ────────────────────────
# Raw text PPL fails because IT model is bad at raw text.
# Try PPL on a completed chat (user+assistant turn).
print("\n--- D2v3: PPL on chat-formatted exchange ---")
chat = [
    {"role": "user", "content": "Write a sentence about AI history."},
    {"role": "assistant", "content":
     "Artificial intelligence research began at a 1956 workshop at Dartmouth College, "
     "sparking decades of progress in machine learning, neural networks, and reasoning."}
]
try:
    # Full chat encoded (no add_generation_prompt since assistant already spoke)
    r = proc.apply_chat_template(
        chat, add_generation_prompt=False,
        return_tensors="pt", tokenize=True
    )
    full_ids = to_input_ids(r).to("cuda")
    print(f"  Full chat tokens: {full_ids.shape[1]}")

    with torch.inference_mode():
        out = mdl(input_ids=full_ids)
    logits = out.logits
    print(f"  Logits shape: {tuple(logits.shape)}")
    sl = logits[:, :-1, :].contiguous().float()
    lb = full_ids[:, 1:].contiguous()
    loss = F.cross_entropy(sl.view(-1, sl.size(-1)), lb.view(-1)).item()
    ppl = float(np.exp(loss))
    print(f"  NLL: {loss:.3f}, PPL: {ppl:.2f}")
    d2_pass = ppl < 500
    print(f"  D2v3: {'PASS ✓' if d2_pass else 'FAIL ✗'}  (target <500)")
except Exception as e:
    import traceback; traceback.print_exc()
    d2_pass = False

# ── D3v3: Single needle (same question as v2) ───────────────
print("\n--- D3v3: Single-needle QA ---")
haystack = (
    "The weather in coastal regions is influenced by ocean currents. "
    "Ancient civilizations developed agriculture. "
    "SECRET FACT: The laboratory password is CRIMSON TIGER 9981. "
    "Mathematics has evolved from counting to algebra. "
    "Music theory encompasses harmony and rhythm. "
    "Volcanic activity shapes landscapes. "
)
msgs = [{"role": "user", "content":
    f"Read the text and answer the question.\n\nTEXT:\n{haystack}\n\n"
    f"QUESTION: What is the laboratory password from the SECRET FACT?\n"
    f"Answer with just the password."}]
try:
    ids = chat_encode(msgs)
    print(f"  Input tokens: {ids.shape[-1]}")
    with torch.inference_mode():
        out = mdl.generate(
            ids, max_new_tokens=30, do_sample=False,
            pad_token_id=tok.eos_token_id or 0,
        )
    resp = tok.decode(out[0, ids.shape[-1]:], skip_special_tokens=True).strip()
    print(f"  Response: {resp!r}")
    d3_pass = all(p in resp.lower() for p in ["crimson", "tiger", "9981"])
    print(f"  D3v3: {'PASS ✓' if d3_pass else 'FAIL ✗'}")
except Exception as e:
    import traceback; traceback.print_exc()
    d3_pass = False

# ── Summary ──────────────────────────────────────────────────
print("\n" + "="*60)
print("DIAGNOSTIC v3 SUMMARY")
print("="*60)
print(f"  D1v3 (chat generation)   : {'✓' if d1_pass else '✗'}")
print(f"  D2v3 (chat-format PPL)   : {'✓' if d2_pass else '✗'}")
print(f"  D3v3 (single needle)     : {'✓' if d3_pass else '✗'}")

print("\n" + "─"*60)
if d1_pass and d3_pass:
    print("DECISION: CHAT INTERFACE WORKS ✓")
    print("  -> NIAH experiment feasible on 26B-A4B")
    print("     (PPL may or may not work, NIAH is the main signal)")
elif d1_pass:
    print("DECISION: Generation works, needle retrieval weak")
    print("  -> 2B capacity limit, 26B-A4B should handle needles")
else:
    print("DECISION: Chat interface still broken")
    print("  -> Write up Phase 1 only, Gemma 4 full eval as future work.")
