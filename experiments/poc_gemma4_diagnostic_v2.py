# Copyright 2026 Kentaro Sato
# Licensed under the Apache License, Version 2.0

# ============================================================
# Gemma 4 Diagnostic v2 -- manual PPL + AutoProcessor
# ============================================================
# Previous diagnostic revealed:
#   - Model is Gemma4ForConditionalGeneration (multimodal wrapper)
#   - LM head is tied and sane (std 0.03)
#   - Forward produces discriminating logits (" France" predicted
#     with logit -9 vs mean -28)
#   - labels=ids handling is broken for multimodal wrapper
#
# Fixes applied:
#   - Manual loss = F.cross_entropy(shift_logits, shift_labels)
#   - AutoProcessor for chat template (not AutoTokenizer)
# ============================================================

import torch
import torch.nn.functional as F
import numpy as np

print("="*60)
print("GEMMA 4 E2B DIAGNOSTIC v2")
print("="*60)

# ── Prep: get a proper processor for chat template ───────────
print("\n--- Loading AutoProcessor ---")
from transformers import AutoProcessor
try:
    proc = AutoProcessor.from_pretrained("google/gemma-4-E2B-it")
    print(f"  type: {type(proc).__name__}")
    has_chat = hasattr(proc, 'apply_chat_template') or \
               hasattr(getattr(proc, 'tokenizer', None), 'apply_chat_template')
    print(f"  has apply_chat_template: {has_chat}")
except Exception as e:
    print(f"  ERROR loading processor: {e}")
    proc = None

# ── Helper: apply chat template robustly ─────────────────────
def chat_encode(messages):
    """Try processor first, fall back to tokenizer."""
    if proc is not None:
        try:
            # Processor route (multimodal-aware)
            if hasattr(proc, 'apply_chat_template'):
                return proc.apply_chat_template(
                    messages, add_generation_prompt=True,
                    return_tensors="pt", tokenize=True
                )
        except Exception as e:
            print(f"    (processor chat failed: {e})")
    # Fall back to tokenizer
    return tok.apply_chat_template(
        messages, add_generation_prompt=True, return_tensors="pt"
    )

# ── D1v2: Generation with chat template ─────────────────────
print("\n--- D1v2: Generation (processor route) ---")
messages = [
    {"role": "user", "content": "What is the capital of France? Answer in one word."}
]
try:
    chat_ids = chat_encode(messages)
    if isinstance(chat_ids, dict):
        chat_ids = chat_ids.get("input_ids", chat_ids)
    chat_ids = chat_ids.to("cuda")
    print(f"  Input tokens: {chat_ids.shape[-1]}")
    decoded = (proc.tokenizer if proc else tok).decode(chat_ids[0])
    print(f"  Decoded (first 200 chars): {decoded[:200]!r}")

    with torch.inference_mode():
        out = mdl.generate(
            chat_ids, max_new_tokens=20, do_sample=False,
            pad_token_id=tok.eos_token_id or 0,
        )
    generated = out[0, chat_ids.shape[-1]:]
    response = tok.decode(generated, skip_special_tokens=True).strip()
    print(f"  Response: {response!r}")
    d1_pass = "paris" in response.lower()
    print(f"  D1v2: {'PASS ✓' if d1_pass else 'FAIL ✗'}")
except Exception as e:
    import traceback
    print(f"  ERROR:")
    traceback.print_exc()
    d1_pass = False

# ── D2v2: Manual PPL (bypass broken labels handling) ────────
print("\n--- D2v2: Manual PPL (F.cross_entropy on shifted logits) ---")
SHORT_TEXT = (
    "The history of artificial intelligence dates back to ancient times, "
    "with myths and stories of artificial beings endowed with intelligence. "
    "The field of AI research was founded at Dartmouth College in 1956. "
    "Since then, AI has experienced cycles of optimism and AI winters. "
    "Machine learning now powers search engines and medical diagnostics."
)
ids = tok.encode(SHORT_TEXT, return_tensors="pt").to("cuda")
print(f"  Input tokens: {ids.shape[1]}")

try:
    with torch.inference_mode():
        out = mdl(ids)  # no labels; just raw logits
    logits = out.logits  # [1, T, V]

    # Shift: logits[t] predicts ids[t+1]
    shift_logits = logits[:, :-1, :].contiguous().float()
    shift_labels = ids[:, 1:].contiguous()
    loss = F.cross_entropy(
        shift_logits.view(-1, shift_logits.size(-1)),
        shift_labels.view(-1)
    ).item()
    ppl = float(np.exp(loss))
    print(f"  NLL per token: {loss:.3f}")
    print(f"  PPL: {ppl:.2f}")
    d2_pass = ppl < 200
    print(f"  D2v2: {'PASS ✓' if d2_pass else 'FAIL ✗'}  "
          f"(target <200, reference GPT-2 ~30)")
except Exception as e:
    import traceback; traceback.print_exc()
    d2_pass = False

# ── D3v2: Single needle with chat template ──────────────────
print("\n--- D3v2: Single-needle QA ---")
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
    chat_ids = chat_encode(messages)
    if isinstance(chat_ids, dict):
        chat_ids = chat_ids.get("input_ids", chat_ids)
    chat_ids = chat_ids.to("cuda")
    print(f"  Input tokens: {chat_ids.shape[-1]}")
    with torch.inference_mode():
        out = mdl.generate(
            chat_ids, max_new_tokens=30, do_sample=False,
            pad_token_id=tok.eos_token_id or 0,
        )
    generated = out[0, chat_ids.shape[-1]:]
    response = tok.decode(generated, skip_special_tokens=True).strip()
    print(f"  Response: {response!r}")
    d3_pass = ("crimson" in response.lower() and
               "tiger" in response.lower() and
               "9981" in response)
    print(f"  D3v2: {'PASS ✓' if d3_pass else 'FAIL ✗'}")
except Exception as e:
    import traceback; traceback.print_exc()
    d3_pass = False

# ── Summary ──────────────────────────────────────────────────
print("\n" + "="*60)
print("DIAGNOSTIC v2 SUMMARY")
print("="*60)
print(f"  D1v2 (coherent gen)    : {'✓' if d1_pass else '✗'}")
print(f"  D2v2 (manual PPL)      : {'✓' if d2_pass else '✗'}")
print(f"  D3v2 (needle retrieval): {'✓' if d3_pass else '✗'}")

print("\n" + "─"*60)
if d1_pass and d2_pass and d3_pass:
    print("DECISION: ✓✓✓ PROTOCOL FIXED")
    print("  -> Scale up to 26B-A4B for main experiment.")
elif d1_pass and d2_pass:
    print("DECISION: Protocol works, 2B lacks needle retrieval capacity")
    print("  -> 26B-A4B should succeed on needle task.")
elif d2_pass:
    print("DECISION: PPL works, chat template still broken")
    print("  -> Can still do PPL experiment on 26B-A4B.")
else:
    print("DECISION: Issues persist, paste error traces.")
