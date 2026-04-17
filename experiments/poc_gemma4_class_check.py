# Copyright 2026 Kentaro Sato
# Licensed under the Apache License, Version 2.0

# ============================================================
# Gemma 4 Model Class + LM Head Inspection
# ============================================================
# Debug why logits are random despite K/V being reasonable.
# Hypothesis: AutoModelForCausalLM loads wrong wrapper for
# multimodal Gemma 4; LM head is disconnected or untied.
# ============================================================

import torch
import transformers

print(f"{'='*60}")
print(f"transformers version: {transformers.__version__}")
print(f"{'='*60}")

# ── C1: What class was actually loaded? ──────────────────────
print(f"\n--- C1: Model class ---")
print(f"  type(mdl).__name__: {type(mdl).__name__}")
print(f"  type(mdl).__mro__ :")
for c in type(mdl).__mro__[:5]:
    print(f"    - {c.__name__}")

# ── C2: Top-level module structure ───────────────────────────
print(f"\n--- C2: Top-level modules ---")
for name, _ in mdl.named_children():
    print(f"  mdl.{name}")

# ── C3: Is there an LM head? Are weights sane? ───────────────
print(f"\n--- C3: LM head inspection ---")
lm_head_candidates = []
for name, mod in mdl.named_modules():
    if "lm_head" in name.lower() or "output_embed" in name.lower():
        lm_head_candidates.append((name, mod))

if not lm_head_candidates:
    print(f"  No 'lm_head' found at top level -- multimodal wrapper suspected")
    # Check common multimodal patterns
    for attr in ["language_model", "text_model", "model"]:
        if hasattr(mdl, attr):
            sub = getattr(mdl, attr)
            print(f"  Found mdl.{attr}: {type(sub).__name__}")
            for sname, _ in sub.named_children():
                print(f"    mdl.{attr}.{sname}")
else:
    for name, mod in lm_head_candidates[:3]:
        w = mod.weight if hasattr(mod, 'weight') else None
        if w is not None:
            print(f"  {name}: shape {tuple(w.shape)}")
            print(f"    mean={w.mean().item():.4f}, std={w.std().item():.4f}")
            print(f"    absmax={w.abs().amax().item():.4f}")
            # Random init usually has std ~0.02
            # Properly loaded usually has std ~0.02-0.05 BUT mean near 0
            # Tied with embeddings: std should match embed table

# ── C4: Compare embedding table vs LM head (tying check) ────
print(f"\n--- C4: Embedding vs LM head tying ---")
try:
    embed = mdl.get_input_embeddings()
    lm_head = mdl.get_output_embeddings()
    print(f"  Input embed : {type(embed).__name__}, shape {tuple(embed.weight.shape)}")
    if lm_head is not None:
        print(f"  Output head : {type(lm_head).__name__}, shape {tuple(lm_head.weight.shape)}")
        same = embed.weight.data_ptr() == lm_head.weight.data_ptr()
        print(f"  Tied (same storage): {same}")
        if not same:
            diff = (embed.weight - lm_head.weight).abs().mean().item()
            print(f"  Weight diff (mean abs): {diff:.6f}")
            print(f"  Embed std : {embed.weight.std().item():.4f}")
            print(f"  Head std  : {lm_head.weight.std().item():.4f}")
    else:
        print(f"  get_output_embeddings() returned None -- issue confirmed")
except Exception as e:
    print(f"  ERROR: {e}")

# ── C5: Forward sanity check ─────────────────────────────────
print(f"\n--- C5: Forward sanity ---")
test_ids = tok.encode("The capital of France is", return_tensors="pt").to("cuda")
with torch.inference_mode():
    out = mdl(test_ids)
print(f"  logits shape: {out.logits.shape}")
print(f"  logits[0,-1] stats: mean={out.logits[0,-1].mean().item():.3f}, "
      f"std={out.logits[0,-1].std().item():.3f}")
print(f"  logits[0,-1] top-5:")
vals, idx = out.logits[0, -1].topk(5)
for v, i in zip(vals.tolist(), idx.tolist()):
    print(f"    {tok.decode([i])!r:20s} logit={v:.3f}")

# ── C6: Config check ─────────────────────────────────────────
print(f"\n--- C6: Config keys ---")
cfg = mdl.config
print(f"  type: {type(cfg).__name__}")
# Multimodal configs often have text_config, vision_config etc.
for k in dir(cfg):
    if k.startswith("_") or callable(getattr(cfg, k, None)):
        continue
    v = getattr(cfg, k, None)
    if hasattr(v, "__dict__") and "Config" in type(v).__name__:
        print(f"  {k}: {type(v).__name__}")
        for kk in ["num_hidden_layers", "hidden_size", "head_dim",
                  "num_attention_heads", "num_key_value_heads",
                  "vocab_size", "tie_word_embeddings"]:
            vv = getattr(v, kk, "?")
            print(f"    .{kk}: {vv}")
    elif k in ["vocab_size", "tie_word_embeddings", "num_hidden_layers",
               "hidden_size", "head_dim"]:
        print(f"  {k}: {v}")
