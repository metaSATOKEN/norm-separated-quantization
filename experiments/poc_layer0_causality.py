# Copyright 2026 Kentaro Sato
# Licensed under the Apache License, Version 2.0

# ============================================================
# Layer 0 Causality Experiment on Qwen2-7B
# ============================================================
# The K/V asymmetry study (Appendix H) and the GPT-J-6B natural
# control (K=18.15 @ L1, clean) both point to Layer 0 specifically
# as the site of pathology in Qwen2-7B. This experiment
# DECISIVELY tests that claim by selectively applying nsep+pchan4
# to different layer subsets:
#
# Methods:
#   1. baseline            (no compression, reference)
#   2. all_naive4          (all layers naive4, catastrophic ref)
#   3. all_nsep            (all layers nsep+pchan4, rescue ref)
#   4. L0_nsep_rest_naive4 (Layer 0 nsep, rest naive4) <-- KEY
#   5. L0_naive4_rest_nsep (Layer 0 naive4, rest nsep) <-- INVERSE
#   6. L0_fp16_rest_naive4 (Layer 0 FP16, rest naive4) <-- UPPER BOUND
#
# Predictions under Layer-0 hypothesis:
#   Method 4 (only fix L0) ≈ baseline           → Layer 0 is sufficient
#   Method 5 (only break L0) ≈ all_naive4       → other layers can't rescue
#   Method 6 (L0 FP16) ≈ Method 4               → L0 alone causes the damage
#
# If the hypothesis is correct, this is the definitive
# mechanistic evidence for the paper.
# ============================================================


# =============================================================
# === CELL L1 === Free + load Qwen2-7B
# =============================================================

try:
    del mdl, tok
except NameError:
    pass
import gc, torch
gc.collect()
if torch.cuda.is_available():
    torch.cuda.empty_cache()

import json, numpy as np
import torch.nn.functional as F
from datetime import datetime
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_ID = "Qwen/Qwen2-7B"

print(f"Loading {MODEL_ID} ...")
tok = AutoTokenizer.from_pretrained(MODEL_ID)
mdl = AutoModelForCausalLM.from_pretrained(
    MODEL_ID, torch_dtype=torch.float16, device_map="auto",
    use_safetensors=True,
)
mdl.eval()
print(f"VRAM used: {torch.cuda.memory_allocated()/1e9:.1f} GB")

# Probe to get layer count
_ids = tok.encode("Hello", return_tensors="pt").to("cuda")
with torch.inference_mode():
    _out = mdl(_ids, use_cache=True)
    _past = _out.past_key_values

if hasattr(_past, 'layers'):
    N_LAYERS = len(_past.layers)
else:
    N_LAYERS = len(_past)
print(f"Qwen2-7B KV cache layers: {N_LAYERS}")
del _out, _past; gc.collect(); torch.cuda.empty_cache()


# =============================================================
# === CELL L2 === Quantization helpers
# =============================================================

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
    if name == "fp16":
        return x  # no compression

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

def compress_cache_selective(past, layer_spec):
    """layer_spec: dict mapping layer index -> method name.
    Layers not in the dict are left untouched (FP16)."""
    for li in range(n_cache_layers(past)):
        method = layer_spec.get(li, "fp16")
        if method == "fp16":
            continue
        ok, ov = get_kv(past, li)
        ok = ok.clone(); ov = ov.clone()
        nk = torch.zeros_like(ok); nv = torch.zeros_like(ov)
        for h in range(ok.shape[1]):
            nk[0, h] = apply_method(ok[0, h], method).to(ok.dtype)
            nv[0, h] = apply_method(ov[0, h], method).to(ov.dtype)
        set_kv(past, li, nk, nv)


# =============================================================
# === CELL L3 === Load WikiText-2 + PPL function
# =============================================================

!pip install -q datasets
from datasets import load_dataset

ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
wtext = "\n\n".join([x for x in ds["text"] if x.strip()])
wtext = wtext[:48000]  # ~12000 tokens
print(f"WikiText-2 text length: {len(wtext)} chars")


def ppl_with_selective(mdl, tok, text, layer_spec_fn,
                       ctx_len=1024, stride=512, max_windows=10):
    """layer_spec_fn(N_LAYERS) -> dict mapping layer idx to method."""
    ids = tok.encode(text, return_tensors="pt").to("cuda")
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

            out_ctx = mdl(context, use_cache=True)
            past = out_ctx.past_key_values
            first_logits = out_ctx.logits[:, -1:, :]

            layer_spec = layer_spec_fn(n_cache_layers(past))
            compress_cache_selective(past, layer_spec)

            out_tgt = mdl(targets, past_key_values=past, use_cache=False)
            rest_logits = out_tgt.logits[:, :-1, :]
            all_logits = torch.cat([first_logits, rest_logits], dim=1)

            log_probs = F.log_softmax(all_logits[0].float(), dim=-1)
            nll = -log_probs.gather(-1, targets[0].unsqueeze(-1)).squeeze(-1)
            total_nll += nll.sum().item()
            total_tokens += targets.shape[1]
            windows_done += 1

            del out_ctx, out_tgt, past
            gc.collect(); torch.cuda.empty_cache()

    mean_nll = total_nll / max(total_tokens, 1)
    ppl = float(np.exp(mean_nll))
    return ppl, total_tokens


# =============================================================
# === CELL L4 === Run 6 methods
# =============================================================

print(f"\n{'='*80}")
print(f"LAYER 0 CAUSALITY EXPERIMENT on Qwen2-7B (WikiText-2)")
print(f"{'='*80}")
print(f"  {'Method':<35} | {'PPL':>10} | {'ΔPPL':>10} | {'tokens':>8}")
print(f"  {'-'*65}")

methods = [
    ("baseline",
     lambda N: {}),                                      # no compression anywhere
    ("all_naive4",
     lambda N: {i: "naive4" for i in range(N)}),         # all layers naive4
    ("all_nsep+pchan4",
     lambda N: {i: "nsep+pchan4" for i in range(N)}),    # all layers nsep
    ("L0_nsep_rest_naive4",
     lambda N: {**{0: "nsep+pchan4"},
                **{i: "naive4" for i in range(1, N)}}),  # Layer 0 nsep, rest naive4
    ("L0_naive4_rest_nsep",
     lambda N: {**{0: "naive4"},
                **{i: "nsep+pchan4" for i in range(1, N)}}),  # inverse
    ("L0_fp16_rest_naive4",
     lambda N: {i: "naive4" for i in range(1, N)}),     # L0 untouched (FP16), rest naive4
]

results = {}
baseline_ppl = None
for name, spec_fn in methods:
    print(f"  Running {name} ...", flush=True)
    ppl, nt = ppl_with_selective(mdl, tok, wtext, spec_fn,
                                  ctx_len=1024, stride=512, max_windows=10)
    if name == "baseline":
        baseline_ppl = ppl
        results[name] = {"ppl": ppl, "tokens": nt}
        dppl_str = "-"
    else:
        dppl = ppl - baseline_ppl
        results[name] = {"ppl": ppl, "tokens": nt, "delta": dppl}
        dppl_str = f"+{dppl:.2f}" if dppl >= 0 else f"{dppl:.2f}"
    print(f"  {name:<35} | {ppl:>10.3f} | {dppl_str:>10} | {nt:>8}")


# =============================================================
# === CELL L5 === Verdict on Layer 0 causality
# =============================================================

print(f"\n{'='*80}")
print("LAYER 0 CAUSALITY VERDICT")
print(f"{'='*80}")

baseline = results["baseline"]["ppl"]
d_all_naive = results["all_naive4"]["delta"]
d_all_nsep = results["all_nsep+pchan4"]["delta"]
d_L0_nsep = results["L0_nsep_rest_naive4"]["delta"]
d_L0_naive = results["L0_naive4_rest_nsep"]["delta"]
d_L0_fp16 = results["L0_fp16_rest_naive4"]["delta"]

print(f"\n  Reference:")
print(f"    baseline         PPL: {baseline:.2f}")
print(f"    all_naive4       ΔPPL: +{d_all_naive:.2f}    (catastrophic reference)")
print(f"    all_nsep+pchan4  ΔPPL: +{d_all_nsep:.2f}    (full rescue reference)")

print(f"\n  Causality tests:")
print(f"    L0_nsep_rest_naive4  ΔPPL: +{d_L0_nsep:.2f}")
print(f"      ↳ Fixing ONLY Layer 0 with nsep+pchan: {'RESCUES' if d_L0_nsep < d_all_naive * 0.1 else 'partial' if d_L0_nsep < d_all_naive * 0.5 else 'insufficient'}")

print(f"\n    L0_naive4_rest_nsep  ΔPPL: +{d_L0_naive:.2f}")
print(f"      ↳ Breaking ONLY Layer 0 with naive4:   {'STILL BREAKS' if d_L0_naive > d_all_naive * 0.5 else 'mostly safe' if d_L0_naive < d_all_nsep * 3 else 'intermediate'}")

print(f"\n    L0_fp16_rest_naive4  ΔPPL: +{d_L0_fp16:.2f}")
print(f"      ↳ Leaving Layer 0 at FP16 (upper bound): {'RESCUES' if d_L0_fp16 < d_all_naive * 0.1 else 'partial'}")

# Verdict
rescue_ratio = 1 - (d_L0_nsep / d_all_naive) if d_all_naive > 1 else 0
print(f"\n  Layer 0 rescue efficiency:")
print(f"    {d_all_naive:.0f} → {d_L0_nsep:.0f} ΔPPL by fixing Layer 0 alone")
print(f"    = {rescue_ratio*100:.1f}% of catastrophic damage attributable to Layer 0")

if rescue_ratio > 0.90:
    print(f"\n  🎯 LAYER 0 IS THE SINGLE POINT OF FAILURE ({rescue_ratio*100:.1f}% rescue)")
    print(f"  >>> Mechanism: K outlier at Layer 0 propagates through all")
    print(f"      subsequent attentions; fixing Layer 0 alone prevents")
    print(f"      the cascade.")
    verdict = "LAYER_0_DOMINANT"
elif rescue_ratio > 0.50:
    print(f"\n  ⚠️  Layer 0 is the DOMINANT factor ({rescue_ratio*100:.1f}% rescue)")
    print(f"  >>> Other layers contribute residual but Layer 0 is primary")
    verdict = "LAYER_0_PRIMARY"
else:
    print(f"\n  ✗ Layer 0 is NOT sufficient ({rescue_ratio*100:.1f}% rescue)")
    print(f"  >>> Pathology is distributed across multiple layers")
    verdict = "DISTRIBUTED"

# Save
final = {
    "experiment": "poc_layer0_causality_qwen2_7b",
    "model": MODEL_ID,
    "timestamp": datetime.now().isoformat(),
    "results": results,
    "layer_0_rescue_ratio": rescue_ratio,
    "verdict": verdict,
}
print(f"\n{'='*80}")
print("FINAL JSON")
print(f"{'='*80}")
print(json.dumps(final, indent=2, ensure_ascii=False))
