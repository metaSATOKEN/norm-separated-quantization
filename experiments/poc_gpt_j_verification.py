# Copyright 2026 Kentaro Sato
# Licensed under the Apache License, Version 2.0

# ============================================================
# GPT-J-6B Pathological Verification (candidate n=2 case)
# ============================================================
# Following the catastrophic hunt, GPT-J-6B exhibits a K outlier
# pattern very similar to Qwen2-7B:
#   K_max = 18.15 at Layer 1 (vs Qwen2-7B 17.23 at Layer 0)
#   Layer 0 K = 9.31 (elevated; adjacent to the K=18.15 spike)
#   Early layers 0-2 all show elevated K (9.3, 18.15, 12.16)
#
# This is the first additional candidate for the pathological
# pattern. Goal: confirm or refute via PPL evaluation, since
# GPT-J-6B is a base model without chat/instruction tuning
# (NIAH would require a different protocol).
#
# Protocol:
#   WikiText-2 sliding-window PPL with cache quantization split.
#   3 methods: baseline, naive4, nsep+pchan4.
#   If naive4 ΔPPL >> 10, GPT-J-6B is confirmed pathological.
# ============================================================


# =============================================================
# === CELL V1 === Free memory + reload GPT-J-6B
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

print("Loading GPT-J-6B ...")
tok = AutoTokenizer.from_pretrained("EleutherAI/gpt-j-6b")
mdl = AutoModelForCausalLM.from_pretrained(
    "EleutherAI/gpt-j-6b",
    torch_dtype=torch.float16, device_map="auto",
    use_safetensors=True,
)
mdl.eval()
print(f"VRAM used: {torch.cuda.memory_allocated()/1e9:.1f} GB")


# =============================================================
# === CELL V2 === Quantization + cache helpers
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


# =============================================================
# === CELL V3 === WikiText-2 PPL evaluation
# =============================================================

!pip install -q datasets
from datasets import load_dataset

ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
wtext = "\n\n".join([x for x in ds["text"] if x.strip()])
wtext = wtext[:48000]  # ~12000 tokens
print(f"WikiText-2 text length: {len(wtext)} chars")


def ppl_with_split(mdl, tok, text, method,
                   ctx_len=1024, stride=512, max_windows=10):
    """PPL via batched teacher-forcing with compressed KV cache."""
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

            if method != "baseline":
                compress_cache(past, method)

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


print("\n" + "="*70)
print("GPT-J-6B WikiText-2 PPL (baseline / naive4 / nsep+pchan4)")
print("="*70)
print(f"  {'Method':>14} | {'PPL':>10} | {'ΔPPL':>10} | {'tokens':>8}")
print(f"  {'-'*52}")

ppl_results = {}
baseline_ppl = None
for method in ["baseline", "naive4", "nsep+pchan4"]:
    print(f"  Running {method} ...", flush=True)
    ppl, nt = ppl_with_split(mdl, tok, wtext, method,
                              ctx_len=1024, stride=512, max_windows=10)
    ppl_results[method] = {"ppl": ppl, "tokens": nt}
    if method == "baseline":
        baseline_ppl = ppl
        dppl_str = "-"
    else:
        dppl = ppl - baseline_ppl
        ppl_results[method]["delta"] = dppl
        dppl_str = f"+{dppl:.2f}" if dppl >= 0 else f"{dppl:.2f}"
    print(f"  {method:>14} | {ppl:>10.3f} | {dppl_str:>10} | {nt:>8}")


# =============================================================
# === CELL V4 === Verdict on catastrophic pathology
# =============================================================

print(f"\n{'='*70}")
print("VERDICT ON GPT-J-6B")
print(f"{'='*70}")

naive_d = ppl_results["naive4"].get("delta", 0)
nsep_d = ppl_results["nsep+pchan4"].get("delta", 0)

print(f"  Baseline PPL       : {ppl_results['baseline']['ppl']:.2f}")
print(f"  naive4 ΔPPL        : {'+' if naive_d>=0 else ''}{naive_d:.2f}")
print(f"  nsep+pchan4 ΔPPL   : {'+' if nsep_d>=0 else ''}{nsep_d:.2f}")
print(f"\n  K_max (from Phase 1): 18.15x @ Layer 1")
print(f"  Layer 0 K            : 9.31x  (elevated)")
print(f"  Layers 0-2 all > 9x : yes (9.31, 18.15, 12.16)")

print(f"\n{'-'*70}")
if naive_d > 100:
    print(f"  🎯🎯🎯 CATASTROPHIC CONFIRMED! naive4 ΔPPL = +{naive_d:.1f}")
    print(f"  >>> GPT-J-6B is the second catastrophic case in our sample")
    print(f"  >>> n=2 achieved. Paper claim refined:")
    print(f"      'K_max >= 15x concentrated at EARLY layers (0-2)'")
elif naive_d > 10:
    print(f"  ⚠️  Significant degradation (+{naive_d:.1f}), not quite catastrophic")
    print(f"  >>> Refines the severity scale")
    print(f"  >>> nsep+pchan4 improvement factor: {naive_d/max(nsep_d, 0.01):.1f}x")
elif naive_d > 2:
    print(f"  ~ Moderate degradation (+{naive_d:.1f})")
    print(f"  >>> Below catastrophic threshold")
else:
    print(f"  ✗ Minimal degradation (+{naive_d:.2f})")
    print(f"  >>> GPT-J-6B does NOT exhibit the full pathological pattern")
    print(f"  >>> K magnitude alone (18.15x) insufficient; Layer 0 position")
    print(f"      (not just Layer 1-2) appears to be required")

# Save
final = {
    "experiment": "poc_gpt_j_verification",
    "model": "EleutherAI/gpt-j-6b",
    "timestamp": datetime.now().isoformat(),
    "outlier_phase1": {
        "k_max": 18.15, "v_max": 5.57,
        "worst_k_layer": 1, "worst_v_layer": 1,
        "layer_0_k": 9.31,
        "top3_k_layers": [[1, 18.15], [2, 12.16], [7, 11.27]],
        "top3_v_layers": [[1, 5.57], [0, 5.49], [6, 4.93]]
    },
    "ppl_wikitext2": ppl_results,
    "verdict": (
        "CATASTROPHIC_CONFIRMED" if naive_d > 100
        else "SIGNIFICANT" if naive_d > 10
        else "MODERATE" if naive_d > 2
        else "CLEAN"
    ),
}
print(f"\n{'='*70}")
print("FINAL JSON")
print(f"{'='*70}")
print(json.dumps(final, indent=2, ensure_ascii=False))
