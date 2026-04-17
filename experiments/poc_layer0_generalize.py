# Copyright 2026 Kentaro Sato
# Licensed under the Apache License, Version 2.0

# ============================================================
# Layer 0 Selective Compression: Generalization Test
# ============================================================
# On Qwen2-7B (catastrophic, ΔPPL +1382), fixing Layer 0 alone
# rescues 99.97% of the damage (poc_layer0_causality).
#
# Question for this experiment: does the Layer-0-dominant
# mechanism extend to models with MODERATE failures, not just
# the catastrophic case?
#
# Test models (in order of expected failure severity):
#   1. Pythia-6.9B   (naive4 ΔPPL +22, moderate fail)  <-- key test
#   2. Pythia-12B    (naive4 ΔPPL +34, stronger fail)
#   3. Qwen2.5-14B   (naive4 ΔPPL +0.55, mild)          <-- sanity check
#
# If Layer 0 selective rescues these too, the mechanism is
# universal across severity levels, strengthening the paper's
# practical-default recommendation.
# ============================================================


# =============================================================
# === CELL G1 === Setup + common helpers
# =============================================================

try:
    del mdl, tok
except NameError:
    pass
import gc, torch
gc.collect()
if torch.cuda.is_available():
    torch.cuda.empty_cache()

!pip install -q -U transformers accelerate hf_transfer sentencepiece datasets
import os
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"

# ─────────────────────────────────────────────────────────────
# >>>>>  PASTE YOUR HF TOKEN  <<<<<
HF_TOKEN = "hf_XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX"
# ─────────────────────────────────────────────────────────────
from huggingface_hub import login
login(token=HF_TOKEN)

import json, numpy as np
import torch.nn.functional as F
from datetime import datetime
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

# ── Quantization + cache helpers ─────────────────────────────
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
        return x

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

# ── WikiText-2 ──────────────────────────────────────────────
ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
wtext = "\n\n".join([x for x in ds["text"] if x.strip()])[:48000]
print(f"WikiText-2 text length: {len(wtext)} chars")

def ppl_with_selective(mdl, tok, text, layer_spec_fn,
                       ctx_len=1024, stride=512, max_windows=10):
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


def run_selective_suite(hf_id, label):
    """Load model, run 5 regimes, return results dict."""
    global mdl, tok
    try:
        del mdl, tok
    except NameError:
        pass
    gc.collect(); torch.cuda.empty_cache()

    print(f"\n{'='*80}")
    print(f"Loading {label} ({hf_id}) ...")
    print(f"{'='*80}")
    tok = AutoTokenizer.from_pretrained(hf_id)
    mdl = AutoModelForCausalLM.from_pretrained(
        hf_id, torch_dtype=torch.float16, device_map="auto",
        use_safetensors=True,
    )
    mdl.eval()
    print(f"VRAM used: {torch.cuda.memory_allocated()/1e9:.1f} GB")

    regimes = [
        ("baseline",
         lambda N: {}),
        ("all_naive4",
         lambda N: {i: "naive4" for i in range(N)}),
        ("all_nsep+pchan4",
         lambda N: {i: "nsep+pchan4" for i in range(N)}),
        ("L0_nsep_rest_naive4",
         lambda N: {**{0: "nsep+pchan4"},
                    **{i: "naive4" for i in range(1, N)}}),
        ("L0_naive4_rest_nsep",
         lambda N: {**{0: "naive4"},
                    **{i: "nsep+pchan4" for i in range(1, N)}}),
    ]

    print(f"\n  {'Regime':<25} | {'PPL':>10} | {'ΔPPL':>10} | {'tokens':>8}")
    print(f"  {'-'*55}")
    results = {}
    baseline_ppl = None
    for name, spec_fn in regimes:
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
            dppl_str = f"+{dppl:.3f}" if dppl >= 0 else f"{dppl:.3f}"
        print(f"  {name:<25} | {ppl:>10.3f} | {dppl_str:>10} | {nt:>8}")

    # Rescue ratio
    d_all_naive = results["all_naive4"]["delta"]
    d_L0_nsep = results["L0_nsep_rest_naive4"]["delta"]
    d_L0_naive = results["L0_naive4_rest_nsep"]["delta"]
    if d_all_naive > 0.5:  # meaningful damage
        rescue_ratio = 1 - (d_L0_nsep / d_all_naive)
    else:
        rescue_ratio = None  # no damage to rescue

    print(f"\n  Layer 0 rescue analysis:")
    if rescue_ratio is not None:
        print(f"    all_naive4          ΔPPL: +{d_all_naive:.3f}")
        print(f"    L0_nsep_rest_naive4 ΔPPL: +{d_L0_nsep:.3f}")
        print(f"    L0_naive4_rest_nsep ΔPPL: +{d_L0_naive:.3f}")
        print(f"    >>> Layer 0 rescue: {rescue_ratio*100:.1f}%")
        if rescue_ratio > 0.90:
            print(f"    >>> 🎯 LAYER 0 IS DOMINANT (>90% rescue)")
        elif rescue_ratio > 0.50:
            print(f"    >>> Layer 0 is PRIMARY (partial rescue)")
        else:
            print(f"    >>> Layer 0 contribution is MINOR")
    else:
        print(f"    Baseline naive4 damage too small to meaningfully measure rescue")

    return {"label": label, "model": hf_id,
            "results": results, "rescue_ratio": rescue_ratio}


# =============================================================
# === CELL G2 === Pythia-6.9B (moderate fail, KEY TEST)
# =============================================================

pythia_69b = run_selective_suite(
    "EleutherAI/pythia-6.9b", "Pythia-6.9B"
)


# =============================================================
# === CELL G3 === Qwen2.5-14B (mild fail, sanity check)
# =============================================================

qwen25_14b = run_selective_suite(
    "Qwen/Qwen2.5-14B", "Qwen2.5-14B"
)


# =============================================================
# === CELL G4 === Pythia-12B (stronger fail, bonus)
# =============================================================

pythia_12b = run_selective_suite(
    "EleutherAI/pythia-12b", "Pythia-12B"
)


# =============================================================
# === CELL G5 === Generalization summary
# =============================================================

all_results = {
    "qwen2_7b_prior": {  # from poc_layer0_causality
        "label": "Qwen2-7B (prior)",
        "all_naive4": 1382.07,
        "L0_nsep_rest_naive4": 0.38,
        "L0_naive4_rest_nsep": 1368.18,
        "rescue_ratio": 0.9997,
        "severity": "catastrophic"
    },
    "pythia_6_9b": pythia_69b,
    "qwen2_5_14b": qwen25_14b,
    "pythia_12b": pythia_12b,
}

print("\n" + "="*100)
print("LAYER 0 SELECTIVE COMPRESSION GENERALIZATION")
print("="*100)
print(f"  {'Model':<20} | {'all_naive4':>12} | {'L0_nsep only':>14} | {'Rescue':>8} | Severity")
print(f"  {'-'*20} + {'-'*12} + {'-'*14} + {'-'*8} + {'-'*15}")

# Qwen2-7B (prior)
qp = all_results["qwen2_7b_prior"]
print(f"  {qp['label']:<20} | +{qp['all_naive4']:>10.2f} | +{qp['L0_nsep_rest_naive4']:>12.2f} | {qp['rescue_ratio']*100:>6.1f}% | {qp['severity']}")

# New results
for key, r in [("pythia_6_9b", pythia_69b),
               ("qwen2_5_14b", qwen25_14b),
               ("pythia_12b", pythia_12b)]:
    d_all = r["results"]["all_naive4"]["delta"]
    d_L0 = r["results"]["L0_nsep_rest_naive4"]["delta"]
    rr = r["rescue_ratio"]
    rr_str = f"{rr*100:.1f}%" if rr is not None else "N/A"
    sev = ("catastrophic" if d_all > 100 else
           "moderate" if d_all > 5 else
           "mild" if d_all > 0.5 else
           "minimal")
    print(f"  {r['label']:<20} | +{d_all:>10.3f} | +{d_L0:>12.3f} | {rr_str:>8} | {sev}")

print(f"\n{'─'*100}")
print("INTERPRETATION:")
print(f"{'─'*100}")

# Check if Layer 0 rescue generalizes
rescues = [pythia_69b["rescue_ratio"], pythia_12b["rescue_ratio"]]
valid_rescues = [r for r in rescues if r is not None and r > 0.5]
if len(valid_rescues) >= 1 and min(valid_rescues) > 0.80:
    print(f"  🎯 Layer 0 rescue GENERALIZES to moderate/strong failures")
    print(f"  >>> Layer 0 dominance is a universal principle, not Qwen2-7B-specific")
elif len(valid_rescues) >= 1 and min(valid_rescues) > 0.50:
    print(f"  ⚠️  Layer 0 rescue is PARTIAL on other models")
    print(f"  >>> Mechanism is primary but not complete on moderate cases")
else:
    print(f"  ✗ Layer 0 selective doesn't generalize beyond Qwen2-7B")
    print(f"  >>> Qwen2-7B catastrophic is qualitatively different from moderate")

# Export
final = {
    "experiment": "poc_layer0_generalize",
    "timestamp": datetime.now().isoformat(),
    "results": all_results,
}
print(f"\n{'='*100}")
print("FINAL JSON")
print(f"{'='*100}")
print(json.dumps(final, indent=2, ensure_ascii=False, default=str))
