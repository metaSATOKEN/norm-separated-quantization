# ============================================================
# Phase 5e: Full Arc Paper Model Sweep
# ============================================================
# Verify nsep+pchan4 across all 14 models from the Arc paper.
# 7 models already verified. Adding the remaining 6.
# (Mistral-7B is gated, so it is skipped.)
#
# New tests:
#   GPT-1       (110M, Post-LN) -- negative control
#   OPT-125m    (125M, Pre-LN)
#   OPT-1.3B    (1.3B, Pre-LN)
#   Pythia-2.8B (2.8B, Pre-LN)
#   OPT-13B     (13B, Pre-LN)
#   Falcon-40B  (40B, Pre-LN) -- barely fits in 102GB VRAM
# ============================================================

# === CELL 1 ===
!pip install -q transformers accelerate hf_transfer
import os; os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"

import gc, json, numpy as np, torch, torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from datetime import datetime

device = "cuda"
print(f"GPU: {torch.cuda.get_device_name()}")
props = torch.cuda.get_device_properties(0)
vram = getattr(props, "total_memory", getattr(props, "total_mem", 0)) / 1e9
print(f"VRAM: {vram:.1f} GB")

# === CELL 2 ===

# ── Quantization ───────────────────────────────────────────────────────────

def qa_perrow(x, b):
    x = x.float()
    qm = 2**(b-1) - 1
    s = x.abs().amax(dim=-1, keepdim=True).clamp(min=1e-12) / qm
    return ((x / s).round().clamp(-qm, qm)) * s

def qa_perchan(x, b):
    x = x.float()
    qm = 2**(b-1) - 1
    s = x.abs().amax(dim=0, keepdim=True).clamp(min=1e-12) / qm
    return ((x / s).round().clamp(-qm, qm)) * s

def norm_sep(x):
    x = x.float()
    n = x.norm(dim=-1, keepdim=True).clamp(min=1e-12)
    return n, x / n

def norm_recon(n, dq):
    dq = dq.float()
    dn = dq.norm(dim=-1, keepdim=True).clamp(min=1e-8)
    return n * (dq / dn)

def apply_method(x, name):
    if name == "naive4":
        return qa_perrow(x, 4)
    if name == "nsep+pchan4":
        n, d = norm_sep(x)
        return norm_recon(n, qa_perchan(d, 4))
    raise ValueError(name)

# ── Cache ──────────────────────────────────────────────────────────────────

def get_kv(past, li):
    if hasattr(past, 'layers'):
        return past.layers[li].keys, past.layers[li].values
    return past[li][0], past[li][1]

def set_kv(past, li, k, v):
    if hasattr(past, 'layers'):
        past.layers[li].keys = k
        past.layers[li].values = v
    return past

def n_cache_layers(past):
    return len(past.layers) if hasattr(past, 'layers') else len(past)

def compress_cache(past, method_name):
    cos_k, cos_v = [], []
    for li in range(n_cache_layers(past)):
        ok, ov = get_kv(past, li)
        ok, ov = ok.clone(), ov.clone()
        nk, nv = torch.zeros_like(ok), torch.zeros_like(ov)
        for h in range(ok.shape[1]):
            nk[0,h] = apply_method(ok[0,h], method_name).to(ok.dtype)
            nv[0,h] = apply_method(ov[0,h], method_name).to(ov.dtype)
            cos_k.append(F.cosine_similarity(ok[0,h].float(), nk[0,h].float(), dim=-1).mean().item())
            cos_v.append(F.cosine_similarity(ov[0,h].float(), nv[0,h].float(), dim=-1).mean().item())
        set_kv(past, li, nk, nv)
    return round(np.mean(cos_k), 6), round(np.mean(cos_v), 6)

def xppl(l, t):
    return float(torch.exp(F.cross_entropy(l, t, reduction="mean")).item())

# ── Text ───────────────────────────────────────────────────────────────────

TEXT = (
    "The old lighthouse keeper climbed the spiral staircase each evening, "
    "carrying a lantern that cast long shadows across the stone walls. "
    "He had performed this ritual for forty years, ever since the automated "
    "systems had failed during the great storm. The sea below crashed "
    "against the rocks with a rhythm that matched his breathing, and he "
    "found comfort in the predictability of waves and wind. Tonight, "
    "however, something was different. A strange light flickered on the "
    "horizon, pulsing with an irregular beat that made him uneasy."
)

CL = 30  # continuation length

# ── Models to test ─────────────────────────────────────────────────────────

MODELS = [
    # Small models (< 1B) -- fast
    ("GPT-1 (Post-LN)", "openai-community/openai-gpt", torch.float32, "Post-LN"),
    ("OPT-125m",        "facebook/opt-125m",            torch.float32, "Pre-LN"),
    # Medium models (1-3B)
    ("OPT-1.3B",        "facebook/opt-1.3b",            torch.float16, "Pre-LN"),
    ("Pythia-2.8B",     "EleutherAI/pythia-2.8b",       torch.float16, "Pre-LN"),
    # Large models (13B+)
    ("OPT-13B",         "facebook/opt-13b",             torch.float16, "Pre-LN"),
    # XL model (40B) -- try last, may OOM
    ("Falcon-40B",      "tiiuae/falcon-40b",            torch.float16, "Pre-LN"),
]

# ── Run ────────────────────────────────────────────────────────────────────

all_results = {}

for mname, hf_id, dtype, ln_type in MODELS:
    print(f"\n{'='*70}")
    print(f"  {mname} ({hf_id}, {ln_type})")
    print(f"{'='*70}")

    try:
        tok = AutoTokenizer.from_pretrained(hf_id)
        mdl = AutoModelForCausalLM.from_pretrained(
            hf_id, torch_dtype=dtype, device_map="auto", use_safetensors=True
        )
    except Exception as e:
        # Fallback: try without safetensors (some older models only have .bin)
        try:
            mdl = AutoModelForCausalLM.from_pretrained(
                hf_id, torch_dtype=dtype, device_map="auto"
            )
        except Exception as e2:
            print(f"  SKIPPED: {e2}")
            continue

    mdl.eval()
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token or "<|endoftext|>"

    # Cache info
    try:
        test_ids = tok.encode("Hello world", return_tensors="pt").to(device)
        with torch.inference_mode():
            test_out = mdl(test_ids, use_cache=True)
        tk, tv = get_kv(test_out.past_key_values, 0)
        n_kv = tk.shape[1]; hd = tk.shape[3]; nl = n_cache_layers(test_out.past_key_values)
        print(f"  {nl} layers, {n_kv} KV heads, head_dim={hd}, cache={tk.dtype}")
        del test_out; gc.collect(); torch.cuda.empty_cache()
    except Exception as e:
        print(f"  Cache probe failed: {e}")
        del mdl; gc.collect(); torch.cuda.empty_cache()
        continue

    vram_used = torch.cuda.memory_allocated() / 1e9
    print(f"  VRAM: {vram_used:.1f} GB")

    # Tokenize
    ids = tok.encode(TEXT, return_tensors="pt").to(device)
    max_len = getattr(mdl.config, "n_positions", None) or getattr(
        mdl.config, "max_position_embeddings", 2048)
    if ids.shape[1] > max_len:
        ids = ids[:, :max_len]
    pfl = ids.shape[1] - CL
    if pfl < 10:
        print(f"  skip (pfl={pfl})")
        del mdl; gc.collect(); torch.cuda.empty_cache()
        continue

    fi = ids[:, :pfl+CL]; tgt = fi[0, pfl:].cpu()

    # Baseline
    with torch.inference_mode():
        bl = mdl(fi, use_cache=False).logits[0, pfl-1:-1].float().cpu()
    bp = xppl(bl, tgt)
    print(f"  Baseline PPL: {bp:.2f}, pfl={pfl}")

    # Layer 0 outlier check
    with torch.inference_mode():
        po = mdl(ids[:,:pfl], use_cache=True)
        k0, _ = get_kv(po.past_key_values, 0)
        col_max = k0[0].float().abs().amax(dim=(0,1))
        ratio = (col_max.max() / col_max.mean()).item()
    print(f"  L0 outlier ratio: {ratio:.1f}x")
    del po; gc.collect(); torch.cuda.empty_cache()

    # naive4 vs nsep+pchan4
    results = {"ln_type": ln_type, "n_layers": nl, "n_kv_heads": n_kv,
               "head_dim": hd, "baseline_ppl": round(bp, 3),
               "outlier_ratio": round(ratio, 2), "methods": {}}

    for mn in ["naive4", "nsep+pchan4"]:
        with torch.inference_mode():
            po = mdl(ids[:,:pfl], use_cache=True)
            past = po.past_key_values
            kc, vc = compress_cache(past, mn)
            co = mdl(ids[:,pfl:pfl+CL], past_key_values=past, use_cache=False)
        cl = co.logits[0].float().cpu()
        ca, ba, ta = cl[:-1], bl[:-1], tgt[1:]
        ml = min(ca.shape[0], ba.shape[0], ta.shape[0])
        dp = xppl(ca[:ml], ta[:ml]) - bp
        results["methods"][mn] = round(dp, 4)
        print(f"  {mn:>14}: ΔPPL={dp:>+10.4f}  Kcos={kc:.4f} Vcos={vc:.4f}")
        del po, co, past; gc.collect(); torch.cuda.empty_cache()

    all_results[mname] = results
    del mdl, tok, bl; gc.collect(); torch.cuda.empty_cache()

# ── COMPLETE TABLE ─────────────────────────────────────────────────────────

print("\n" + "="*70)
print("  COMPLETE ARC PAPER MODEL SWEEP")
print("="*70)

# Merge with prior results
prior = {
    "GPT-2":        {"params": "124M",  "ln": "Pre-LN",  "hd": 64,  "naive4": 0.64,   "nsep_pc4": 0.56},
    "Pythia-410M":  {"params": "410M",  "ln": "Pre-LN",  "hd": 64,  "naive4": 77.55,  "nsep_pc4": 12.62},
    "Qwen2-0.5B":   {"params": "0.5B",  "ln": "RMSNorm", "hd": 64,  "naive4": 0.64,   "nsep_pc4": 0.28},
    "Pythia-6.9B":  {"params": "6.9B",  "ln": "Pre-LN",  "hd": 128, "naive4": 22.26,  "nsep_pc4": 0.27},
    "Qwen2-7B":     {"params": "7B",    "ln": "RMSNorm", "hd": 128, "naive4": 238.23, "nsep_pc4": 0.32},
    "Pythia-12B":   {"params": "12B",   "ln": "Pre-LN",  "hd": 128, "naive4": 27.28,  "nsep_pc4": 1.82},
    "Qwen2.5-14B":  {"params": "14B",   "ln": "RMSNorm", "hd": 128, "naive4": 0.30,   "nsep_pc4": 0.26},
}

# Add new results
for mname, mdata in all_results.items():
    prior[mname] = {
        "params": "--",
        "ln": mdata["ln_type"],
        "hd": mdata["head_dim"],
        "naive4": mdata["methods"].get("naive4", "?"),
        "nsep_pc4": mdata["methods"].get("nsep+pchan4", "?"),
    }

print(f"\n  {'Model':<20} {'Params':<7} {'LN':<8} {'hd':<4} {'naive4':>9} {'nsep+pc4':>9} {'improve':>8}")
print(f"  {'-'*70}")
for name, d in prior.items():
    n = d["naive4"]; s = d["nsep_pc4"]
    if isinstance(n, (int,float)) and isinstance(s, (int,float)) and s != 0:
        imp = f"{abs(n)/abs(s):.0f}x" if abs(s) > 0.01 else "--"
    else:
        imp = "?"
    n_s = f"{n:>+9.2f}" if isinstance(n, (int,float)) else f"{'?':>9}"
    s_s = f"{s:>+9.2f}" if isinstance(s, (int,float)) else f"{'?':>9}"
    print(f"  {name:<20} {d['params']:<7} {d['ln']:<8} {d['hd']:<4} {n_s} {s_s} {imp:>8}")

print("\n" + "="*70)
print("JSON OUTPUT:")
print("="*70)
output = {
    "experiment": "phase5e_full_arc_sweep",
    "timestamp": datetime.now().isoformat(),
    "device": torch.cuda.get_device_name(),
    "new_models": all_results,
    "complete_table": prior,
}
print(json.dumps(output, indent=2, ensure_ascii=False))
