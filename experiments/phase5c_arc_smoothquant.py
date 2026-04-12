# ============================================================
# Phase 5c: Arc Prior + Outlier-Aware Quantization
# ============================================================
# Finding 8: The outlier (absmax=167) in Qwen2-7B Layer 0 destroys INT4.
# We use a SmoothQuant-style approach to separate outlier channels
# and combine it with the Arc prior (norm separation) for verification.
#
# Methods:
#   1. naive:     per-row absmax INT4 (baseline)
#   2. nsep:      norm-sep + per-row absmax INT4
#   3. outlier:   outlier channels in fp16 + rest INT4
#   4. nsep+out:  norm-sep -> outlier-sep -> rest INT4 (full combo)
#   5. perchan:   per-channel absmax INT4
#   6. nsep+pchan: norm-sep + per-channel INT4
#
# Test on Qwen2-7B (outlier-heavy) and Pythia-6.9B (control)
# ============================================================

# === CELL 1 ===
!pip install -q transformers accelerate hf_transfer
import os; os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"

import gc, json, numpy as np, torch, torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from datetime import datetime

device = "cuda"
print(f"GPU: {torch.cuda.get_device_name()}")

# === CELL 2 ===

# ── Quantization Methods ───────────────────────────────────────────────────

def qa_perrow(x, b):
    """Standard per-row absmax quantization."""
    x = x.float()
    qm = 2**(b-1) - 1
    s = x.abs().amax(dim=-1, keepdim=True).clamp(min=1e-12) / qm
    return ((x / s).round().clamp(-qm, qm)) * s

def qa_perchan(x, b):
    """Per-channel absmax quantization (each dim gets its own scale)."""
    x = x.float()
    qm = 2**(b-1) - 1
    s = x.abs().amax(dim=0, keepdim=True).clamp(min=1e-12) / qm  # (1, head_dim)
    return ((x / s).round().clamp(-qm, qm)) * s

def qa_outlier(x, b, n_outlier=4):
    """Outlier-aware: keep top-n_outlier channels in fp16, quantize rest."""
    x = x.float()
    # Identify outlier channels by column-wise absmax
    col_absmax = x.abs().amax(dim=0)  # (head_dim,)
    _, outlier_idx = col_absmax.topk(n_outlier)
    mask = torch.zeros(x.shape[1], dtype=torch.bool, device=x.device)
    mask[outlier_idx] = True

    result = x.clone()
    # Quantize non-outlier channels
    non_outlier = x[:, ~mask]
    if non_outlier.numel() > 0:
        result[:, ~mask] = qa_perrow(non_outlier, b)
    # Outlier channels stay in fp16 (unchanged)
    return result

def norm_sep(x):
    """Separate norm from direction."""
    x = x.float()
    n = x.norm(dim=-1, keepdim=True).clamp(min=1e-12)
    d = x / n
    return n, d

def norm_recon(n, d_q):
    """Reconstruct from norm + quantized direction."""
    d_q = d_q.float()
    dn = d_q.norm(dim=-1, keepdim=True).clamp(min=1e-8)
    return n * (d_q / dn)

# ── Combined Methods ───────────────────────────────────────────────────────

def compress_naive(x, b):
    return qa_perrow(x, b)

def compress_nsep(x, b):
    n, d = norm_sep(x)
    return norm_recon(n, qa_perrow(d, b))

def compress_outlier(x, b, n_out=4):
    return qa_outlier(x, b, n_out)

def compress_nsep_outlier(x, b, n_out=4):
    n, d = norm_sep(x)
    d_q = qa_outlier(d, b, n_out)
    return norm_recon(n, d_q)

def compress_perchan(x, b):
    return qa_perchan(x, b)

def compress_nsep_perchan(x, b):
    n, d = norm_sep(x)
    return norm_recon(n, qa_perchan(d, b))

METHODS = {
    "naive4":       lambda x: compress_naive(x, 4),
    "nsep4":        lambda x: compress_nsep(x, 4),
    "outlier4":     lambda x: compress_outlier(x, 4, n_out=4),
    "outlier8":     lambda x: compress_outlier(x, 4, n_out=8),
    "nsep+out4":    lambda x: compress_nsep_outlier(x, 4, n_out=4),
    "nsep+out8":    lambda x: compress_nsep_outlier(x, 4, n_out=8),
    "perchan4":     lambda x: compress_perchan(x, 4),
    "nsep+pchan4":  lambda x: compress_nsep_perchan(x, 4),
    # Also test INT3 to see if outlier-aware fixes the anomaly
    "naive3":       lambda x: compress_naive(x, 3),
    "nsep+out4_3":  lambda x: compress_nsep_outlier(x, 3, n_out=4),
}

# ── Cache Helpers ──────────────────────────────────────────────────────────

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

def compress_cache(past, method_fn):
    cos_k, cos_v = [], []
    for li in range(n_cache_layers(past)):
        ok, ov = get_kv(past, li)
        ok, ov = ok.clone(), ov.clone()
        B, nh, sl, hd = ok.shape
        nk, nv = torch.zeros_like(ok), torch.zeros_like(ov)
        for h in range(nh):
            nk[0,h] = method_fn(ok[0,h]).to(ok.dtype)
            nv[0,h] = method_fn(ov[0,h]).to(ov.dtype)
            cos_k.append(F.cosine_similarity(ok[0,h].float(), nk[0,h].float(), dim=-1).mean().item())
            cos_v.append(F.cosine_similarity(ov[0,h].float(), nv[0,h].float(), dim=-1).mean().item())
        set_kv(past, li, nk, nv)
    return round(np.mean(cos_k), 6), round(np.mean(cos_v), 6)

def xppl(l, t):
    return float(torch.exp(F.cross_entropy(l, t, reduction="mean")).item())

# ── Text ───────────────────────────────────────────────────────────────────

TEXT = (
    "The history of AI began with myths of artificial beings. Philosophers "
    "described thinking as symbol manipulation, leading to digital computers "
    "in the 1940s. The field was founded at Dartmouth in 1956. Researchers "
    "predicted human-level AI within a generation and received millions in "
    "funding. By 1973, governments cut funding, causing the first AI winter. "
    "Expert systems revived the field in the early 1980s."
)

CL = 40

# ── Run on Both Models ─────────────────────────────────────────────────────

MODEL_CONFIGS = [
    ("Qwen2-7B", "Qwen/Qwen2-7B"),
    ("Pythia-6.9B", "EleutherAI/pythia-6.9b"),
]

all_results = {}

for mname, hf_id in MODEL_CONFIGS:
    print(f"\n{'='*70}")
    print(f"  {mname}")
    print(f"{'='*70}")

    tok = AutoTokenizer.from_pretrained(hf_id)
    mdl = AutoModelForCausalLM.from_pretrained(
        hf_id, torch_dtype=torch.float16, device_map="auto", use_safetensors=True
    )
    mdl.eval()
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    ids = tok.encode(TEXT, return_tensors="pt").to(device)
    pfl = ids.shape[1] - CL
    if pfl < 15:
        print(f"  skip (pfl={pfl})")
        del mdl; gc.collect(); torch.cuda.empty_cache()
        continue

    fi = ids[:, :pfl+CL]; tgt = fi[0, pfl:].cpu()

    with torch.inference_mode():
        bl = mdl(fi, use_cache=False).logits[0, pfl-1:-1].float().cpu()
    bp = xppl(bl, tgt)
    print(f"  Baseline PPL: {bp:.2f}, prefill={pfl}")

    # ── Layer 0 analysis first ──
    print(f"\n  Layer 0 outlier analysis:")
    with torch.inference_mode():
        po = mdl(ids[:,:pfl], use_cache=True)
        k0, v0 = get_kv(po.past_key_values, 0)
        k0_flat = k0[0].float()  # (nh, sl, hd)
        col_absmax = k0_flat.abs().amax(dim=(0,1))  # (hd,)
        top_vals, top_idx = col_absmax.topk(8)
        print(f"    Top 8 channel absmax: {[f'{v:.1f}' for v in top_vals.tolist()]}")
        print(f"    Top 8 channel indices: {top_idx.tolist()}")
        print(f"    Mean channel absmax: {col_absmax.mean():.2f}")
        print(f"    Outlier ratio (max/mean): {col_absmax.max()/col_absmax.mean():.1f}x")
    del po; gc.collect(); torch.cuda.empty_cache()

    # ── Full method comparison ──
    print(f"\n  Method comparison:")
    results = []

    for method_name, method_fn in METHODS.items():
        with torch.inference_mode():
            po = mdl(ids[:,:pfl], use_cache=True)
            past = po.past_key_values
            kc, vc = compress_cache(past, method_fn)
            co = mdl(ids[:,pfl:pfl+CL], past_key_values=past, use_cache=False)
        cl = co.logits[0].float().cpu()
        ca, ba, ta = cl[:-1], bl[:-1], tgt[1:]
        ml = min(ca.shape[0], ba.shape[0], ta.shape[0])
        dp = xppl(ca[:ml], ta[:ml]) - bp

        results.append({
            "method": method_name, "dppl": round(dp, 4),
            "kc": kc, "vc": vc,
        })
        print(f"    {method_name:>14}: ΔPPL={dp:>+10.4f}  Kcos={kc:.4f} Vcos={vc:.4f}")
        del po, co, past; gc.collect(); torch.cuda.empty_cache()

    # ── Layer-0-only compression (to isolate the fix) ──
    print(f"\n  Layer 0 only (isolating the outlier fix):")
    l0_results = []
    for method_name, method_fn in [
        ("naive4", METHODS["naive4"]),
        ("nsep4", METHODS["nsep4"]),
        ("outlier4", METHODS["outlier4"]),
        ("nsep+out4", METHODS["nsep+out4"]),
        ("perchan4", METHODS["perchan4"]),
        ("nsep+pchan4", METHODS["nsep+pchan4"]),
    ]:
        with torch.inference_mode():
            po = mdl(ids[:,:pfl], use_cache=True)
            past = po.past_key_values
            # Compress ONLY layer 0
            ok, ov = get_kv(past, 0)
            ok_c, ov_c = ok.clone(), ov.clone()
            nk, nv = torch.zeros_like(ok_c), torch.zeros_like(ov_c)
            for h in range(ok_c.shape[1]):
                nk[0,h] = method_fn(ok_c[0,h]).to(ok.dtype)
                nv[0,h] = method_fn(ov_c[0,h]).to(ov.dtype)
            set_kv(past, 0, nk, nv)
            co = mdl(ids[:,pfl:pfl+CL], past_key_values=past, use_cache=False)
        cl = co.logits[0].float().cpu()
        ca, ba, ta = cl[:-1], bl[:-1], tgt[1:]
        ml = min(ca.shape[0], ba.shape[0], ta.shape[0])
        dp = xppl(ca[:ml], ta[:ml]) - bp

        l0_results.append({"method": method_name, "dppl": round(dp, 4)})
        print(f"    {method_name:>14}: ΔPPL={dp:>+10.4f}")
        del po, co, past; gc.collect(); torch.cuda.empty_cache()

    # ── Generation ──
    print(f"\n  Generation (top-p=0.9, temp=0.8):")
    gen_ids = tok.encode(TEXT, return_tensors="pt").to(device)[:, :60]
    gen_results = []
    for method_name in ["baseline", "naive4", "nsep+out4", "nsep+pchan4"]:
        method_fn = METHODS.get(method_name)
        with torch.inference_mode():
            po = mdl(gen_ids, use_cache=True)
            past = po.past_key_values
            if method_fn:
                compress_cache(past, method_fn)
            g = []; logits = po.logits[0, -1:]
            for _ in range(60):
                probs = F.softmax(logits.float() / 0.8, dim=-1)
                sp, si = probs.sort(descending=True)
                cm = sp.cumsum(dim=-1) - sp > 0.9; sp[cm] = 0
                sp = sp / sp.sum()
                ix = torch.multinomial(sp, 1); nt = si.gather(-1, ix)
                g.append(nt[0,0].item())
                if nt[0,0].item() == tok.eos_token_id: break
                o = mdl(nt, past_key_values=past, use_cache=True)
                past = o.past_key_values; logits = o.logits[0, -1:]
        txt = tok.decode(g, skip_special_tokens=True)
        ws = txt.split()
        tri = [tuple(ws[i:i+3]) for i in range(len(ws)-2)] if len(ws)>=4 else []
        uq = len(set(tri))/len(tri) if tri else 1.0
        gen_results.append({"method": method_name, "text": txt[:120], "uq": round(uq, 3)})
        rep = " [REP!]" if uq < 0.5 else ""
        print(f"    {method_name:>14}: uq={uq:.3f}{rep}  \"{txt[:70]}\"")
        del po, past; gc.collect(); torch.cuda.empty_cache()

    all_results[mname] = {
        "baseline_ppl": round(bp, 3),
        "prefill": pfl,
        "full_model": results,
        "layer0_only": l0_results,
        "generation": gen_results,
    }
    del mdl, tok; gc.collect(); torch.cuda.empty_cache()

# ── Summary ────────────────────────────────────────────────────────────────
print("\n" + "="*70)
print("  SUMMARY: Arc + Outlier-Aware Quantization")
print("="*70)

for mname, mdata in all_results.items():
    print(f"\n  {mname} (baseline PPL={mdata['baseline_ppl']}):")
    print(f"    {'Method':>14} {'Full ΔPPL':>12} {'L0-only ΔPPL':>14}")
    full = {r["method"]: r["dppl"] for r in mdata["full_model"]}
    l0 = {r["method"]: r["dppl"] for r in mdata["layer0_only"]}
    for m in ["naive4", "nsep4", "outlier4", "nsep+out4", "perchan4", "nsep+pchan4"]:
        f_val = full.get(m, "--")
        l_val = l0.get(m, "--")
        f_str = f"{f_val:>+12.4f}" if isinstance(f_val, float) else f"{'--':>12}"
        l_str = f"{l_val:>+14.4f}" if isinstance(l_val, float) else f"{'--':>14}"
        print(f"    {m:>14} {f_str} {l_str}")

print("\n" + "="*70)
print("JSON OUTPUT:")
print("="*70)
output = {
    "experiment": "phase5c_arc_smoothquant",
    "timestamp": datetime.now().isoformat(),
    "results": all_results,
}
print(json.dumps(output, indent=2, ensure_ascii=False))
