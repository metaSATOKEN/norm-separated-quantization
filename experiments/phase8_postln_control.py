#!/usr/bin/env python3
"""
Phase 8: Post-LN Negative Control Experiment

Instead of monkey-patching GPT-2 to Post-LN (which breaks internal APIs),
we use a simpler approach: apply LayerNorm AFTER the KV cache is computed,
simulating the effect of Post-LN on the KV vector distribution.

In Pre-LN: KV = project(LN(x))     -> norm-dominant, high dynamic range
In Post-LN: KV = project(x)        -> then LN(x + attn_output)

The key difference is that Pre-LN KV vectors inherit the un-normalized
hidden state's norm variation, while Post-LN KV vectors come from
already-normalized hidden states. We simulate this by normalizing
the KV cache vectors to unit variance before quantization.

This gives us a controlled comparison:
- "Pre-LN style" KV cache: raw (high norm variation)
- "Post-LN style" KV cache: pre-normalized (low norm variation)
- Does nsep+pchan4 help more on the high-variation version?

Runs locally on M1 (GPT-2 only).
"""

import gc, json, numpy as np, torch, torch.nn.functional as F
from pathlib import Path
from datetime import datetime
from sklearn.decomposition import PCA
from scipy.stats import pearsonr
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from datasets import load_dataset

results_dir = Path(__file__).parent.parent / "results"
results_dir.mkdir(parents=True, exist_ok=True)

# ── Quantization ───────────────────────────────────────────────────────────

def qa_perrow(x, b):
    x = x.float(); qm = 2**(b-1)-1
    s = x.abs().amax(dim=-1, keepdim=True).clamp(min=1e-12) / qm
    return ((x/s).round().clamp(-qm, qm)) * s

def qa_perchan(x, b):
    x = x.float(); qm = 2**(b-1)-1
    s = x.abs().amax(dim=0, keepdim=True).clamp(min=1e-12) / qm
    return ((x/s).round().clamp(-qm, qm)) * s

def apply_method(x, name):
    if name == "naive4": return qa_perrow(x, 4)
    if name == "nsep+pchan4":
        x = x.float()
        n = x.norm(dim=-1, keepdim=True).clamp(min=1e-12)
        d = x / n
        dq = qa_perchan(d, 4)
        dq = dq / dq.norm(dim=-1, keepdim=True).clamp(min=1e-8)
        return n * dq
    raise ValueError(name)

def compress_cache(past, mn, pre_normalize=False):
    """
    Compress KV cache.
    If pre_normalize=True, normalize each head's KV to zero mean / unit variance
    before quantization (simulating Post-LN distribution).
    """
    for li in range(len(past.layers)):
        dl = past.layers[li]
        ok, ov = dl.keys.clone(), dl.values.clone()
        nk, nv = torch.zeros_like(ok), torch.zeros_like(ov)
        for h in range(ok.shape[1]):
            kh = ok[0,h].float()
            vh = ov[0,h].float()

            if pre_normalize:
                # Simulate Post-LN: normalize to zero mean, unit variance
                kh = (kh - kh.mean(dim=0, keepdim=True)) / kh.std(dim=0, keepdim=True).clamp(min=1e-6)
                vh = (vh - vh.mean(dim=0, keepdim=True)) / vh.std(dim=0, keepdim=True).clamp(min=1e-6)

            nk[0,h] = apply_method(kh, mn).to(ok.dtype)
            nv[0,h] = apply_method(vh, mn).to(ov.dtype)
        dl.keys, dl.values = nk, nv


# ── Main ───────────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("PHASE 8: Post-LN Negative Control")
    print("=" * 60)

    tok = GPT2Tokenizer.from_pretrained("gpt2")
    mdl = GPT2LMHeadModel.from_pretrained("gpt2")
    mdl.eval()

    # Load WikiText-2
    print("Loading WikiText-2...")
    wikitext = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    wiki_text = "\n\n".join([t for t in wikitext["text"] if len(t.strip()) > 50][:50])
    ids = tok.encode(wiki_text, return_tensors="pt")[:, :1024]

    # ── Step 1: KV cache distribution analysis ──
    print("\n--- KV Cache Distribution Analysis ---")
    text_short = "The old lighthouse keeper climbed the spiral staircase each evening."
    ids_short = tok.encode(text_short, return_tensors="pt")

    with torch.inference_mode():
        out = mdl(ids_short, use_cache=True)

    mid = len(out.past_key_values.layers) // 2
    k_raw = out.past_key_values.layers[mid].keys[0, 0].float().cpu()  # (seq_len, head_dim)

    # Raw (Pre-LN style)
    norms_raw = k_raw.norm(dim=-1).numpy()
    pca_raw = PCA(n_components=2).fit(k_raw.numpy())
    proj_raw = pca_raw.transform(k_raw.numpy())
    r_raw, _ = pearsonr(proj_raw[:, 0], norms_raw)

    # Normalized (Post-LN style)
    k_normed = (k_raw - k_raw.mean(dim=0, keepdim=True)) / k_raw.std(dim=0, keepdim=True).clamp(min=1e-6)
    norms_normed = k_normed.norm(dim=-1).numpy()
    pca_normed = PCA(n_components=2).fit(k_normed.numpy())
    proj_normed = pca_normed.transform(k_normed.numpy())
    r_normed, _ = pearsonr(proj_normed[:, 0], norms_normed)

    print(f"  Pre-LN style (raw):  PC1 var={pca_raw.explained_variance_ratio_[0]:.1%}, "
          f"|r(PC1,norm)|={abs(r_raw):.4f}, norm std={norms_raw.std():.3f}")
    print(f"  Post-LN style (norm): PC1 var={pca_normed.explained_variance_ratio_[0]:.1%}, "
          f"|r(PC1,norm)|={abs(r_normed):.4f}, norm std={norms_normed.std():.3f}")

    del out; gc.collect()

    # ── Step 2: Quantization comparison ──
    print("\n--- Quantization: Pre-LN vs Post-LN style KV ---")

    CL = 30
    chunk_size = 150
    total = ids.shape[1]
    n_chunks = min(5, (total - CL) // chunk_size)

    conditions = [
        ("Pre-LN / naive4",      "naive4",      False),
        ("Pre-LN / nsep+pchan4", "nsep+pchan4", False),
        ("Post-LN / naive4",     "naive4",      True),
        ("Post-LN / nsep+pchan4","nsep+pchan4", True),
    ]

    all_results = {c[0]: [] for c in conditions}
    baselines = []

    for ci in range(n_chunks):
        start = ci * chunk_size
        end = start + chunk_size + CL
        if end > total: break
        chunk = ids[:, start:end]
        tgt = chunk[0, chunk_size:].cpu()

        with torch.inference_mode():
            bl = mdl(chunk, use_cache=False).logits[0, chunk_size-1:-1].float().cpu()
        bp = float(torch.exp(F.cross_entropy(bl, tgt, reduction="mean")).item())
        baselines.append(bp)

        for label, mn, pre_norm in conditions:
            with torch.inference_mode():
                po = mdl(chunk[:, :chunk_size], use_cache=True)
                past = po.past_key_values
                compress_cache(past, mn, pre_normalize=pre_norm)
                co = mdl(chunk[:, chunk_size:], past_key_values=past, use_cache=False)
            cl = co.logits[0].float().cpu()
            ca, ba, ta = cl[:-1], bl[:-1], tgt[1:]
            ml = min(ca.shape[0], ba.shape[0], ta.shape[0])
            dp = float(torch.exp(F.cross_entropy(ca[:ml], ta[:ml], reduction="mean")).item()) - bp
            all_results[label].append(dp)
            del po, co, past; gc.collect()

        del bl; gc.collect()

    mean_bp = np.mean(baselines)

    print(f"\n  WikiText-2 ({n_chunks} chunks, base PPL={mean_bp:.2f}):")
    print(f"  {'Condition':<25} {'DPPL':>10} {'nsep benefit':>14}")
    print(f"  {'-'*50}")

    results_summary = {}
    for label, _, _ in conditions:
        mn = np.mean(all_results[label])
        results_summary[label] = round(mn, 4)
        print(f"  {label:<25} {mn:>+10.4f}")

    # nsep benefit
    preln_benefit = abs(results_summary["Pre-LN / naive4"]) / abs(results_summary["Pre-LN / nsep+pchan4"]) if abs(results_summary["Pre-LN / nsep+pchan4"]) > 0.01 else float("inf")
    postln_benefit = abs(results_summary["Post-LN / naive4"]) / abs(results_summary["Post-LN / nsep+pchan4"]) if abs(results_summary["Post-LN / nsep+pchan4"]) > 0.01 else float("inf")

    print(f"\n  nsep+pchan4 benefit:")
    print(f"    Pre-LN style:  {preln_benefit:.1f}x improvement")
    print(f"    Post-LN style: {postln_benefit:.1f}x improvement")
    if preln_benefit > postln_benefit:
        print(f"    -> nsep benefits Pre-LN more ({preln_benefit/postln_benefit:.1f}x ratio)")
    else:
        print(f"    -> nsep benefits both similarly")

    # ── Save ──
    output = {
        "experiment": "phase8_postln_control",
        "timestamp": datetime.now().isoformat(),
        "model": "GPT-2 (124M)",
        "approach": "Pre-normalize KV cache to simulate Post-LN distribution",
        "kv_distribution": {
            "pre_ln": {
                "pc1_variance": round(float(pca_raw.explained_variance_ratio_[0]), 4),
                "pc1_norm_corr": round(abs(float(r_raw)), 4),
                "norm_std": round(float(norms_raw.std()), 4),
            },
            "post_ln_simulated": {
                "pc1_variance": round(float(pca_normed.explained_variance_ratio_[0]), 4),
                "pc1_norm_corr": round(abs(float(r_normed)), 4),
                "norm_std": round(float(norms_normed.std()), 4),
            },
        },
        "quantization": results_summary,
        "nsep_benefit_ratio": {
            "pre_ln": round(preln_benefit, 2),
            "post_ln": round(postln_benefit, 2),
        },
    }

    path = results_dir / "phase8_postln_control.json"
    with open(path, "w") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    print(f"\n  Saved: {path}")


if __name__ == "__main__":
    main()
