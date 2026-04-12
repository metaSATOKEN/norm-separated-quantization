#!/usr/bin/env python3
"""
Phase 6: Figure 1 (Distribution Histogram) + WikiText-2 PPL
ローカル M1 版。GPT-2 のみ。
"""

import gc, json, sys, numpy as np, torch, torch.nn.functional as F
from pathlib import Path
from datetime import datetime

# matplotlib setup
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
matplotlib.rcParams['font.size'] = 11

from transformers import GPT2LMHeadModel, GPT2Tokenizer
from datasets import load_dataset

results_dir = Path(__file__).parent.parent / "results"
results_dir.mkdir(parents=True, exist_ok=True)
paper_dir = Path(__file__).parent.parent / "paper"
paper_dir.mkdir(parents=True, exist_ok=True)

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

def compress_cache(past, method_name):
    for li in range(len(past.layers)):
        dl = past.layers[li]
        ok, ov = dl.keys.clone(), dl.values.clone()
        nk, nv = torch.zeros_like(ok), torch.zeros_like(ov)
        for h in range(ok.shape[1]):
            nk[0,h] = apply_method(ok[0,h], method_name).to(ok.dtype)
            nv[0,h] = apply_method(ov[0,h], method_name).to(ov.dtype)
        dl.keys, dl.values = nk, nv


# ════════════════════════════════════════════════════════════════════════════
# PART A: Figure 1
# ════════════════════════════════════════════════════════════════════════════
print("=" * 60)
print("PART A: Figure 1 — KV Vector Distribution (GPT-2)")
print("=" * 60)

tok = GPT2Tokenizer.from_pretrained("gpt2")
mdl = GPT2LMHeadModel.from_pretrained("gpt2")
mdl.eval()

TEXT = (
    "The old lighthouse keeper climbed the spiral staircase each evening, "
    "carrying a lantern that cast long shadows across the stone walls. "
    "He had performed this ritual for forty years, ever since the automated "
    "systems had failed during the great storm. The sea below crashed "
    "against the rocks with a rhythm that matched his breathing, and he "
    "found comfort in the predictability of waves and wind."
)

ids = tok.encode(TEXT, return_tensors="pt")

with torch.inference_mode():
    out = mdl(ids, use_cache=True)
past = out.past_key_values

# Layer 0 (embedding-adjacent, often has outliers)
k0 = past.layers[0].keys[0].float()  # (n_heads, seq_len, head_dim)
k0_vals = k0.cpu().numpy().flatten()
norms0 = k0.norm(dim=-1, keepdim=True).clamp(min=1e-12)
dirs0 = (k0 / norms0).cpu().numpy().flatten()

# Mid layer
mid = len(past.layers) // 2
k_mid = past.layers[mid].keys[0].float()
k_mid_vals = k_mid.cpu().numpy().flatten()
norms_mid = k_mid.norm(dim=-1, keepdim=True).clamp(min=1e-12)
dirs_mid = (k_mid / norms_mid).cpu().numpy().flatten()

del out, past; gc.collect()

print(f"  Layer 0:  raw range [{k0_vals.min():.1f}, {k0_vals.max():.1f}], "
      f"normsep range [{dirs0.min():.3f}, {dirs0.max():.3f}]")
print(f"  Layer {mid}: raw range [{k_mid_vals.min():.1f}, {k_mid_vals.max():.1f}], "
      f"normsep range [{dirs_mid.min():.3f}, {dirs_mid.max():.3f}]")

# ── Plot ──
fig, axes = plt.subplots(2, 2, figsize=(12, 8))

ax = axes[0, 0]
ax.hist(k0_vals, bins=200, color='#e74c3c', alpha=0.7, edgecolor='none', density=True)
ax.set_title('(a) Layer 0 Key — Raw Values')
ax.set_xlabel('Value'); ax.set_ylabel('Density')
rng = max(abs(k0_vals.min()), abs(k0_vals.max())) * 1.1
ax.set_xlim(-rng, rng)
ax.annotate(f'range: [{k0_vals.min():.1f}, {k0_vals.max():.1f}]\nstd: {k0_vals.std():.2f}',
            xy=(0.97, 0.95), xycoords='axes fraction', ha='right', va='top',
            fontsize=10, bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))

ax = axes[0, 1]
ax.hist(dirs0, bins=200, color='#2ecc71', alpha=0.7, edgecolor='none', density=True)
ax.set_title('(b) Layer 0 Key — After Norm Separation')
ax.set_xlabel('Value'); ax.set_ylabel('Density')
ax.set_xlim(-0.5, 0.5)
ax.annotate(f'range: [{dirs0.min():.3f}, {dirs0.max():.3f}]\nstd: {dirs0.std():.4f}',
            xy=(0.97, 0.95), xycoords='axes fraction', ha='right', va='top',
            fontsize=10, bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))

ax = axes[1, 0]
ax.hist(k_mid_vals, bins=200, color='#3498db', alpha=0.7, edgecolor='none', density=True)
ax.set_title(f'(c) Layer {mid} Key — Raw Values')
ax.set_xlabel('Value'); ax.set_ylabel('Density')
rng2 = max(abs(k_mid_vals.min()), abs(k_mid_vals.max())) * 1.1
ax.set_xlim(-rng2, rng2)
ax.annotate(f'range: [{k_mid_vals.min():.1f}, {k_mid_vals.max():.1f}]\nstd: {k_mid_vals.std():.2f}',
            xy=(0.97, 0.95), xycoords='axes fraction', ha='right', va='top',
            fontsize=10, bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))

ax = axes[1, 1]
ax.hist(dirs_mid, bins=200, color='#2ecc71', alpha=0.7, edgecolor='none', density=True)
ax.set_title(f'(d) Layer {mid} Key — After Norm Separation')
ax.set_xlabel('Value'); ax.set_ylabel('Density')
ax.set_xlim(-0.5, 0.5)
ax.annotate(f'range: [{dirs_mid.min():.3f}, {dirs_mid.max():.3f}]\nstd: {dirs_mid.std():.4f}',
            xy=(0.97, 0.95), xycoords='axes fraction', ha='right', va='top',
            fontsize=10, bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))

plt.suptitle('Figure 1: Dynamic Range Compression via Norm Separation (GPT-2)',
             fontsize=13, fontweight='bold')
plt.tight_layout()

fig1_path = paper_dir / "figure1_distribution.png"
fig1_pdf = paper_dir / "figure1_distribution.pdf"
plt.savefig(fig1_path, dpi=200, bbox_inches='tight')
plt.savefig(fig1_pdf, bbox_inches='tight')
plt.close()
print(f"  Saved: {fig1_path}")
print(f"  Saved: {fig1_pdf}")

# ── Compact 1-row version ──
fig2, axes2 = plt.subplots(1, 2, figsize=(10, 3.5))

ax = axes2[0]
ax.hist(k0_vals, bins=150, color='#e74c3c', alpha=0.7, edgecolor='none', density=True)
ax.set_title('Raw KV Values (Layer 0)')
ax.set_xlabel('Value'); ax.set_xlim(-rng, rng)
ax.annotate(f'range: [{k0_vals.min():.1f}, {k0_vals.max():.1f}]',
            xy=(0.5, 0.95), xycoords='axes fraction', ha='center', va='top', fontsize=9)

ax = axes2[1]
ax.hist(dirs0, bins=150, color='#2ecc71', alpha=0.7, edgecolor='none', density=True)
ax.set_title('After Norm Separation (Layer 0)')
ax.set_xlabel('Value'); ax.set_xlim(-0.5, 0.5)
ax.annotate(f'range: [{dirs0.min():.3f}, {dirs0.max():.3f}]',
            xy=(0.5, 0.95), xycoords='axes fraction', ha='center', va='top', fontsize=9)

plt.suptitle('Norm Separation Compresses Dynamic Range (GPT-2)', fontsize=12, fontweight='bold')
plt.tight_layout()
fig2_path = paper_dir / "figure1_compact.png"
plt.savefig(fig2_path, dpi=200, bbox_inches='tight')
plt.close()
print(f"  Saved: {fig2_path}")


# ════════════════════════════════════════════════════════════════════════════
# PART B: WikiText-2 PPL
# ════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("PART B: WikiText-2 PPL Benchmark (GPT-2)")
print("=" * 60)

print("  Loading WikiText-2...")
dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
wiki_text = "\n\n".join([t for t in dataset["text"] if len(t.strip()) > 50][:50])
print(f"  WikiText-2: {len(wiki_text)} chars")

ids_w = tok.encode(wiki_text, return_tensors="pt")
max_len = 1024
if ids_w.shape[1] > max_len:
    ids_w = ids_w[:, :max_len]
total = ids_w.shape[1]
print(f"  Tokens: {total}")

CL = 30
chunk_size = 150
n_chunks = min(5, (total - CL) // chunk_size)
print(f"  Chunks: {n_chunks} x {chunk_size} tokens")

results_wiki = {"naive4": [], "nsep+pchan4": []}
baselines = []

for ci in range(n_chunks):
    start = ci * chunk_size
    pfl = chunk_size
    end = start + pfl + CL
    if end > total: break

    chunk = ids_w[:, start:end]
    tgt = chunk[0, pfl:].cpu()

    with torch.inference_mode():
        bl = mdl(chunk, use_cache=False).logits[0, pfl-1:-1].float().cpu()
    bp = float(torch.exp(F.cross_entropy(bl, tgt, reduction="mean")).item())
    baselines.append(bp)

    for mn in ["naive4", "nsep+pchan4"]:
        with torch.inference_mode():
            po = mdl(chunk[:, :pfl], use_cache=True)
            past = po.past_key_values
            compress_cache(past, mn)
            co = mdl(chunk[:, pfl:], past_key_values=past, use_cache=False)
        cl = co.logits[0].float().cpu()
        ca, ba, ta = cl[:-1], bl[:-1], tgt[1:]
        ml = min(ca.shape[0], ba.shape[0], ta.shape[0])
        dp = float(torch.exp(F.cross_entropy(ca[:ml], ta[:ml], reduction="mean")).item()) - bp
        results_wiki[mn].append(dp)
        del po, co, past; gc.collect()

    del bl; gc.collect()

mean_bp = np.mean(baselines)
mean_naive = np.mean(results_wiki["naive4"])
mean_nsep = np.mean(results_wiki["nsep+pchan4"])
imp = abs(mean_naive) / abs(mean_nsep) if abs(mean_nsep) > 0.01 else float("inf")

print(f"\n  WikiText-2 Results (GPT-2, {n_chunks} chunks avg):")
print(f"    Baseline PPL:     {mean_bp:.2f}")
print(f"    naive4 ΔPPL:      {mean_naive:+.4f}")
print(f"    nsep+pchan4 ΔPPL: {mean_nsep:+.4f}  ({imp:.1f}x improvement)")

# Per-chunk detail
print(f"\n  Per-chunk detail:")
for ci in range(n_chunks):
    print(f"    chunk {ci}: base={baselines[ci]:.1f}  "
          f"naive4={results_wiki['naive4'][ci]:+.3f}  "
          f"nsep+pc={results_wiki['nsep+pchan4'][ci]:+.3f}")

# ── Save JSON ──
output = {
    "experiment": "phase6_figure1_wikitext",
    "timestamp": datetime.now().isoformat(),
    "figure1": {
        "model": "GPT-2",
        "layer0_raw_range": [float(k0_vals.min()), float(k0_vals.max())],
        "layer0_normsep_range": [float(dirs0.min()), float(dirs0.max())],
        "layer0_raw_std": float(k0_vals.std()),
        "layer0_normsep_std": float(dirs0.std()),
        "dynamic_range_compression": round(float(k0_vals.std() / dirs0.std()), 1),
    },
    "wikitext2": {
        "model": "GPT-2",
        "n_chunks": n_chunks,
        "baseline_ppl": round(mean_bp, 2),
        "naive4_dppl": round(mean_naive, 4),
        "nsep_pchan4_dppl": round(mean_nsep, 4),
        "improvement": round(imp, 2),
        "per_chunk_naive4": [round(v, 4) for v in results_wiki["naive4"]],
        "per_chunk_nsep_pchan4": [round(v, 4) for v in results_wiki["nsep+pchan4"]],
    },
}

json_path = results_dir / "phase6_figure1_wikitext.json"
with open(json_path, "w") as f:
    json.dump(output, f, indent=2, ensure_ascii=False)

print(f"\n  Saved: {json_path}")
print(f"\n  Done!")
