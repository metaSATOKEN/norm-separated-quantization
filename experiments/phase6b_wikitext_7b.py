# ============================================================
# Phase 6b: WikiText-2 PPL (7B Models) + Figure 1 (Qwen2-7B)
# ============================================================
# Run on Colab Blackwell GPU.
#
# Part A: WikiText-2 PPL on Pythia-6.9B + Qwen2-7B
#         -- Added to paper Section 5.5 table
# Part B: Figure 1 (KV distribution histogram) on Qwen2-7B
#         -- Backup / replacement for GPT-2 version
#
# Design: Load one model at a time -- run experiment -- delete.
#         Even if one model fails, the other's results survive.
# ============================================================

# === CELL 1: Setup ===
!pip install -q transformers accelerate hf_transfer datasets matplotlib

import os
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"

import gc, json, numpy as np, torch, torch.nn.functional as F
import matplotlib
import matplotlib.pyplot as plt
matplotlib.rcParams['font.size'] = 11
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from datetime import datetime

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {device}")
if device == "cuda":
    print(f"GPU: {torch.cuda.get_device_name()}")
    props = torch.cuda.get_device_properties(0)
    vram = getattr(props, "total_memory", getattr(props, "total_mem", 0)) / 1e9
    print(f"VRAM: {vram:.1f} GB")
import transformers
print(f"transformers: {transformers.__version__}")

# Load WikiText-2 (tiny dataset, instant)
print("\nLoading WikiText-2...")
wikitext = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
wiki_text = "\n\n".join([t for t in wikitext["text"] if len(t.strip()) > 50][:50])
print(f"WikiText-2: {len(wiki_text)} chars ready")

# === CELL 2: Experiments ===

# ── Quantization (same as all previous phases, FIXED dim=-1) ───────────────

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
    if name == "naive4":
        return qa_perrow(x, 4)
    if name == "nsep+pchan4":
        x = x.float()
        n = x.norm(dim=-1, keepdim=True).clamp(min=1e-12)  # FIXED: dim=-1
        d = x / n
        dq = qa_perchan(d, 4)
        dq = dq / dq.norm(dim=-1, keepdim=True).clamp(min=1e-8)
        return n * dq
    raise ValueError(name)

# ── Cache helpers (DynamicCache + tuple compatible) ────────────────────────

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
    for li in range(n_cache_layers(past)):
        ok, ov = get_kv(past, li)
        ok, ov = ok.clone(), ov.clone()
        nk, nv = torch.zeros_like(ok), torch.zeros_like(ov)
        for h in range(ok.shape[1]):
            nk[0,h] = apply_method(ok[0,h], method_name).to(ok.dtype)
            nv[0,h] = apply_method(ov[0,h], method_name).to(ov.dtype)
        set_kv(past, li, nk, nv)

# ── WikiText-2 evaluation function ────────────────────────────────────────

def eval_wikitext(model_name, hf_id, dtype):
    """Run WikiText-2 evaluation for a single model. Returns results dict or None on failure."""
    print(f"\n{'━'*60}")
    print(f"  {model_name} -- WikiText-2")
    print(f"{'━'*60}")

    try:
        tok = AutoTokenizer.from_pretrained(hf_id)
        mdl = AutoModelForCausalLM.from_pretrained(
            hf_id, torch_dtype=dtype, device_map="auto", use_safetensors=True
        )
    except Exception as e:
        try:
            mdl = AutoModelForCausalLM.from_pretrained(
                hf_id, torch_dtype=dtype, device_map="auto"
            )
        except Exception as e2:
            print(f"  FAILED to load: {e2}")
            return None

    mdl.eval()
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token or "<|endoftext|>"

    # Tokenize WikiText-2
    ids = tok.encode(wiki_text, return_tensors="pt").to(device)
    max_len = getattr(mdl.config, "n_positions", None) or getattr(
        mdl.config, "max_position_embeddings", 2048)
    if ids.shape[1] > max_len:
        ids = ids[:, :max_len]
    total = ids.shape[1]

    CL = 30
    chunk_size = 150
    n_chunks = min(5, (total - CL) // chunk_size)
    print(f"  Tokens: {total}, Chunks: {n_chunks} x {chunk_size}")

    results = {"naive4": [], "nsep+pchan4": []}
    baselines = []

    for ci in range(n_chunks):
        start = ci * chunk_size
        end = start + chunk_size + CL
        if end > total:
            break

        chunk = ids[:, start:end]
        tgt = chunk[0, chunk_size:].cpu()

        # Baseline
        with torch.inference_mode():
            bl = mdl(chunk, use_cache=False).logits[0, chunk_size-1:-1].float().cpu()
        bp = float(torch.exp(F.cross_entropy(bl, tgt, reduction="mean")).item())
        baselines.append(bp)

        # Methods
        for mn in ["naive4", "nsep+pchan4"]:
            with torch.inference_mode():
                po = mdl(chunk[:, :chunk_size], use_cache=True)
                past = po.past_key_values
                compress_cache(past, mn)
                co = mdl(chunk[:, chunk_size:], past_key_values=past, use_cache=False)
            cl = co.logits[0].float().cpu()
            ca, ba, ta = cl[:-1], bl[:-1], tgt[1:]
            ml = min(ca.shape[0], ba.shape[0], ta.shape[0])
            dp = float(torch.exp(F.cross_entropy(ca[:ml], ta[:ml], reduction="mean")).item()) - bp
            results[mn].append(dp)
            del po, co, past; gc.collect()
            if device == "cuda": torch.cuda.empty_cache()

        del bl; gc.collect()
        print(f"    chunk {ci}: base={bp:.1f}  naive4={results['naive4'][-1]:+.3f}  nsep+pc={results['nsep+pchan4'][-1]:+.3f}")

    mean_bp = np.mean(baselines)
    mean_naive = np.mean(results["naive4"])
    mean_nsep = np.mean(results["nsep+pchan4"])
    imp = abs(mean_naive) / abs(mean_nsep) if abs(mean_nsep) > 0.01 else float("inf")

    print(f"\n  RESULT ({model_name}, {n_chunks} chunks avg):")
    print(f"    Baseline PPL:     {mean_bp:.2f}")
    print(f"    naive4 ΔPPL:      {mean_naive:+.4f}")
    print(f"    nsep+pchan4 ΔPPL: {mean_nsep:+.4f}  ({imp:.1f}x)")

    result = {
        "model": model_name, "hf_id": hf_id,
        "baseline_ppl": round(mean_bp, 2),
        "naive4_dppl": round(mean_naive, 4),
        "nsep_pchan4_dppl": round(mean_nsep, 4),
        "improvement": round(imp, 2),
        "n_chunks": n_chunks,
        "per_chunk": {
            "baselines": [round(v, 2) for v in baselines],
            "naive4": [round(v, 4) for v in results["naive4"]],
            "nsep_pchan4": [round(v, 4) for v in results["nsep+pchan4"]],
        },
    }

    del mdl, tok; gc.collect()
    if device == "cuda": torch.cuda.empty_cache()
    return result

# ════════════════════════════════════════════════════════════════════════════
# PART A: WikiText-2 PPL
# ════════════════════════════════════════════════════════════════════════════
print("=" * 60)
print("PART A: WikiText-2 PPL Benchmark (7B Models)")
print("=" * 60)

all_wiki_results = []

# Model 1: Pythia-6.9B
r1 = eval_wikitext("Pythia-6.9B", "EleutherAI/pythia-6.9b", torch.float16)
if r1:
    all_wiki_results.append(r1)

# Model 2: Qwen2-7B
r2 = eval_wikitext("Qwen2-7B", "Qwen/Qwen2-7B", torch.float16)
if r2:
    all_wiki_results.append(r2)

# ── Combined table with GPT-2 (from Phase 6) ──
print(f"\n{'='*60}")
print("WikiText-2 Complete Table (including GPT-2 from Phase 6)")
print(f"{'='*60}")

# GPT-2 from yesterday
gpt2_result = {"model": "GPT-2", "baseline_ppl": 70.36, "naive4_dppl": 1.5955, "nsep_pchan4_dppl": 1.1281, "improvement": 1.41}
all_for_table = [gpt2_result] + all_wiki_results

print(f"\n  {'Model':<15} {'Base PPL':>9} {'naive4':>10} {'nsep+pc':>10} {'improve':>8}")
print(f"  {'-'*55}")
for r in all_for_table:
    n = r["naive4_dppl"]; s = r["nsep_pchan4_dppl"]
    imp = f"{abs(n)/abs(s):.1f}x" if abs(s) > 0.01 else "--"
    print(f"  {r['model']:<15} {r['baseline_ppl']:>9.2f} {n:>+10.4f} {s:>+10.4f} {imp:>8}")

# ════════════════════════════════════════════════════════════════════════════
# PART B: Figure 1 with Qwen2-7B (if it loaded successfully)
# ════════════════════════════════════════════════════════════════════════════

if r2 is not None:
    print(f"\n{'='*60}")
    print("PART B: Figure 1 -- Qwen2-7B KV Distribution")
    print(f"{'='*60}")

    tok_q = AutoTokenizer.from_pretrained("Qwen/Qwen2-7B")
    mdl_q = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen2-7B", torch_dtype=torch.float16, device_map="auto", use_safetensors=True
    )
    mdl_q.eval()
    if tok_q.pad_token is None: tok_q.pad_token = tok_q.eos_token

    TEXT = (
        "The old lighthouse keeper climbed the spiral staircase each evening, "
        "carrying a lantern that cast long shadows across the stone walls. "
        "He had performed this ritual for forty years, ever since the automated "
        "systems had failed during the great storm."
    )
    fig_ids = tok_q.encode(TEXT, return_tensors="pt").to(device)

    with torch.inference_mode():
        fig_out = mdl_q(fig_ids, use_cache=True)
    past = fig_out.past_key_values

    # Layer 0
    k0 = get_kv(past, 0)[0][0].float()  # (n_heads, seq_len, head_dim)
    k0_vals = k0.cpu().numpy().flatten()
    norms0 = k0.norm(dim=-1, keepdim=True).clamp(min=1e-12)
    dirs0 = (k0 / norms0).cpu().numpy().flatten()

    # Mid layer
    mid = n_cache_layers(past) // 2
    k_mid = get_kv(past, mid)[0][0].float()
    k_mid_vals = k_mid.cpu().numpy().flatten()
    norms_mid = k_mid.norm(dim=-1, keepdim=True).clamp(min=1e-12)
    dirs_mid = (k_mid / norms_mid).cpu().numpy().flatten()

    del fig_out, past, mdl_q, tok_q; gc.collect(); torch.cuda.empty_cache()

    print(f"  Layer 0:  raw [{k0_vals.min():.1f}, {k0_vals.max():.1f}] std={k0_vals.std():.2f}")
    print(f"            sep [{dirs0.min():.3f}, {dirs0.max():.3f}] std={dirs0.std():.4f}")
    print(f"  Layer {mid}: raw [{k_mid_vals.min():.1f}, {k_mid_vals.max():.1f}] std={k_mid_vals.std():.2f}")
    print(f"            sep [{dirs_mid.min():.3f}, {dirs_mid.max():.3f}] std={dirs_mid.std():.4f}")
    print(f"  Dynamic range compression: {k0_vals.std() / dirs0.std():.1f}x")

    # Plot
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    ax = axes[0, 0]
    ax.hist(k0_vals, bins=200, color='#e74c3c', alpha=0.7, edgecolor='none', density=True)
    ax.set_title('(a) Layer 0 Key -- Raw Values')
    ax.set_xlabel('Value'); ax.set_ylabel('Density')
    rng = max(abs(k0_vals.min()), abs(k0_vals.max())) * 1.1
    ax.set_xlim(-rng, rng)
    ax.annotate(f'range: [{k0_vals.min():.0f}, {k0_vals.max():.0f}]\nstd: {k0_vals.std():.1f}',
                xy=(0.97, 0.95), xycoords='axes fraction', ha='right', va='top',
                fontsize=10, bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))

    ax = axes[0, 1]
    ax.hist(dirs0, bins=200, color='#2ecc71', alpha=0.7, edgecolor='none', density=True)
    ax.set_title('(b) Layer 0 Key -- After Norm Separation')
    ax.set_xlabel('Value'); ax.set_ylabel('Density')
    ax.set_xlim(-0.5, 0.5)
    ax.annotate(f'range: [{dirs0.min():.2f}, {dirs0.max():.2f}]\nstd: {dirs0.std():.3f}',
                xy=(0.97, 0.95), xycoords='axes fraction', ha='right', va='top',
                fontsize=10, bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))

    ax = axes[1, 0]
    ax.hist(k_mid_vals, bins=200, color='#3498db', alpha=0.7, edgecolor='none', density=True)
    ax.set_title(f'(c) Layer {mid} Key -- Raw Values')
    ax.set_xlabel('Value'); ax.set_ylabel('Density')
    rng2 = max(abs(k_mid_vals.min()), abs(k_mid_vals.max())) * 1.1
    ax.set_xlim(-rng2, rng2)
    ax.annotate(f'range: [{k_mid_vals.min():.1f}, {k_mid_vals.max():.1f}]\nstd: {k_mid_vals.std():.2f}',
                xy=(0.97, 0.95), xycoords='axes fraction', ha='right', va='top',
                fontsize=10, bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))

    ax = axes[1, 1]
    ax.hist(dirs_mid, bins=200, color='#2ecc71', alpha=0.7, edgecolor='none', density=True)
    ax.set_title(f'(d) Layer {mid} Key -- After Norm Separation')
    ax.set_xlabel('Value'); ax.set_ylabel('Density')
    ax.set_xlim(-0.5, 0.5)
    ax.annotate(f'range: [{dirs_mid.min():.2f}, {dirs_mid.max():.2f}]\nstd: {dirs_mid.std():.3f}',
                xy=(0.97, 0.95), xycoords='axes fraction', ha='right', va='top',
                fontsize=10, bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))

    plt.suptitle('Figure 1: Dynamic Range Compression via Norm Separation (Qwen2-7B)',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig('figure1_qwen2_7b.png', dpi=200, bbox_inches='tight')
    plt.savefig('figure1_qwen2_7b.pdf', bbox_inches='tight')
    plt.show()
    print("  Saved: figure1_qwen2_7b.png / .pdf")

    r2["figure1"] = {
        "layer0_raw_range": [float(k0_vals.min()), float(k0_vals.max())],
        "layer0_normsep_range": [float(dirs0.min()), float(dirs0.max())],
        "layer0_raw_std": float(k0_vals.std()),
        "layer0_normsep_std": float(dirs0.std()),
        "dynamic_range_compression": round(float(k0_vals.std() / dirs0.std()), 1),
    }
else:
    print("\nPART B skipped (Qwen2-7B failed to load)")

# ════════════════════════════════════════════════════════════════════════════
# JSON Output
# ════════════════════════════════════════════════════════════════════════════
print(f"\n{'='*60}")
print("JSON OUTPUT:")
print(f"{'='*60}")

output = {
    "experiment": "phase6b_wikitext_7b",
    "timestamp": datetime.now().isoformat(),
    "device": device,
    "wikitext2_results": all_wiki_results,
    "complete_table": all_for_table,
}
print(json.dumps(output, indent=2, ensure_ascii=False))
