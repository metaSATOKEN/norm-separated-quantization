#!/usr/bin/env python3
"""
Phase 4b: K/V Asymmetric + Norm-Separated Quantization

Phase 4 findings:
  - norm_pca outperforms plain PCA by 13.4x at k=48
  - Key cosine (0.92-0.99) >> Value cosine (0.82-0.90): K is more resilient to compression
  -> Verify K/V asymmetric compression + norm-separated quantization

Experiments:
  A. K/V asymmetric compression: strong compression (small k) for K, gentle compression (large k) for V
  B. Norm-separated quantization: separate norm -> quantize direction vector
     - baseline: direct quantization (naive INT8/INT4)
     - proposed: norm separation -> direction quantization -> reconstruction
  C. Combination: asymmetric + norm-separated quantization

Quantization scheme:
  - absmax symmetric: x_q = round(x / scale * (2^(b-1)-1))
  - per-token granularity (each token vector quantized independently)
"""

import sys
import gc
import json
import time
from pathlib import Path
from datetime import datetime

import numpy as np
import torch
import torch.nn.functional as F
from transformers import GPT2LMHeadModel, GPT2Tokenizer


# ── Config ──────────────────────────────────────────────────────────────────

PREFILL_TEXT = (
    "The old lighthouse keeper climbed the spiral staircase each evening, "
    "carrying a lantern that cast long shadows across the stone walls. "
    "He had performed this ritual for forty years, ever since the automated "
    "systems had failed during the great storm. The sea below crashed "
    "against the rocks with a rhythm that matched his breathing, and he "
    "found comfort in the predictability of waves and wind. Tonight, "
    "however, something was different. A strange light flickered on the "
    "horizon, pulsing with an irregular beat that made him uneasy. "
    "He set the lantern down on the iron railing and squinted into the "
    "darkness. The light grew brighter, then dimmed, then brightened again. "
    "The keeper reached for his telescope, an old brass instrument that had "
    "belonged to his grandfather, and trained it on the mysterious source. "
    "What he saw made him drop the telescope in astonishment. A ship, "
    "unlike any he had ever seen, hovered silently above the waves. Its "
    "hull gleamed with an otherworldly light, and from its deck, figures "
    "moved with an eerie grace that seemed to defy the laws of physics."
)

CONTINUATION_TEXT = (
    " The keeper stumbled backward, nearly losing his footing on the wet "
    "stone steps. His mind raced through every possible explanation, but "
    "none could account for what he was witnessing. The ship began to "
    "descend slowly toward the water, its light intensifying until the "
    "entire cove was illuminated as brightly as midday."
)


# ── Quantization Primitives ─────────────────────────────────────────────────

def quantize_absmax(x: torch.Tensor, bits: int) -> torch.Tensor:
    """
    Absmax symmetric quantization -> dequantization (simulated).
    Per-token granularity: each row quantized independently.
    """
    qmax = 2 ** (bits - 1) - 1
    # Per-row scale
    scale = x.abs().amax(dim=-1, keepdim=True).clamp(min=1e-12) / qmax
    x_q = (x / scale).round().clamp(-qmax, qmax)
    return x_q * scale  # dequantized


def quantize_norm_separated(x: torch.Tensor, bits: int) -> torch.Tensor:
    """
    Arc-style norm-separated quantization:
    1. Separate norm
    2. Quantize direction vector (unit-ish) with better dynamic range
    3. Reconstruct: norm * quantized_direction
    """
    norms = x.norm(dim=-1, keepdim=True).clamp(min=1e-12)
    directions = x / norms

    # Direction vectors are near unit norm -- tighter dynamic range -- better quantization
    directions_q = quantize_absmax(directions, bits)

    # Re-normalize to prevent drift
    directions_q = directions_q / directions_q.norm(dim=-1, keepdim=True).clamp(min=1e-12)

    # Norms stored at full precision (1 float per token, negligible cost)
    return norms * directions_q


# ── PCA Compression (from Phase 4) ─────────────────────────────────────────

def pca_truncate(x: torch.Tensor, k: int) -> torch.Tensor:
    mean = x.mean(dim=0, keepdim=True)
    centered = x - mean
    U, S, Vt = torch.linalg.svd(centered, full_matrices=False)
    actual_k = min(k, U.shape[1])
    return U[:, :actual_k] @ torch.diag(S[:actual_k]) @ Vt[:actual_k] + mean


def norm_pca_truncate(x: torch.Tensor, k: int) -> torch.Tensor:
    norms = x.norm(dim=-1, keepdim=True).clamp(min=1e-12)
    dirs = x / norms
    mean_dir = dirs.mean(dim=0, keepdim=True)
    centered = dirs - mean_dir
    U, S, Vt = torch.linalg.svd(centered, full_matrices=False)
    actual_k = min(k, U.shape[1])
    recon = U[:, :actual_k] @ torch.diag(S[:actual_k]) @ Vt[:actual_k] + mean_dir
    recon = recon / recon.norm(dim=-1, keepdim=True).clamp(min=1e-12)
    return norms * recon


# ── KV Cache Manipulation ──────────────────────────────────────────────────

def compress_kv(past, config: dict):
    """
    Apply compression config to KV cache.

    config keys:
      key_method: "none" | "pca" | "norm_pca" | "quant" | "norm_quant" | "pca+quant" | "norm_pca+quant"
      key_k: int (for pca methods)
      key_bits: int (for quant methods)
      value_method: same options
      value_k: int
      value_bits: int
      compress_layers: list[int] | None (None = all)
    """
    n_layers = len(past.layers)
    compress_layers = config.get("compress_layers") or list(range(n_layers))

    stats = {"per_layer": []}

    for li in range(n_layers):
        dl = past.layers[li]
        if li not in compress_layers:
            stats["per_layer"].append({"layer": li, "compressed": False})
            continue

        orig_k = dl.keys.clone()   # (B, n_heads, seq_len, head_dim)
        orig_v = dl.values.clone()
        B, n_heads, seq_len, hd = orig_k.shape

        new_k = torch.zeros_like(orig_k)
        new_v = torch.zeros_like(orig_v)

        cos_k_list, cos_v_list = [], []

        for h in range(n_heads):
            kh = orig_k[0, h]  # (seq_len, head_dim)
            vh = orig_v[0, h]

            new_k[0, h] = _apply_method(kh, config.get("key_method", "none"),
                                         config.get("key_k"), config.get("key_bits"))
            new_v[0, h] = _apply_method(vh, config.get("value_method", "none"),
                                         config.get("value_k"), config.get("value_bits"))

            cos_k_list.append(F.cosine_similarity(kh, new_k[0, h], dim=-1).mean().item())
            cos_v_list.append(F.cosine_similarity(vh, new_v[0, h], dim=-1).mean().item())

        dl.keys = new_k
        dl.values = new_v

        stats["per_layer"].append({
            "layer": li, "compressed": True,
            "key_cosine": round(float(np.mean(cos_k_list)), 6),
            "value_cosine": round(float(np.mean(cos_v_list)), 6),
        })

    return stats


def _apply_method(x, method, k=None, bits=None):
    if method == "none" or method is None:
        return x
    elif method == "pca":
        return pca_truncate(x, k)
    elif method == "norm_pca":
        return norm_pca_truncate(x, k)
    elif method == "quant":
        return quantize_absmax(x, bits)
    elif method == "norm_quant":
        return quantize_norm_separated(x, bits)
    elif method == "pca+quant":
        return quantize_absmax(pca_truncate(x, k), bits)
    elif method == "norm_pca+quant":
        return quantize_norm_separated(norm_pca_truncate(x, k), bits)
    else:
        raise ValueError(f"Unknown method: {method}")


# ── Evaluation ──────────────────────────────────────────────────────────────

def _ppl(logits, targets):
    return float(torch.exp(F.cross_entropy(logits, targets, reduction="mean")).item())

def _kl(base, comp):
    p = F.log_softmax(base, dim=-1)
    q = F.log_softmax(comp, dim=-1)
    return float(F.kl_div(q, p, log_target=True, reduction="batchmean").item())

def _top5(base, comp):
    tb = base.topk(5, dim=-1).indices
    tc = comp.topk(5, dim=-1).indices
    return float(np.mean([
        len(set(tb[t].tolist()) & set(tc[t].tolist())) / 5
        for t in range(tb.shape[0])
    ]))


def evaluate_config(model, prefill_ids, continuation_ids, config: dict,
                    baseline_logits=None, baseline_ppl=None) -> dict:
    """Evaluate a single compression config."""
    full_ids = torch.cat([prefill_ids, continuation_ids], dim=1)
    pf_len = prefill_ids.shape[1]
    target = full_ids[0, pf_len:].cpu()

    if baseline_logits is None:
        with torch.inference_mode():
            base_out = model(full_ids, use_cache=False)
        baseline_logits = base_out.logits[0, pf_len - 1:-1].float().cpu()
        baseline_ppl = _ppl(baseline_logits, target)

    with torch.inference_mode():
        pf_out = model(prefill_ids, use_cache=True)
        past = pf_out.past_key_values
        stats = compress_kv(past, config)
        cont_out = model(continuation_ids, past_key_values=past, use_cache=False)

    comp_logits = cont_out.logits[0].float().cpu()

    cl = comp_logits[:-1]
    bl = baseline_logits[:-1]
    tgt = target[1:]
    ml = min(cl.shape[0], bl.shape[0], tgt.shape[0])
    cl, bl, tgt = cl[:ml], bl[:ml], tgt[:ml]

    cppl = _ppl(cl, tgt)

    return {
        "config": config,
        "baseline_ppl": round(baseline_ppl, 3),
        "compressed_ppl": round(cppl, 3),
        "delta_ppl": round(cppl - baseline_ppl, 4),
        "kl_divergence": round(_kl(bl, cl), 6),
        "top5_overlap": round(_top5(bl, cl), 4),
        "cache_stats": stats,
    }


def estimate_memory(config: dict, n_layers=12, n_heads=12, seq_len=200, head_dim=64):
    """Estimate effective bits per element for a config."""
    def bits_for(method, k=None, bits=None):
        if method in ("none", None):
            return 32.0  # FP32
        elif method in ("pca", "norm_pca"):
            # Store k coefficients + basis (amortized)
            return 32.0 * (k / head_dim)
        elif method in ("quant", "norm_quant"):
            # quantized bits + scale overhead
            overhead = 32.0 / head_dim  # 1 scale per row
            if method == "norm_quant":
                overhead += 32.0 / head_dim  # 1 norm per row
            return bits + overhead
        elif method in ("pca+quant", "norm_pca+quant"):
            pca_ratio = k / head_dim
            overhead = 32.0 / k  # scale per row of k-dim
            if "norm" in method:
                overhead += 32.0 / k
            return bits * pca_ratio + overhead
        return 32.0

    key_bpe = bits_for(config.get("key_method"), config.get("key_k"), config.get("key_bits"))
    val_bpe = bits_for(config.get("value_method"), config.get("value_k"), config.get("value_bits"))
    avg_bpe = (key_bpe + val_bpe) / 2
    compression = 32.0 / avg_bpe

    return {
        "key_bits_per_elem": round(key_bpe, 2),
        "value_bits_per_elem": round(val_bpe, 2),
        "avg_bits_per_elem": round(avg_bpe, 2),
        "effective_compression": round(compression, 2),
    }


# ── Main ────────────────────────────────────────────────────────────────────

def main():
    print("=" * 70)
    print("PHASE 4b: K/V Asymmetric + Norm-Separated Quantization")
    print("=" * 70)

    results_dir = Path(__file__).parent.parent / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    model = GPT2LMHeadModel.from_pretrained("gpt2")
    model.eval()

    prefill_ids = tokenizer.encode(PREFILL_TEXT, return_tensors="pt")
    continuation_ids = tokenizer.encode(CONTINUATION_TEXT, return_tensors="pt")
    full_ids = torch.cat([prefill_ids, continuation_ids], dim=1)
    pf_len = prefill_ids.shape[1]

    # Compute baseline once
    with torch.inference_mode():
        base_out = model(full_ids, use_cache=False)
    base_logits = base_out.logits[0, pf_len - 1:-1].float().cpu()
    target = full_ids[0, pf_len:].cpu()
    base_ppl = _ppl(base_logits, target)
    print(f"  Baseline PPL: {base_ppl:.3f}")
    print(f"  Prefill: {pf_len}, Continuation: {continuation_ids.shape[1]}")

    out = {
        "experiment": "phase4b_asymmetric_quantization",
        "version": "v2.0",
        "timestamp": datetime.now().isoformat(),
        "model": "GPT-2",
        "baseline_ppl": round(base_ppl, 3),
    }

    # ════════════════════════════════════════════════════════════════════════
    # Experiment A: K/V Asymmetric PCA
    # ════════════════════════════════════════════════════════════════════════
    print(f"\n{'━' * 60}")
    print(f"  Exp A: K/V Asymmetric PCA (norm_pca)")
    print(f"  Key gets strong compression, Value gets gentle")
    print(f"{'━' * 60}")

    asymmetric_results = []
    # (key_k, value_k) pairs
    asym_pairs = [
        # Reference: symmetric
        (16, 16, "symmetric k=16"),
        (32, 32, "symmetric k=32"),
        (48, 48, "symmetric k=48"),
        # Asymmetric: aggressive K, gentle V
        (8,  48, "asym K=8 V=48"),
        (8,  56, "asym K=8 V=56"),
        (16, 48, "asym K=16 V=48"),
        (16, 56, "asym K=16 V=56"),
        (32, 48, "asym K=32 V=48"),
        (32, 56, "asym K=32 V=56"),
        # Extreme
        (4,  56, "asym K=4 V=56"),
        (4,  60, "asym K=4 V=60"),
        (8,  60, "asym K=8 V=60"),
    ]

    for kk, vk, label in asym_pairs:
        config = {
            "key_method": "norm_pca", "key_k": kk,
            "value_method": "norm_pca", "value_k": vk,
        }
        r = evaluate_config(model, prefill_ids, continuation_ids, config,
                           base_logits, base_ppl)
        mem = estimate_memory(config)
        r["memory"] = mem
        r["label"] = label
        asymmetric_results.append(r)

        print(f"  {label:>20}: ΔPPL={r['delta_ppl']:>+8.4f}  "
              f"top5={r['top5_overlap']:.3f}  "
              f"~{mem['effective_compression']:.1f}x")

    out["asymmetric_pca"] = asymmetric_results

    # ════════════════════════════════════════════════════════════════════════
    # Experiment B: Quantization comparison
    # ════════════════════════════════════════════════════════════════════════
    print(f"\n{'━' * 60}")
    print(f"  Exp B: Quantization -- naive vs norm-separated")
    print(f"{'━' * 60}")

    quant_results = []
    for bits in [8, 4, 3, 2]:
        # Naive quantization
        cfg_naive = {
            "key_method": "quant", "key_bits": bits,
            "value_method": "quant", "value_bits": bits,
        }
        r_naive = evaluate_config(model, prefill_ids, continuation_ids, cfg_naive,
                                  base_logits, base_ppl)
        r_naive["label"] = f"naive INT{bits}"
        r_naive["memory"] = estimate_memory(cfg_naive)
        quant_results.append(r_naive)

        # Norm-separated quantization
        cfg_norm = {
            "key_method": "norm_quant", "key_bits": bits,
            "value_method": "norm_quant", "value_bits": bits,
        }
        r_norm = evaluate_config(model, prefill_ids, continuation_ids, cfg_norm,
                                 base_logits, base_ppl)
        r_norm["label"] = f"norm_sep INT{bits}"
        r_norm["memory"] = estimate_memory(cfg_norm)
        quant_results.append(r_norm)

        ratio = abs(r_naive["delta_ppl"]) / abs(r_norm["delta_ppl"]) if r_norm["delta_ppl"] != 0 else float("inf")
        print(f"  INT{bits}: naive ΔPPL={r_naive['delta_ppl']:>+8.4f}  "
              f"norm_sep ΔPPL={r_norm['delta_ppl']:>+8.4f}  "
              f"advantage={ratio:.1f}x")

    out["quantization"] = quant_results

    # ════════════════════════════════════════════════════════════════════════
    # Experiment C: Asymmetric + quantization (K/V different bits)
    # ════════════════════════════════════════════════════════════════════════
    print(f"\n{'━' * 60}")
    print(f"  Exp C: K/V Asymmetric quantization")
    print(f"{'━' * 60}")

    asym_quant_results = []
    asym_quant_configs = [
        # (key_method, key_bits, val_method, val_bits, label)
        ("norm_quant", 4, "norm_quant", 8, "K:norm4 V:norm8"),
        ("norm_quant", 3, "norm_quant", 8, "K:norm3 V:norm8"),
        ("norm_quant", 2, "norm_quant", 8, "K:norm2 V:norm8"),
        ("norm_quant", 4, "norm_quant", 4, "K:norm4 V:norm4 (sym)"),
        ("norm_quant", 3, "norm_quant", 4, "K:norm3 V:norm4"),
        ("norm_quant", 2, "norm_quant", 4, "K:norm2 V:norm4"),
        ("quant", 4, "quant", 8, "K:naive4 V:naive8"),
        ("quant", 4, "quant", 4, "K:naive4 V:naive4 (sym)"),
    ]

    for km, kb, vm, vb, label in asym_quant_configs:
        cfg = {"key_method": km, "key_bits": kb, "value_method": vm, "value_bits": vb}
        r = evaluate_config(model, prefill_ids, continuation_ids, cfg,
                           base_logits, base_ppl)
        r["label"] = label
        r["memory"] = estimate_memory(cfg)
        asym_quant_results.append(r)

        print(f"  {label:>25}: ΔPPL={r['delta_ppl']:>+8.4f}  "
              f"~{r['memory']['avg_bits_per_elem']:.1f}bpe  "
              f"top5={r['top5_overlap']:.3f}")

    out["asymmetric_quantization"] = asym_quant_results

    # ════════════════════════════════════════════════════════════════════════
    # Experiment D: Combined PCA + quantization
    # ════════════════════════════════════════════════════════════════════════
    print(f"\n{'━' * 60}")
    print(f"  Exp D: PCA + Quantization combined")
    print(f"{'━' * 60}")

    combined_results = []
    combined_configs = [
        # Best from each axis
        {"key_method": "norm_pca+quant", "key_k": 32, "key_bits": 8,
         "value_method": "norm_pca+quant", "value_k": 48, "value_bits": 8,
         "label": "norm_pca(K32V48)+INT8"},
        {"key_method": "norm_pca+quant", "key_k": 16, "key_bits": 8,
         "value_method": "norm_pca+quant", "value_k": 48, "value_bits": 8,
         "label": "norm_pca(K16V48)+INT8"},
        {"key_method": "norm_pca+quant", "key_k": 32, "key_bits": 4,
         "value_method": "norm_pca+quant", "value_k": 48, "value_bits": 8,
         "label": "norm_pca(K32V48)+K4V8"},
        {"key_method": "norm_pca+quant", "key_k": 16, "key_bits": 4,
         "value_method": "norm_pca+quant", "value_k": 48, "value_bits": 4,
         "label": "norm_pca(K16V48)+INT4"},
        # Aggressive
        {"key_method": "norm_pca+quant", "key_k": 8, "key_bits": 4,
         "value_method": "norm_pca+quant", "value_k": 32, "value_bits": 8,
         "label": "norm_pca(K8V32)+K4V8"},
    ]

    for cfg in combined_configs:
        label = cfg.pop("label")
        r = evaluate_config(model, prefill_ids, continuation_ids, cfg,
                           base_logits, base_ppl)
        r["label"] = label
        cfg["label"] = label
        r["memory"] = estimate_memory(cfg)
        combined_results.append(r)

        print(f"  {label:>30}: ΔPPL={r['delta_ppl']:>+8.4f}  "
              f"~{r['memory']['effective_compression']:.1f}x  "
              f"top5={r['top5_overlap']:.3f}")

    out["combined_pca_quant"] = combined_results

    # ════════════════════════════════════════════════════════════════════════
    # Summary: Best configs
    # ════════════════════════════════════════════════════════════════════════
    all_configs = (asymmetric_results + quant_results +
                   asym_quant_results + combined_results)

    # Sort by |DPPL| < 1.0, then by compression
    viable = [r for r in all_configs if abs(r["delta_ppl"]) < 1.0]
    viable.sort(key=lambda r: -r["memory"]["effective_compression"])

    near_viable = [r for r in all_configs if 1.0 <= abs(r["delta_ppl"]) < 3.0]
    near_viable.sort(key=lambda r: abs(r["delta_ppl"]))

    out["summary"] = {
        "viable_configs": [
            {"label": r["label"], "delta_ppl": r["delta_ppl"],
             "compression": r["memory"]["effective_compression"],
             "avg_bpe": r["memory"]["avg_bits_per_elem"]}
            for r in viable
        ],
        "near_viable_configs": [
            {"label": r["label"], "delta_ppl": r["delta_ppl"],
             "compression": r["memory"]["effective_compression"],
             "avg_bpe": r["memory"]["avg_bits_per_elem"]}
            for r in near_viable[:10]
        ],
    }

    out["elapsed_seconds"] = round(time.time() - t0, 1)

    path = results_dir / "phase4b_asymmetric_quantization.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)

    # ── Final Summary ──
    print(f"\n{'━' * 70}")
    print(f"  VIABLE (|ΔPPL| < 1.0):")
    print(f"{'━' * 70}")
    if viable:
        for r in viable:
            print(f"  {r['label']:>35}: ΔPPL={r['delta_ppl']:>+8.4f}  "
                  f"{r['memory']['effective_compression']:.1f}x  "
                  f"{r['memory']['avg_bits_per_elem']:.1f}bpe")
    else:
        print(f"  (none)")

    print(f"\n  NEAR-VIABLE (1.0 ≤ |ΔPPL| < 3.0):")
    for r in near_viable[:5]:
        print(f"  {r['label']:>35}: ΔPPL={r['delta_ppl']:>+8.4f}  "
              f"{r['memory']['effective_compression']:.1f}x  "
              f"{r['memory']['avg_bits_per_elem']:.1f}bpe")

    print(f"\n  Saved: {path}")
    print(f"  Elapsed: {out['elapsed_seconds']:.0f}s")


if __name__ == "__main__":
    main()
