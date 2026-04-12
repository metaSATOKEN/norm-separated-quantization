#!/usr/bin/env python3
"""
Phase 2 (Lightweight): Rank-Performance Curve

Phase 1 の結果を受けて、高 k 領域まで sweep し曲線の形を確認する。
目的: sweet spot（knee）があるか、単調に下がるだけかを定量化。

k 候補: 2, 4, 8, 16, 32, 64, 128, 192, 256, 384, 512
対象: GPT-2 の耐性上位層（L8, L9, L10）+ 中間層（L6）

結果: JSON + 標準出力に曲線データ
"""

import sys
import os
import gc
import json
import time
from pathlib import Path
from datetime import datetime

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from sklearn.decomposition import PCA

sys.path.insert(0, str(Path(__file__).parent.parent))
from compression.compressors import (
    compute_pca_basis,
    compute_random_basis,
    compress_norm_pca,
    decompress_norm_pca,
)


# ── Config ──────────────────────────────────────────────────────────────────

K_VALUES = [2, 4, 8, 16, 32, 64, 128, 192, 256, 384, 512]
TARGET_LAYERS = [6, 8, 9, 10]  # mid + best from Phase 1

EVAL_TEXT = (
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
    "What he saw made him drop the telescope in astonishment."
)

BASIS_TEXT = (
    "In the beginning, the universe was a vast expanse of darkness and silence. "
    "Then came the first light, a tiny spark that grew into a blazing inferno. "
    "Stars formed from clouds of hydrogen and helium, their nuclear furnaces "
    "igniting one by one across the cosmic void. Galaxies swirled into being, "
    "spiraling arms of light stretching across millions of light-years. "
    "On a small blue planet orbiting an unremarkable yellow star, molecules "
    "began to combine in increasingly complex patterns. Life emerged from "
    "the primordial soup, evolving over billions of years into the rich "
    "tapestry of organisms that now inhabit every corner of the Earth."
)


# ── Also test direct PCA truncation (without norm split) ────────────────────

class DirectPCAHook:
    """
    Alternative: 直接 PCA truncation（norm分離なし）
    h ≈ mean + sum(c_i * v_i) for top-k PCA components
    """
    def __init__(self, pca_components, pca_mean):
        self.V = pca_components   # (k, d_model)
        self.mean = pca_mean      # (d_model,)
        self.cosine_drifts = []

    def __call__(self, module, input, output):
        if isinstance(output, tuple):
            h = output[0]
            rest = output[1:]
        else:
            h = output
            rest = None

        original_h = h.clone()
        B, T, D = h.shape

        for b in range(B):
            h_b = h[b].float()  # (T, D)
            centered = h_b - self.mean.unsqueeze(0)
            coeffs = centered @ self.V.T        # (T, k)
            reconstructed = coeffs @ self.V + self.mean.unsqueeze(0)  # (T, D)

            cos = F.cosine_similarity(original_h[b], reconstructed, dim=-1)
            self.cosine_drifts.extend((1.0 - cos).detach().cpu().tolist())

            h[b] = reconstructed

        if rest is not None:
            return (h,) + rest
        return h


class NormPCAHook:
    """norm分離版 (Phase 1 と同じ)"""
    def __init__(self, basis, mean_direction):
        self.basis = basis
        self.mean_direction = mean_direction
        self.cosine_drifts = []

    def __call__(self, module, input, output):
        if isinstance(output, tuple):
            h = output[0]
            rest = output[1:]
        else:
            h = output
            rest = None

        original_h = h.clone()
        B, T, D = h.shape

        for b in range(B):
            h_b = h[b].float()
            norms = h_b.norm(dim=1)
            h_normed = h_b / norms.unsqueeze(1).clamp(min=1e-12)

            proj = h_normed @ self.basis.T  # (T, k)
            direction_approx = proj @ self.basis + self.mean_direction.unsqueeze(0)
            direction_approx = direction_approx / direction_approx.norm(
                dim=1, keepdim=True
            ).clamp(min=1e-12)
            reconstructed = norms.unsqueeze(1) * direction_approx

            cos = F.cosine_similarity(original_h[b], reconstructed, dim=-1)
            self.cosine_drifts.extend((1.0 - cos).detach().cpu().tolist())

            h[b] = reconstructed

        if rest is not None:
            return (h,) + rest
        return h


def compute_perplexity(logits, input_ids):
    shift_logits = logits[:-1]
    shift_labels = input_ids[1:]
    loss = F.cross_entropy(shift_logits, shift_labels, reduction="mean")
    return float(torch.exp(loss).item())


def compute_top_k_overlap(logits_a, logits_b, k=5):
    top_a = logits_a.topk(k, dim=-1).indices
    top_b = logits_b.topk(k, dim=-1).indices
    overlaps = []
    for t in range(top_a.shape[0]):
        a_set = set(top_a[t].tolist())
        b_set = set(top_b[t].tolist())
        overlaps.append(len(a_set & b_set) / k)
    return float(np.mean(overlaps))


def main():
    print("=" * 70)
    print("PHASE 2 (Lightweight): Rank-Performance Curve")
    print("=" * 70)

    results_dir = Path(__file__).parent.parent / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    start_time = time.time()

    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    model = AutoModelForCausalLM.from_pretrained("gpt2", torch_dtype=torch.float32)
    model.eval()

    eval_ids = tokenizer.encode(EVAL_TEXT, return_tensors="pt")
    basis_ids = tokenizer.encode(BASIS_TEXT, return_tensors="pt")
    T = eval_ids.shape[1]

    # Baseline
    with torch.inference_mode():
        baseline_logits = model(eval_ids).logits[0].float().cpu()
    baseline_ppl = compute_perplexity(baseline_logits, eval_ids[0])
    print(f"  Baseline PPL: {baseline_ppl:.2f}, T={T}")

    # Collect hidden states for basis computation (on basis_text)
    collected_basis = {}
    def make_capture(layer_idx):
        def hook(module, input, output):
            if isinstance(output, tuple):
                collected_basis[layer_idx] = output[0][0].float().detach()
            else:
                collected_basis[layer_idx] = output[0].float().detach()
        return hook

    handles = []
    layers = model.transformer.h
    for li in TARGET_LAYERS:
        handles.append(layers[li].register_forward_hook(make_capture(li)))
    with torch.inference_mode():
        model(basis_ids)
    for h in handles:
        h.remove()

    all_results = {
        "experiment": "phase2_rank_performance_curve",
        "version": "v2.0",
        "timestamp": datetime.now().isoformat(),
        "model": "GPT-2",
        "baseline_ppl": round(baseline_ppl, 3),
        "eval_tokens": T,
        "d_model": 768,
        "layers": [],
    }

    for li in TARGET_LAYERS:
        print(f"\n  Layer {li}:")
        h_basis = collected_basis[li]  # (T_basis, 768)

        # Prepare PCA basis (full)
        norms = h_basis.norm(dim=1, keepdim=True).clamp(min=1e-12)
        h_normed = h_basis / norms
        mean_dir = h_normed.mean(dim=0)

        h_centered = h_normed - h_normed.mean(dim=0, keepdim=True)
        U, S, Vt = torch.linalg.svd(h_centered, full_matrices=False)
        full_pca_basis = Vt  # (min(T,d), d_model)

        # Prepare direct PCA basis
        h_for_direct = h_basis
        direct_mean = h_for_direct.mean(dim=0)
        h_dc = h_for_direct - direct_mean.unsqueeze(0)
        Ud, Sd, Vd = torch.linalg.svd(h_dc, full_matrices=False)
        full_direct_basis = Vd

        layer_entry = {"layer": li, "curves": []}

        for k in K_VALUES:
            max_k = min(full_pca_basis.shape[0], full_direct_basis.shape[0])
            if k > max_k:
                continue

            row = {"k": k, "compression_ratio": round(768 / (1 + k), 1)}

            # ── Method 1: norm + PCA ──
            pca_k = full_pca_basis[:k]
            hook = NormPCAHook(pca_k, mean_dir)
            handle = layers[li].register_forward_hook(hook)
            with torch.inference_mode():
                logits = model(eval_ids).logits[0].float().cpu()
            handle.remove()

            ppl = compute_perplexity(logits, eval_ids[0])
            top5 = compute_top_k_overlap(baseline_logits, logits)
            cos_drift = float(np.mean(hook.cosine_drifts))

            row["norm_pca"] = {
                "ppl": round(ppl, 3),
                "delta_ppl": round(ppl - baseline_ppl, 3),
                "top5_overlap": round(top5, 4),
                "mean_cosine_drift": round(cos_drift, 6),
            }

            # ── Method 2: direct PCA truncation ──
            direct_k = full_direct_basis[:k]
            hook2 = DirectPCAHook(direct_k, direct_mean)
            handle = layers[li].register_forward_hook(hook2)
            with torch.inference_mode():
                logits2 = model(eval_ids).logits[0].float().cpu()
            handle.remove()

            ppl2 = compute_perplexity(logits2, eval_ids[0])
            top52 = compute_top_k_overlap(baseline_logits, logits2)
            cos_drift2 = float(np.mean(hook2.cosine_drifts))

            row["direct_pca"] = {
                "ppl": round(ppl2, 3),
                "delta_ppl": round(ppl2 - baseline_ppl, 3),
                "top5_overlap": round(top52, 4),
                "mean_cosine_drift": round(cos_drift2, 6),
            }

            better = "DIRECT" if ppl2 < ppl else "NORM"
            print(f"    k={k:>3} ({row['compression_ratio']:>5.1f}x): "
                  f"norm+PCA ΔPPL={ppl-baseline_ppl:>+10.1f}  "
                  f"direct ΔPPL={ppl2-baseline_ppl:>+10.1f}  "
                  f"[{better}] top5: {top5:.3f}/{top52:.3f}")

            layer_entry["curves"].append(row)

        all_results["layers"].append(layer_entry)

    # ── Analysis: find knee / sweet spot ──
    analysis = {"sweet_spots": [], "conclusion": ""}
    for layer_data in all_results["layers"]:
        li = layer_data["layer"]
        for curve_pt in layer_data["curves"]:
            k = curve_pt["k"]
            # Check both methods
            for method in ["norm_pca", "direct_pca"]:
                dppl = curve_pt[method]["delta_ppl"]
                if dppl < 5.0:
                    analysis["sweet_spots"].append({
                        "layer": li, "k": k, "method": method,
                        "delta_ppl": dppl,
                        "compression_ratio": curve_pt["compression_ratio"],
                    })

    if analysis["sweet_spots"]:
        analysis["conclusion"] = (
            "Sweet spots found with ΔPPL < 5.0. "
            "Hidden state compression may be viable at higher k."
        )
    else:
        analysis["conclusion"] = (
            "No sweet spot found even at k=512. "
            "Hidden state full-replacement compression is not viable. "
            "Recommend pivoting to KV cache compression (Phase 4)."
        )

    all_results["analysis"] = analysis
    all_results["elapsed_seconds"] = round(time.time() - start_time, 1)

    out_path = results_dir / "phase2_rank_performance_curve.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)

    print(f"\n{'=' * 70}")
    print(f"  ANALYSIS: {analysis['conclusion']}")
    if analysis["sweet_spots"]:
        for sp in analysis["sweet_spots"]:
            print(f"    L{sp['layer']} k={sp['k']} {sp['method']}: "
                  f"ΔPPL={sp['delta_ppl']:+.2f} ({sp['compression_ratio']:.1f}x)")
    print(f"  Saved: {out_path}")
    print(f"  Elapsed: {all_results['elapsed_seconds']:.0f}s")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
