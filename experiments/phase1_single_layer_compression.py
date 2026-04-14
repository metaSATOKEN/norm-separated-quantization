#!/usr/bin/env python3
# Copyright 2026 Kentaro Sato
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Phase 1: Single-Layer Compression Tolerance (Arc-Compression v2.0)

Insert a compress -> decompress forward hook into each layer
and measure how much compression each layer can tolerate.

Comparison conditions:
  - baseline:         no compression
  - norm_only:        preserve norm only (k=0)
  - norm_pca_k:       norm + PCA top-k
  - norm_random_k:    norm + random top-k (baseline comparison)

k candidates: 2, 4, 8, 16, 32, 64

Evaluation metrics:
  - DPPL (perplexity change)
  - logits KL divergence
  - top-5 token overlap
  - hidden-state cosine drift
  - reconstruction MSE

GO conditions:
  - For some mid-layers, even small k yields tolerable DPPL
  - PCA is clearly superior to random

NO-GO conditions:
  - Severe degradation across all layers
  - PCA and random perform equally
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
    compress_norm_only,
    decompress_norm_only,
)


# ── Config ──────────────────────────────────────────────────────────────────

MODELS = [
    {
        "name": "GPT-2",
        "hf_id": "gpt2",
        "n_layers": 12,
        "d_model": 768,
    },
]

K_VALUES = [2, 4, 8, 16, 32, 64]

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

# Separate text for computing PCA basis (avoid data leakage)
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


# ── Evaluation Metrics ──────────────────────────────────────────────────────

def compute_perplexity(logits: torch.Tensor, input_ids: torch.Tensor) -> float:
    """Compute perplexity from logits and target token ids."""
    # Shift: logits[t] predicts input_ids[t+1]
    shift_logits = logits[:-1]
    shift_labels = input_ids[1:]
    loss = F.cross_entropy(shift_logits, shift_labels, reduction="mean")
    return float(torch.exp(loss).item())


def compute_kl_divergence(
    logits_baseline: torch.Tensor,
    logits_compressed: torch.Tensor,
) -> float:
    """KL(baseline || compressed) averaged over tokens."""
    p = F.log_softmax(logits_baseline, dim=-1)
    q = F.log_softmax(logits_compressed, dim=-1)
    kl = F.kl_div(q, p, log_target=True, reduction="batchmean")
    return float(kl.item())


def compute_top_k_overlap(
    logits_baseline: torch.Tensor,
    logits_compressed: torch.Tensor,
    k: int = 5,
) -> float:
    """Average top-k token overlap between baseline and compressed."""
    top_base = logits_baseline.topk(k, dim=-1).indices  # (T, k)
    top_comp = logits_compressed.topk(k, dim=-1).indices
    overlaps = []
    for t in range(top_base.shape[0]):
        base_set = set(top_base[t].tolist())
        comp_set = set(top_comp[t].tolist())
        overlaps.append(len(base_set & comp_set) / k)
    return float(np.mean(overlaps))


# ── Hook-based Compression ──────────────────────────────────────────────────

class CompressionHook:
    """Forward hook that compresses and reconstructs hidden states at a layer."""

    def __init__(self, basis, mean_direction, method="norm_pca"):
        self.basis = basis
        self.mean_direction = mean_direction
        self.method = method
        self.cosine_drifts = []
        self.mse_values = []

    def __call__(self, module, input, output):
        # output is typically a tuple; hidden states are the first element
        if isinstance(output, tuple):
            h = output[0]
            rest = output[1:]
        else:
            h = output
            rest = None

        original_h = h.clone()
        B, T, D = h.shape

        for b in range(B):
            h_b = h[b]  # (T, D)

            if self.method == "norm_only":
                compressed = compress_norm_only(h_b)
                reconstructed = decompress_norm_only(
                    compressed, self.mean_direction
                )
            else:  # norm_pca or norm_random
                compressed = compress_norm_pca(h_b, self.basis)
                reconstructed = decompress_norm_pca(
                    compressed, self.basis, self.mean_direction
                )

            # Metrics
            cos_sim = F.cosine_similarity(
                original_h[b], reconstructed, dim=-1
            )
            self.cosine_drifts.extend(
                (1.0 - cos_sim).detach().cpu().tolist()
            )
            mse = ((original_h[b] - reconstructed) ** 2).mean().item()
            self.mse_values.append(mse)

            h[b] = reconstructed

        if rest is not None:
            return (h,) + rest
        return h


def run_baseline(model, input_ids):
    """Run model without compression, get baseline logits."""
    with torch.inference_mode():
        out = model(input_ids)
    return out.logits[0].float().cpu()


def run_with_compression(
    model,
    input_ids,
    layer_idx: int,
    basis: torch.Tensor,
    mean_direction: torch.Tensor,
    method: str = "norm_pca",
) -> dict:
    """Run model with compression hook at specified layer, return metrics."""
    # Get the transformer layer module
    if hasattr(model, "transformer"):
        # GPT-2 style
        layers = model.transformer.h
    elif hasattr(model, "gpt_neox"):
        # Pythia style
        layers = model.gpt_neox.layers
    elif hasattr(model, "model") and hasattr(model.model, "layers"):
        # Qwen / Llama style
        layers = model.model.layers
    else:
        raise ValueError(f"Unknown model architecture: {type(model)}")

    target_layer = layers[layer_idx]
    hook = CompressionHook(basis, mean_direction, method=method)
    handle = target_layer.register_forward_hook(hook)

    try:
        with torch.inference_mode():
            out = model(input_ids)
        logits = out.logits[0].float().cpu()
    finally:
        handle.remove()

    return {
        "logits": logits,
        "cosine_drifts": hook.cosine_drifts,
        "mse_values": hook.mse_values,
    }


# ── PCA Basis Preparation ──────────────────────────────────────────────────

def prepare_basis_for_layer(
    model,
    tokenizer,
    layer_idx: int,
    text: str,
    max_k: int = 64,
) -> dict:
    """Extract hidden states at a layer and compute PCA + random bases."""
    input_ids = tokenizer.encode(text, return_tensors="pt")
    max_len = getattr(model.config, "n_positions", None) or getattr(
        model.config, "max_position_embeddings", 2048
    )
    if input_ids.shape[1] > max_len:
        input_ids = input_ids[:, :max_len]

    # Collect hidden states via hook
    collected = {}

    def capture_hook(module, input, output):
        if isinstance(output, tuple):
            collected["h"] = output[0][0].float().detach()
        else:
            collected["h"] = output[0].float().detach()

    if hasattr(model, "transformer"):
        layers = model.transformer.h
    elif hasattr(model, "gpt_neox"):
        layers = model.gpt_neox.layers
    elif hasattr(model, "model") and hasattr(model.model, "layers"):
        layers = model.model.layers
    else:
        raise ValueError(f"Unknown architecture: {type(model)}")

    handle = layers[layer_idx].register_forward_hook(capture_hook)
    with torch.inference_mode():
        model(input_ids)
    handle.remove()

    h = collected["h"]  # (T, d_model)
    d_model = h.shape[1]
    actual_k = min(max_k, d_model, h.shape[0] - 1)

    # Norm-normalize for basis computation
    norms = h.norm(dim=1, keepdim=True).clamp(min=1e-12)
    h_normed = h / norms
    mean_direction = h_normed.mean(dim=0)

    # PCA basis
    pca_basis = compute_pca_basis(h, actual_k)

    # Random basis
    random_basis = compute_random_basis(d_model, actual_k, seed=42)

    return {
        "pca_basis": pca_basis,
        "random_basis": random_basis,
        "mean_direction": mean_direction,
        "actual_k": actual_k,
    }


# ── Main Experiment ─────────────────────────────────────────────────────────

def run_phase1(model_cfg: dict) -> dict:
    """Run Phase 1 for a single model."""
    model_name = model_cfg["name"]
    hf_id = model_cfg["hf_id"]
    n_layers = model_cfg["n_layers"]
    d_model = model_cfg["d_model"]

    print(f"\n{'━' * 60}")
    print(f"  {model_name} ({hf_id})")
    print(f"  {n_layers} layers, d_model={d_model}")
    print(f"{'━' * 60}")

    tokenizer = AutoTokenizer.from_pretrained(hf_id)
    model = AutoModelForCausalLM.from_pretrained(
        hf_id, torch_dtype=torch.float32
    )
    model.eval()
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token or "<|endoftext|>"

    # Prepare eval tokens
    eval_ids = tokenizer.encode(EVAL_TEXT, return_tensors="pt")
    max_len = getattr(model.config, "n_positions", None) or getattr(
        model.config, "max_position_embeddings", 2048
    )
    if eval_ids.shape[1] > max_len:
        eval_ids = eval_ids[:, :max_len]

    T = eval_ids.shape[1]
    print(f"  Eval tokens: {T}")

    # ── Baseline ──
    print(f"  Running baseline...")
    baseline_logits = run_baseline(model, eval_ids)
    baseline_ppl = compute_perplexity(baseline_logits, eval_ids[0])
    print(f"  Baseline PPL: {baseline_ppl:.2f}")

    # ── Per-layer compression ──
    layer_results = []

    # Test all layers (skip embedding layer 0)
    test_layers = list(range(n_layers))

    for li in test_layers:
        print(f"\n  Layer {li}/{n_layers-1}:")
        t0 = time.time()

        # Prepare bases
        basis_data = prepare_basis_for_layer(
            model, tokenizer, li, BASIS_TEXT, max_k=max(K_VALUES)
        )
        pca_basis = basis_data["pca_basis"]
        random_basis = basis_data["random_basis"]
        mean_dir = basis_data["mean_direction"]

        layer_entry = {
            "layer": li,
            "conditions": [],
        }

        # (A) norm-only
        result = run_with_compression(
            model, eval_ids, li, None, mean_dir, method="norm_only"
        )
        ppl = compute_perplexity(result["logits"], eval_ids[0])
        kl = compute_kl_divergence(baseline_logits, result["logits"])
        top5 = compute_top_k_overlap(baseline_logits, result["logits"])
        cos_drift = float(np.mean(result["cosine_drifts"]))
        mse = float(np.mean(result["mse_values"]))

        layer_entry["conditions"].append({
            "method": "norm_only",
            "k": 0,
            "compression_ratio": float(d_model),  # d_model -> 1
            "ppl": round(ppl, 3),
            "delta_ppl": round(ppl - baseline_ppl, 3),
            "kl_divergence": round(kl, 6),
            "top5_overlap": round(top5, 4),
            "mean_cosine_drift": round(cos_drift, 6),
            "reconstruction_mse": round(mse, 4),
        })
        print(f"    norm_only:     ΔPPL={ppl-baseline_ppl:+.2f}, "
              f"KL={kl:.4f}, top5={top5:.3f}")

        # (B) PCA and Random for each k
        for k in K_VALUES:
            if k > basis_data["actual_k"]:
                continue

            # PCA
            pca_k = pca_basis[:k]
            result = run_with_compression(
                model, eval_ids, li, pca_k, mean_dir, method="norm_pca"
            )
            ppl_pca = compute_perplexity(result["logits"], eval_ids[0])
            kl_pca = compute_kl_divergence(baseline_logits, result["logits"])
            top5_pca = compute_top_k_overlap(baseline_logits, result["logits"])
            cos_pca = float(np.mean(result["cosine_drifts"]))
            mse_pca = float(np.mean(result["mse_values"]))

            compression_ratio = d_model / (1 + k)  # norm(1) + k coefficients

            layer_entry["conditions"].append({
                "method": "norm_pca",
                "k": k,
                "compression_ratio": round(compression_ratio, 1),
                "ppl": round(ppl_pca, 3),
                "delta_ppl": round(ppl_pca - baseline_ppl, 3),
                "kl_divergence": round(kl_pca, 6),
                "top5_overlap": round(top5_pca, 4),
                "mean_cosine_drift": round(cos_pca, 6),
                "reconstruction_mse": round(mse_pca, 4),
            })

            # Random
            rand_k = random_basis[:k]
            result = run_with_compression(
                model, eval_ids, li, rand_k, mean_dir, method="norm_random"
            )
            ppl_rand = compute_perplexity(result["logits"], eval_ids[0])
            kl_rand = compute_kl_divergence(baseline_logits, result["logits"])
            top5_rand = compute_top_k_overlap(baseline_logits, result["logits"])
            cos_rand = float(np.mean(result["cosine_drifts"]))
            mse_rand = float(np.mean(result["mse_values"]))

            layer_entry["conditions"].append({
                "method": "norm_random",
                "k": k,
                "compression_ratio": round(compression_ratio, 1),
                "ppl": round(ppl_rand, 3),
                "delta_ppl": round(ppl_rand - baseline_ppl, 3),
                "kl_divergence": round(kl_rand, 6),
                "top5_overlap": round(top5_rand, 4),
                "mean_cosine_drift": round(cos_rand, 6),
                "reconstruction_mse": round(mse_rand, 4),
            })

            pca_wins = "PCA" if ppl_pca < ppl_rand else "RND"
            print(f"    k={k:>2}: PCA ΔPPL={ppl_pca-baseline_ppl:+7.2f}  "
                  f"RND ΔPPL={ppl_rand-baseline_ppl:+7.2f}  "
                  f"[{pca_wins}] top5: {top5_pca:.3f}/{top5_rand:.3f}")

        elapsed = time.time() - t0
        print(f"    ({elapsed:.1f}s)")

        layer_results.append(layer_entry)

    del model
    gc.collect()
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()

    return {
        "model_name": model_name,
        "hf_id": hf_id,
        "n_layers": n_layers,
        "d_model": d_model,
        "eval_tokens": T,
        "baseline_ppl": round(baseline_ppl, 3),
        "k_values_tested": K_VALUES,
        "layers": layer_results,
    }


# ── Gate 2 & 3 Evaluation ──────────────────────────────────────────────────

def evaluate_gates(results: dict) -> dict:
    """
    Gate 2: Layer Tolerance -- are there layers that survive compression?
    Gate 3: Structural Advantage -- is PCA residual stronger than random?
    """
    n_layers = results["n_layers"]

    # Gate 2: Any mid layer with DPPL < 1.0 at k=8?
    tolerant_layers = []
    for layer_data in results["layers"]:
        li = layer_data["layer"]
        # Skip first/last 2 layers
        if li < 2 or li >= n_layers - 2:
            continue
        for cond in layer_data["conditions"]:
            if cond["method"] == "norm_pca" and cond["k"] == 8:
                if abs(cond["delta_ppl"]) < 1.0:
                    tolerant_layers.append({
                        "layer": li,
                        "k": 8,
                        "delta_ppl": cond["delta_ppl"],
                        "compression_ratio": cond["compression_ratio"],
                    })

    gate2_pass = len(tolerant_layers) > 0

    # Gate 3: PCA < random in DPPL for majority of (layer, k) pairs?
    pca_wins = 0
    random_wins = 0
    ties = 0
    comparisons = []

    for layer_data in results["layers"]:
        li = layer_data["layer"]
        pca_by_k = {}
        rand_by_k = {}
        for cond in layer_data["conditions"]:
            if cond["method"] == "norm_pca":
                pca_by_k[cond["k"]] = cond
            elif cond["method"] == "norm_random":
                rand_by_k[cond["k"]] = cond

        for k in K_VALUES:
            if k in pca_by_k and k in rand_by_k:
                p_dppl = pca_by_k[k]["delta_ppl"]
                r_dppl = rand_by_k[k]["delta_ppl"]
                if abs(p_dppl) < abs(r_dppl):
                    pca_wins += 1
                    winner = "PCA"
                elif abs(r_dppl) < abs(p_dppl):
                    random_wins += 1
                    winner = "RANDOM"
                else:
                    ties += 1
                    winner = "TIE"
                comparisons.append({
                    "layer": li, "k": k,
                    "pca_delta_ppl": p_dppl,
                    "random_delta_ppl": r_dppl,
                    "winner": winner,
                })

    total = pca_wins + random_wins + ties
    pca_win_rate = pca_wins / total if total > 0 else 0
    gate3_pass = pca_win_rate > 0.6  # PCA wins >60% of comparisons

    return {
        "gate2": {
            "gate": "Gate 2: Layer Tolerance",
            "decision": "GO" if gate2_pass else "NO-GO",
            "question": "Are there layers that survive compression? (DPPL < 1.0 at k=8)",
            "tolerant_mid_layers": tolerant_layers,
            "n_tolerant": len(tolerant_layers),
        },
        "gate3": {
            "gate": "Gate 3: Structural Advantage",
            "decision": "GO" if gate3_pass else "NO-GO",
            "question": "Is PCA residual stronger than random?",
            "pca_wins": pca_wins,
            "random_wins": random_wins,
            "ties": ties,
            "total_comparisons": total,
            "pca_win_rate": round(pca_win_rate, 3),
        },
    }


# ── Main ────────────────────────────────────────────────────────────────────

def main():
    print("=" * 70)
    print("PHASE 1: Single-Layer Compression Tolerance")
    print("         Arc-Compression v2.0")
    print("=" * 70)

    results_dir = Path(__file__).parent.parent / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    start_time = time.time()
    all_results = {
        "experiment": "phase1_single_layer_compression",
        "version": "v2.0",
        "timestamp": datetime.now().isoformat(),
        "models": [],
    }

    for mcfg in MODELS:
        model_result = run_phase1(mcfg)
        gates = evaluate_gates(model_result)
        model_result["gates"] = gates
        all_results["models"].append(model_result)

        # Print gate summary
        print(f"\n{'━' * 70}")
        print(f"  GATE 2: {gates['gate2']['decision']} — "
              f"{gates['gate2']['n_tolerant']} tolerant mid-layers found")
        if gates['gate2']['tolerant_mid_layers']:
            for tl in gates['gate2']['tolerant_mid_layers']:
                print(f"    Layer {tl['layer']}: ΔPPL={tl['delta_ppl']:+.3f} "
                      f"(compression {tl['compression_ratio']:.0f}x)")
        print(f"  GATE 3: {gates['gate3']['decision']} — "
              f"PCA wins {gates['gate3']['pca_win_rate']:.0%} "
              f"({gates['gate3']['pca_wins']}/{gates['gate3']['total_comparisons']})")
        print(f"{'━' * 70}")

    all_results["elapsed_seconds"] = round(time.time() - start_time, 1)

    out_path = results_dir / "phase1_single_layer_compression.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)

    print(f"\n  Results saved: {out_path}")
    print(f"  Elapsed: {all_results['elapsed_seconds']:.0f}s")


if __name__ == "__main__":
    main()
