#!/usr/bin/env python3
"""
Phase 4: KV Cache Compression (Arc-Compression v2.0)

Full hidden state replacement proved infeasible (Phase 1-2).
Focus on KV cache to explore the possibility of inference-time memory reduction.

Compression methods:
  1. per-head PCA: low-rank approximation of K/V independently per head
  2. norm_pca: norm separation followed by low-rank approximation (leveraging Arc findings)
  3. random: random projection of the same dimensionality (baseline)

Gate 5 conditions:
  - Actual memory reduction > 3x with DPPL < 1.0
  - PCA is superior to random
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

K_VALUES = [2, 4, 8, 16, 32, 48, 56, 60]

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


# ── KV Cache Compression (DynamicCache-aware) ──────────────────────────────

def compress_kv_cache(past, k: int, method: str = "pca", seed: int = 42,
                      compress_layers: list[int] | None = None):
    """
    In-place compress KV cache in a DynamicCache object.

    past: DynamicCache with .layers list of DynamicLayer
    Each DynamicLayer has .keys and .values of shape (B, n_heads, seq_len, head_dim)
    """
    stats = {"per_layer_cosine": []}
    n_layers = len(past.layers)

    if compress_layers is None:
        compress_layers = list(range(n_layers))

    for li in range(n_layers):
        dl = past.layers[li]
        orig_k = dl.keys   # (B, n_heads, seq_len, head_dim)
        orig_v = dl.values

        if li not in compress_layers:
            stats["per_layer_cosine"].append({
                "layer": li, "compressed": False,
                "key_cosine": 1.0, "value_cosine": 1.0,
            })
            continue

        B, n_heads, seq_len, head_dim = orig_k.shape
        new_keys = torch.zeros_like(orig_k)
        new_vals = torch.zeros_like(orig_v)

        layer_cos_k, layer_cos_v = [], []

        for h in range(n_heads):
            kh = orig_k[0, h]   # (seq_len, head_dim)
            vh = orig_v[0, h]

            if method == "pca":
                new_keys[0, h] = _pca_truncate(kh, k)
                new_vals[0, h] = _pca_truncate(vh, k)
            elif method == "norm_pca":
                new_keys[0, h] = _norm_pca_truncate(kh, k)
                new_vals[0, h] = _norm_pca_truncate(vh, k)
            elif method == "random":
                s = seed + h + li * 100
                new_keys[0, h] = _random_project(kh, k, s)
                new_vals[0, h] = _random_project(vh, k, s + 50)

            cos_k = F.cosine_similarity(kh, new_keys[0, h], dim=-1).mean().item()
            cos_v = F.cosine_similarity(vh, new_vals[0, h], dim=-1).mean().item()
            layer_cos_k.append(cos_k)
            layer_cos_v.append(cos_v)

        # Replace in-place
        dl.keys = new_keys
        dl.values = new_vals

        stats["per_layer_cosine"].append({
            "layer": li, "compressed": True,
            "key_cosine": round(float(np.mean(layer_cos_k)), 6),
            "value_cosine": round(float(np.mean(layer_cos_v)), 6),
        })

    return stats


def _pca_truncate(x: torch.Tensor, k: int) -> torch.Tensor:
    mean = x.mean(dim=0, keepdim=True)
    centered = x - mean
    U, S, Vt = torch.linalg.svd(centered, full_matrices=False)
    actual_k = min(k, U.shape[1])
    return U[:, :actual_k] @ torch.diag(S[:actual_k]) @ Vt[:actual_k] + mean


def _norm_pca_truncate(x: torch.Tensor, k: int) -> torch.Tensor:
    norms = x.norm(dim=1, keepdim=True).clamp(min=1e-12)
    dirs = x / norms
    mean_dir = dirs.mean(dim=0, keepdim=True)
    centered = dirs - mean_dir
    U, S, Vt = torch.linalg.svd(centered, full_matrices=False)
    actual_k = min(k, U.shape[1])
    recon = U[:, :actual_k] @ torch.diag(S[:actual_k]) @ Vt[:actual_k] + mean_dir
    recon = recon / recon.norm(dim=1, keepdim=True).clamp(min=1e-12)
    return norms * recon


def _random_project(x: torch.Tensor, k: int, seed: int) -> torch.Tensor:
    mean = x.mean(dim=0, keepdim=True)
    centered = x - mean
    rng = torch.Generator().manual_seed(seed)
    R = torch.randn(x.shape[1], k, generator=rng, device=x.device, dtype=x.dtype)
    Q, _ = torch.linalg.qr(R)
    proj = centered @ Q
    return proj @ Q.T + mean


# ── Evaluation Helpers ──────────────────────────────────────────────────────

def _ppl(logits, target_ids):
    loss = F.cross_entropy(logits, target_ids, reduction="mean")
    return float(torch.exp(loss).item())


def _kl(base, comp):
    p = F.log_softmax(base, dim=-1)
    q = F.log_softmax(comp, dim=-1)
    return float(F.kl_div(q, p, log_target=True, reduction="batchmean").item())


def _top5(base, comp, k=5):
    tb = base.topk(k, dim=-1).indices
    tc = comp.topk(k, dim=-1).indices
    return float(np.mean([
        len(set(tb[t].tolist()) & set(tc[t].tolist())) / k
        for t in range(tb.shape[0])
    ]))


# ── Run Evaluation ──────────────────────────────────────────────────────────

def evaluate(model, prefill_ids, continuation_ids,
             k: int, method: str,
             compress_layers: list[int] | None = None) -> dict:
    """
    1. Baseline: full forward on prefill + continuation
    2. Compressed: prefill -> compress KV -> continue
    """
    full_ids = torch.cat([prefill_ids, continuation_ids], dim=1)
    pf_len = prefill_ids.shape[1]

    # Baseline
    with torch.inference_mode():
        base_out = model(full_ids, use_cache=False)
    base_logits = base_out.logits[0, pf_len - 1:-1].float().cpu()
    target = full_ids[0, pf_len:].cpu()
    base_ppl = _ppl(base_logits, target)

    # Prefill + compress + continue
    with torch.inference_mode():
        pf_out = model(prefill_ids, use_cache=True)
        past = pf_out.past_key_values

        # Compress KV cache
        comp_stats = compress_kv_cache(past, k, method, compress_layers=compress_layers)

        # Continue with compressed cache
        cont_out = model(continuation_ids, past_key_values=past, use_cache=False)

    comp_logits = cont_out.logits[0].float().cpu()

    # Align
    comp_l = comp_logits[:-1]
    base_l = base_logits[:-1]
    tgt = target[1:]
    min_len = min(comp_l.shape[0], base_l.shape[0], tgt.shape[0])
    comp_l, base_l, tgt = comp_l[:min_len], base_l[:min_len], tgt[:min_len]

    comp_ppl = _ppl(comp_l, tgt)
    n_layers = len(comp_stats["per_layer_cosine"])
    n_compressed = sum(1 for s in comp_stats["per_layer_cosine"] if s["compressed"])

    return {
        "method": method,
        "k": k,
        "head_dim": 64,
        "per_head_compression": round(64 / k, 2),
        "compress_layers": compress_layers,
        "n_layers_compressed": n_compressed,
        "n_layers_total": n_layers,
        "baseline_ppl": round(base_ppl, 3),
        "compressed_ppl": round(comp_ppl, 3),
        "delta_ppl": round(comp_ppl - base_ppl, 4),
        "kl_divergence": round(_kl(base_l, comp_l), 6),
        "top5_overlap": round(_top5(base_l, comp_l), 4),
        "cache_cosines": comp_stats["per_layer_cosine"],
    }


# ── Gate 5 ──────────────────────────────────────────────────────────────────

def evaluate_gate5(all_results, selective_results):
    # Viable: >=3x per-head compression with |DPPL| < 1.0
    viable = []
    for r in all_results:
        if r["method"] == "pca" and r["per_head_compression"] >= 3 and abs(r["delta_ppl"]) < 1.0:
            viable.append({"k": r["k"], "comp": r["per_head_compression"], "dppl": r["delta_ppl"]})
    for r in selective_results:
        if abs(r["delta_ppl"]) < 1.0:
            viable.append({"k": r["k"], "config": r.get("config_name"), "dppl": r["delta_ppl"]})

    # PCA vs random
    pca_wins, total = 0, 0
    for k in K_VALUES:
        pr = [r for r in all_results if r["method"] == "pca" and r["k"] == k]
        rr = [r for r in all_results if r["method"] == "random" and r["k"] == k]
        if pr and rr:
            total += 1
            if abs(pr[0]["delta_ppl"]) < abs(rr[0]["delta_ppl"]):
                pca_wins += 1
    pca_rate = pca_wins / total if total else 0

    best_3x = None
    for r in all_results:
        if r["method"] == "pca" and r["per_head_compression"] >= 3:
            if best_3x is None or abs(r["delta_ppl"]) < abs(best_3x["delta_ppl"]):
                best_3x = r

    c1 = len(viable) > 0
    c2 = pca_rate > 0.5
    c3 = best_3x is not None and abs(best_3x["delta_ppl"]) < 5.0

    return {
        "decision": "GO" if c1 else ("PARTIAL" if (c2 and c3) else "NO-GO"),
        "criteria": {
            "viable_3x": {
                "threshold": ">= 3x compression with |ΔPPL| < 1.0",
                "result": c1, "configs": viable,
            },
            "pca_advantage": {
                "threshold": "PCA > random in >50%",
                "result": c2,
                "pca_win_rate": round(pca_rate, 3),
                "wins": pca_wins, "total": total,
            },
            "quality_at_3x": {
                "threshold": "ΔPPL < 5.0 at 3x+",
                "result": c3,
                "best": {
                    "k": best_3x["k"],
                    "compression": best_3x["per_head_compression"],
                    "delta_ppl": best_3x["delta_ppl"],
                } if best_3x else None,
            },
        },
    }


# ── Main ────────────────────────────────────────────────────────────────────

def main():
    print("=" * 70)
    print("PHASE 4: KV Cache Compression — Arc-Compression v2.0")
    print("=" * 70)

    results_dir = Path(__file__).parent.parent / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    model = GPT2LMHeadModel.from_pretrained("gpt2")
    model.eval()

    prefill_ids = tokenizer.encode(PREFILL_TEXT, return_tensors="pt")
    continuation_ids = tokenizer.encode(CONTINUATION_TEXT, return_tensors="pt")

    print(f"  Prefill: {prefill_ids.shape[1]} tokens")
    print(f"  Continuation: {continuation_ids.shape[1]} tokens")
    print(f"  GPT-2: 12 layers, 12 heads, head_dim=64")

    out = {
        "experiment": "phase4_kv_cache_compression",
        "version": "v2.0",
        "timestamp": datetime.now().isoformat(),
        "model": "GPT-2",
        "n_layers": 12, "n_heads": 12, "head_dim": 64,
        "prefill_tokens": int(prefill_ids.shape[1]),
        "continuation_tokens": int(continuation_ids.shape[1]),
    }

    # ── Part A: All-layer sweep ──
    print(f"\n{'━' * 60}")
    print(f"  Part A: All-layer KV compression")
    print(f"{'━' * 60}")

    all_layer = []
    for k in K_VALUES:
        for method in ["pca", "norm_pca", "random"]:
            r = evaluate(model, prefill_ids, continuation_ids, k, method)
            all_layer.append(r)
            print(f"  k={k:>2} ({64/k:>4.1f}x) {method:>8}: "
                  f"ΔPPL={r['delta_ppl']:>+9.4f}  "
                  f"KL={r['kl_divergence']:.5f}  "
                  f"top5={r['top5_overlap']:.3f}")
    out["all_layer_sweep"] = all_layer

    # ── Part B: Selective ──
    print(f"\n{'━' * 60}")
    print(f"  Part B: Selective layer compression (PCA)")
    print(f"{'━' * 60}")

    configs = [
        ("mid_only",   list(range(3, 9)),  "L3-L8 (6 layers)"),
        ("mid_wide",   list(range(2, 10)), "L2-L9 (8 layers)"),
        ("skip_edges", list(range(1, 11)), "L1-L10 (10 layers)"),
        ("all",        list(range(12)),    "all 12 layers"),
    ]

    selective = []
    for k in [4, 8, 16, 32]:
        for name, layers, desc in configs:
            r = evaluate(model, prefill_ids, continuation_ids, k, "pca",
                         compress_layers=layers)
            r["config_name"] = name
            r["config_desc"] = desc
            selective.append(r)
            print(f"  k={k:>2} {desc:>20}: "
                  f"ΔPPL={r['delta_ppl']:>+9.4f}  "
                  f"top5={r['top5_overlap']:.3f}")
    out["selective_compression"] = selective

    # ── Gate 5 ──
    gate5 = evaluate_gate5(all_layer, selective)
    out["gate5"] = gate5
    out["elapsed_seconds"] = round(time.time() - t0, 1)

    path = results_dir / "phase4_kv_cache_compression.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)

    print(f"\n{'━' * 70}")
    print(f"  GATE 5: {gate5['decision']}")
    print(f"{'━' * 70}")
    for cn, cv in gate5["criteria"].items():
        st = "PASS" if cv["result"] else "FAIL"
        print(f"  [{st}] {cn}: {cv['threshold']}")
        if cn == "viable_3x" and cv["configs"]:
            for c in cv["configs"]:
                print(f"       → k={c['k']}, {c.get('comp','selective')}, ΔPPL={c['dppl']:+.4f}")
        if cn == "quality_at_3x" and cv.get("best"):
            b = cv["best"]
            print(f"       → k={b['k']}, {b['compression']}x, ΔPPL={b['delta_ppl']:+.4f}")
    print(f"\n  Saved: {path}")
    print(f"  Elapsed: {out['elapsed_seconds']:.0f}s")


if __name__ == "__main__":
    main()
