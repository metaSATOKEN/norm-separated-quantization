#!/usr/bin/env python3
"""
Phase 4b Cross-Model: Robustness Verification

Phase 4b の主要 Finding を GPT-2 以外のモデルで再現確認。

対象モデル:
  - GPT-2 (124M): 12 layers, 12 KV heads, head_dim=64 (MHA)
  - Pythia-410M:   24 layers, 16 KV heads, head_dim=64 (MHA)
  - Qwen2-0.5B:    24 layers,  2 KV heads, head_dim=64 (GQA!)

検証する Finding:
  1. norm_pca > plain PCA in KV cache (Finding 1)
  2. norm_sep quantization > naive quantization at low bits (Finding 2)
  3. K cosine > V cosine across models (Finding 3)
  4. PCA + quantization combo (Finding 4)

GQA note:
  Qwen2 は KV heads=2 のため、KV cache が元々小さい。
  per-head PCA の seq_len 方向の低ランク近似は依然有効だが、
  圧縮の実用的意味合いが異なる。
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
from transformers import AutoModelForCausalLM, AutoTokenizer


# ── Config ──────────────────────────────────────────────────────────────────

MODELS = [
    {"name": "GPT-2",       "hf_id": "gpt2",                    "n_layers": 12},
    {"name": "Pythia-410M", "hf_id": "EleutherAI/pythia-410m",  "n_layers": 24},
    {"name": "Qwen2-0.5B",  "hf_id": "Qwen/Qwen2-0.5B",        "n_layers": 24},
]

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


# ── Compression Primitives ──────────────────────────────────────────────────

def pca_truncate(x, k):
    mean = x.mean(dim=0, keepdim=True)
    centered = x - mean
    U, S, Vt = torch.linalg.svd(centered, full_matrices=False)
    ak = min(k, U.shape[1])
    return U[:, :ak] @ torch.diag(S[:ak]) @ Vt[:ak] + mean

def norm_pca_truncate(x, k):
    norms = x.norm(dim=-1, keepdim=True).clamp(min=1e-12)
    dirs = x / norms
    mean_dir = dirs.mean(dim=0, keepdim=True)
    centered = dirs - mean_dir
    U, S, Vt = torch.linalg.svd(centered, full_matrices=False)
    ak = min(k, U.shape[1])
    recon = U[:, :ak] @ torch.diag(S[:ak]) @ Vt[:ak] + mean_dir
    recon = recon / recon.norm(dim=-1, keepdim=True).clamp(min=1e-12)
    return norms * recon

def random_project(x, k, seed=42):
    mean = x.mean(dim=0, keepdim=True)
    centered = x - mean
    rng = torch.Generator().manual_seed(seed)
    R = torch.randn(x.shape[1], k, generator=rng, device=x.device, dtype=x.dtype)
    Q, _ = torch.linalg.qr(R)
    return (centered @ Q) @ Q.T + mean

def quantize_absmax(x, bits):
    qmax = 2 ** (bits - 1) - 1
    scale = x.abs().amax(dim=-1, keepdim=True).clamp(min=1e-12) / qmax
    return ((x / scale).round().clamp(-qmax, qmax)) * scale

def quantize_norm_separated(x, bits):
    norms = x.norm(dim=-1, keepdim=True).clamp(min=1e-12)
    dirs = x / norms
    dirs_q = quantize_absmax(dirs, bits)
    dirs_q = dirs_q / dirs_q.norm(dim=-1, keepdim=True).clamp(min=1e-12)
    return norms * dirs_q

def apply_method(x, method, k=None, bits=None, seed=42):
    if method == "none": return x
    if method == "pca": return pca_truncate(x, k)
    if method == "norm_pca": return norm_pca_truncate(x, k)
    if method == "random": return random_project(x, k, seed)
    if method == "quant": return quantize_absmax(x, bits)
    if method == "norm_quant": return quantize_norm_separated(x, bits)
    if method == "norm_pca+quant":
        return quantize_norm_separated(norm_pca_truncate(x, k), bits)
    raise ValueError(method)


# ── KV Cache Compression ───────────────────────────────────────────────────

def compress_kv(past, config, seed=42):
    n_layers = len(past.layers)
    stats = {"per_layer": []}

    for li in range(n_layers):
        dl = past.layers[li]
        orig_k, orig_v = dl.keys.clone(), dl.values.clone()
        B, n_heads, seq_len, hd = orig_k.shape

        new_k = torch.zeros_like(orig_k)
        new_v = torch.zeros_like(orig_v)
        cos_k_list, cos_v_list = [], []

        for h in range(n_heads):
            s = seed + h + li * 100
            new_k[0, h] = apply_method(
                orig_k[0, h], config.get("key_method", "none"),
                config.get("key_k"), config.get("key_bits"), s)
            new_v[0, h] = apply_method(
                orig_v[0, h], config.get("value_method", "none"),
                config.get("value_k"), config.get("value_bits"), s + 50)

            cos_k_list.append(F.cosine_similarity(orig_k[0, h], new_k[0, h], dim=-1).mean().item())
            cos_v_list.append(F.cosine_similarity(orig_v[0, h], new_v[0, h], dim=-1).mean().item())

        dl.keys = new_k
        dl.values = new_v

        stats["per_layer"].append({
            "layer": li,
            "key_cosine": round(float(np.mean(cos_k_list)), 6),
            "value_cosine": round(float(np.mean(cos_v_list)), 6),
        })

    return stats


# ── Evaluation ──────────────────────────────────────────────────────────────

def ppl(logits, targets):
    return float(torch.exp(F.cross_entropy(logits, targets, reduction="mean")).item())

def kl_div(base, comp):
    p = F.log_softmax(base, dim=-1)
    q = F.log_softmax(comp, dim=-1)
    return float(F.kl_div(q, p, log_target=True, reduction="batchmean").item())

def top5_overlap(base, comp):
    tb = base.topk(5, dim=-1).indices
    tc = comp.topk(5, dim=-1).indices
    return float(np.mean([
        len(set(tb[t].tolist()) & set(tc[t].tolist())) / 5
        for t in range(tb.shape[0])
    ]))


def evaluate(model, prefill_ids, continuation_ids, config, base_logits, base_ppl_val):
    with torch.inference_mode():
        pf_out = model(prefill_ids, use_cache=True)
        past = pf_out.past_key_values
        stats = compress_kv(past, config)
        cont_out = model(continuation_ids, past_key_values=past, use_cache=False)

    cl = cont_out.logits[0].float().cpu()
    full_ids = torch.cat([prefill_ids, continuation_ids], dim=1)
    target = full_ids[0, prefill_ids.shape[1]:].cpu()

    cl_a = cl[:-1]
    bl_a = base_logits[:-1]
    tgt = target[1:]
    ml = min(cl_a.shape[0], bl_a.shape[0], tgt.shape[0])
    cl_a, bl_a, tgt = cl_a[:ml], bl_a[:ml], tgt[:ml]

    return {
        "delta_ppl": round(ppl(cl_a, tgt) - base_ppl_val, 4),
        "kl": round(kl_div(bl_a, cl_a), 5),
        "top5": round(top5_overlap(bl_a, cl_a), 4),
        "stats": stats,
    }


# ── Per-model Experiment ────────────────────────────────────────────────────

def run_model(mcfg):
    name = mcfg["name"]
    hf_id = mcfg["hf_id"]

    tokenizer = AutoTokenizer.from_pretrained(hf_id)
    model = AutoModelForCausalLM.from_pretrained(hf_id, torch_dtype=torch.float32)
    model.eval()
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token or "<|endoftext|>"

    prefill_ids = tokenizer.encode(PREFILL_TEXT, return_tensors="pt")
    continuation_ids = tokenizer.encode(CONTINUATION_TEXT, return_tensors="pt")
    full_ids = torch.cat([prefill_ids, continuation_ids], dim=1)
    pf_len = prefill_ids.shape[1]

    # Model info from cache
    with torch.inference_mode():
        out = model(full_ids, use_cache=True)
    base_logits = out.logits[0, pf_len - 1:-1].float().cpu()
    target = full_ids[0, pf_len:].cpu()
    base_ppl_val = ppl(base_logits, target)

    past = out.past_key_values
    dl0 = past.layers[0]
    n_kv_heads = dl0.keys.shape[1]
    head_dim = dl0.keys.shape[3]
    n_layers = len(past.layers)
    del out, past

    print(f"\n{'━' * 60}")
    print(f"  {name} ({hf_id})")
    print(f"  {n_layers} layers, {n_kv_heads} KV heads, head_dim={head_dim}")
    print(f"  Prefill: {pf_len}, Continuation: {continuation_ids.shape[1]}")
    print(f"  Baseline PPL: {base_ppl_val:.3f}")
    print(f"{'━' * 60}")

    model_result = {
        "model": name, "hf_id": hf_id,
        "n_layers": n_layers, "n_kv_heads": n_kv_heads, "head_dim": head_dim,
        "prefill_tokens": int(pf_len),
        "continuation_tokens": int(continuation_ids.shape[1]),
        "baseline_ppl": round(base_ppl_val, 3),
    }

    # ── Finding 1: norm_pca vs pca vs random ──
    print(f"\n  [Finding 1] norm_pca vs pca vs random:")
    f1_results = []
    for k in [8, 16, 32, 48]:
        row = {"k": k}
        for method in ["pca", "norm_pca", "random"]:
            cfg = {"key_method": method, "key_k": k, "value_method": method, "value_k": k}
            r = evaluate(model, prefill_ids, continuation_ids, cfg, base_logits, base_ppl_val)
            row[method] = r["delta_ppl"]
        ratio = abs(row["pca"]) / abs(row["norm_pca"]) if row["norm_pca"] != 0 else float("inf")
        print(f"    k={k:>2}: PCA={row['pca']:>+9.3f}  norm_pca={row['norm_pca']:>+9.3f}  "
              f"random={row['random']:>+9.3f}  advantage={ratio:.1f}x")
        row["norm_pca_advantage"] = round(ratio, 2)
        f1_results.append(row)
    model_result["finding1_pca_comparison"] = f1_results

    # ── Finding 2: quantization naive vs norm_sep ──
    print(f"\n  [Finding 2] Quantization: naive vs norm_sep:")
    f2_results = []
    for bits in [8, 4, 3, 2]:
        cfg_n = {"key_method": "quant", "key_bits": bits, "value_method": "quant", "value_bits": bits}
        cfg_s = {"key_method": "norm_quant", "key_bits": bits, "value_method": "norm_quant", "value_bits": bits}
        rn = evaluate(model, prefill_ids, continuation_ids, cfg_n, base_logits, base_ppl_val)
        rs = evaluate(model, prefill_ids, continuation_ids, cfg_s, base_logits, base_ppl_val)
        adv = abs(rn["delta_ppl"]) / abs(rs["delta_ppl"]) if rs["delta_ppl"] != 0 else float("inf")
        print(f"    INT{bits}: naive={rn['delta_ppl']:>+9.4f}  norm_sep={rs['delta_ppl']:>+9.4f}  "
              f"advantage={adv:.2f}x")
        f2_results.append({
            "bits": bits, "naive": rn["delta_ppl"], "norm_sep": rs["delta_ppl"],
            "advantage": round(adv, 3),
        })
    model_result["finding2_quantization"] = f2_results

    # ── Finding 3: K vs V cosine ──
    print(f"\n  [Finding 3] K vs V cosine (k=16 norm_pca):")
    cfg_kv = {"key_method": "norm_pca", "key_k": 16, "value_method": "norm_pca", "value_k": 16}
    r_kv = evaluate(model, prefill_ids, continuation_ids, cfg_kv, base_logits, base_ppl_val)

    k_cosines = [s["key_cosine"] for s in r_kv["stats"]["per_layer"]]
    v_cosines = [s["value_cosine"] for s in r_kv["stats"]["per_layer"]]
    k_mean = float(np.mean(k_cosines))
    v_mean = float(np.mean(v_cosines))
    k_gt_v = sum(1 for kc, vc in zip(k_cosines, v_cosines) if kc > vc)

    print(f"    Key mean cosine:   {k_mean:.6f}")
    print(f"    Value mean cosine: {v_mean:.6f}")
    print(f"    Key > Value: {k_gt_v}/{n_layers} layers")

    model_result["finding3_kv_asymmetry"] = {
        "key_mean_cosine": round(k_mean, 6),
        "value_mean_cosine": round(v_mean, 6),
        "key_gt_value_layers": k_gt_v,
        "total_layers": n_layers,
        "per_layer": r_kv["stats"]["per_layer"],
    }

    # ── Finding 4: PCA + quant combo ──
    print(f"\n  [Finding 4] Best combos:")
    f4_results = []
    combos = [
        ("norm_sep INT4",              {"key_method": "norm_quant", "key_bits": 4,
                                        "value_method": "norm_quant", "value_bits": 4}),
        ("norm_pca(K32V48)+K4V8",      {"key_method": "norm_pca+quant", "key_k": 32, "key_bits": 4,
                                        "value_method": "norm_pca+quant", "value_k": 48, "value_bits": 8}),
        ("norm_pca(K16V48)+INT8",      {"key_method": "norm_pca+quant", "key_k": 16, "key_bits": 8,
                                        "value_method": "norm_pca+quant", "value_k": 48, "value_bits": 8}),
        ("naive INT4",                 {"key_method": "quant", "key_bits": 4,
                                        "value_method": "quant", "value_bits": 4}),
    ]
    for label, cfg in combos:
        r = evaluate(model, prefill_ids, continuation_ids, cfg, base_logits, base_ppl_val)
        print(f"    {label:>30}: ΔPPL={r['delta_ppl']:>+9.4f}  top5={r['top5']:.3f}")
        f4_results.append({"label": label, "delta_ppl": r["delta_ppl"], "top5": r["top5"]})
    model_result["finding4_combos"] = f4_results

    del model
    gc.collect()
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()

    return model_result


# ── Main ────────────────────────────────────────────────────────────────────

def main():
    print("=" * 70)
    print("PHASE 4b CROSS-MODEL: Robustness Verification")
    print("=" * 70)

    results_dir = Path(__file__).parent.parent / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    out = {
        "experiment": "phase4b_cross_model",
        "version": "v2.0",
        "timestamp": datetime.now().isoformat(),
        "models": [],
    }

    for mcfg in MODELS:
        result = run_model(mcfg)
        out["models"].append(result)

    # ── Cross-model Summary ──
    print(f"\n{'━' * 70}")
    print(f"  CROSS-MODEL SUMMARY")
    print(f"{'━' * 70}")

    # Finding 1: norm_pca advantage
    print(f"\n  Finding 1: norm_pca advantage over PCA (average across k)")
    for m in out["models"]:
        advs = [r["norm_pca_advantage"] for r in m["finding1_pca_comparison"]]
        print(f"    {m['model']:>15}: mean={np.mean(advs):.1f}x  "
              f"[{', '.join(f'{a:.1f}x' for a in advs)}]")

    # Finding 2: norm_sep advantage by bits
    print(f"\n  Finding 2: norm_sep quantization advantage")
    for bits in [8, 4, 3, 2]:
        row = []
        for m in out["models"]:
            entry = [r for r in m["finding2_quantization"] if r["bits"] == bits][0]
            row.append(f"{m['model']}: {entry['advantage']:.2f}x")
        print(f"    INT{bits}: {' | '.join(row)}")

    # Finding 3: K > V
    print(f"\n  Finding 3: Key > Value cosine")
    for m in out["models"]:
        f3 = m["finding3_kv_asymmetry"]
        print(f"    {m['model']:>15}: K={f3['key_mean_cosine']:.4f}  "
              f"V={f3['value_mean_cosine']:.4f}  "
              f"K>V in {f3['key_gt_value_layers']}/{f3['total_layers']} layers")

    # Finding 4: Best combo ΔPPL
    print(f"\n  Finding 4: norm_sep INT4 ΔPPL across models")
    for m in out["models"]:
        ns4 = [r for r in m["finding4_combos"] if r["label"] == "norm_sep INT4"][0]
        n4  = [r for r in m["finding4_combos"] if r["label"] == "naive INT4"][0]
        print(f"    {m['model']:>15}: norm_sep={ns4['delta_ppl']:>+8.4f}  "
              f"naive={n4['delta_ppl']:>+8.4f}")

    # Robustness verdict
    print(f"\n{'━' * 70}")
    verdicts = {}
    # F1: norm_pca > pca in all models?
    f1_ok = all(
        np.mean([r["norm_pca_advantage"] for r in m["finding1_pca_comparison"]]) > 1.0
        for m in out["models"]
    )
    verdicts["finding1_norm_pca_advantage"] = f1_ok

    # F2: norm_sep advantage increases with lower bits in all models?
    f2_ok = True
    for m in out["models"]:
        qs = m["finding2_quantization"]
        # Check INT4 advantage > INT8 advantage
        adv8 = [r for r in qs if r["bits"] == 8][0]["advantage"]
        adv4 = [r for r in qs if r["bits"] == 4][0]["advantage"]
        if adv4 <= adv8:
            f2_ok = False

    verdicts["finding2_low_bit_advantage_increases"] = f2_ok

    # F3: K > V in majority of layers for all models?
    f3_ok = all(
        m["finding3_kv_asymmetry"]["key_gt_value_layers"] >
        m["finding3_kv_asymmetry"]["total_layers"] / 2
        for m in out["models"]
    )
    verdicts["finding3_key_gt_value"] = f3_ok

    out["robustness_verdicts"] = verdicts

    for fn, ok in verdicts.items():
        status = "REPRODUCED" if ok else "NOT REPRODUCED"
        print(f"  [{status}] {fn}")

    out["elapsed_seconds"] = round(time.time() - t0, 1)

    path = results_dir / "phase4b_cross_model.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)

    print(f"\n  Saved: {path}")
    print(f"  Elapsed: {out['elapsed_seconds']:.0f}s")


if __name__ == "__main__":
    main()
