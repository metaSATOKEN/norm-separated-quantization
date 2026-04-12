#!/usr/bin/env python3
"""
Phase 0: Structure Verification (Arc-Compression v2.0)

Arc論文の幾何学構造を再現し、圧縮前提が成立するか検証する。
結果は全てJSONで出力。

検証項目:
  1. PC1 variance explained (>90% を期待)
  2. corr(PC1, norm) (|r| > 0.96 を期待)
  3. norm-normalization 後の PC1 variance collapse
  4. residual subspace の position / difficulty signal
  5. 複数テキストタイプでの安定性

GO条件:
  - |corr(PC1, norm)| > 0.96
  - norm-normalization により PC1 variance が大幅低下
  - residual で少なくとも position signal が確認される

NO-GO条件:
  - norm dominance が再現しない
  - residual 構造がノイズ的で安定しない
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
from transformers import AutoModelForCausalLM, AutoTokenizer
from sklearn.decomposition import PCA
from scipy.stats import pearsonr, f_oneway
from collections import defaultdict


# ── Config ──────────────────────────────────────────────────────────────────

MODELS = [
    {
        "name": "GPT-2",
        "hf_id": "gpt2",
        "n_layers": 12,
        "d_model": 768,
        "ln_type": "Pre-LN",
    },
    {
        "name": "Pythia-410M",
        "hf_id": "EleutherAI/pythia-410m",
        "n_layers": 24,
        "d_model": 1024,
        "ln_type": "Pre-LN",
    },
    {
        "name": "Qwen2-0.5B",
        "hf_id": "Qwen/Qwen2-0.5B",
        "n_layers": 24,
        "d_model": 896,
        "ln_type": "Pre-LN (RMSNorm)",
    },
]

SAMPLE_TEXTS = {
    "narrative": (
        "The old lighthouse keeper climbed the spiral staircase each evening, "
        "carrying a lantern that cast long shadows across the stone walls. "
        "He had performed this ritual for forty years, ever since the automated "
        "systems had failed during the great storm. The sea below crashed "
        "against the rocks with a rhythm that matched his breathing, and he "
        "found comfort in the predictability of waves and wind. Tonight, "
        "however, something was different. A strange light flickered on the "
        "horizon, pulsing with an irregular beat that made him uneasy. "
        "He set the lantern down on the iron railing and squinted into the "
        "darkness. The light grew brighter, then dimmed, then brightened again."
    ),
    "technical": (
        "Transformer architectures process sequential input through layers of "
        "multi-head self-attention and feed-forward networks. Each attention "
        "head computes query, key, and value projections, producing a weighted "
        "sum over value vectors. The weights are determined by the softmax of "
        "scaled dot-product similarities between queries and keys. Residual "
        "connections and layer normalization stabilize training across many "
        "layers, enabling models with billions of parameters to converge. "
        "Pre-layer normalization places the normalization before the attention "
        "and feed-forward sublayers, which has been shown to improve gradient "
        "flow and training stability in deeper models."
    ),
    "mixed": (
        "Yesterday I ran pip install torch and it took forever. The package "
        "manager downloaded 2.3 GB of CUDA libraries. Meanwhile, my cat sat "
        "on the keyboard and typed import antigravity which actually works in "
        "Python. She also managed to mass-delete my virtual environment. I had "
        "to recreate it from scratch using python3 -m venv .venv and reinstall "
        "everything. The whole process took about 45 minutes, which gave me "
        "time to read about the latest advances in efficient inference for "
        "large language models, particularly KV cache compression techniques."
    ),
}

FUNC_WORDS = {
    "the", "a", "an", "of", "in", "to", "and", "is", "was", "are", "were",
    "be", "been", "being", "have", "has", "had", "do", "does", "did", "will",
    "would", "could", "should", "may", "might", "shall", "can", "for", "with",
    "on", "at", "from", "by", "as", "or", "but", "not", "no", "if", "that",
    "this", "it", "its", "he", "she", "they", "we", "you", "his", "her",
    "their", "our", "my", "your", "which", "what", "who", "how", "when",
    "where", "than", "then", "so", "i",
}
PUNCT = set(".,!?;:'\"--()")


def classify_pos(tok_str: str) -> str:
    t = tok_str.strip().lower()
    if t in PUNCT or all(c in PUNCT for c in t):
        return "PUNCT"
    if t in FUNC_WORDS:
        return "FUNC"
    return "CONTENT"


# ── Core Analysis ───────────────────────────────────────────────────────────

def extract_all_layers(model_id: str, text: str) -> dict:
    """Extract hidden states from all layers + compute logits."""
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id, output_hidden_states=True, torch_dtype=torch.float32,
    )
    model.eval()

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token or "<|endoftext|>"

    ids = tokenizer.encode(text, return_tensors="pt")
    max_len = getattr(model.config, "n_positions", None) or getattr(
        model.config, "max_position_embeddings", 2048
    )
    if ids.shape[1] > max_len:
        ids = ids[:, :max_len]

    with torch.inference_mode():
        out = model(ids)

    # hidden_states[0] = embedding, [1..n_layers] = layer outputs
    hidden = {
        i: out.hidden_states[i][0].float().cpu().numpy()
        for i in range(len(out.hidden_states))
    }
    logits = out.logits[0].float().cpu()

    tokens = [tokenizer.decode([ids[0, t].item()]) for t in range(ids.shape[1])]

    del out, model
    gc.collect()
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()

    return {
        "hidden_states": hidden,
        "logits": logits,
        "tokens": tokens,
        "input_ids": ids[0].numpy(),
    }


def analyze_layer(
    h: np.ndarray,
    positions: np.ndarray,
    surprisals: np.ndarray,
    pos_cats: list[str],
    n_pcs: int = 10,
) -> dict:
    """Analyze one layer's hidden states: PCA, norm correlation, residual probing."""
    T, d = h.shape
    norms = np.linalg.norm(h, axis=1)
    K = min(n_pcs, d, T - 1)

    # ── Raw PCA ──
    pca = PCA(n_components=K)
    proj = pca.fit_transform(h)
    ve = pca.explained_variance_ratio_.tolist()

    # PC-feature correlations
    pc_correlations = []
    for pc_i in range(K):
        pv = proj[:, pc_i]
        r_norm, p_norm = pearsonr(pv, norms)
        r_pos, p_pos = pearsonr(pv, positions)
        r_surp, p_surp = pearsonr(pv, surprisals)

        # POS F-statistic
        groups = defaultdict(list)
        for t in range(T):
            groups[pos_cats[t]].append(pv[t])
        valid = [v for v in groups.values() if len(v) >= 5]
        f_stat = float(f_oneway(*valid)[0]) if len(valid) >= 2 else 0.0

        pc_correlations.append({
            "pc": pc_i + 1,
            "variance_explained": ve[pc_i],
            "corr_norm": float(r_norm),
            "p_norm": float(p_norm),
            "corr_position": float(r_pos),
            "p_position": float(p_pos),
            "corr_surprisal": float(r_surp),
            "p_surprisal": float(p_surp),
            "pos_f_statistic": f_stat,
        })

    # ── Norm-normalized PCA ──
    h_normed = h / (np.linalg.norm(h, axis=1, keepdims=True) + 1e-12)
    pca_norm = PCA(n_components=K)
    proj_norm = pca_norm.fit_transform(h_normed)
    ve_norm = pca_norm.explained_variance_ratio_.tolist()

    norm_pc_correlations = []
    for pc_i in range(K):
        pv = proj_norm[:, pc_i]
        r_pos, _ = pearsonr(pv, positions)
        r_surp, _ = pearsonr(pv, surprisals)

        groups = defaultdict(list)
        for t in range(T):
            groups[pos_cats[t]].append(pv[t])
        valid = [v for v in groups.values() if len(v) >= 5]
        f_stat = float(f_oneway(*valid)[0]) if len(valid) >= 2 else 0.0

        norm_pc_correlations.append({
            "pc": pc_i + 1,
            "variance_explained": ve_norm[pc_i],
            "corr_position": float(r_pos),
            "corr_surprisal": float(r_surp),
            "pos_f_statistic": f_stat,
        })

    return {
        "T": T,
        "d_model": d,
        "norm_stats": {
            "mean": float(norms.mean()),
            "std": float(norms.std()),
            "min": float(norms.min()),
            "max": float(norms.max()),
        },
        "raw_pca": {
            "variance_explained": ve,
            "pc1_variance": ve[0],
            "top2_variance": sum(ve[:2]),
            "pc_correlations": pc_correlations,
        },
        "normalized_pca": {
            "variance_explained": ve_norm,
            "pc1_variance": ve_norm[0],
            "top2_variance": sum(ve_norm[:2]),
            "pc1_variance_collapse": ve[0] - ve_norm[0],
            "pc_correlations": norm_pc_correlations,
        },
    }


def compute_surprisals(logits: torch.Tensor, input_ids: np.ndarray) -> np.ndarray:
    """Compute per-token surprisals from logits."""
    T = input_ids.shape[0]
    log_probs = torch.log_softmax(logits, dim=-1)
    surprisals = np.zeros(T)
    for t in range(T - 1):
        surprisals[t + 1] = -log_probs[t, input_ids[t + 1]].item()
    surprisals[0] = surprisals[1]
    return surprisals


# ── Gate Evaluation ─────────────────────────────────────────────────────────

def evaluate_gate1(results: dict) -> dict:
    """
    Gate 1: 構造存在 — norm-dominant geometry は再現するか

    Criteria:
      - |corr(PC1, norm)| > 0.96 (across all models/texts)
      - norm-normalization により PC1 variance が大幅低下
      - residual で少なくとも position signal が確認される
    """
    all_corrs = []
    all_collapses = []
    position_signals = []
    details = []

    for model_result in results["models"]:
        model_name = model_result["model_name"]
        for text_result in model_result["texts"]:
            text_type = text_result["text_type"]
            mid = text_result["mid_layer"]

            layer_data = mid["analysis"]
            pc1_corr_norm = abs(layer_data["raw_pca"]["pc_correlations"][0]["corr_norm"])
            pc1_collapse = layer_data["normalized_pca"]["pc1_variance_collapse"]

            # Best position signal in normalized residual (top 3 PCs)
            best_pos_r = max(
                abs(pc["corr_position"])
                for pc in layer_data["normalized_pca"]["pc_correlations"][:3]
            )

            all_corrs.append(pc1_corr_norm)
            all_collapses.append(pc1_collapse)
            position_signals.append(best_pos_r)

            details.append({
                "model": model_name,
                "text": text_type,
                "layer": mid["layer_index"],
                "pc1_corr_norm": round(pc1_corr_norm, 4),
                "pc1_variance_raw": round(layer_data["raw_pca"]["pc1_variance"], 4),
                "pc1_variance_normalized": round(layer_data["normalized_pca"]["pc1_variance"], 4),
                "pc1_variance_collapse": round(pc1_collapse, 4),
                "best_position_signal": round(best_pos_r, 4),
            })

    min_corr = min(all_corrs)
    mean_corr = sum(all_corrs) / len(all_corrs)
    min_collapse = min(all_collapses)
    min_pos_signal = min(position_signals)

    # GO/NO-GO判定
    norm_dominant = min_corr > 0.96
    variance_collapses = min_collapse > 0.3  # PC1 variance drops by at least 30pp
    position_exists = min_pos_signal > 0.3

    go = norm_dominant and variance_collapses
    # position is a "should have" but not strict NO-GO

    return {
        "gate": "Gate 1: Structure Existence",
        "decision": "GO" if go else "NO-GO",
        "criteria": {
            "norm_dominant": {
                "threshold": "|corr(PC1, norm)| > 0.96",
                "result": norm_dominant,
                "min_corr": round(min_corr, 4),
                "mean_corr": round(mean_corr, 4),
            },
            "variance_collapse": {
                "threshold": "PC1 variance collapse > 0.30",
                "result": variance_collapses,
                "min_collapse": round(min_collapse, 4),
            },
            "position_signal": {
                "threshold": "best |corr(position)| > 0.30 in residual",
                "result": position_exists,
                "min_signal": round(min_pos_signal, 4),
                "note": "advisory, not strict GO/NO-GO",
            },
        },
        "details": details,
    }


# ── Main ────────────────────────────────────────────────────────────────────

def main():
    print("=" * 70)
    print("PHASE 0: Structure Verification — Arc-Compression v2.0")
    print("=" * 70)

    results_dir = Path(__file__).parent.parent / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    start_time = time.time()
    all_results = {
        "experiment": "phase0_structure_verification",
        "version": "v2.0",
        "timestamp": datetime.now().isoformat(),
        "models": [],
    }

    for mcfg in MODELS:
        model_name = mcfg["name"]
        hf_id = mcfg["hf_id"]
        n_layers = mcfg["n_layers"]

        print(f"\n{'━' * 60}")
        print(f"  Model: {model_name} ({hf_id})")
        print(f"  Layers: {n_layers}, d_model: {mcfg['d_model']}, LN: {mcfg['ln_type']}")
        print(f"{'━' * 60}")

        model_result = {
            "model_name": model_name,
            "hf_id": hf_id,
            "n_layers": n_layers,
            "d_model": mcfg["d_model"],
            "ln_type": mcfg["ln_type"],
            "texts": [],
        }

        for text_key, text in SAMPLE_TEXTS.items():
            print(f"\n  Text: {text_key}")
            t0 = time.time()

            data = extract_all_layers(hf_id, text)
            T = len(data["tokens"])
            positions = np.arange(T, dtype=float)
            surprisals = compute_surprisals(data["logits"], data["input_ids"])
            pos_cats = [classify_pos(tok) for tok in data["tokens"]]

            print(f"    Tokens: {T}, extracted in {time.time()-t0:.1f}s")

            # Analyze mid layer (main target for compression)
            mid_idx = n_layers // 2
            h_mid = data["hidden_states"][mid_idx + 1]  # +1 for embedding offset
            print(f"    Analyzing mid layer {mid_idx} (shape: {h_mid.shape})")
            mid_analysis = analyze_layer(h_mid, positions, surprisals, pos_cats)

            # Quick summary
            pc1_var = mid_analysis["raw_pca"]["pc1_variance"]
            pc1_norm_r = abs(mid_analysis["raw_pca"]["pc_correlations"][0]["corr_norm"])
            pc1_norm_var = mid_analysis["normalized_pca"]["pc1_variance"]
            print(f"    PC1 variance: {pc1_var:.1%}")
            print(f"    |corr(PC1, norm)|: {pc1_norm_r:.4f}")
            print(f"    After norm-norm: PC1 variance {pc1_var:.1%} -> {pc1_norm_var:.1%}")

            # Layer-wise PC1-norm correlation (quick sweep)
            layerwise = []
            target_layers = sorted(set([
                0,                    # embedding
                1,                    # first layer
                n_layers // 4,
                n_layers // 2,
                3 * n_layers // 4,
                n_layers - 1,         # last layer
                n_layers,             # final output
            ]))

            for li in target_layers:
                if li not in data["hidden_states"]:
                    continue
                h_l = data["hidden_states"][li]
                norms_l = np.linalg.norm(h_l, axis=1)
                pca_l = PCA(n_components=min(2, h_l.shape[1], T - 1))
                proj_l = pca_l.fit_transform(h_l)
                r_l, _ = pearsonr(proj_l[:, 0], norms_l)
                layerwise.append({
                    "layer": li,
                    "label": "embedding" if li == 0 else f"layer_{li-1}" if li <= n_layers else "output",
                    "pc1_variance": float(pca_l.explained_variance_ratio_[0]),
                    "corr_pc1_norm": float(r_l),
                    "abs_corr_pc1_norm": float(abs(r_l)),
                })

            text_result = {
                "text_type": text_key,
                "n_tokens": T,
                "mid_layer": {
                    "layer_index": mid_idx,
                    "analysis": mid_analysis,
                },
                "layerwise_pc1_norm": layerwise,
            }
            model_result["texts"].append(text_result)

            del data
            gc.collect()

        all_results["models"].append(model_result)

    # ── Gate 1 Evaluation ──
    gate1 = evaluate_gate1(all_results)
    all_results["gate1"] = gate1
    all_results["elapsed_seconds"] = round(time.time() - start_time, 1)

    # ── Save JSON ──
    out_path = results_dir / "phase0_structure_verification.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)

    print(f"\n{'=' * 70}")
    print(f"  Results saved: {out_path}")
    print(f"  Elapsed: {all_results['elapsed_seconds']:.0f}s")
    print(f"{'=' * 70}")

    # ── Print Gate 1 Summary ──
    print(f"\n{'━' * 70}")
    print(f"  GATE 1 DECISION: {gate1['decision']}")
    print(f"{'━' * 70}")
    for crit_name, crit in gate1["criteria"].items():
        status = "PASS" if crit["result"] else "FAIL"
        print(f"  [{status}] {crit_name}: {crit['threshold']}")
    print()
    for d in gate1["details"]:
        print(f"    {d['model']:>15} | {d['text']:>10} | L{d['layer']:>2} | "
              f"|r|={d['pc1_corr_norm']:.4f} | "
              f"PC1: {d['pc1_variance_raw']:.1%} -> {d['pc1_variance_normalized']:.1%} | "
              f"pos_signal={d['best_position_signal']:.3f}")

    return gate1["decision"]


if __name__ == "__main__":
    decision = main()
    sys.exit(0 if decision == "GO" else 1)
