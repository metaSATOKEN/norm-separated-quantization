# Experiment Scripts

## Environment

- **Local (Phase 0-4b, 6-local)**: Apple M1, 16GB RAM. GPT-2 and small models only.
- **Colab GPU (Phase 5-7)**: NVIDIA RTX PRO 6000 Blackwell (102GB) or T4 (15GB). Required for 7B+ models.

## Phase Overview

| Phase | Script | Environment | What it tests |
|-------|--------|-------------|---------------|
| 0 | `phase0_structure_verification.py` | Local | Arc structure reproduction |
| 1 | `phase1_single_layer_compression.py` | Local | Hidden-state compression (negative result) |
| 2 | `phase2_rank_performance_curve.py` | Local | Rank-performance curve |
| 4 | `phase4_kv_cache_compression.py` | Local | KV cache norm-sep vs PCA |
| 4b | `phase4b_asymmetric_quantization.py` | Local | Asymmetric + norm-sep quantization |
| 4b | `phase4b_cross_model.py` | Local | Cross-model verification (GPT-2, Pythia-410M, Qwen2-0.5B) |
| 5c | `phase5c_arc_smoothquant.py` | Colab | Arc + outlier-aware fusion (Qwen2-7B, Pythia-6.9B) |
| 5b | `phase5b_qwen_anomaly.py` | Colab | INT3 > INT4 anomaly investigation |
| 5e | `phase5e_full_arc_sweep.py` | Colab | 12-model sweep (124M-40B) |
| 6 | `phase6_figure1_wikitext.py` | Local | Figure 1 + WikiText-2 (GPT-2) |
| 6b | `phase6b_wikitext_7b.py` | Colab | WikiText-2 (Pythia-6.9B, Qwen2-7B) + Figure 1 |
| 7 | `phase7_appendix_experiments.py` | Colab | Long context, KIVI comparison, memory |
| 7b | `phase7b_longctx_qwen_mistral.py` | Colab | Long context (Qwen2-7B, Mistral-7B, 256-4096t) |

## Running Local Experiments

```bash
pip install -r requirements.txt
python phase0_structure_verification.py
python phase4b_asymmetric_quantization.py
python phase6_figure1_wikitext.py
python generate_figures.py
```

## Running Colab Experiments

1. Open Google Colab with GPU runtime
2. Copy-paste the script content into cells
3. Split at `# === CELL 1 ===` / `# === CELL 2 ===` markers
4. Run cells sequentially

## Output

All experiments output JSON results to stdout. Save them to `results/` for archival.
