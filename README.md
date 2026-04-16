# Norm-Separated Quantization

**A Training-Free Fix for KV Cache INT4 Failures**

[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)
[![Paper](https://img.shields.io/badge/Paper-PDF-red.svg)](paper/arc_compression.pdf)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.19590278.svg)](https://doi.org/10.5281/zenodo.19590278)

## TL;DR

Naive INT4 quantization of KV caches fails catastrophically on some models (ΔPPL = +8293 on Qwen2-7B at 4096 tokens). We fix this by separating the L2 norm before quantizing:

```python
# Before: naive INT4 (can fail catastrophically)
scale = x.abs().amax(dim=-1, keepdim=True) / 7
x_q = (x / scale).round().clamp(-7, 7) * scale

# After: norm-separated per-channel INT4 (always safe)
norm = x.norm(dim=-1, keepdim=True)
direction = x / norm
scale = direction.abs().amax(dim=0, keepdim=True) / 7  # per-channel
direction_q = (direction / scale).round().clamp(-7, 7) * scale
direction_q = direction_q / direction_q.norm(dim=-1, keepdim=True)
x_q = norm * direction_q
```

**Result**: ΔPPL +8293 → +0.19 at 4096 tokens (44,000x improvement). Never hurts models where naive INT4 already works.

## Key Results

### WikiText-2 (7 models, 124M to 14B)

| Model | Params | naive INT4 | nsep+pchan | Improvement |
|-------|--------|-----------|------------|-------------|
| GPT-2 | 124M | +1.60 | +1.13 | 1.4x |
| Pythia-2.8B | 2.8B | +14.74 | +1.99 | **7.4x** |
| Pythia-6.9B | 6.9B | +22.56 | +1.21 | **18.7x** |
| Mistral-7B | 7B | +0.10 | +0.04 | 2.6x |
| Qwen2-7B | 7B | +811.61 | +0.43 | **1885x** |
| Pythia-12B | 12B | +34.22 | +4.01 | **8.5x** |
| Qwen2.5-14B | 14B | +0.55 | +0.45 | 1.2x |

### Long Context (4096 tokens)

| Model | naive INT4 | nsep+pchan | Improvement |
|-------|-----------|------------|-------------|
| Qwen2-7B | +8293 | +0.19 | **44,000x** |
| Mistral-7B | +0.11 | +0.12 | 1x (no harm) |

### 12-Model Sweep (124M to 40B)

Validated on GPT-2, OPT (125M, 1.3B, 13B), Pythia (410M, 2.8B, 6.9B, 12B), Qwen2 (0.5B, 7B), Qwen2.5-14B, Mistral-7B, and Falcon-40B. nsep+pchan achieves ΔPPL < 4.1 on all models. See [paper](paper/arc_compression.pdf) Table 2.

## Why It Works

Two independent problems cause naive INT4 to fail:

1. **Token-wise norm variation**: KV vector norms vary 2-5x across tokens, making per-row quantization scales inconsistent
2. **Activation outlier channels**: Specific dimensions have values 10-100x larger than average, corrupting the quantization scale

Norm separation fixes (1) by decoupling magnitude from direction. Per-channel quantization fixes (2) by giving each dimension its own scale. **Neither alone is sufficient** — on Qwen2-7B, nsep alone gives 4.1x improvement, perchan alone gives 2.4x, but the combination gives **744x**.

## Simulated vs Real Quantization

The main experiment code uses **simulated (fake) quantization**: values are quantized to INT4 and immediately dequantized back to floating point. This is **standard practice in quantization research** (KIVI, SmoothQuant, GPTQ use the same approach) and accurately measures the quality impact (ΔPPL) of quantization.

### Real INT4 Packing PoC

We also provide a **real INT4 packing implementation** (`experiments/poc_real_int4.py`, `poc_real_int4_7b.py`) that stores quantized values in packed `uint8` tensors (2 values per byte), achieving **actual memory reduction**:

| Model | FP16 KV | Real INT4 | Compression | naive4 ΔPPL | nsep+pchan (real) |
|-------|---------|-----------|-------------|-------------|-------------------|
| GPT-2 | 1.4 MB | 0.4 MB | **3.43x** | — | -3.86 |
| Qwen2-7B | 3.6 MB | 1.0 MB | **3.65x** | +401.3 | **+0.29** |

**Fake vs real ΔPPL difference: < 0.005** — packing is lossless. The quality results from simulated quantization are fully reproduced with actual memory reduction.

A production deployment would additionally require a fused CUDA kernel for quantize-on-write and dequantize-on-read to eliminate the Python overhead.

## Paper

📄 **[Norm-Separated Quantization: A Training-Free Fix for KV Cache INT4 Failures](paper/arc_compression.pdf)** (14 pages, 5 figures)

## Repository Structure

```
norm-separated-quantization/
├── paper/                      # LaTeX source, PDF, figures
├── experiments/                # All experiment scripts
│   ├── phase0_*.py            # Structure verification (local, M1)
│   ├── phase1_*.py            # Hidden-state compression (local)
│   ├── phase4*_*.py           # KV cache compression (local)
│   ├── phase5*_*.py           # 7B+ scaling (Colab GPU)
│   ├── phase6*_*.py           # WikiText-2 benchmarks
│   ├── phase7*_*.py           # Appendix experiments
│   ├── generate_figures.py    # Reproduce all paper figures
│   ├── compressors.py         # Compression primitives
│   └── requirements.txt
├── results/                   # All experiment results (JSON)
├── docs/                      # Experiment report, plan
├── LICENSE                    # Apache 2.0
└── README.md
```

## Quick Start

### Reproduce paper figures (no GPU needed)

```bash
cd experiments
pip install -r requirements.txt
python generate_figures.py
```

### Run local experiments (M1 Mac, 16GB)

```bash
# Phase 0: Verify arc structure
python phase0_structure_verification.py

# Phase 4b: KV cache quantization (GPT-2)
python phase4b_asymmetric_quantization.py

# WikiText-2 benchmark (GPT-2)
python phase6_figure1_wikitext.py
```

### Run GPU experiments (Colab)

Copy-paste scripts from `experiments/phase5*.py` or `phase7*.py` into Google Colab cells. Split at the `# === CELL 1 ===` / `# === CELL 2 ===` markers.

## Models Tested

| Model | Params | KV Heads | head_dim | Arch | Source |
|-------|--------|----------|----------|------|--------|
| GPT-2 | 124M | 12 | 64 | MHA | `gpt2` |
| OPT-125m | 125M | 12 | 64 | MHA | `facebook/opt-125m` |
| Pythia-410M | 410M | 16 | 64 | MHA | `EleutherAI/pythia-410m` |
| Qwen2-0.5B | 0.5B | 2 | 64 | GQA | `Qwen/Qwen2-0.5B` |
| OPT-1.3B | 1.3B | 32 | 64 | MHA | `facebook/opt-1.3b` |
| Pythia-2.8B | 2.8B | 32 | 80 | MHA | `EleutherAI/pythia-2.8b` |
| Pythia-6.9B | 6.9B | 32 | 128 | MHA | `EleutherAI/pythia-6.9b` |
| Mistral-7B | 7B | 8 | 128 | GQA | `mistralai/Mistral-7B-v0.1` |
| Qwen2-7B | 7B | 4 | 128 | GQA | `Qwen/Qwen2-7B` |
| Pythia-12B | 12B | 40 | 128 | MHA | `EleutherAI/pythia-12b` |
| OPT-13B | 13B | 40 | 128 | MHA | `facebook/opt-13b` |
| Qwen2.5-14B | 14B | 8 | 128 | GQA | `Qwen/Qwen2.5-14B` |
| Falcon-40B | 40B | 128 | 64 | MHA | `tiiuae/falcon-40b` |

## Citation

```bibtex
@article{sato2026normsep,
  title={Norm-Separated Quantization: A Training-Free Fix for KV Cache INT4 Failures},
  author={Sato, Kentaro},
  year={2026},
  doi={10.5281/zenodo.19590278},
  url={https://doi.org/10.5281/zenodo.19590278}
}
```

## Related Work

This work builds on the geometric observations from [The Arc and Its Thickness](https://github.com/metaSATOKEN/manifold_topology_experiment) (Sato, 2026), which established that Pre-LN Transformer hidden states concentrate on a norm-dominant subspace.

## License

Apache License 2.0. See [LICENSE](LICENSE).
