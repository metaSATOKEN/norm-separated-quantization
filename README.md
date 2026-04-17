# Norm-Separated Quantization

**A Training-Free Fix for KV Cache INT4 Failures**

[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)
[![Paper](https://img.shields.io/badge/Paper-PDF-red.svg)](paper/arc_compression.pdf)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.19602981.svg)](https://doi.org/10.5281/zenodo.19602981)

## TL;DR

Naive absmax INT4 quantization of KV caches works on many models but fails catastrophically on certain ones (ΔPPL +8293 on Qwen2-7B at 4096 tokens). We fix the catastrophic cases by separating the L2 norm before quantizing:

```python
# Before: naive INT4 (catastrophic on some models)
scale = x.abs().amax(dim=-1, keepdim=True) / 7
x_q = (x / scale).round().clamp(-7, 7) * scale

# After: norm-separated per-channel INT4
norm = x.norm(dim=-1, keepdim=True)
direction = x / norm
scale = direction.abs().amax(dim=0, keepdim=True) / 7  # per-channel
direction_q = (direction / scale).round().clamp(-7, 7) * scale
direction_q = direction_q / direction_q.norm(dim=-1, keepdim=True)
x_q = norm * direction_q
```

**Pathological case** (Qwen2-7B): ΔPPL +8293 → +0.19 at 4096 tokens (44,000× improvement in this case). **Other models in our 12-model evaluation**: we did not observe catastrophic degradation from nsep+pchan (worst case: +0.24 ΔPPL on OPT-125m).

## Key Results

### WikiText-2 (8 models, 124M to 14B)

| Model | Params | naive INT4 | nsep+pchan | Improvement |
|-------|--------|-----------|------------|-------------|
| GPT-2 | 124M | +1.60 | +1.13 | 1.4× |
| Pythia-410M | 410M | +171.13 | +23.46 | **7.3×** |
| Pythia-2.8B | 2.8B | +14.74 | +1.99 | **7.4×** |
| Pythia-6.9B | 6.9B | +22.56 | +1.21 | **18.7×** |
| Mistral-7B | 7B | +0.10 | +0.04 | 2.6× |
| Qwen2-7B | 7B | +811.61 | +0.43 | **1885×** |
| Pythia-12B | 12B | +34.22 | +4.01 | **8.5×** |
| Qwen2.5-14B | 14B | +0.55 | +0.45 | 1.2× |

### Long Context (4096 tokens)

| Model | naive INT4 | nsep+pchan | Improvement |
|-------|-----------|------------|-------------|
| Qwen2-7B | +8293 | +0.19 | **44,000× (pathological case)** |
| Mistral-7B | +0.11 | +0.12 | matched |

### Needle-in-Haystack: Factual Retrieval

Does KV cache quantization make models forget facts buried in context?

**Single-needle** (one secret code hidden in 185–1413 token haystack):

| Model | Outlier ratio | naive INT4 | nsep+pchan |
|-------|:---:|:---:|:---:|
| Qwen2-7B | 8.6× | **0/15** | **15/15** |
| Pythia-6.9B | 4.6× | 15/15 | 15/15 |
| Mistral-7B | 3.1× | 15/15 | 15/15 |

**Multi-needle** (3–5 secrets across 280–2459 tokens):

| Model | Outlier ratio | baseline | naive INT4 | nsep+pchan |
|-------|:---:|:---:|:---:|:---:|
| **Qwen2-7B** | **8.6×** | **26/26** | **0/26** | **26/26** |
| Qwen2.5-14B | 3.5× | 26/26 | 26/26 | 26/26 |

On Qwen2-7B, naive INT4 causes **complete loss of all embedded facts** (0/26). nsep+pchan4 **fully recovers every needle** (26/26). Models with low outlier ratios retain all needles even with naive INT4.

## K/V Outlier Asymmetry Study (11 Models, 6 Vendors)

A follow-up study on 11 additional open models looked at Key and Value outlier ratios separately, to better characterize when naive INT4 actually breaks. Full table and analysis in **Appendix H** of the paper.

| Model | K_max | V_max | Layer 0 K | baseline | naive4 | nsep+pchan4 |
|-------|:----:|:----:|:---:|:---:|:---:|:---:|
| **Qwen2-7B** (pathological) | **17.23** | 6.09 | **17.23** | 26/26 | **0/26** | **26/26** |
| Qwen2.5-14B | 10.65 | 7.71 | — | 26/26 | 26/26 | 26/26 |
| Mistral-7B | 6.17 | 16.47 | — | 15/15 | 15/15 | 15/15 |
| Gemma 4 E2B-it | 5.56 | 7.48 | — | 16/16 | 16/16 | 16/16 |
| Gemma 4 26B-A4B (MoE) | 6.82 | 10.22 | — | 21/21 | 21/21 | 21/21 |
| Gemma 4 31B-it | 8.92 | **39.63** | 6.59 | 21/21 | 21/21 | 21/21 |
| Phi-3-mini-4k | 5.85 | 24.37 | 4.33 | 21/21 | 21/21 | 21/21 |
| Phi-3-medium-4k | 5.67 | **34.99** | 2.82 | 21/21 | 21/21 | 21/21 |
| Llama-3.1-8B-Instruct | 8.81 | 4.32 | 3.09 | 21/21 | 21/21 | 21/21 |
| Llama-3.2-3B-Instruct† | 6.89 | 4.74 | 5.84 | 16/21 | 16/21 | 16/21 |
| DeepSeek-LLM-7B-Chat‡ | 9.07 | 10.04 | 7.45 | 9/21 | 10/21 | 6/21 |

† Safety refusal on the longest-context config (identical across all three methods — compression does not cause the refusal).
‡ Marginal multi-needle capability at FP16 baseline (9/21); all methods within baseline variance. Capability-limited, not quantization-limited.

**Observations from our sample** (n=1 catastrophic case):

- **V outliers up to 39.63× were tolerated** without NIAH degradation (Gemma 4 31B Layer 12, measured 21/21 at baseline/naive4/nsep).
- **Late-layer K outliers up to 10.65× were tolerated** (Qwen2.5-14B Layer 30 of 48, 26/26 at naive4).
- The one catastrophic case (Qwen2-7B) coincides with **K_max ≈ 17× concentrated at Layer 0**, the first attention in the network. We present this as a **candidate pathology signature**, not an established threshold — a single catastrophic case is statistically insufficient to claim a necessary condition.
- **Training recipe appears to dominate architecture** for outlier patterns: MHA vs GQA vs MQA does not predict the pattern; but different vendors (Meta / Google / Microsoft / Mistral / DeepSeek / Alibaba) produce visibly different signatures.

## Why It Works

Two independent problems cause naive INT4 to fail:

1. **Token-wise norm variation**: KV vector norms vary 2–5× across tokens, making per-row quantization scales inconsistent.
2. **Activation outlier channels**: Specific dimensions have values 10–100× larger than average, corrupting the quantization scale for the other dimensions.

Norm separation fixes (1) by decoupling magnitude from direction. Per-channel quantization fixes (2) by giving each dimension its own scale. **Neither alone is sufficient** — on Qwen2-7B, nsep alone gives 4.1× improvement, per-channel alone gives 2.4×, but the combination gives **744×**.

The K/V asymmetry observed in Appendix H is consistent with this mechanism: K determines *where* attention is placed (so errors in K corrupt the attention pattern), while V determines *what value is read at the already-chosen positions* (so errors in V produce noisy values without changing attention structure). Layer 0 K errors are especially harmful because they propagate through every subsequent attention.

## Simulated vs Real Quantization

The main experiment code uses **simulated (fake) quantization**: values are quantized to INT4 and immediately dequantized back to floating point. This is **standard practice in quantization research** (KIVI, SmoothQuant, GPTQ use the same approach) and accurately measures the quality impact (ΔPPL).

### Real INT4 Packing PoC

We also provide a **real INT4 packing implementation** (`experiments/poc_real_int4.py`, `poc_real_int4_7b.py`) that stores quantized values in packed `uint8` tensors (2 values per byte), achieving **actual memory reduction**:

| Model | FP16 KV | Real INT4 | Compression | naive4 ΔPPL | nsep+pchan (real) |
|-------|---------|-----------|-------------|-------------|-------------------|
| GPT-2 | 1.4 MB | 0.4 MB | **3.43×** | — | −3.86 |
| Qwen2-7B | 3.6 MB | 1.0 MB | **3.65×** | +401.3 | **+0.29** |

**Fake vs real ΔPPL difference: < 0.005** — packing is lossless within our measurement. The quality results from simulated quantization are reproduced with actual memory reduction.

A production deployment would additionally require a fused CUDA kernel for quantize-on-write and dequantize-on-read to eliminate the Python overhead.

## Overhead

- **Memory**: < 0.1% of KV cache (one FP16 norm scalar per token).
- **Latency**: Our unoptimized Python implementation adds **+21% decode latency** on 7B-class models. Given the low arithmetic intensity (one L2 norm + one division per vector), we expect this to approach zero with a fused CUDA kernel. We have not yet implemented such a kernel.

## Paper

📄 **[Norm-Separated Quantization: A Training-Free Fix for KV Cache INT4 Failures](paper/arc_compression.pdf)** (21 pages, 5 figures, 8 appendices)

**V3 additions** (April 17, 2026):
- Appendix H: K/V Outlier Asymmetry across 11 models from six vendors (Meta, Google, Microsoft, Mistral AI, DeepSeek, Alibaba), with mechanistic interpretation of K-vs-V asymmetry and refined characterization of the catastrophic failure pattern as a *candidate* signature.
- Strengthened Limitations section with explicit statistical-power disclaimer (n=1 catastrophic case).
- Honest latency reporting (+21% Python decode) with expected path to zero via fused kernels.

## Repository Structure

```
norm-separated-quantization/
├── paper/                              # LaTeX source, PDF, figures
│   ├── arc_compression.tex
│   ├── arc_compression.pdf             # latest (V3)
│   ├── arc_compression_v2.pdf          # archival (pre-Appendix H)
│   └── arc_compression_v3.pdf          # archival (V3 snapshot)
├── experiments/
│   ├── phase0_*.py … phase8_*.py       # Main experimental pipeline
│   ├── generate_figures.py             # Reproduce all paper figures
│   ├── compressors.py                  # Compression primitives
│   ├── poc_real_int4*.py               # Real INT4 packing PoC
│   ├── poc_needle*.py                  # Needle-in-Haystack experiments
│   ├── poc_multi_needle*.py            # Multi-needle retrieval
│   ├── poc_gemma4_*.py                 # Gemma 4 family (E2B / 26B-A4B / 31B)
│   ├── poc_k_vs_v_outlier.py           # K vs V asymmetry: 5 models
│   ├── poc_phi3_deepseek_kv_outlier.py # Phi-3 + DeepSeek K/V outliers
│   ├── poc_llama_kv_outlier_colab.py   # Llama 3.1 8B + 3.2 3B K/V outliers
│   ├── poc_niah_verification_5models.py# NIAH for all 5 new models
│   └── requirements.txt
├── results/                            # All experiment results (JSON)
├── docs/                               # Experiment report, plan
├── LICENSE                             # Apache 2.0
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

# Real INT4 packing PoC
python poc_real_int4.py
```

### Run GPU experiments (Colab)

Copy-paste scripts from `experiments/phase5*.py`, `phase7*.py`, or `poc_*.py` into Google Colab cells. Split at the `# === CELL 1 ===` / `# === CELL 2 ===` markers. The Gemma 4 / Phi-3 / Llama / K-vs-V scripts assume Colab Pro (95 GB VRAM) for the larger models; smaller variants run on a free T4 tier.

## Models Evaluated

### Main sweep (12 models; PPL and NIAH)

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

### K/V asymmetry study (6 additional models)

| Model | Params | KV Heads | head_dim | Arch | Source |
|-------|--------|----------|----------|------|--------|
| Gemma 4 E2B-it | 2B | 1 | 256 | MQA | `google/gemma-4-E2B-it` |
| Gemma 4 26B-A4B-it | 26B (4B active) | — | — | MoE | `google/gemma-4-26B-A4B-it` |
| Gemma 4 31B-it | 31B | — | — | dense | `google/gemma-4-31B-it` |
| Phi-3-mini-4k-instruct | 3.8B | 32 | 96 | MHA | `microsoft/Phi-3-mini-4k-instruct` |
| Phi-3-medium-4k-instruct | 14B | 10 | 128 | GQA | `microsoft/Phi-3-medium-4k-instruct` |
| DeepSeek-LLM-7B-Chat | 7B | 32 | 128 | MHA | `deepseek-ai/deepseek-llm-7b-chat` |
| Llama-3.1-8B-Instruct | 8B | 8 | 128 | GQA | `meta-llama/Llama-3.1-8B-Instruct` |
| Llama-3.2-3B-Instruct | 3B | 8 | 128 | GQA | `meta-llama/Llama-3.2-3B-Instruct` |

## Citation

```bibtex
@article{sato2026normsep,
  title={Norm-Separated Quantization: A Training-Free Fix for KV Cache INT4 Failures},
  author={Sato, Kentaro},
  year={2026},
  doi={10.5281/zenodo.19602981},
  url={https://doi.org/10.5281/zenodo.19602981}
}
```

## Related Work

This work builds on the geometric observations from [The Arc and Its Thickness](https://github.com/metaSATOKEN/manifold_topology_experiment) (Sato, 2026), which established that Pre-LN Transformer hidden states concentrate on a norm-dominant subspace.

For related KV cache compression methods, see the paper's Related Work section (KIVI, KVQuant, GEAR, H2O, StreamingLLM, SnapKV, Scissorhands).

## License

Apache License 2.0. See [LICENSE](LICENSE).
