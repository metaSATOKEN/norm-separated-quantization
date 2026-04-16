# Zenodo Description

## Title

Norm-Separated Quantization: A Training-Free Fix for KV Cache INT4 Failures

## Authors

Kentaro Sato

## Description (English)

### The Problem

Large language models (LLMs) store key-value (KV) vectors in memory during text generation to avoid redundant computation. At long context lengths, this KV cache becomes the dominant memory bottleneck — a 7B-parameter model processing 4096 tokens requires over 2 GB of KV cache alone. INT4 quantization (storing each value in 4 bits instead of 16) is a standard solution, reducing memory by 4x. However, we find that naive INT4 quantization **fails catastrophically on certain models**, increasing perplexity by +8293 on Qwen2-7B at 4096 tokens — effectively destroying the model's output.

### The Fix

We propose **norm-separated quantization (nsep+pchan)** — a simple preprocessing step that decomposes each KV vector into its magnitude (L2 norm, stored exactly) and direction (quantized to INT4 with per-channel scaling). This addresses two independent failure modes: (1) token-wise norm variation that inflates the quantization dynamic range, and (2) activation outlier channels that corrupt the quantization scale.

The method is **4 lines of code**, requires **no training or calibration**, and adds **negligible computational overhead** (~4 MB for 1024 tokens, <1% of KV cache).

### Results

- **44,000x improvement** on the worst case (Qwen2-7B at 4096 tokens: ΔPPL +8293 → +0.19)
- **1885x improvement** on WikiText-2 (Qwen2-7B: ΔPPL +812 → +0.43)
- **Never degrades** models where naive INT4 already works (worst case: +0.24 ΔPPL)
- Validated on **12 Pre-LN models from 124M to 40B parameters** (GPT-2, Pythia, OPT, Qwen, Mistral, Falcon)
- **WikiText-2 benchmark** on 7 models (124M to 14B)
- Long-context stability verified up to 4096 tokens

### Practical Impact

For inference providers and on-device deployment:
- Drop-in replacement for naive INT4 KV cache quantization
- Eliminates unpredictable per-model quantization failures
- Enables reliable 4x KV cache memory reduction across all Pre-LN architectures
- Compatible with existing methods (KIVI, GEAR, SmoothQuant) as a preprocessing step

### Repository Contents

- `paper/` — Full paper (15 pages, 5 figures, 25 references) with LaTeX source
- `experiments/` — All experiment scripts (reproducible on M1 Mac + Google Colab)
- `results/` — Complete experimental results in JSON format (19 experiments)
- `docs/` — Detailed experiment report and original experiment plan

### Keywords

KV cache compression, quantization, large language models, INT4, activation outliers, norm separation, Pre-LN Transformer, inference optimization

## License

Apache License 2.0

## DOI

10.5281/zenodo.19602981

## Related Identifiers

- GitHub repository: https://github.com/metaSATOKEN/norm-separated-quantization
- Based on: "The Arc and Its Thickness: Geometric Decomposition of Pre-LayerNorm Transformer Hidden States" (Sato, 2026)

---

## Description (Japanese / 日本語)

### 課題

大規模言語モデル（LLM）はテキスト生成時にKey-Value（KV）ベクトルをメモリに保持しますが、長文コンテキストではこのKVキャッシュが最大のメモリボトルネックとなります。INT4量子化（各値を16ビットから4ビットに圧縮）は標準的な対策ですが、一部のモデルで**壊滅的に失敗**することを発見しました（Qwen2-7B・4096トークンで perplexity が+8293悪化）。

### 提案手法

**Norm-Separated Quantization（nsep+pchan）** — 各KVベクトルをL2ノルム（そのまま保存）と方向（INT4量子化）に分解する前処理ステップ。トークン間のノルム変動と活性化外れ値チャネルという2つの独立した失敗要因に同時に対処します。

実装は**4行のコード追加**のみ。学習・キャリブレーション不要、計算オーバーヘッドは無視可能。

### 主要結果

- Qwen2-7B（4096トークン）: naive INT4 ΔPPL +8293 → nsep+pchan **+0.19**（**44,000倍改善**）
- 12モデル（124M〜40B）で検証、naive INT4が正常なモデルでは一切悪化しない
- WikiText-2ベンチマーク: 7モデル（124M〜14B）で確認

### 事業的インパクト

- 推論事業者・エッジデバイス展開向けの即座に適用可能なドロップインソリューション
- モデルごとの量子化失敗を予測不要で排除
- 既存手法（KIVI、GEAR、SmoothQuant）の前処理ステップとして統合可能
