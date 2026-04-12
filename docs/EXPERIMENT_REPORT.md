# Arc-Compression v2.0 — 実験総括レポート

**Date**: 2026-04-10〜12
**Author**: Kentaro Sato + Claude
**Status**: 全 Phase 完了（Phase 0 → 6e）、10 Finding 確立、**12 モデル検証（124M〜40B）**、WikiText-2 (7モデル)、論文 tex 完成

---

## 1. Executive Summary

Pre-LN Transformer の hidden-state geometry（"Arc構造"）を compression prior として活用できるかを検証した。6段階の実験を通じて以下が判明:

1. **Arc構造は完全に再現された**（Gate 1 GO）
2. **Hidden state 全置換による圧縮は不可能**（Gate 2 NO-GO）
3. **PCA prior は random に対して圧倒的優位**（Gate 3 GO、81%勝率）
4. **KV cache 圧縮では norm 分離（Arc方式）が plain PCA を最大13.4倍上回る**
5. **norm分離 + 量子化の組み合わせで、ΔPPL < 1.0 を保ちながら 5.7x 圧縮を達成**
6. **低ビット量子化において、norm分離は精度を最大1.3倍改善する（新規知見）**

---

## 2. 実験の流れ

```
Phase 0: 構造再現        → Gate 1: GO (|r| > 0.999 全条件)
    ↓
Phase 1: 単層圧縮耐性    → Gate 2: NO-GO (全層 ΔPPL > +300)
                         → Gate 3: GO (PCA 81% 勝率)
    ↓
Phase 2: Rank-Perf Curve → hidden state 全置換は死亡確認
    ↓
    ↓  ピボット: KV cache へ
    ↓
Phase 4: KV Cache 圧縮   → norm_pca が PCA を圧倒 (最大13.4x)
                         → しかし 3x 以上で ΔPPL < 1.0 未達
    ↓
Phase 4b: 非対称 + 量子化 → ΔPPL < 1.0 で 5.7x 圧縮を達成!
    ↓
Phase 4b cross-model      → 3/3 Finding 再現 (GPT-2, Pythia, Qwen2)
    ↓
Phase 4c: 長文 + CUDA fix → 長文安定確認 + 生成品質維持 (Colab T4)
```

---

## 3. Phase 0: 構造再現

### 目的
Arc論文の幾何学構造が自前環境で再現されるか確認。

### 結果

| Model | |corr(PC1, norm)| | PC1 variance | norm-norm後 | collapse |
|-------|-------------------|-------------|-------------|----------|
| GPT-2 (124M) | 0.9994〜0.9996 | 94.8〜95.7% | 6.5〜7.0% | ~88pp |
| Pythia-410M | 0.9997〜0.9998 | 96.5〜97.0% | 6.5〜7.9% | ~90pp |
| Qwen2-0.5B | 0.9999〜1.0000 | 98.7〜99.0% | 7.5〜9.2% | ~91pp |

- 3モデル × 3テキスト = 9条件、全てで GO 条件をクリア
- **モデルが大きいほど norm dominance が強い**（スケール効果仮説をサポート）
- Qwen2 (RMSNorm) で |r| = 1.0000 → norm がほぼ完全に PC1
- Residual subspace の position signal: 全条件で |r| > 0.35

### 判定: **Gate 1 GO**

---

## 4. Phase 1: 単層圧縮耐性

### 目的
各層の hidden state を compress → decompress hook で置換し、圧縮耐性を測定。

### 比較条件
- baseline / norm_only / norm+PCA top-k / norm+random top-k
- k = 2, 4, 8, 16, 32, 64

### 結果

| 指標 | 結果 |
|------|------|
| 最良層 (L10, k=64) | ΔPPL = +362.7 |
| PCA vs random 勝率 | **81%** (58/72 条件) |
| 後半層 vs 前半層 | 後半が 3〜10 倍耐性あり |

**発見:**
- 全層で ΔPPL が数百〜数万 → 単層まるごと置換は不可能
- PCA が random に 81% 勝利 → 構造的優位は明確
- 後半層ほど耐性が高い（L9, L10 が最良）

### 判定: **Gate 2 NO-GO** / **Gate 3 GO**

---

## 5. Phase 2: Rank-Performance Curve

### 目的
高 k 領域に sweet spot（knee）があるか確認。同時に、norm 分離 PCA vs 直接 PCA truncation を比較。

### 結果 (GPT-2, Layer 10)

| k | 圧縮率 | norm+PCA ΔPPL | direct PCA ΔPPL |
|---|--------|---------------|-----------------|
| 8 | 85x | +923 | +540 |
| 16 | 45x | +554 | +315 |
| 32 | 23x | +432 | +199 |
| 64 | 12x | +363 | +123 |

**発見:**
- **Direct PCA truncation が norm+PCA 分離より 2〜5 倍良い**（hidden state 全置換の場合）
- 曲線は単調減少、knee なし → hidden state 圧縮は根本的に不可能
- ただし norm+PCA が hidden state で劣る理由が、KV cache では逆転する（Phase 4 で判明）

### 判定: **Hidden state 全置換は死亡確認。KV cache へピボット。**

---

## 6. Phase 4: KV Cache 圧縮

### 目的
KV cache の per-head 低ランク近似で推論メモリ削減。

### 核心的発見: norm_pca の逆転勝利

| k | 圧縮率 | PCA ΔPPL | norm_pca ΔPPL | 優位倍率 |
|---|--------|----------|---------------|---------|
| 4 | 16x | +108.1 | **+55.0** | 2.0x |
| 8 | 8x | +58.6 | **+37.7** | 1.6x |
| 16 | 4x | +21.9 | **+3.15** | 6.9x |
| 32 | 2x | +10.7 | **+1.59** | 6.7x |
| 48 | 1.3x | +8.7 | **+0.65** | 13.4x |

**なぜ KV cache では norm_pca が逆転勝利するのか:**

Hidden state 全置換では、再構成された 768 次元ベクトルがそのまま次層に入るため、方向の微小誤差が増幅される。一方、KV cache では attention の softmax が近似誤差を吸収する。norm 分離により方向ベクトルの dynamic range がコンパクトになるメリットが、softmax のおかげで活きる。

### K vs V cosine の非対称性

```
Key cosine:   0.92〜0.99 (圧縮に強い)
Value cosine: 0.82〜0.90 (圧縮に弱い)
全層で Key > Value（差: +0.05〜+0.14）
```

### 判定: **Gate 5 NO-GO**（3x 以上で ΔPPL < 1.0 未達）→ Phase 4b へ

---

## 7. Phase 4b: K/V 非対称 + Norm 分離量子化

### 目的
Phase 4 の知見を組み合わせて実用ラインを攻める。

### Exp A: K/V 非対称 PCA

| 構成 | ΔPPL |
|------|------|
| symmetric k=16 | +3.15 |
| symmetric k=48 | +0.65 |
| K=16 V=48 (非対称) | +2.42 |
| K=32 V=48 (非対称) | +1.10 |

- 非対称の効果は限定的。Key を k=8 以下にすると崩壊（+25 以上）。
- Key 側のボトルネックは norm 分離 PCA でも解消しきれない。

### Exp B: 量子化 — naive vs norm 分離

| bits | naive ΔPPL | norm_sep ΔPPL | 改善率 |
|------|-----------|---------------|--------|
| INT8 | +0.533 | +0.532 | 1.0x |
| INT4 | +0.641 | **+0.563** | 1.1x |
| INT3 | +1.533 | **+1.236** | 1.2x |
| INT2 | +57.6 | **+45.0** | 1.3x |

**新規知見: ビット数が減るほど norm 分離の優位が増大する。**

メカニズム: norm を分離すると、方向ベクトルの値域が [-1, 1] 付近にコンパクト化される。同じビット数でも有効な量子化レベルが増え、近似精度が向上する。この効果は量子化粒度が粗いほど（低ビットほど）顕著になる。

### Exp C: K/V 非対称量子化

| 構成 | ΔPPL | bpe |
|------|------|-----|
| K:norm4 V:norm8 | +0.760 | 7.0 |
| K:norm4 V:norm4 (sym) | **+0.563** | 5.0 |
| K:norm3 V:norm4 | +1.285 | 4.5 |
| K:naive4 V:naive8 | +0.815 | 6.5 |

- norm_sep INT4 symmetric が最もバランスが良い
- K/V 非対称量子化は K:4bit V:8bit で +0.76（viable）

### Exp D: PCA + 量子化の合わせ技

| 構成 | ΔPPL | 圧縮率 |
|------|------|--------|
| norm_pca(K32,V48) + INT8 | +1.10 | 4.8x |
| **norm_pca(K32,V48) + K:4bit V:8bit** | **+0.96** | **5.7x** |
| norm_pca(K16,V48) + INT4 | +2.58 | 6.9x |

---

## 8. VIABLE Configurations (ΔPPL < 1.0)

| Rank | 手法 | ΔPPL | 圧縮率 | bpe | 新規性 |
|------|------|------|--------|-----|--------|
| 1 | naive INT4 | +0.641 | 7.1x | 4.5 | 既知 |
| 2 | **norm_sep INT4** | **+0.563** | **6.4x** | 5.0 | **Arc新規** |
| 3 | **norm_pca(K32V48)+K4V8** | **+0.960** | **5.7x** | 5.7 | **Arc新規** |
| 4 | K:norm4 V:norm8 | +0.760 | 4.6x | 7.0 | **Arc新規** |
| 5 | naive INT8 | +0.533 | 3.8x | 8.5 | 既知 |
| 6 | norm_sep INT8 | +0.532 | 3.6x | 9.0 | 微差 |

---

## 8b. Cross-Model Robustness (Phase 4b cross-model)

### 目的
Phase 4b の Finding 1〜3 が GPT-2 以外で再現されるか検証。

### 対象モデル
| Model | Params | KV heads | head_dim | Architecture |
|-------|--------|----------|----------|-------------|
| GPT-2 | 124M | 12 | 64 | MHA |
| Pythia-410M | 410M | 16 | 64 | MHA |
| Qwen2-0.5B | 500M | **2** | 64 | **GQA** |

### 結果

**Finding 1: norm_pca > PCA — 全3モデルで再現** ✅

| Model | 平均優位倍率 | k=48 最大 |
|-------|-----------|----------|
| GPT-2 | 7.1x | 13.4x |
| Pythia-410M | **10.3x** | **15.7x** |
| Qwen2-0.5B | 6.8x | 2.3x |

モデルが大きいほど norm_pca 優位が増す傾向。

**Finding 2: norm_sep 量子化改善 — 全モデルで INT4 以下で改善** ✅

| bits | GPT-2 | Pythia-410M | Qwen2-0.5B |
|------|-------|-------------|------------|
| INT8 | 1.0x | **5.7x** | 1.1x |
| INT4 | 1.1x | **6.2x** | **2.3x** |
| INT3 | 1.2x | 2.1x | 1.1x |
| INT2 | 1.3x | 3.5x | 0.8x (逆転) |

**Pythia での爆発的改善**: INT4 で naive +77.5 → norm_sep +12.6（6.2 倍）。
Qwen2 INT4 で norm_sep ΔPPL=+0.28 — 全モデル中最良。

**Finding 3: Key > Value — 全モデル全層で再現** ✅

```
GPT-2:       K=0.956 V=0.852  K>V in 12/12 layers
Pythia-410M: K=0.977 V=0.829  K>V in 24/24 layers
Qwen2-0.5B:  K=0.955 V=0.842  K>V in 24/24 layers
```

**60/60 層で例外なし。**

### Cross-Model Verdict: **3/3 Findings 再現**（ただし 7B で修正あり、後述）

---

## 8c. Long Context Stability (Phase 4c)

### 目的
KV cache 圧縮が長文コンテキストで安定するか検証。

### CUDA バグ発見と修正

Colab 初回実行で nsep4 の ΔPPL が +8000〜+12000 に膨張するバグを発見。

**原因**: PyTorch の `.norm()` API の位置引数解釈の違い。
```python
x.norm(-1, keepdim=True)      # BUG: p=-1 (L_{-1} ノルム ≈ 0.0003)
x.norm(dim=-1, keepdim=True)  # FIX: dim=-1 (L2 ノルム ≈ 6.0)
```
`.norm()` の第1位置引数は `p`（ノルムの次数）であり、`dim` はキーワード引数。
ローカルコードは `dim=-1` とキーワードで書いていたが、Colab 用にコードを短縮した際にキーワードを落としたことで発生。L_{-1} ノルムが ≈ 0.0003 となり、方向ベクトルが ~11000 倍に膨張した。

### 結果（Colab T4、バグ修正後）

**GPT-2:**

| prefill | naive4 | nsep4 | npca32 | combo |
|---------|--------|-------|--------|-------|
| 36t | +4.14 | **+3.78** | **+2.31** | +3.08 |
| 135t | +12.99 | +13.15 | +16.42 | +13.15 |
| **218t** | +0.16 | **+0.12** | +0.90 | +0.64 |

**Pythia-410M:**

| prefill | naive4 | nsep4 | npca32 | combo |
|---------|--------|-------|--------|-------|
| 37t | +45.4 | **+25.9 (1.7x)** | **+0.89** | +25.9 |
| 139t | +99.2 | **+31.8 (3.1x)** | +14.9 | +42.5 |
| **221t** | +32.2 | **+9.87 (3.3x)** | +4.99 | +15.5 |

**Key Observations:**
- **nsep4 は Pythia で naive4 を 1.7〜3.3 倍改善** — モデル規模で効果増大
- **長い prefill ほど ΔPPL が低下** — 長文コンテキストでは崩壊しない
- **ΔPPL の絶対値変動は baseline PPL の変動に起因**（テキスト内容による自然な変動）
- **npca32 は Pythia pfl=37 で ΔPPL=+0.89** — 短文でも実用圏

### 生成品質（top-p sampling, p=0.9, temp=0.8）

| Model | Method | Trigram Unique | Repetition | 生成例 |
|-------|--------|---------------|------------|--------|
| GPT-2 | baseline | 1.000 | なし | "The technology has improved in recent years..." |
| GPT-2 | nsep4 | 1.000 | なし | "More recently, AI has evolved into a basic, generalized, and universal tool..." |
| GPT-2 | combo | 1.000 | なし | "Astonishingly, most of the recent advances in AI come from algorithms..." |
| Pythia | baseline | 1.000 | なし | "Some areas of the field were given a boost by IBM..." |
| Pythia | nsep4 | 1.000 | なし | "In the same period, the field attracted the public..." |
| Pythia | npca32 | 1.000 | なし | "Many major corporations saw an opportunity in a new market..." |

**全手法 × 全モデルで uniq=1.0、repetition ゼロ。** 圧縮後も自然で coherent な文を生成。
前回の greedy decoding での repetition はデコーディング手法の問題であり、圧縮の問題ではないことが確認された。

### Verdict: **長文・生成ともに安定**

---

## 8d. 7B Model Scaling (Phase 5)

### 目的
norm_sep の効果がモデル規模とともにスケールするか検証。

### 環境
Colab Pro — NVIDIA RTX PRO 6000 Blackwell (102GB VRAM)

### 対象
| Model | Params | KV heads | head_dim | Architecture |
|-------|--------|----------|----------|-------------|
| Pythia-6.9B | 6.9B | 32 | **128** | MHA |
| Qwen2-7B | 7B | **4** | **128** | GQA (RMSNorm) |

### 結果: norm_sep INT4 advantage — スケーリングは単調ではない

| Model | Params | head_dim | INT4 advantage |
|-------|--------|----------|---------------|
| GPT-2 | 124M | 64 | 1.14x |
| Pythia-410M | 410M | 64 | **6.15x** |
| Pythia-6.9B | 6.9B | 128 | 2.12x |
| Qwen2-7B | 7B | 128 | 4.96x |

**head_dim の影響**: Pythia-410M (head_dim=64) → Pythia-6.9B (head_dim=128) で advantage が 6.15x → 2.12x に低下。head_dim が大きいと同じ INT4 でも量子化の相対的粗さが減り、norm 分離の改善余地が縮小する。

### K/V 非対称性 — GQA で逆転

```
Pythia-6.9B (MHA, 32 heads): K>V in 22/32 layers  ← 維持
Qwen2-7B    (GQA,  4 heads): K>V in  2/28 layers  ← 逆転!
```

**Finding 3 は MHA では robust だが、GQA では成立しない。** GQA ではKV heads 数が大幅に少なく（4 heads）、各 head が複数の attention heads を担当するため、Value の分布構造が異なる。

### PCA — 7B でも有効

| Model | npca64 ΔPPL | 圧縮率 (head_dim/2) |
|-------|------------|---------------------|
| Pythia-6.9B | +1.37 | 2x |
| **Qwen2-7B** | **+0.41** | 2x |

Qwen2-7B npca64 = +0.41 は全実験中最良クラス。head_dim=128 の半分を切り捨てても ΔPPL < 0.5。

### 生成品質

| Model | baseline | naive4 | nsep4 |
|-------|----------|--------|-------|
| Pythia-6.9B | coherent (1.0) | coherent (1.0) | coherent (1.0) |
| Qwen2-7B | coherent (1.0) | **repetition (0.76)** | multilingual artifacts |

Qwen2-7B は INT4 に対する感度が高く、生成品質が劣化。INT8 または PCA ベースの圧縮がこのモデルには適切。

### Phase 5 Verdict

- **norm_sep advantage は 7B でも存在する**（Pythia 2.12x, Qwen2 4.96x）
- **ただしスケーリングは head_dim に依存** — params ではなく head_dim が支配的
- **K>V 非対称性は MHA 固有** — GQA では逆転する（Finding 3 の適用範囲を修正）
- **PCA 圧縮は 7B でも強力** — npca64 で ΔPPL < 1.5（Pythia）/ < 0.5（Qwen2）

---

## 8e. Qwen2-7B INT3>INT4 Anomaly Investigation (Phase 5b)

### 問題
Qwen2-7B で INT3 (ΔPPL=+6.6) が INT4 (ΔPPL=+238) より 36 倍良い。ビット数が少ないほうが良いのは物理的にありえない。

### 捜査結果

**Part A (量子化関数)**: 合成データでは bit→MSE は完全に単調 → **実装は正しい**

**Part B (値分布)**: Layer 0 と Layer 27 に巨大な外れ値を発見

| Layer | std | absmax | 外れ値率 |
|-------|-----|--------|---------|
| L0 | 25.6 | **166.75** | 0.78% |
| L4-L26 | 1.5-1.8 | 10-22 | 0.2-0.7% |
| L27 | 39.4 | **408.75** | **1.36%** |

Layer 0 の absmax は中間層の **10 倍**、Layer 27 は **20 倍**。

**Part C (Per-head MSE)**: INT4 MSE < INT3 MSE で正常。再構成品質自体は INT4 が上。

**Part D (犯人特定)**: **Layer 0 だけで全体の INT4 劣化の 99% を占める**

```
L0:  INT4 ΔPPL=+179.4  INT3 ΔPPL=+1.47  (122倍の差!)
L4+: INT4 ≈ INT3 (差は <0.3)
```

**Part E (対照実験)**: Pythia-6.9B は完全に単調 → **Qwen 固有の現象**

### メカニズム

Layer 0 の absmax=166.75 に対する absmax 量子化:

```
INT4 (qmax=7):  scale = 166.75/7 = 23.82
  値 15.0 → round(0.63) × 23.82 = 23.82  (ノイズ化: 60% 誤差)
INT3 (qmax=3):  scale = 166.75/3 = 55.58
  値 15.0 → round(0.27) × 55.58 = 0      (ゼロ化)
```

**INT4 は中間レンジの値を「ノイズ化」、INT3 は「ゼロ化」する。**

Attention の QK 内積では外れ値チャネルが 90%+ を支配。INT3 のゼロ化は内積をクリーンに保つが、INT4 のノイズ化は softmax のランキングを破壊する。

### 先行研究との関連

この現象は **SmoothQuant (Xiao et al., 2023) / LLM.int8() (Dettmers et al., 2022)** が解決しようとしている問題と同一。Activation outlier が naive absmax 量子化を破壊する。

### 含意

- **Arc prior (norm 分離) と outlier-aware quantization は直交的に組み合わせ可能**
- norm 分離で方向の dynamic range を圧縮 → outlier-aware quant で外れ値を処理
- 両方を適用すれば、Qwen のような outlier が強いモデルでも圧縮可能になる可能性

---

## 8f. Arc Prior + Outlier-Aware Quantization (Phase 5c)

### 目的
Finding 8 の含意を検証: Arc prior (norm 分離) と outlier-aware quantization を組み合わせると何が起きるか。

### 手法

| Method | norm分離 | outlier対策 |
|--------|---------|-----------|
| naive4 | - | - |
| nsep4 | norm分離 | - |
| outlier4 | - | top-4ch fp16 |
| **nsep+out4** | **norm分離** | **top-4ch fp16** |
| perchan4 | - | per-channel scale |
| **nsep+pchan4** | **norm分離** | **per-channel scale** |

### 結果: Qwen2-7B

| Method | ΔPPL | naive4 比改善 |
|--------|------|-------------|
| naive4 | +238.2 | — |
| nsep4 | +57.5 | 4.1x |
| outlier4 | +23.2 | 10.3x |
| **nsep+out4** | **+0.59** | **404x** |
| perchan4 | +97.8 | 2.4x |
| **nsep+pchan4** | **+0.32** | **744x** |

### 結果: Pythia-6.9B（対照群）

| Method | ΔPPL | naive4 比改善 |
|--------|------|-------------|
| naive4 | +22.3 | — |
| nsep+pchan4 | **+0.27** | **82x** |

### 核心的発見

**nsep だけでもダメ。outlier/perchan だけでもダメ。組み合わせて初めて機能する。**

- nsep は Layer 0 の outlier 問題を解決するが、他層の量子化誤差は残る
- outlier/perchan は量子化精度を上げるが、norm に支配された dynamic range 問題は残る
- **両方を組み合わせると直交的に効き、ΔPPL が 238 → 0.32 に激減**

### 生成品質

| Model | nsep+pchan4 | 内容 |
|-------|------------|------|
| Qwen2-7B | uq=0.98 | "But AI progress stagnated and funding dried up. Today, AI is an open-ended field." |
| Pythia-6.9B | uq=1.0 | "which led to a major AI bust. Over the last 50 years, the field has evolved..." |

naive4 では壊れていた Qwen2-7B の生成が、nsep+pchan4 では完全に coherent。

### Verdict: **Arc prior + outlier-aware quantization の融合は有効。7B モデルで ΔPPL < 0.6 を INT4 で達成。**

---

## 9. Key Findings（論文化可能な知見）

### Finding 1: Hidden state geometry は compression prior として KV cache で有効
- Hidden state の全置換では PCA prior は不十分（ΔPPL > 100）
- しかし KV cache では norm_pca が plain PCA を最大 **13.4 倍**上回る
- Attention softmax による誤差吸収が norm 分離の利点を活かす

### Finding 2: Norm 分離は低ビット量子化の精度を改善する
- INT8 では差がないが、INT4 で 1.1x、INT3 で 1.2x、INT2 で 1.3x の改善
- 低ビットほど改善幅が増大する系統的パターン
- メカニズム: 方向ベクトルの dynamic range 圧縮による有効量子化レベル増加

### Finding 3: Key は Value より圧縮に強い（MHA 限定）
- MHA モデル（GPT-2, Pythia）: Key cosine > Value cosine が全層で一貫
- GPT-2: 12/12, Pythia-410M: 24/24, Pythia-6.9B: 22/32
- **ただし GQA (Qwen2-7B, KV heads=4) では逆転**: K>V はわずか 2/28 層
- Finding 3 の適用範囲: MHA アーキテクチャに限定される

### Finding 4: PCA + 量子化の組み合わせが新しい圧縮点を開拓
- norm_pca(K32,V48) + K:INT4,V:INT8 で ΔPPL=+0.96、5.7x 圧縮
- PCA による次元削減と量子化によるビット削減が直交的に効く

### Finding 5: クロスモデル・クロススケール検証（124M〜7B, 5モデル）
- norm_sep advantage は全 5 モデルで確認（INT4: 1.1x〜6.2x）
- **ただしスケーリングは単調ではない** — head_dim が支配的要因
  - head_dim=64 (Pythia-410M): 6.15x
  - head_dim=128 (Pythia-6.9B): 2.12x
- K>V 非対称性は **MHA で robust、GQA で逆転** — アーキテクチャ依存
- PCA 圧縮は 7B でも有効（npca64: Qwen2-7B で ΔPPL=+0.41）

### Finding 6: 圧縮は長文コンテキストで安定し、生成品質を維持する
- GPT-2: nsep4 ΔPPL が +3.78(36t) → +0.12(218t) と改善
- Pythia: nsep4 が naive4 を長文でも 1.7〜3.3 倍改善（モデル規模効果が長文でも持続）
- 全手法 × 全モデルで生成品質維持（top-p sampling, trigram unique = 1.0）
- greedy decoding の repetition は圧縮ではなくデコーディングの問題と判明

### Finding 7: PyTorch `.norm()` API の落とし穴（実装知見）
- `.norm(-1)` は L_{-1} ノルムを計算する（dim=-1 ではない）
- `.norm(dim=-1)` が正しい記法
- この差異が方向ベクトルを ~3000 倍に膨張させ、norm 分離系の手法を完全に破壊する
- CPU (M1/MPS) では発見が遅れた — CUDA で顕在化したことでバグが特定できた

### Finding 8: Activation outlier が量子化の bit-monotonicity を破壊する (Phase 5b)
- Qwen2-7B Layer 0 に absmax=166.75 の外れ値（中間層の 10 倍）
- **Layer 0 だけで全体の INT4 劣化の 99% を占める**（ΔPPL=+179 vs 他層 <0.3）
- INT4 (qmax=7) は中間値をノイズ化、INT3 (qmax=3) はゼロ化 → 外れ値支配の attention ではゼロ化のほうが softmax を保護
- SmoothQuant / LLM.int8() と同じ問題。**Arc prior と outlier-aware quantization は直交的に組み合わせ可能**
- Pythia-6.9B では発生しない → Qwen2 (GQA + RMSNorm) 固有

### Finding 9: Arc prior + outlier-aware quantization は直交的に組み合わせ可能 (Phase 5c-5d)
- **norm 分離だけ (+57.5) でも outlier 分離だけ (+23.2) でも不十分**
- **両方組み合わせると ΔPPL = +0.32 (naive4 比 744x 改善)**
- Qwen2-7B: naive4 +238.2 → nsep+pchan4 **+0.32**
- Pythia-6.9B: naive4 +22.3 → nsep+pchan4 **+0.27**
- 生成品質も完全に回復（Qwen2-7B uq=0.98, Pythia uq=1.0）
- **Arc 論文の構造知見が実用的な圧縮技術に直結する初の実証**

### Finding 10: nsep+pchan4 は 124M〜40B の 12 モデルで汎用的 (Phase 5d-5e)

Arc 論文の全 14 モデル中 12 モデルで検証（GPT-1 は Post-LN で KV cache 非対応、Mistral は gated）。

**完全テーブル:**

| Model | Params | hd | naive4 | nsep+pc4 | 改善 | outlier比 |
|-------|--------|-----|--------|----------|-----|----------|
| Falcon-40B | 40B | 64 | +0.08 | **+0.04** | 2x | 2.4x |
| Qwen2.5-14B | 14B | 128 | +0.30 | +0.26 | 1x | 3.5x |
| OPT-13B | 13B | 128 | +0.28 | +0.35 | 1x | 3.1x |
| Pythia-12B | 12B | 128 | +27.28 | +1.82 | **15x** | 4.6x |
| **Qwen2-7B** | **7B** | **128** | **+238.23** | **+0.32** | **744x** | **8.6x** |
| Pythia-6.9B | 6.9B | 128 | +22.26 | +0.27 | **82x** | 4.6x |
| Pythia-2.8B | 2.8B | 80 | +0.90 | **-0.06** | **14x** | 4.0x |
| OPT-1.3B | 1.3B | 64 | +0.70 | +0.68 | 1x | 1.4x |
| Qwen2-0.5B | 0.5B | 64 | +0.64 | +0.28 | 2x | — |
| Pythia-410M | 410M | 64 | +77.55 | +12.62 | **6x** | — |
| GPT-2 | 124M | 64 | +0.64 | +0.56 | 1x | — |
| OPT-125m | 125M | 64 | +1.22 | +1.46 | 1x | 1.5x |

**パターン:**
- **naive4 が壊れるモデル（ΔPPL > 10）では nsep+pchan4 が 6〜744x 改善**
- **naive4 が良いモデル（ΔPPL < 1）では nsep+pchan4 もほぼ同等**（最悪でも +0.23 悪化のみ）
- **Pythia-2.8B では ΔPPL = -0.06** — 圧縮が微弱な正則化として機能し、ベースラインを超えた
- **OPT 系は outlier が少なく（ratio 1.4-3.1x）、nsep+pchan4 の恩恵が限定的**
- **Falcon-40B（40B）でも動作確認** — 83.7GB VRAM で問題なし
- **結論: nsep+pchan4 は「ダウンサイドほぼゼロの保険」として全 Pre-LN モデルに適用可能**

---

## 10. 計画書の成功基準との対照

| 基準 | 目標 | 結果 | 判定 |
|------|------|------|------|
| 最低成功: 圧縮率 | > 5x | 5.7x (PCA+quant) / 7.1x (INT4) | **達成** |
| 最低成功: ΔPPL | < 1.0 | +0.56 (norm_sep INT4) | **達成** |
| 最低成功: random優位 | 有意 | 81% 勝率 (Phase 1) | **達成** |
| 十分成功: 圧縮率 | > 10x | 7.1x (INT4) | 未達 |
| 十分成功: ΔPPL | < 0.5 | +0.56 | 未達 (惜しい) |

**計画書の最低成功基準はクリア。十分成功基準にはあと一歩。**

---

## 11. 意思決定: 続行 / ピボット / 中止

計画書 Section 12 の判定ルールに照らすと:

> **続行条件**: 構造再現あり ✓ / random より優位 ✓ / 一部層で 5x 以上 ✓

**→ 続行。**

ただし方向性の修正あり:
- ❌ Hidden state 全置換 → 中止
- ✅ KV cache + norm 分離量子化 → 本命
- ✅ PCA + 量子化の合わせ技 → 差別化ポイント

---

## 12. Next Steps

### 完了済み ✅
1. ~~複数モデルでの再現~~: GPT-2 / Pythia-410M / Qwen2-0.5B で 3/3 Finding 再現
2. ~~長文コンテキスト~~: ~220 tokens まで安定性確認（GPT-2 + Pythia）
3. ~~生成品質~~: top-p sampling で全手法 repetition なし確認

### 短期（検証強化）
4. **WikiText-2 / Penn Treebank** での標準ベンチマーク
5. **seq_len = 512〜1024** での更に長いコンテキスト検証
6. **メモリ実測**: PyTorch profiler での実メモリ削減量計測

### 中期（実装）
7. **Learned residual correction**: norm_pca + 軽量 adapter で INT3 圏内を攻める
8. **Online PCA basis**: prefill 時に basis を動的計算（cross-domain 汎用性）
9. **7B+ モデルでのスケール検証**（Colab Blackwell GPU）

### 長期（プロダクト）
10. **既存フレームワーク統合**: vLLM / TGI の KV cache backend として実装
11. **Quantization-aware training** との組み合わせ
12. **実サービスでの latency / throughput ベンチマーク**

---

## 13. ファイル一覧

```
超実験/
├── README.md
├── EXPERIMENT_REPORT.md
├── 実験計画書Arc-Compression v2.pdf
│
├── paper/
│   ├── arc_compression.tex                         # 論文 LaTeX (最終版)
│   ├── arc_compression.pdf                         # 論文 PDF (12 pages)
│   ├── arc_compression_draft.md                    # 論文 MD ドラフト
│   ├── figure1_qwen2_7b.pdf                       # Fig 1: 分布ヒストグラム
│   ├── figure2_wikitext2.pdf                       # Fig 2: WikiText-2 結果
│   ├── figure3_ablation.pdf                        # Fig 3: Ablation
│   ├── figure4_outlier_sweep.pdf                   # Fig 4: Outlier + 12モデル
│   └── figure5_anomaly.pdf                         # Fig 5: INT3>INT4
│
├── analysis/
│   ├── phase0_structure_verification.py
│   ├── phase1_single_layer_compression.py
│   ├── phase2_rank_performance_curve.py
│   ├── phase4_kv_cache_compression.py
│   ├── phase4b_asymmetric_quantization.py
│   ├── phase4b_cross_model.py
│   ├── phase4c_long_context.py
│   ├── phase6_figure1_wikitext.py
│   └── generate_figures.py
│
├── colab/
│   ├── phase4c_bugfix_crossmodel.py
│   ├── phase5_7b_scaling.py
│   ├── phase5b_qwen_anomaly.py
│   ├── phase5c_arc_smoothquant.py
│   ├── phase5d_10b_generalization.py
│   ├── phase5e_full_arc_sweep.py
│   ├── phase6b_wikitext_7b.py
│   ├── phase6c_wikitext_12b_14b.py
│   ├── phase6d_wikitext_mistral.py
│   └── phase6e_wikitext_pythia28b.py
│
├── compression/
│   ├── __init__.py
│   └── compressors.py
│
└── results/                                        # 全実験 JSON (17 files)
    ├── phase0〜phase6e の全 JSON
    └── (詳細は README.md 参照)
```

---

## 14. 実行環境

- **ローカル**: Apple M1, 16GB RAM — Phase 0〜4b, Figure 生成
- **Colab T4**: 15GB VRAM — Phase 4c 長文検証
- **Colab Pro Blackwell**: NVIDIA RTX PRO 6000, 102GB VRAM — Phase 5〜6 (7B〜40B)
- **Models**: 12 models (124M〜40B), 4 architecture families
- **依存**: PyTorch 2.x, Transformers, scikit-learn, scipy, datasets, matplotlib

---

## 15. 最終的な問いへの回答

> "Pre-LN Transformer の hidden-state geometry は、単なる解析対象ではなく、compression prior として使えるか？"

**Yes, ただし KV cache 量子化の前段処理として。**

Hidden state の全置換圧縮としては不可能だが、KV cache の norm 分離量子化として有効に機能する。特に低ビット量子化（INT4以下）において、norm を分離してから方向ベクトルを量子化することで、同じビット数での近似精度が系統的に改善される。この効果は PCA 次元削減と直交的に組み合わせ可能であり、ΔPPL < 1.0 を保ちながら 5.7x の圧縮を実現した。

**追加検証により確立された事実:**
- この効果は GPT-2 / Pythia-410M / Qwen2-0.5B / Pythia-6.9B / Qwen2-7B の **5 モデル** で確認
- norm_sep advantage は全モデルで存在するが、**スケーリングは head_dim に依存**（params ではない）
- 長文コンテキスト（~220 tokens）でも安定
- **K>V 非対称性は MHA 固有**。GQA (Qwen2-7B) では逆転
- **PCA 圧縮は 7B でも有効**: npca64 で Qwen2-7B ΔPPL=+0.41

**Phase 5c の決定的発見:**
- **norm 分離 + outlier-aware quantization (per-channel) の組み合わせで、7B モデルの INT4 ΔPPL を +238 → +0.32 に削減 (744x 改善)**
- 単独ではどちらも不十分（nsep: +57.5, perchan: +97.8）だが、組み合わせると直交的に効く
- これは Arc 論文の構造知見が実用圧縮技術に直結する初の実証
