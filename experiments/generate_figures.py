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
Generate all paper figures from existing JSON results.
No model loading needed -- purely visualization.
"""

import json, numpy as np
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

results_dir = Path(__file__).parent.parent / "results"
paper_dir = Path(__file__).parent.parent / "paper"
paper_dir.mkdir(parents=True, exist_ok=True)

# ── Style ──
plt.rcParams.update({
    'font.size': 11,
    'axes.titlesize': 12,
    'axes.labelsize': 11,
    'figure.dpi': 150,
})
C_NAIVE = '#e74c3c'
C_NSEP = '#2ecc71'
C_MID = '#3498db'
C_COMBO = '#9b59b6'

# ════════════════════════════════════════════════════════════════════════════
# Figure 2: WikiText-2 ΔPPL Bar Chart (7 models)
# ════════════════════════════════════════════════════════════════════════════
print("Figure 2: WikiText-2 ΔPPL comparison...")

models = ["GPT-2\n(124M)", "Pythia\n2.8B", "Pythia\n6.9B", "Mistral\n7B", "Qwen2\n7B", "Pythia\n12B", "Qwen2.5\n14B"]
naive4 = [1.60, 14.74, 22.56, 0.10, 811.61, 34.22, 0.55]
nsep_pc = [1.13, 1.99, 1.21, 0.04, 0.43, 4.01, 0.45]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 4.5), gridspec_kw={'width_ratios': [3, 2]})

# (a) Log-scale bar chart
x = np.arange(len(models))
w = 0.35
bars1 = ax1.bar(x - w/2, naive4, w, label='naive INT4', color=C_NAIVE, alpha=0.8, edgecolor='white')
bars2 = ax1.bar(x + w/2, nsep_pc, w, label='nsep+pchan INT4', color=C_NSEP, alpha=0.8, edgecolor='white')
ax1.set_yscale('log')
ax1.set_ylabel('ΔPPL (log scale)')
ax1.set_xticks(x)
ax1.set_xticklabels(models, fontsize=9)
ax1.legend(fontsize=10, loc='upper left')
ax1.set_title('(a) WikiText-2: naive INT4 vs nsep+pchan INT4')
ax1.axhline(y=1.0, color='gray', linestyle='--', alpha=0.5, linewidth=0.8)
ax1.set_ylim(0.01, 2000)

# Add improvement labels on top
for i in range(len(models)):
    if naive4[i] > 5:
        imp = naive4[i] / nsep_pc[i]
        ax1.annotate(f'{imp:.0f}x', xy=(x[i], max(naive4[i], nsep_pc[i])),
                    ha='center', va='bottom', fontsize=8, fontweight='bold', color='#2c3e50')

# (b) Improvement factor
improvements = [n/s for n, s in zip(naive4, nsep_pc)]
colors = [C_NSEP if imp > 5 else C_MID for imp in improvements]
ax2.barh(x, improvements, color=colors, alpha=0.8, edgecolor='white')
ax2.set_xscale('log')
ax2.set_xlabel('Improvement (naive4 / nsep+pchan4)')
ax2.set_yticks(x)
ax2.set_yticklabels(models, fontsize=9)
ax2.set_title('(b) Improvement Factor')
ax2.axvline(x=1.0, color='gray', linestyle='--', alpha=0.5)
for i, imp in enumerate(improvements):
    ax2.text(imp * 1.2, i, f'{imp:.1f}x', va='center', fontsize=9, fontweight='bold')

plt.tight_layout()
plt.savefig(paper_dir / "figure2_wikitext2.png", dpi=200, bbox_inches='tight')
plt.savefig(paper_dir / "figure2_wikitext2.pdf", bbox_inches='tight')
plt.close()
print(f"  Saved: figure2_wikitext2.png/pdf")


# ════════════════════════════════════════════════════════════════════════════
# Figure 3: Ablation -- Orthogonal Composition (Qwen2-7B)
# ════════════════════════════════════════════════════════════════════════════
print("Figure 3: Ablation (Qwen2-7B)...")

methods = ['naive4', 'nsep4', 'outlier4\n(4ch)', 'perchan4', 'nsep+out4', 'nsep+\npchan4']
dppl = [238.23, 57.45, 23.17, 97.77, 0.59, 0.32]
has_nsep = [False, True, False, False, True, True]
has_outlier = [False, False, True, False, True, True]

fig, ax = plt.subplots(figsize=(10, 5))

colors = []
for ns, ol in zip(has_nsep, has_outlier):
    if ns and ol: colors.append(C_COMBO)
    elif ns: colors.append(C_NSEP)
    elif ol: colors.append(C_MID)
    else: colors.append(C_NAIVE)

bars = ax.bar(range(len(methods)), dppl, color=colors, alpha=0.85, edgecolor='white', width=0.7)
ax.set_yscale('log')
ax.set_ylabel('ΔPPL (log scale)')
ax.set_xticks(range(len(methods)))
ax.set_xticklabels(methods, fontsize=10)
ax.set_title('Figure 3: Ablation -- Neither Technique Alone Suffices (Qwen2-7B, INT4)')

# Add value labels
for i, (bar, val) in enumerate(zip(bars, dppl)):
    ax.text(bar.get_x() + bar.get_width()/2, val * 1.3,
            f'+{val:.1f}' if val > 1 else f'+{val:.2f}',
            ha='center', va='bottom', fontsize=10, fontweight='bold')

# Legend
from matplotlib.patches import Patch
legend_elements = [
    Patch(facecolor=C_NAIVE, alpha=0.85, label='No treatment'),
    Patch(facecolor=C_NSEP, alpha=0.85, label='Norm-sep only'),
    Patch(facecolor=C_MID, alpha=0.85, label='Outlier-aware only'),
    Patch(facecolor=C_COMBO, alpha=0.85, label='Both (nsep+pchan/out)'),
]
ax.legend(handles=legend_elements, loc='upper right', fontsize=9)

ax.axhline(y=1.0, color='gray', linestyle='--', alpha=0.4, linewidth=0.8)
ax.set_ylim(0.1, 500)

plt.tight_layout()
plt.savefig(paper_dir / "figure3_ablation.png", dpi=200, bbox_inches='tight')
plt.savefig(paper_dir / "figure3_ablation.pdf", bbox_inches='tight')
plt.close()
print(f"  Saved: figure3_ablation.png/pdf")


# ════════════════════════════════════════════════════════════════════════════
# Figure 4: Outlier Ratio vs Improvement (12-model sweep)
# ════════════════════════════════════════════════════════════════════════════
print("Figure 4: Outlier ratio vs improvement...")

# From phase5e results (curated text, 12 models)
model_data = [
    ("GPT-2",       0.64,   0.56,   None,  "MHA", "124M"),
    ("OPT-125m",    1.22,   1.46,   1.49,  "MHA", "125M"),
    ("Pythia-410M", 77.55,  12.62,  None,  "MHA", "410M"),
    ("Qwen2-0.5B",  0.64,   0.28,   None,  "GQA", "0.5B"),
    ("OPT-1.3B",    0.70,   0.68,   1.44,  "MHA", "1.3B"),
    ("Pythia-2.8B", 0.90,   -0.06,  4.02,  "MHA", "2.8B"),
    ("Pythia-6.9B", 22.26,  0.27,   4.56,  "MHA", "6.9B"),
    ("Qwen2-7B",    238.23, 0.32,   8.60,  "GQA", "7B"),
    ("Pythia-12B",  27.28,  1.82,   4.56,  "MHA", "12B"),
    ("OPT-13B",     0.28,   0.35,   3.10,  "MHA", "13B"),
    ("Qwen2.5-14B", 0.30,   0.26,   3.48,  "GQA", "14B"),
    ("Falcon-40B",  0.08,   0.04,   2.35,  "MHA", "40B"),
]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# (a) naive4 ΔPPL vs outlier ratio
for name, naive, nsep, ratio, arch, params in model_data:
    if ratio is None:
        continue
    marker = 'o' if arch == "MHA" else 's'
    color = C_NAIVE if naive > 10 else C_MID
    ax1.scatter(ratio, naive, s=80, marker=marker, c=color, edgecolors='black', linewidths=0.5, zorder=5)

# Manual label placement to avoid overlaps
label_positions = {
    "OPT-125m": (8, 12), "OPT-1.3B": (-45, 12), "Pythia-2.8B": (8, -5),
    "OPT-13B": (-45, -12), "Qwen2.5-14B": (8, -12), "Falcon-40B": (8, 5),
    "Pythia-12B": (-55, 5), "Pythia-6.9B": (8, 5), "Qwen2-7B": (-55, -5),
}
for name, naive, nsep, ratio, arch, params in model_data:
    if ratio is None:
        continue
    ox, oy = label_positions.get(name, (5, 5))
    ax1.annotate(f'{name}', xy=(ratio, naive), xytext=(ox, oy),
                textcoords='offset points', fontsize=7, ha='left')

ax1.set_xlabel('Layer 0 Outlier Ratio')
ax1.set_ylabel('naive INT4 ΔPPL')
ax1.set_yscale('log')
ax1.set_title('(a) Outlier Ratio Predicts naive INT4 Failure')
ax1.axvline(x=4.0, color='red', linestyle='--', alpha=0.4, label='Threshold ~4x')
ax1.legend(fontsize=9)

# (b) 12-model sweep: naive vs nsep+pchan
names_12 = [d[0] for d in model_data]
naive_12 = [d[1] for d in model_data]
nsep_12 = [d[2] for d in model_data]

x12 = np.arange(len(names_12))
w12 = 0.35
ax2.bar(x12 - w12/2, naive_12, w12, label='naive INT4', color=C_NAIVE, alpha=0.8, edgecolor='white')
ax2.bar(x12 + w12/2, [max(0.01, v) for v in nsep_12], w12, label='nsep+pchan', color=C_NSEP, alpha=0.8, edgecolor='white')
ax2.set_yscale('log')
ax2.set_ylabel('ΔPPL (log scale)')
ax2.set_xticks(x12)
ax2.set_xticklabels([d[0].replace("Pythia-","P-").replace("Qwen2","Q2").replace("Falcon","Flc") for d in model_data],
                     fontsize=7, rotation=45, ha='right')
ax2.set_title('(b) 12-Model Sweep: Curated Text')
ax2.legend(fontsize=9)
ax2.set_ylim(0.01, 500)
ax2.axhline(y=1.0, color='gray', linestyle='--', alpha=0.4)

plt.tight_layout()
plt.savefig(paper_dir / "figure4_outlier_sweep.png", dpi=200, bbox_inches='tight')
plt.savefig(paper_dir / "figure4_outlier_sweep.pdf", bbox_inches='tight')
plt.close()
print(f"  Saved: figure4_outlier_sweep.png/pdf")


# ════════════════════════════════════════════════════════════════════════════
# Figure 5: INT3 > INT4 Anomaly (Qwen2-7B)
# ════════════════════════════════════════════════════════════════════════════
print("Figure 5: INT3>INT4 anomaly...")

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

# (a) Per-layer ΔPPL (INT4 vs INT3) on Qwen2-7B
layers = [0, 4, 8, 12, 16, 20, 24, 27]
int4_dppl = [179.45, 0.11, 0.12, 0.09, 0.07, 0.10, 0.11, 0.25]
int3_dppl = [1.47, 0.04, 0.15, 0.10, 0.16, 0.13, 0.13, 0.28]

ax1.bar([i-0.2 for i in range(len(layers))], int4_dppl, 0.35,
        label='INT4', color=C_NAIVE, alpha=0.8, edgecolor='white')
ax1.bar([i+0.2 for i in range(len(layers))], int3_dppl, 0.35,
        label='INT3', color=C_MID, alpha=0.8, edgecolor='white')
ax1.set_yscale('log')
ax1.set_ylabel('ΔPPL (single layer compressed)')
ax1.set_xticks(range(len(layers)))
ax1.set_xticklabels([f'L{l}' for l in layers])
ax1.set_title('(a) Per-Layer ΔPPL: INT4 vs INT3 (Qwen2-7B)')
ax1.legend(fontsize=9, loc='center right')
ax1.set_ylim(0.01, 800)
ax1.annotate('Layer 0: INT4 = +179\n             INT3 = +1.5\n             (122x worse)',
            xy=(0.2, 179.45), xytext=(2.5, 15),
            arrowprops=dict(arrowstyle='->', color='red', lw=1.5),
            fontsize=9, color='red', fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.4', facecolor='white', edgecolor='red', alpha=0.9))

# (b) Layer 0 value distribution (showing outlier)
np.random.seed(42)
normal_vals = np.random.normal(0, 2, 5000)
outlier_vals = np.random.normal(0, 30, 500)
all_vals = np.concatenate([normal_vals, outlier_vals])
all_vals = np.clip(all_vals, -170, 170)

ax2.hist(all_vals, bins=200, color=C_NAIVE, alpha=0.5, density=True, edgecolor='none', label='All channels')
ax2.hist(normal_vals, bins=100, color=C_NSEP, alpha=0.5, density=True, edgecolor='none', label='Normal channels')
ax2.set_xlabel('Value')
ax2.set_ylabel('Density')
ax2.set_xlim(-180, 180)
ax2.set_title('(b) Layer 0 Key Distribution (Qwen2-7B, schematic)')
ax2.legend(fontsize=9, loc='upper right')
ax2.annotate('Outlier channels\n(absmax ~167)', xy=(120, 0.002), xytext=(100, 0.04),
            arrowprops=dict(arrowstyle='->', color='red'),
            fontsize=9, color='red', fontweight='bold')

plt.tight_layout()
plt.savefig(paper_dir / "figure5_anomaly.png", dpi=200, bbox_inches='tight')
plt.savefig(paper_dir / "figure5_anomaly.pdf", bbox_inches='tight')
plt.close()
print(f"  Saved: figure5_anomaly.png/pdf")


# ════════════════════════════════════════════════════════════════════════════
print(f"\nAll figures saved to {paper_dir}/")
print("  figure1_distribution.png/pdf  -- already generated (Qwen2-7B)")
print("  figure2_wikitext2.png/pdf     -- WikiText-2 bar chart")
print("  figure3_ablation.png/pdf      -- ablation (orthogonal composition)")
print("  figure4_outlier_sweep.png/pdf -- outlier ratio + 12-model sweep")
print("  figure5_anomaly.png/pdf       -- INT3>INT4 anomaly")
