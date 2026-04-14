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

# ============================================================
# Phase 5b: Qwen2-7B INT3 > INT4 Anomaly Investigation
# ============================================================
# Qwen2-7B shows INT3 (DPPL=+6.6) far better than INT4 (DPPL=+238).
# This is physically impossible (fewer bits should not be better),
# so we investigate the root cause.
#
# Hypotheses:
#   H1: absmax quantization clipping is harming INT4
#       (discretization granularity issue with qmax=7 vs qmax=3)
#   H2: Qwen2 KV cache has outlier channels, and INT4 is more sensitive
#   H3: With GQA's few KV heads (4), specific heads are breaking
#   H4: Non-trivial quantization behavior at head_dim=128
#   H5: Implementation bug (miscalculation in INT3/INT4)
#
# Verification:
#   Part A: Quantization function correctness check (bit sweep, reconstruction MSE)
#   Part B: KV cache value distribution / outlier analysis
#   Part C: Identify which head/layer breaks
#   Part D: Control experiment (check if Pythia-6.9B shows the same issue)
# ============================================================

# === CELL 1: Setup ===
!pip install -q transformers accelerate hf_transfer
import os
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"

import gc, json, numpy as np, torch, torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from datetime import datetime

device = "cuda"
print(f"GPU: {torch.cuda.get_device_name()}")

# === CELL 2: Investigation ===

# ── Quantization (same as Phase 5) ──
def qa(x, b):
    x = x.float()
    qm = 2**(b-1) - 1
    s = x.abs().amax(dim=-1, keepdim=True).clamp(min=1e-12) / qm
    return ((x / s).round().clamp(-qm, qm)) * s

def qn(x, b):
    x = x.float()
    n = x.norm(dim=-1, keepdim=True).clamp(min=1e-12)
    d = x / n
    r = qa(d, b)
    rn = r.norm(dim=-1, keepdim=True).clamp(min=1e-8)
    return n * (r / rn)

# ── Cache access helpers ──
def get_kv(past, li):
    if hasattr(past, 'layers'):
        dl = past.layers[li]
        return dl.keys, dl.values
    return past[li][0], past[li][1]

def set_kv(past, li, k, v):
    if hasattr(past, 'layers'):
        past.layers[li].keys = k
        past.layers[li].values = v
    return past

def n_layers(past):
    return len(past.layers) if hasattr(past, 'layers') else len(past)

def xppl(l, t):
    return float(torch.exp(F.cross_entropy(l, t, reduction="mean")).item())


TEXT = (
    "The history of AI began with myths of artificial beings. Philosophers "
    "described thinking as symbol manipulation, leading to digital computers "
    "in the 1940s. The field was founded at Dartmouth in 1956. Researchers "
    "predicted human-level AI within a generation and received millions in "
    "funding. By 1973, governments cut funding, causing the first AI winter. "
    "Expert systems revived the field in the early 1980s."
)

# ════════════════════════════════════════════════════════════════════════════
# Part A: Quantization function correctness check
# ════════════════════════════════════════════════════════════════════════════
print("="*60)
print("Part A: Quantization Sanity Check")
print("="*60)

# Test with a synthetic vector
torch.manual_seed(42)
x_test = torch.randn(50, 128).cuda()  # (seq_len=50, head_dim=128)

print("\nSynthetic data (randn, head_dim=128):")
for bits in [8, 7, 6, 5, 4, 3, 2]:
    qm = 2**(bits-1) - 1
    x_q = qa(x_test, bits)
    mse = ((x_test.float() - x_q)**2).mean().item()
    cos = F.cosine_similarity(x_test.float(), x_q, dim=-1).mean().item()
    print(f"  INT{bits} (qmax={qm:>3}): MSE={mse:.6f}  cosine={cos:.6f}")

print("\n  Expected: MSE and cosine should monotonically improve with more bits")
print("  If INT3 < INT4 in MSE, something is wrong with qa()")

# ════════════════════════════════════════════════════════════════════════════
# Part B: Qwen2-7B KV cache value distribution analysis
# ════════════════════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("Part B: Qwen2-7B KV Cache Distribution")
print("="*60)

tok_q = AutoTokenizer.from_pretrained("Qwen/Qwen2-7B")
mdl_q = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen2-7B", torch_dtype=torch.float16, device_map="auto",
    use_safetensors=True,
)
mdl_q.eval()
if tok_q.pad_token is None:
    tok_q.pad_token = tok_q.eos_token

ids_q = tok_q.encode(TEXT, return_tensors="pt").to(device)
CL = 40
pfl = ids_q.shape[1] - CL
fi = ids_q[:, :pfl+CL]; tgt = fi[0, pfl:].cpu()

with torch.inference_mode():
    po = mdl_q(ids_q[:,:pfl], use_cache=True)
    past = po.past_key_values

    # Baseline logits
    bl = mdl_q(fi, use_cache=False).logits[0, pfl-1:-1].float().cpu()
bp = xppl(bl, tgt)
print(f"Baseline PPL: {bp:.2f}, prefill={pfl}")

print(f"\nKV cache structure: {n_layers(past)} layers")
k0, v0 = get_kv(past, 0)
print(f"Shape: K={k0.shape}, V={v0.shape}")
print(f"dtype: {k0.dtype}")

# Per-layer distribution analysis
print(f"\nPer-layer value distribution (Key):")
print(f"  {'Layer':>5} {'mean':>8} {'std':>8} {'min':>8} {'max':>8} {'abs_max':>8} {'|max/mean|':>10} {'outlier%':>8}")

layer_stats = []
for li in range(n_layers(past)):
    k, v = get_kv(past, li)
    k_flat = k.float().reshape(-1)
    v_flat = v.float().reshape(-1)
    k_abs = k_flat.abs()

    mean_k = k_flat.mean().item()
    std_k = k_flat.std().item()
    max_k = k_flat.max().item()
    min_k = k_flat.min().item()
    absmax_k = k_abs.max().item()

    # Outlier: |x| > 5*std
    outlier_pct = (k_abs > 5 * std_k).float().mean().item() * 100

    layer_stats.append({
        "layer": li, "mean": mean_k, "std": std_k,
        "min": min_k, "max": max_k, "absmax": absmax_k,
        "outlier_pct": outlier_pct,
    })

    if li % 4 == 0 or li == n_layers(past)-1:
        print(f"  L{li:>3}  {mean_k:>+8.4f} {std_k:>8.4f} {min_k:>8.3f} {max_k:>8.3f} "
              f"{absmax_k:>8.3f} {absmax_k/(std_k+1e-12):>10.1f} {outlier_pct:>7.2f}%")

# ════════════════════════════════════════════════════════════════════════════
# Part C: Per-head quantization error -- which head breaks at INT4?
# ════════════════════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("Part C: Per-Head Quantization Error (Qwen2-7B)")
print("="*60)

print("\nPer-head MSE at INT3 vs INT4 (layer 0):")
k0, v0 = get_kv(past, 0)
B, nh, sl, hd = k0.shape
print(f"  n_kv_heads={nh}, seq_len={sl}, head_dim={hd}")

for h in range(nh):
    kh = k0[0, h].float()  # (sl, hd)
    k_q4 = qa(kh, 4)
    k_q3 = qa(kh, 3)
    mse4 = ((kh - k_q4)**2).mean().item()
    mse3 = ((kh - k_q3)**2).mean().item()
    cos4 = F.cosine_similarity(kh, k_q4, dim=-1).mean().item()
    cos3 = F.cosine_similarity(kh, k_q3, dim=-1).mean().item()
    absmax = kh.abs().amax(dim=-1).mean().item()

    flag = " <- INT3 better MSE!" if mse3 < mse4 else ""
    print(f"  Head {h}: INT4 MSE={mse4:.6f} cos={cos4:.4f}  |  "
          f"INT3 MSE={mse3:.6f} cos={cos3:.4f}  |  absmax={absmax:.3f}{flag}")

# ════════════════════════════════════════════════════════════════════════════
# Part D: DPPL per layer -- which layer causes the explosion?
# ════════════════════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("Part D: Per-Layer ΔPPL Contribution (INT4 vs INT3)")
print("="*60)

print("\nCompressing ONE layer at a time, measuring ΔPPL:")
print(f"  {'Layer':>5} {'INT4 ΔPPL':>12} {'INT3 ΔPPL':>12} {'INT4/INT3':>10} {'Flag'}")

per_layer_dppl = []
# Sample every 4th layer to save time
sample_layers = list(range(0, n_layers(past), 4)) + [n_layers(past)-1]
sample_layers = sorted(set(sample_layers))

for li in sample_layers:
    results_layer = {}
    for bits, label in [(4, "int4"), (3, "int3")]:
        with torch.inference_mode():
            po2 = mdl_q(ids_q[:,:pfl], use_cache=True)
            p2 = po2.past_key_values

            # Compress ONLY this one layer
            ok, ov = get_kv(p2, li)
            ok_c, ov_c = ok.clone(), ov.clone()
            nk, nv = torch.zeros_like(ok_c), torch.zeros_like(ov_c)
            for h in range(ok_c.shape[1]):
                nk[0,h] = qa(ok_c[0,h], bits).to(ok.dtype)
                nv[0,h] = qa(ov_c[0,h], bits).to(ov.dtype)
            set_kv(p2, li, nk, nv)

            co2 = mdl_q(ids_q[:,pfl:pfl+CL], past_key_values=p2, use_cache=False)
        cl2 = co2.logits[0].float().cpu()
        ca2, ba2, ta2 = cl2[:-1], bl[:-1], tgt[1:]
        ml2 = min(ca2.shape[0], ba2.shape[0], ta2.shape[0])
        dp = xppl(ca2[:ml2], ta2[:ml2]) - bp
        results_layer[label] = dp
        del po2, p2, co2; gc.collect(); torch.cuda.empty_cache()

    ratio = results_layer["int4"] / results_layer["int3"] if results_layer["int3"] != 0 else float("inf")
    flag = ""
    if results_layer["int4"] > 10 and ratio > 5:
        flag = " <- INT4 EXPLOSION"
    elif results_layer["int3"] < results_layer["int4"] * 0.5:
        flag = " <- INT3 much better"

    print(f"  L{li:>3}  {results_layer['int4']:>+12.4f} {results_layer['int3']:>+12.4f} "
          f"{ratio:>10.2f}x{flag}")
    per_layer_dppl.append({"layer": li, **results_layer, "ratio": round(ratio, 3)})

# ════════════════════════════════════════════════════════════════════════════
# Part E: Pythia-6.9B control -- does the same anomaly occur?
# ════════════════════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("Part E: Pythia-6.9B Control (same test)")
print("="*60)

del mdl_q; gc.collect(); torch.cuda.empty_cache()

tok_p = AutoTokenizer.from_pretrained("EleutherAI/pythia-6.9b")
mdl_p = AutoModelForCausalLM.from_pretrained(
    "EleutherAI/pythia-6.9b", torch_dtype=torch.float16, device_map="auto",
    use_safetensors=True,
)
mdl_p.eval()
if tok_p.pad_token is None:
    tok_p.pad_token = tok_p.eos_token

ids_p = tok_p.encode(TEXT, return_tensors="pt").to(device)
pfl_p = ids_p.shape[1] - CL
fi_p = ids_p[:, :pfl_p+CL]; tgt_p = fi_p[0, pfl_p:].cpu()

with torch.inference_mode():
    bl_p = mdl_p(fi_p, use_cache=False).logits[0, pfl_p-1:-1].float().cpu()
bp_p = xppl(bl_p, tgt_p)
print(f"Baseline PPL: {bp_p:.2f}, prefill={pfl_p}")

# Full model INT sweep
print(f"\nFull-model compression:")
for bits in [8, 7, 6, 5, 4, 3, 2]:
    with torch.inference_mode():
        po = mdl_p(ids_p[:,:pfl_p], use_cache=True)
        past_p = po.past_key_values
        for li in range(n_layers(past_p)):
            ok, ov = get_kv(past_p, li)
            ok_c, ov_c = ok.clone(), ov.clone()
            nk, nv = torch.zeros_like(ok_c), torch.zeros_like(ov_c)
            for h in range(ok_c.shape[1]):
                nk[0,h] = qa(ok_c[0,h], bits).to(ok.dtype)
                nv[0,h] = qa(ov_c[0,h], bits).to(ov.dtype)
            set_kv(past_p, li, nk, nv)
        co = mdl_p(ids_p[:,pfl_p:pfl_p+CL], past_key_values=past_p, use_cache=False)
    cl = co.logits[0].float().cpu()
    ca, ba, ta = cl[:-1], bl_p[:-1], tgt_p[1:]
    ml = min(ca.shape[0], ba.shape[0], ta.shape[0])
    dp = xppl(ca[:ml], ta[:ml]) - bp_p
    print(f"  INT{bits}: ΔPPL={dp:>+10.4f}")
    del po, past_p, co; gc.collect(); torch.cuda.empty_cache()

del mdl_p; gc.collect(); torch.cuda.empty_cache()

# ════════════════════════════════════════════════════════════════════════════
# Summary
# ════════════════════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("INVESTIGATION SUMMARY")
print("="*60)
print("""
Check the following:
1. Part A: Does synthetic MSE monotonically decrease with more bits?
   YES -> qa() is correct, problem is data-specific
   NO  -> qa() has a bug

2. Part B: Are there outlier channels in Qwen2 KV cache?
   outlier% > 1% -> outlier channels dominate INT4 error

3. Part C: Does any single head show INT3 < INT4 in MSE?
   This would be physically impossible -> likely a rounding artifact

4. Part D: Which layer causes INT4 explosion?
   If concentrated in few layers -> layer-specific outlier problem

5. Part E: Does Pythia show monotonic bit->DPPL?
   YES -> problem is Qwen-specific (GQA/RMSNorm/outlier)
   NO  -> problem is general to head_dim=128
""")

# JSON output
output = {
    "experiment": "phase5b_qwen_anomaly",
    "timestamp": datetime.now().isoformat(),
    "layer_stats": layer_stats,
    "per_layer_dppl": per_layer_dppl,
}
print("\nJSON OUTPUT:")
print(json.dumps(output, indent=2, ensure_ascii=False))
