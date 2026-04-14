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
# Phase 7: Appendix Experiments
# ============================================================
# Three additional experiments to address reviewer concerns:
#   A. Long context stability (1k-4k tokens)
#   B. KIVI-style baseline comparison
#   C. Memory measurement
#
# Target: Pythia-6.9B (consistent failure case, reliable download)
# ============================================================

# === CELL 1 ===
!pip install -q transformers accelerate hf_transfer datasets
import os; os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"

import gc, json, time, numpy as np, torch, torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from datetime import datetime

device = "cuda"
print(f"GPU: {torch.cuda.get_device_name()}")

# Load WikiText-2 for long context
print("Loading WikiText-2...")
wikitext = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
# Concatenate for long sequences
wiki_long = " ".join([t for t in wikitext["text"] if len(t.strip()) > 20])
print(f"WikiText-2 concatenated: {len(wiki_long)} chars")

# === CELL 2 ===

# ── Quantization Methods ───────────────────────────────────────────────────

def qa_perrow(x, b):
    x = x.float(); qm = 2**(b-1)-1
    s = x.abs().amax(dim=-1, keepdim=True).clamp(min=1e-12) / qm
    return ((x/s).round().clamp(-qm, qm)) * s

def qa_perchan(x, b):
    x = x.float(); qm = 2**(b-1)-1
    s = x.abs().amax(dim=0, keepdim=True).clamp(min=1e-12) / qm
    return ((x/s).round().clamp(-qm, qm)) * s

def apply_method(x, name):
    if name == "naive4": return qa_perrow(x, 4)
    if name == "nsep+pchan4":
        x = x.float()
        n = x.norm(dim=-1, keepdim=True).clamp(min=1e-12)
        d = x / n
        dq = qa_perchan(d, 4)
        dq = dq / dq.norm(dim=-1, keepdim=True).clamp(min=1e-8)
        return n * dq
    # KIVI-style: K=INT2 per-channel, V=INT4 per-channel
    if name == "kivi_k2v4_key": return qa_perchan(x, 2)
    if name == "kivi_k2v4_val": return qa_perchan(x, 4)
    # KIVI + nsep: norm-sep then KIVI
    if name == "nsep+kivi_key":
        x = x.float()
        n = x.norm(dim=-1, keepdim=True).clamp(min=1e-12)
        d = x / n
        dq = qa_perchan(d, 2)
        dq = dq / dq.norm(dim=-1, keepdim=True).clamp(min=1e-8)
        return n * dq
    if name == "nsep+kivi_val":
        x = x.float()
        n = x.norm(dim=-1, keepdim=True).clamp(min=1e-12)
        d = x / n
        dq = qa_perchan(d, 4)
        dq = dq / dq.norm(dim=-1, keepdim=True).clamp(min=1e-8)
        return n * dq
    raise ValueError(name)

# ── Cache helpers ──────────────────────────────────────────────────────────

def get_kv(past, li):
    if hasattr(past, 'layers'): return past.layers[li].keys, past.layers[li].values
    return past[li][0], past[li][1]

def set_kv(past, li, k, v):
    if hasattr(past, 'layers'):
        past.layers[li].keys = k; past.layers[li].values = v
    return past

def n_cache_layers(past):
    return len(past.layers) if hasattr(past, 'layers') else len(past)

def compress_cache(past, key_method, val_method):
    for li in range(n_cache_layers(past)):
        ok, ov = get_kv(past, li)
        ok, ov = ok.clone(), ov.clone()
        nk, nv = torch.zeros_like(ok), torch.zeros_like(ov)
        for h in range(ok.shape[1]):
            nk[0,h] = apply_method(ok[0,h], key_method).to(ok.dtype)
            nv[0,h] = apply_method(ov[0,h], val_method).to(ov.dtype)
        set_kv(past, li, nk, nv)

# ── Evaluation ─────────────────────────────────────────────────────────────

def eval_ppl(model, ids, pfl, cl):
    """Evaluate ΔPPL for a given prefill/continuation split, multiple methods."""
    fi = ids[:, :pfl+cl]; tgt = fi[0, pfl:].cpu()
    with torch.inference_mode():
        bl = model(fi, use_cache=False).logits[0, pfl-1:-1].float().cpu()
    bp = float(torch.exp(F.cross_entropy(bl, tgt, reduction="mean")).item())

    results = {}
    METHODS = [
        ("naive4",       "naive4",         "naive4"),
        ("nsep+pchan4",  "nsep+pchan4",    "nsep+pchan4"),
        ("kivi_k2v4",    "kivi_k2v4_key",  "kivi_k2v4_val"),
        ("nsep+kivi",    "nsep+kivi_key",  "nsep+kivi_val"),
    ]
    for label, km, vm in METHODS:
        with torch.inference_mode():
            po = model(ids[:, :pfl], use_cache=True)
            past = po.past_key_values
            compress_cache(past, km, vm)
            co = model(ids[:, pfl:pfl+cl], past_key_values=past, use_cache=False)
        cl_out = co.logits[0].float().cpu()
        ca, ba, ta = cl_out[:-1], bl[:-1], tgt[1:]
        ml = min(ca.shape[0], ba.shape[0], ta.shape[0])
        dp = float(torch.exp(F.cross_entropy(ca[:ml], ta[:ml], reduction="mean")).item()) - bp
        results[label] = round(dp, 4)
        del po, co, past; gc.collect(); torch.cuda.empty_cache()

    del bl; gc.collect()
    return bp, results

# ── Load model ─────────────────────────────────────────────────────────────

print("Loading Pythia-6.9B...")
tok = AutoTokenizer.from_pretrained("EleutherAI/pythia-6.9b")
mdl = AutoModelForCausalLM.from_pretrained(
    "EleutherAI/pythia-6.9b", torch_dtype=torch.float16, device_map="auto", use_safetensors=True
)
mdl.eval()
if tok.pad_token is None: tok.pad_token = tok.eos_token

# Tokenize long text
ids_long = tok.encode(wiki_long, return_tensors="pt").to(device)
max_pos = getattr(mdl.config, "max_position_embeddings", 2048)
if ids_long.shape[1] > max_pos:
    ids_long = ids_long[:, :max_pos]
total_tokens = ids_long.shape[1]
print(f"Total tokens: {total_tokens} (max_pos={max_pos})")

# ════════════════════════════════════════════════════════════════════════════
# PART A: Long Context Stability (256, 512, 1024, 1536, 2000)
# ════════════════════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("PART A: Long Context Stability (Pythia-6.9B)")
print("="*60)

CL = 40  # continuation length
PREFILL_LENGTHS = [256, 512, 1024, 1536]
# Filter to what we actually have
PREFILL_LENGTHS = [p for p in PREFILL_LENGTHS if p + CL <= total_tokens]

long_ctx_results = []
for pfl in PREFILL_LENGTHS:
    bp, results = eval_ppl(mdl, ids_long, pfl, CL)
    long_ctx_results.append({"pfl": pfl, "bp": round(bp, 2), "methods": results})
    print(f"  pfl={pfl:>5}: base={bp:.2f}  naive4={results['naive4']:>+8.3f}  "
          f"nsep+pc={results['nsep+pchan4']:>+8.3f}  "
          f"kivi={results['kivi_k2v4']:>+8.3f}  "
          f"nsep+kivi={results['nsep+kivi']:>+8.3f}")

# Stability summary
print(f"\n  Stability across context lengths:")
for mn in ["naive4", "nsep+pchan4", "kivi_k2v4", "nsep+kivi"]:
    vals = [r["methods"][mn] for r in long_ctx_results]
    print(f"    {mn:>14}: {' → '.join(f'{v:+.2f}' for v in vals)}")

# ════════════════════════════════════════════════════════════════════════════
# PART B: KIVI Comparison
# ════════════════════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("PART B: KIVI-Style Baseline Comparison")
print("="*60)

print("""
  KIVI (Liu et al., 2024): Key=INT2 per-channel, Value=INT4 per-channel
  nsep+pchan4: Key=INT4 nsep+perchan, Value=INT4 nsep+perchan
  nsep+kivi: Key=INT2 nsep+perchan, Value=INT4 nsep+perchan (KIVI + norm-sep)

  Effective bits per element:
    naive4:      4 bpe (Key=4, Value=4)
    nsep+pchan4: ~4 bpe (Key=4, Value=4, +negligible norm)
    kivi_k2v4:   3 bpe (Key=2, Value=4)
    nsep+kivi:   ~3 bpe (Key=2, Value=4, +negligible norm)
""")

# Use pfl=1024 for comparison
pfl_comp = min(1024, total_tokens - CL)
bp_comp, results_comp = eval_ppl(mdl, ids_long, pfl_comp, CL)

print(f"  pfl={pfl_comp}, base PPL={bp_comp:.2f}")
print(f"  {'Method':<15} {'ΔPPL':>10} {'bpe':>6} {'Notes'}")
print(f"  {'-'*45}")
rows = [
    ("naive4",       results_comp["naive4"],       "4.0", "per-row absmax"),
    ("nsep+pchan4",  results_comp["nsep+pchan4"],  "~4.0", "norm-sep + per-channel"),
    ("kivi_k2v4",    results_comp["kivi_k2v4"],    "3.0", "K:INT2, V:INT4 per-channel"),
    ("nsep+kivi",    results_comp["nsep+kivi"],     "~3.0", "norm-sep + KIVI-style"),
]
for name, dp, bpe, note in rows:
    print(f"  {name:<15} {dp:>+10.4f} {bpe:>6} {note}")

# ════════════════════════════════════════════════════════════════════════════
# PART C: Memory Measurement
# ════════════════════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("PART C: Memory Measurement")
print("="*60)

# Measure KV cache memory at different sequence lengths
print("\n  KV cache memory (Pythia-6.9B, fp16):")
print(f"  {'seq_len':>8} {'FP16 KV':>12} {'INT4 est.':>12} {'Ratio':>8}")
print(f"  {'-'*45}")

n_layers_m = 32
n_kv_heads_m = 32
head_dim_m = 128

for seq_len in [512, 1024, 2048, 4096, 8192]:
    fp16_bytes = 2 * n_layers_m * n_kv_heads_m * seq_len * head_dim_m * 2  # 2 for K+V, 2 bytes per fp16
    int4_bytes = 2 * n_layers_m * n_kv_heads_m * seq_len * head_dim_m * 0.5  # 0.5 bytes per int4
    nsep_overhead = 2 * n_layers_m * n_kv_heads_m * seq_len * 2  # 1 fp16 norm per (layer, head, token)
    nsep_total = int4_bytes + nsep_overhead
    ratio = fp16_bytes / nsep_total

    print(f"  {seq_len:>8} {fp16_bytes/1e6:>10.1f}MB {nsep_total/1e6:>10.1f}MB {ratio:>7.1f}x")

# Actual measurement with model
print("\n  Actual VRAM measurement:")
torch.cuda.reset_peak_memory_stats()
gc.collect(); torch.cuda.empty_cache()

mem_before = torch.cuda.memory_allocated()

# Generate KV cache for 1024 tokens
test_ids = ids_long[:, :1024]
with torch.inference_mode():
    out = mdl(test_ids, use_cache=True)
    past = out.past_key_values

mem_after = torch.cuda.memory_allocated()
kv_mem = mem_after - mem_before

# Measure compressed
k0, v0 = get_kv(past, 0)
print(f"  KV cache (1024 tokens):")
print(f"    FP16 actual: {kv_mem/1e6:.1f} MB")
print(f"    Per-layer shape: K={k0.shape}, V={v0.shape}")
nsep_oh = 2 * n_layers_m * n_kv_heads_m * 1024 * 2  # bytes
int4_est = kv_mem / 4
print(f"    Theoretical INT4: {int4_est/1e6:.1f} MB (4x reduction)")
print(f"    nsep overhead: {nsep_oh/1e6:.2f} MB")
print(f"    nsep+INT4 total: {(int4_est + nsep_oh)/1e6:.1f} MB")
print(f"    Effective ratio: {kv_mem / (int4_est + nsep_oh):.2f}x")

# Latency measurement
print("\n  Latency (prefill 512 tokens + decode 40 tokens):")
test_ids_lat = ids_long[:, :512]

# Baseline
torch.cuda.synchronize()
t0 = time.time()
for _ in range(3):
    with torch.inference_mode():
        out = mdl(test_ids_lat, use_cache=True)
        past = out.past_key_values
        for step in range(40):
            next_tok = out.logits[:, -1:].argmax(dim=-1)
            out = mdl(next_tok, past_key_values=past, use_cache=True)
            past = out.past_key_values
    del out, past; gc.collect(); torch.cuda.empty_cache()
torch.cuda.synchronize()
baseline_time = (time.time() - t0) / 3

# With compression
torch.cuda.synchronize()
t0 = time.time()
for _ in range(3):
    with torch.inference_mode():
        out = mdl(test_ids_lat, use_cache=True)
        past = out.past_key_values
        compress_cache(past, "nsep+pchan4", "nsep+pchan4")
        for step in range(40):
            next_tok = out.logits[:, -1:].argmax(dim=-1)
            out = mdl(next_tok, past_key_values=past, use_cache=True)
            past = out.past_key_values
    del out, past; gc.collect(); torch.cuda.empty_cache()
torch.cuda.synchronize()
nsep_time = (time.time() - t0) / 3

overhead_pct = (nsep_time - baseline_time) / baseline_time * 100
print(f"    Baseline:     {baseline_time:.3f}s")
print(f"    nsep+pchan4:  {nsep_time:.3f}s")
print(f"    Overhead:     {overhead_pct:+.1f}%")

del mdl, tok; gc.collect(); torch.cuda.empty_cache()

# ════════════════════════════════════════════════════════════════════════════
# JSON Output
# ════════════════════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("JSON OUTPUT:")
print("="*60)
output = {
    "experiment": "phase7_appendix",
    "timestamp": datetime.now().isoformat(),
    "model": "Pythia-6.9B",
    "long_context": long_ctx_results,
    "kivi_comparison": {
        "pfl": pfl_comp, "bp": round(bp_comp, 2),
        "results": results_comp,
    },
    "memory": {
        "model": "Pythia-6.9B",
        "kv_cache_fp16_1024tok_mb": round(kv_mem/1e6, 1),
        "overhead_percent": round(overhead_pct, 1),
    },
}
print(json.dumps(output, indent=2, ensure_ascii=False))
