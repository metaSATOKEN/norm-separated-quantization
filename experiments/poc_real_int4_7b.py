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
# PoC: Real INT4 Packing -- Qwen2-7B (Colab GPU)
# ============================================================
# Validates real INT4 packing on the paper's headline model.
# Confirms that packed INT4 produces identical DPPL to fake quant
# and measures actual memory reduction.
# ============================================================

# === CELL 1 ===
!pip install -q transformers accelerate hf_transfer datasets
import os; os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"

import gc, json, numpy as np, torch, torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from datetime import datetime

device = "cuda"
print(f"GPU: {torch.cuda.get_device_name()}")

# === CELL 2 ===

# ── INT4 Pack / Unpack ─────────────────────────────────────────────────────

def pack_int4(values_int):
    """Pack INT4 (-7..+7) into uint8 (2 per byte)."""
    unsigned = (values_int + 8).to(torch.uint8)
    low = unsigned[..., 0::2]
    high = unsigned[..., 1::2]
    return low | (high << 4)

def unpack_int4(packed, original_size):
    """Unpack uint8 back to INT4."""
    low = (packed & 0x0F).to(torch.int8) - 8
    high = ((packed >> 4) & 0x0F).to(torch.int8) - 8
    result = torch.stack([low, high], dim=-1).reshape(*packed.shape[:-1], -1)
    return result[..., :original_size] if result.shape[-1] > original_size else result

# ── Fake quantization (for comparison) ─────────────────────────────────────

def fake_nsep_pchan4(x):
    x = x.float()
    n = x.norm(dim=-1, keepdim=True).clamp(min=1e-12)
    d = x / n
    scale = d.abs().amax(dim=0, keepdim=True).clamp(min=1e-12) / 7
    dq = (d / scale).round().clamp(-7, 7) * scale
    dq = dq / dq.norm(dim=-1, keepdim=True).clamp(min=1e-8)
    return n * dq

def fake_naive4(x):
    x = x.float()
    scale = x.abs().amax(dim=-1, keepdim=True).clamp(min=1e-12) / 7
    return ((x / scale).round().clamp(-7, 7)) * scale

# ── Real packed cache ──────────────────────────────────────────────────────

class PackedKVCache:
    def __init__(self):
        self.layers = []

    def compress_and_store(self, keys, values):
        B, nh, sl, hd = keys.shape
        layer = {}
        for role, tensor in [("k", keys), ("v", values)]:
            x = tensor[0].float()
            norms = x.norm(dim=-1, keepdim=True).clamp(min=1e-12)
            direction = x / norms
            scale = direction.abs().amax(dim=1, keepdim=True).clamp(min=1e-12) / 7
            quantized = (direction / scale).round().clamp(-7, 7).to(torch.int8)
            packed = pack_int4(quantized)
            layer[f"norms_{role}"] = norms.half()
            layer[f"packed_{role}"] = packed
            layer[f"scale_{role}"] = scale.half()
        self.layers.append(layer)

    def decompress(self, layer_idx):
        layer = self.layers[layer_idx]
        results = []
        for role in ["k", "v"]:
            norms = layer[f"norms_{role}"].float()
            packed = layer[f"packed_{role}"]
            scale = layer[f"scale_{role}"].float()
            hd = scale.shape[-1]
            quantized = unpack_int4(packed, hd).float()
            direction = quantized * scale
            dir_norm = direction.norm(dim=-1, keepdim=True).clamp(min=1e-8)
            direction = direction / dir_norm
            results.append((norms * direction).half().unsqueeze(0))
        return results[0], results[1]

    def memory_bytes(self):
        total_packed, total_norms, total_scales = 0, 0, 0
        for layer in self.layers:
            for role in ["k", "v"]:
                total_packed += layer[f"packed_{role}"].numel() * 1
                total_norms += layer[f"norms_{role}"].numel() * 2
                total_scales += layer[f"scale_{role}"].numel() * 2
        return {"packed": total_packed, "norms": total_norms, "scales": total_scales,
                "total": total_packed + total_norms + total_scales}

# ── Cache helpers ──────────────────────────────────────────────────────────

def get_kv(past, li):
    if hasattr(past, 'layers'): return past.layers[li].keys, past.layers[li].values
    return past[li][0], past[li][1]

def set_kv(past, li, k, v):
    if hasattr(past, 'layers'):
        past.layers[li].keys = k; past.layers[li].values = v

def n_cache_layers(past):
    return len(past.layers) if hasattr(past, 'layers') else len(past)

# ── Main ───────────────────────────────────────────────────────────────────

TEXT = (
    "The history of AI began with myths of artificial beings. Philosophers "
    "described thinking as symbol manipulation, leading to digital computers "
    "in the 1940s. The field was founded at Dartmouth in 1956. Researchers "
    "predicted human-level AI within a generation and received millions in "
    "funding. By 1973, governments cut funding, causing the first AI winter. "
    "Expert systems revived the field in the early 1980s."
)

print("Loading Qwen2-7B...")
tok = AutoTokenizer.from_pretrained("Qwen/Qwen2-7B")
mdl = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen2-7B", torch_dtype=torch.float16, device_map="auto", use_safetensors=True
)
mdl.eval()
if tok.pad_token is None: tok.pad_token = tok.eos_token

ids = tok.encode(TEXT, return_tensors="pt").to(device)
CL = 30
pfl = ids.shape[1] - CL

print(f"Tokens: {ids.shape[1]}, prefill: {pfl}, continuation: {CL}")

# Baseline
with torch.inference_mode():
    bl = mdl(ids, use_cache=False).logits[0, pfl-1:-1].float().cpu()
tgt = ids[0, pfl:].cpu()
base_ppl = float(torch.exp(F.cross_entropy(bl, tgt, reduction="mean")).item())
print(f"Baseline PPL: {base_ppl:.2f}")

# ── Get KV cache info ──
with torch.inference_mode():
    out = mdl(ids[:, :pfl], use_cache=True)
past = out.past_key_values
nl = n_cache_layers(past)
k0, v0 = get_kv(past, 0)
nh, sl, hd = k0.shape[1], k0.shape[2], k0.shape[3]
print(f"KV cache: {nl} layers, {nh} heads, seq_len={sl}, head_dim={hd}")

fp16_bytes = 2 * nl * nh * sl * hd * 2
print(f"FP16 KV: {fp16_bytes/1e6:.2f} MB")

# ── Real INT4 packing ──
packed_cache = PackedKVCache()
for li in range(nl):
    k, v = get_kv(past, li)
    packed_cache.compress_and_store(k.clone(), v.clone())

mem = packed_cache.memory_bytes()
print(f"Real INT4: {mem['total']/1e6:.2f} MB (packed={mem['packed']/1e6:.2f}, norms={mem['norms']/1e6:.2f}, scales={mem['scales']/1e6:.2f})")
print(f"Compression: {fp16_bytes / mem['total']:.2f}x")

del out, past; gc.collect(); torch.cuda.empty_cache()

# ── Evaluate: naive4 (fake) ──
print("\n--- Evaluation ---")

with torch.inference_mode():
    po = mdl(ids[:, :pfl], use_cache=True); past = po.past_key_values
    for li in range(nl):
        k, v = get_kv(past, li)
        nk, nv = torch.zeros_like(k), torch.zeros_like(v)
        for h in range(nh):
            nk[0,h] = fake_naive4(k[0,h]).to(k.dtype)
            nv[0,h] = fake_naive4(v[0,h]).to(v.dtype)
        set_kv(past, li, nk, nv)
    co = mdl(ids[:, pfl:], past_key_values=past, use_cache=False)
cl = co.logits[0].float().cpu()
ca, ba, ta = cl[:-1], bl[:-1], tgt[1:]
ml = min(ca.shape[0], ba.shape[0], ta.shape[0])
naive_dppl = float(torch.exp(F.cross_entropy(ca[:ml], ta[:ml], reduction="mean")).item()) - base_ppl
print(f"naive4 (fake):        ΔPPL = {naive_dppl:+.4f}")
del po, co, past; gc.collect(); torch.cuda.empty_cache()

# ── Evaluate: nsep+pchan4 (fake) ──
with torch.inference_mode():
    po = mdl(ids[:, :pfl], use_cache=True); past = po.past_key_values
    for li in range(nl):
        k, v = get_kv(past, li)
        nk, nv = torch.zeros_like(k), torch.zeros_like(v)
        for h in range(nh):
            nk[0,h] = fake_nsep_pchan4(k[0,h]).to(k.dtype)
            nv[0,h] = fake_nsep_pchan4(v[0,h]).to(v.dtype)
        set_kv(past, li, nk, nv)
    co = mdl(ids[:, pfl:], past_key_values=past, use_cache=False)
cl = co.logits[0].float().cpu()
ca = cl[:-1]
fake_dppl = float(torch.exp(F.cross_entropy(ca[:ml], ta[:ml], reduction="mean")).item()) - base_ppl
print(f"nsep+pchan4 (fake):   ΔPPL = {fake_dppl:+.4f}")
del po, co, past; gc.collect(); torch.cuda.empty_cache()

# ── Evaluate: nsep+pchan4 (REAL packed) ──
with torch.inference_mode():
    po = mdl(ids[:, :pfl], use_cache=True); past = po.past_key_values
    for li in range(nl):
        real_k, real_v = packed_cache.decompress(li)
        set_kv(past, li, real_k.to(device), real_v.to(device))
    co = mdl(ids[:, pfl:], past_key_values=past, use_cache=False)
cl = co.logits[0].float().cpu()
ca = cl[:-1]
real_dppl = float(torch.exp(F.cross_entropy(ca[:ml], ta[:ml], reduction="mean")).item()) - base_ppl
print(f"nsep+pchan4 (REAL):   ΔPPL = {real_dppl:+.4f}")
del po, co, past; gc.collect(); torch.cuda.empty_cache()

# ── Summary ──
print(f"\n{'='*60}")
print(f"SUMMARY: Qwen2-7B Real INT4 Packing")
print(f"{'='*60}")
print(f"  FP16 KV:           {fp16_bytes/1e6:.2f} MB")
print(f"  Real INT4:         {mem['total']/1e6:.2f} MB ({fp16_bytes/mem['total']:.2f}x)")
print(f"  naive4 (fake):     ΔPPL = {naive_dppl:+.4f}")
print(f"  nsep+pchan (fake): ΔPPL = {fake_dppl:+.4f}")
print(f"  nsep+pchan (REAL): ΔPPL = {real_dppl:+.4f}")
print(f"  Fake vs Real diff: {abs(fake_dppl - real_dppl):.6f}")
print(f"  Packing lossless:  {abs(fake_dppl - real_dppl) < 0.1}")
print(f"{'='*60}")

print(f"\nJSON: {json.dumps({'model':'Qwen2-7B','fp16_mb':round(fp16_bytes/1e6,2),'int4_mb':round(mem['total']/1e6,2),'compression':round(fp16_bytes/mem['total'],2),'naive4':round(naive_dppl,4),'fake_nsep':round(fake_dppl,4),'real_nsep':round(real_dppl,4),'diff':round(abs(fake_dppl-real_dppl),6)})}")
