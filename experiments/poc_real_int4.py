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
PoC: Real INT4 Packing for nsep+pchan KV Cache

Demonstrates actual memory reduction by packing INT4 values
into uint8 tensors (2 values per byte).

Compares:
  1. FP16 KV cache (baseline, 16 bits per element)
  2. Fake INT4 (simulated, still 16 bits in memory)
  3. Real INT4 packed (4 bits per element + FP16 scales + FP16 norms)

Validates that real packing produces identical DPPL to fake quantization.
"""

import gc, sys, json, numpy as np, torch, torch.nn.functional as F
from pathlib import Path
from datetime import datetime
from transformers import GPT2LMHeadModel, GPT2Tokenizer

# ── Real INT4 Pack / Unpack ────────────────────────────────────────────────

def pack_int4(values_int: torch.Tensor) -> torch.Tensor:
    """
    Pack INT4 values (-7 to +7) into uint8 (2 values per byte).
    Input: (*, N) int8 tensor with values in [-7, 7]
    Output: (*, N//2) uint8 tensor

    Packing: low nibble = values[..., 0::2], high nibble = values[..., 1::2]
    Values are stored as unsigned (offset by 8): stored = value + 8 (range 1-15)
    """
    # Offset to unsigned: -7..+7 -> 1..15 (0 reserved for exact zero)
    unsigned = (values_int + 8).to(torch.uint8)  # 1..15
    # Pack pairs into single bytes
    low = unsigned[..., 0::2]   # even indices
    high = unsigned[..., 1::2]  # odd indices
    packed = low | (high << 4)
    return packed


def unpack_int4(packed: torch.Tensor, original_size: int) -> torch.Tensor:
    """
    Unpack uint8 back to INT4 values.
    Output: (*, original_size) int8 tensor with values in [-7, 7]
    """
    low = (packed & 0x0F).to(torch.int8) - 8
    high = ((packed >> 4) & 0x0F).to(torch.int8) - 8
    # Interleave back
    result = torch.stack([low, high], dim=-1).reshape(*packed.shape[:-1], -1)
    # Trim if original size was odd
    if result.shape[-1] > original_size:
        result = result[..., :original_size]
    return result


# ── nsep+pchan with Real Packing ──────────────────────────────────────────

class PackedKVCache:
    """
    Real INT4 KV cache with norm separation.

    Storage per (head, token):
      - norm: 1 x FP16 (2 bytes)
      - direction_packed: head_dim/2 x uint8 (head_dim/2 bytes)
      - scale: head_dim x FP16 (shared across tokens, 2*head_dim bytes per head)

    Total per element: 4 bits + amortized scale + 1 norm
    """

    def __init__(self):
        self.layers = []  # list of {norms_k, packed_k, scale_k, norms_v, packed_v, scale_v}

    def compress_and_store(self, keys: torch.Tensor, values: torch.Tensor):
        """
        Compress one layer's KV cache to real INT4.
        keys/values: (batch, n_heads, seq_len, head_dim) float16
        """
        B, nh, sl, hd = keys.shape
        assert B == 1, "Batch > 1 not supported in PoC"

        layer = {}
        for role, tensor in [("k", keys), ("v", values)]:
            x = tensor[0].float()  # (nh, sl, hd)

            # 1. Norm separation
            norms = x.norm(dim=-1, keepdim=True).clamp(min=1e-12)  # (nh, sl, 1)
            direction = x / norms  # (nh, sl, hd)

            # 2. Per-channel scale (shared across all tokens in this head)
            scale = direction.abs().amax(dim=1, keepdim=True).clamp(min=1e-12) / 7  # (nh, 1, hd)

            # 3. Quantize to INT4 levels
            quantized = (direction / scale).round().clamp(-7, 7).to(torch.int8)  # (nh, sl, hd)

            # 4. Pack into uint8
            packed = pack_int4(quantized)  # (nh, sl, hd//2)

            layer[f"norms_{role}"] = norms.half()   # FP16
            layer[f"packed_{role}"] = packed         # uint8
            layer[f"scale_{role}"] = scale.half()    # FP16

        self.layers.append(layer)

    def decompress(self, layer_idx: int) -> tuple:
        """
        Decompress one layer back to float16 KV tensors.
        Returns: (keys, values) each (1, n_heads, seq_len, head_dim) float16
        """
        layer = self.layers[layer_idx]
        results = []

        for role in ["k", "v"]:
            norms = layer[f"norms_{role}"].float()     # (nh, sl, 1)
            packed = layer[f"packed_{role}"]            # (nh, sl, hd//2)
            scale = layer[f"scale_{role}"].float()      # (nh, 1, hd)

            hd = scale.shape[-1]
            quantized = unpack_int4(packed, hd).float()  # (nh, sl, hd)

            # Dequantize
            direction = quantized * scale               # (nh, sl, hd)

            # Re-normalize
            dir_norm = direction.norm(dim=-1, keepdim=True).clamp(min=1e-8)
            direction = direction / dir_norm

            # Reconstruct
            x = norms * direction
            results.append(x.half().unsqueeze(0))  # (1, nh, sl, hd)

        return results[0], results[1]

    def memory_bytes(self) -> dict:
        """Calculate actual memory usage."""
        total_packed = 0
        total_norms = 0
        total_scales = 0

        for layer in self.layers:
            for role in ["k", "v"]:
                total_packed += layer[f"packed_{role}"].numel() * 1   # uint8 = 1 byte
                total_norms += layer[f"norms_{role}"].numel() * 2    # FP16 = 2 bytes
                total_scales += layer[f"scale_{role}"].numel() * 2   # FP16 = 2 bytes

        return {
            "packed_bytes": total_packed,
            "norms_bytes": total_norms,
            "scales_bytes": total_scales,
            "total_bytes": total_packed + total_norms + total_scales,
        }


# ── Fake quantization (for comparison) ─────────────────────────────────────

def fake_nsep_pchan4(x):
    x = x.float()
    n = x.norm(dim=-1, keepdim=True).clamp(min=1e-12)
    d = x / n
    scale = d.abs().amax(dim=0, keepdim=True).clamp(min=1e-12) / 7
    dq = (d / scale).round().clamp(-7, 7) * scale
    dq = dq / dq.norm(dim=-1, keepdim=True).clamp(min=1e-8)
    return n * dq


# ── Main ───────────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("PoC: Real INT4 Packing for nsep+pchan")
    print("=" * 60)

    tok = GPT2Tokenizer.from_pretrained("gpt2")
    mdl = GPT2LMHeadModel.from_pretrained("gpt2")
    mdl.eval()

    TEXT = (
        "The old lighthouse keeper climbed the spiral staircase each evening, "
        "carrying a lantern that cast long shadows across the stone walls. "
        "He had performed this ritual for forty years, ever since the automated "
        "systems had failed during the great storm. The sea below crashed "
        "against the rocks with a rhythm that matched his breathing."
    )

    ids = tok.encode(TEXT, return_tensors="pt")
    T = ids.shape[1]
    CL = 20
    pfl = T - CL

    # ── Step 1: Baseline PPL ──
    print(f"\nTokens: {T}, prefill: {pfl}, continuation: {CL}")

    with torch.inference_mode():
        bl = mdl(ids, use_cache=False).logits[0, pfl-1:-1].float().cpu()
    tgt = ids[0, pfl:].cpu()
    base_ppl = float(torch.exp(F.cross_entropy(bl, tgt, reduction="mean")).item())
    print(f"Baseline PPL: {base_ppl:.2f}")

    # ── Step 2: Get KV cache ──
    with torch.inference_mode():
        out = mdl(ids[:, :pfl], use_cache=True)
    past = out.past_key_values
    n_layers = len(past.layers)
    k0 = past.layers[0].keys
    nh, sl, hd = k0.shape[1], k0.shape[2], k0.shape[3]
    print(f"KV cache: {n_layers} layers, {nh} heads, seq_len={sl}, head_dim={hd}")

    # ── Step 3: Memory comparison ──
    print(f"\n--- Memory Comparison ---")

    fp16_bytes = 2 * n_layers * nh * sl * hd * 2  # K+V, 2 bytes per fp16
    print(f"FP16 KV cache:  {fp16_bytes:>10,} bytes ({fp16_bytes/1024:.1f} KB)")

    # Real INT4 packing
    packed_cache = PackedKVCache()
    for li in range(n_layers):
        dl = past.layers[li]
        packed_cache.compress_and_store(dl.keys.clone(), dl.values.clone())

    mem = packed_cache.memory_bytes()
    print(f"Real INT4 packed:")
    print(f"  Packed data:  {mem['packed_bytes']:>10,} bytes")
    print(f"  Norms (FP16): {mem['norms_bytes']:>10,} bytes")
    print(f"  Scales (FP16):{mem['scales_bytes']:>10,} bytes")
    print(f"  Total:        {mem['total_bytes']:>10,} bytes ({mem['total_bytes']/1024:.1f} KB)")
    print(f"  Compression:  {fp16_bytes / mem['total_bytes']:.2f}x")

    # ── Step 4: Verify quality (real packed == fake quantized) ──
    print(f"\n--- Quality Verification ---")

    # Fake quantization path
    with torch.inference_mode():
        out_fake = mdl(ids[:, :pfl], use_cache=True)
        past_fake = out_fake.past_key_values
        for li in range(n_layers):
            dl = past_fake.layers[li]
            ok, ov = dl.keys.clone(), dl.values.clone()
            nk, nv = torch.zeros_like(ok), torch.zeros_like(ov)
            for h in range(nh):
                nk[0,h] = fake_nsep_pchan4(ok[0,h]).to(ok.dtype)
                nv[0,h] = fake_nsep_pchan4(ov[0,h]).to(ov.dtype)
            dl.keys, dl.values = nk, nv
        co_fake = mdl(ids[:, pfl:], past_key_values=past_fake, use_cache=False)
    fake_logits = co_fake.logits[0].float().cpu()
    ca, ba, ta = fake_logits[:-1], bl[:-1], tgt[1:]
    ml = min(ca.shape[0], ba.shape[0], ta.shape[0])
    fake_dppl = float(torch.exp(F.cross_entropy(ca[:ml], ta[:ml], reduction="mean")).item()) - base_ppl
    del out_fake, past_fake, co_fake; gc.collect()

    # Real packed path
    with torch.inference_mode():
        out_real = mdl(ids[:, :pfl], use_cache=True)
        past_real = out_real.past_key_values
        for li in range(n_layers):
            real_k, real_v = packed_cache.decompress(li)
            past_real.layers[li].keys = real_k
            past_real.layers[li].values = real_v
        co_real = mdl(ids[:, pfl:], past_key_values=past_real, use_cache=False)
    real_logits = co_real.logits[0].float().cpu()
    ca2 = real_logits[:-1]
    real_dppl = float(torch.exp(F.cross_entropy(ca2[:ml], ta[:ml], reduction="mean")).item()) - base_ppl
    del out_real, past_real, co_real; gc.collect()

    # Compare
    print(f"Fake INT4 ΔPPL:   {fake_dppl:+.4f}")
    print(f"Real INT4 ΔPPL:   {real_dppl:+.4f}")
    print(f"Difference:       {abs(fake_dppl - real_dppl):.6f}")

    match = abs(fake_dppl - real_dppl) < 0.01
    print(f"Match: {'YES -- real packing is lossless vs fake' if match else 'NO -- packing introduced error'}")

    # ── Step 5: Cosine similarity between fake and real ──
    cos_sim = F.cosine_similarity(
        fake_logits.flatten().unsqueeze(0),
        real_logits.flatten().unsqueeze(0)
    ).item()
    print(f"Logits cosine similarity: {cos_sim:.8f}")

    # ── Summary ──
    print(f"\n{'='*60}")
    print(f"SUMMARY")
    print(f"{'='*60}")
    print(f"  FP16 KV cache:     {fp16_bytes/1024:>8.1f} KB")
    print(f"  Real INT4 packed:  {mem['total_bytes']/1024:>8.1f} KB")
    print(f"  Compression ratio: {fp16_bytes / mem['total_bytes']:.2f}x")
    print(f"  Fake ΔPPL:         {fake_dppl:+.4f}")
    print(f"  Real ΔPPL:         {real_dppl:+.4f}")
    print(f"  Packing lossless:  {match}")
    print(f"{'='*60}")

    output = {
        "experiment": "poc_real_int4",
        "timestamp": datetime.now().isoformat(),
        "model": "GPT-2",
        "fp16_bytes": fp16_bytes,
        "real_int4_bytes": mem["total_bytes"],
        "compression_ratio": round(fp16_bytes / mem["total_bytes"], 2),
        "fake_dppl": round(fake_dppl, 4),
        "real_dppl": round(real_dppl, 4),
        "packing_lossless": match,
        "logits_cosine": round(cos_sim, 8),
    }
    results_dir = Path(__file__).parent.parent / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    path = results_dir / "poc_real_int4.json"
    with open(path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved: {path}")


if __name__ == "__main__":
    main()
