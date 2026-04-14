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
# Phase 7b: Long Context -- Qwen2-7B + Mistral-7B
# ============================================================
# Reviewer request: test long context on
#   Qwen2-7B (catastrophic failure case)
#   Mistral-7B (already-works case)
# to prove nsep+pchan is safe across the full spectrum.
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

print("Loading WikiText-2...")
wikitext = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
wiki_long = " ".join([t for t in wikitext["text"] if len(t.strip()) > 20])
print(f"WikiText-2 concatenated: {len(wiki_long)} chars")

# === CELL 2 ===

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
    raise ValueError(name)

def get_kv(past, li):
    if hasattr(past, 'layers'): return past.layers[li].keys, past.layers[li].values
    return past[li][0], past[li][1]

def set_kv(past, li, k, v):
    if hasattr(past, 'layers'):
        past.layers[li].keys = k; past.layers[li].values = v
    return past

def n_cache_layers(past):
    return len(past.layers) if hasattr(past, 'layers') else len(past)

def compress_cache(past, mn):
    for li in range(n_cache_layers(past)):
        ok, ov = get_kv(past, li)
        ok, ov = ok.clone(), ov.clone()
        nk, nv = torch.zeros_like(ok), torch.zeros_like(ov)
        for h in range(ok.shape[1]):
            nk[0,h] = apply_method(ok[0,h], mn).to(ok.dtype)
            nv[0,h] = apply_method(ov[0,h], mn).to(ov.dtype)
        set_kv(past, li, nk, nv)

def eval_longctx(model_name, hf_id, dtype):
    print(f"\n{'━'*60}")
    print(f"  {model_name} -- Long Context")
    print(f"{'━'*60}")

    tok = AutoTokenizer.from_pretrained(hf_id)
    mdl = AutoModelForCausalLM.from_pretrained(
        hf_id, torch_dtype=dtype, device_map="auto", use_safetensors=True
    )
    mdl.eval()
    if tok.pad_token is None: tok.pad_token = tok.eos_token or "<|endoftext|>"

    ids = tok.encode(wiki_long, return_tensors="pt").to(device)
    max_pos = getattr(mdl.config, "max_position_embeddings", 2048)
    if ids.shape[1] > max_pos: ids = ids[:, :max_pos]
    total = ids.shape[1]
    print(f"  Tokens: {total} (max_pos={max_pos})")

    CL = 40
    PFLS = [256, 512, 1024, 1536, 2048, 4096]
    PFLS = [p for p in PFLS if p + CL <= total]

    results = []
    for pfl in PFLS:
        fi = ids[:, :pfl+CL]; tgt = fi[0, pfl:].cpu()
        with torch.inference_mode():
            bl = mdl(fi, use_cache=False).logits[0, pfl-1:-1].float().cpu()
        bp = float(torch.exp(F.cross_entropy(bl, tgt, reduction="mean")).item())

        row = {"pfl": pfl, "bp": round(bp, 2), "methods": {}}
        for mn in ["naive4", "nsep+pchan4"]:
            with torch.inference_mode():
                po = mdl(ids[:, :pfl], use_cache=True)
                past = po.past_key_values
                compress_cache(past, mn)
                co = mdl(ids[:, pfl:pfl+CL], past_key_values=past, use_cache=False)
            cl = co.logits[0].float().cpu()
            ca, ba, ta = cl[:-1], bl[:-1], tgt[1:]
            ml = min(ca.shape[0], ba.shape[0], ta.shape[0])
            dp = float(torch.exp(F.cross_entropy(ca[:ml], ta[:ml], reduction="mean")).item()) - bp
            row["methods"][mn] = round(dp, 4)
            del po, co, past; gc.collect(); torch.cuda.empty_cache()

        results.append(row)
        n4 = row["methods"]["naive4"]; ns = row["methods"]["nsep+pchan4"]
        print(f"  pfl={pfl:>5}: base={bp:.2f}  naive4={n4:>+9.3f}  nsep+pc={ns:>+9.3f}")
        del bl; gc.collect()

    del mdl, tok; gc.collect(); torch.cuda.empty_cache()
    return results

# ── Run ──
all_results = {}

r1 = eval_longctx("Qwen2-7B", "Qwen/Qwen2-7B", torch.float16)
all_results["Qwen2-7B"] = r1

r2 = eval_longctx("Mistral-7B", "mistralai/Mistral-7B-v0.1", torch.float16)
all_results["Mistral-7B"] = r2

# ── Summary ──
print(f"\n{'='*60}")
print("LONG CONTEXT SUMMARY")
print(f"{'='*60}")

for mname, data in all_results.items():
    print(f"\n  {mname}:")
    print(f"  {'pfl':>6} {'base':>6} {'naive4':>9} {'nsep+pc':>9}")
    for r in data:
        print(f"  {r['pfl']:>6} {r['bp']:>6.1f} {r['methods']['naive4']:>+9.3f} {r['methods']['nsep+pchan4']:>+9.3f}")

print(f"\n{'='*60}")
print("JSON OUTPUT:")
print(f"{'='*60}")
output = {
    "experiment": "phase7b_longctx_qwen_mistral",
    "timestamp": datetime.now().isoformat(),
    "results": all_results,
}
print(json.dumps(output, indent=2, ensure_ascii=False))
