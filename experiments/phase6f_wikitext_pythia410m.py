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
# Phase 6f: WikiText-2 -- Pythia-410M
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
wiki_text = "\n\n".join([t for t in wikitext["text"] if len(t.strip()) > 50][:50])
print(f"WikiText-2: {len(wiki_text)} chars")

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

print("Loading Pythia-410M...")
tok = AutoTokenizer.from_pretrained("EleutherAI/pythia-410m")
mdl = AutoModelForCausalLM.from_pretrained(
    "EleutherAI/pythia-410m", torch_dtype=torch.float32, device_map="auto", use_safetensors=True
)
mdl.eval()
if tok.pad_token is None: tok.pad_token = tok.eos_token

ids = tok.encode(wiki_text, return_tensors="pt").to(device)
max_len = getattr(mdl.config, "max_position_embeddings", 2048)
if ids.shape[1] > max_len: ids = ids[:, :max_len]

CL = 30; chunk_size = 150; total = ids.shape[1]
n_chunks = min(5, (total - CL) // chunk_size)
print(f"Tokens: {total}, Chunks: {n_chunks}")

results = {"naive4": [], "nsep+pchan4": []}; baselines = []

for ci in range(n_chunks):
    start = ci * chunk_size; end = start + chunk_size + CL
    if end > total: break
    chunk = ids[:, start:end]; tgt = chunk[0, chunk_size:].cpu()
    with torch.inference_mode():
        bl = mdl(chunk, use_cache=False).logits[0, chunk_size-1:-1].float().cpu()
    bp = float(torch.exp(F.cross_entropy(bl, tgt, reduction="mean")).item())
    baselines.append(bp)
    for mn in ["naive4", "nsep+pchan4"]:
        with torch.inference_mode():
            po = mdl(chunk[:, :chunk_size], use_cache=True)
            past = po.past_key_values; compress_cache(past, mn)
            co = mdl(chunk[:, chunk_size:], past_key_values=past, use_cache=False)
        cl = co.logits[0].float().cpu()
        ca, ba, ta = cl[:-1], bl[:-1], tgt[1:]
        ml = min(ca.shape[0], ba.shape[0], ta.shape[0])
        dp = float(torch.exp(F.cross_entropy(ca[:ml], ta[:ml], reduction="mean")).item()) - bp
        results[mn].append(dp)
        del po, co, past; gc.collect(); torch.cuda.empty_cache()
    del bl; gc.collect()
    print(f"  chunk {ci}: base={bp:.1f}  naive4={results['naive4'][-1]:+.3f}  nsep+pc={results['nsep+pchan4'][-1]:+.3f}")

mn_bp = np.mean(baselines); mn_n = np.mean(results["naive4"]); mn_s = np.mean(results["nsep+pchan4"])
imp = abs(mn_n) / abs(mn_s) if abs(mn_s) > 0.01 else float("inf")

print(f"\nRESULT: base={mn_bp:.2f}  naive4={mn_n:+.4f}  nsep+pc={mn_s:+.4f}  ({imp:.1f}x)")
print(f"\nJSON: {json.dumps({'model':'Pythia-410M','bp':round(mn_bp,2),'n4':round(mn_n,4),'ns':round(mn_s,4),'imp':round(imp,2)})}")
