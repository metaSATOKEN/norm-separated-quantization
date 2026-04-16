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
# PoC: Needle-in-Haystack x KV Cache Quantization
# ============================================================
# Does KV cache quantization make the model "forget" a fact
# buried in a long context? Does nsep+pchan4 preserve it?
#
# Setup:
#   1. Build a "haystack" of filler text (~500-2000 tokens)
#   2. Insert a "needle" fact at various positions
#   3. Ask the model to recall the needle
#   4. Compare: FP16 vs naive4 vs nsep+pchan4
#
# Colab GPU recommended for Qwen2-7B.
# GPT-2 version runs locally.
# ============================================================

# === CELL 1 ===
!pip install -q transformers accelerate hf_transfer
import os; os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"

import gc, json, numpy as np, torch, torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from datetime import datetime

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {device}")
if device == "cuda":
    print(f"GPU: {torch.cuda.get_device_name()}")

# === CELL 2 ===

# ── Quantization ───────────────────────────────────────────────────────────

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

# ── Haystack Construction ──────────────────────────────────────────────────

FILLER_PARAGRAPHS = [
    "The weather in coastal regions is influenced by ocean currents and prevailing winds. "
    "Maritime climates tend to have moderate temperatures year-round, with cool summers "
    "and mild winters. Precipitation is distributed relatively evenly throughout the year.",

    "Ancient civilizations developed sophisticated agricultural techniques to sustain "
    "growing populations. Irrigation systems, crop rotation, and selective breeding were "
    "among the innovations that transformed human societies from nomadic to sedentary.",

    "The study of mathematics has evolved over millennia, from simple counting systems "
    "to abstract algebra and topology. Each era built upon the discoveries of its "
    "predecessors, creating an ever-expanding body of knowledge.",

    "Music theory encompasses the rules and practices of harmony, melody, and rhythm. "
    "Different cultures have developed distinct musical traditions, from the pentatonic "
    "scales of East Asia to the complex polyrhythms of West Africa.",

    "Volcanic activity shapes landscapes through both destructive and creative forces. "
    "Eruptions can devastate entire regions, but the mineral-rich soils they produce "
    "often become some of the most fertile agricultural land on Earth.",

    "The development of printing technology revolutionized the spread of information. "
    "From Gutenberg's movable type to modern digital publishing, each advance has "
    "democratized access to knowledge and transformed society.",

    "Marine ecosystems support an incredible diversity of life forms. Coral reefs, "
    "often called the rainforests of the sea, host thousands of species in complex "
    "interdependent relationships that scientists are still working to understand.",

    "The principles of aerodynamics govern the design of aircraft, automobiles, and "
    "even buildings. Understanding how air flows around objects has enabled engineers "
    "to create increasingly efficient and stable structures.",

    "Traditional medicine practices from around the world have contributed to modern "
    "pharmacology. Many widely used drugs were originally derived from plants that "
    "indigenous peoples had used for centuries to treat various ailments.",

    "The architecture of ancient Rome continues to influence building design today. "
    "Innovations such as the arch, the dome, and concrete construction techniques "
    "enabled the Romans to create structures of unprecedented scale and durability.",
]

NEEDLE = "The secret code for the experiment is: BLUE ELEPHANT 7742."
QUESTION = "What is the secret code for the experiment?"
EXPECTED = "BLUE ELEPHANT 7742"


def build_haystack(n_paragraphs, needle_position_ratio):
    """
    Build haystack with needle inserted at a specific relative position.
    needle_position_ratio: 0.0 = start, 0.5 = middle, 1.0 = end
    """
    # Repeat filler paragraphs as needed
    paragraphs = []
    for i in range(n_paragraphs):
        paragraphs.append(FILLER_PARAGRAPHS[i % len(FILLER_PARAGRAPHS)])

    # Insert needle
    insert_idx = int(len(paragraphs) * needle_position_ratio)
    insert_idx = max(0, min(insert_idx, len(paragraphs)))
    paragraphs.insert(insert_idx, NEEDLE)

    haystack = " ".join(paragraphs)
    prompt = haystack + "\n\nQuestion: " + QUESTION + "\nAnswer:"
    return prompt


# ── Needle Retrieval ───────────────────────────────────────────────────────

def test_needle_retrieval(model, tokenizer, prompt, method_name, max_new=30):
    """
    Generate response after compressing KV cache.
    Returns generated text.
    """
    ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
    prompt_len = ids.shape[1]

    max_pos = getattr(model.config, "max_position_embeddings",
              getattr(model.config, "n_positions", 2048))
    if prompt_len > max_pos - max_new:
        ids = ids[:, -(max_pos - max_new):]
        prompt_len = ids.shape[1]

    with torch.inference_mode():
        # Prefill
        out = model(ids, use_cache=True)
        past = out.past_key_values

        # Compress KV cache (skip for baseline)
        if method_name != "baseline":
            compress_cache(past, method_name)

        # Generate
        generated = []
        next_logits = out.logits[:, -1:]
        for _ in range(max_new):
            next_tok = next_logits.argmax(dim=-1)
            generated.append(next_tok[0, 0].item())
            if next_tok[0, 0].item() == tokenizer.eos_token_id:
                break
            out = model(next_tok, past_key_values=past, use_cache=True)
            past = out.past_key_values
            next_logits = out.logits[:, -1:]

    text = tokenizer.decode(generated, skip_special_tokens=True).strip()
    del out, past; gc.collect()
    if device == "cuda": torch.cuda.empty_cache()
    return text


def check_needle_found(response, expected=EXPECTED):
    """Check if the needle fact appears in the response."""
    response_lower = response.lower()
    expected_lower = expected.lower()
    # Check for exact match or key parts
    if expected_lower in response_lower:
        return True, "exact"
    # Check partial: "blue elephant" and "7742"
    has_animal = "blue elephant" in response_lower
    has_code = "7742" in response_lower
    if has_animal and has_code:
        return True, "partial_both"
    if has_animal or has_code:
        return True, "partial_one"
    return False, "missed"


# ── Main Experiment ────────────────────────────────────────────────────────

MODEL_CONFIGS = [
    ("Qwen2-7B", "Qwen/Qwen2-7B", torch.float16),
]

# Fallback for local: use GPT-2
if device == "cpu":
    MODEL_CONFIGS = [("GPT-2", "gpt2", torch.float32)]

METHODS = ["baseline", "naive4", "nsep+pchan4"]
HAYSTACK_SIZES = [5, 10, 20]  # number of filler paragraphs
NEEDLE_POSITIONS = [0.0, 0.25, 0.5, 0.75, 1.0]  # relative position

all_results = {}

for model_name, hf_id, dtype in MODEL_CONFIGS:
    print(f"\n{'='*60}")
    print(f"  {model_name} -- Needle in Haystack")
    print(f"{'='*60}")

    tok = AutoTokenizer.from_pretrained(hf_id)
    mdl = AutoModelForCausalLM.from_pretrained(
        hf_id, torch_dtype=dtype, device_map="auto", use_safetensors=True
    )
    mdl.eval()
    if tok.pad_token is None: tok.pad_token = tok.eos_token or "<|endoftext|>"

    model_results = []

    for n_para in HAYSTACK_SIZES:
        for pos in NEEDLE_POSITIONS:
            prompt = build_haystack(n_para, pos)
            n_tokens = len(tok.encode(prompt))

            row = {
                "n_paragraphs": n_para, "needle_position": pos,
                "total_tokens": n_tokens, "methods": {}
            }

            print(f"\n  paragraphs={n_para}, pos={pos:.0%}, tokens={n_tokens}")

            for method in METHODS:
                response = test_needle_retrieval(mdl, tok, prompt, method)
                found, match_type = check_needle_found(response)

                row["methods"][method] = {
                    "response": response[:100],
                    "found": found,
                    "match_type": match_type,
                }

                status = "FOUND" if found else "MISSED"
                print(f"    {method:>14}: [{status:>6}] ({match_type}) \"{response[:60]}\"")

            model_results.append(row)

    all_results[model_name] = model_results
    del mdl, tok; gc.collect()
    if device == "cuda": torch.cuda.empty_cache()

# ── Summary ────────────────────────────────────────────────────────────────

print(f"\n{'='*60}")
print("NEEDLE RETRIEVAL SUMMARY")
print(f"{'='*60}")

for model_name, results in all_results.items():
    print(f"\n  {model_name}:")
    print(f"  {'Config':<25} {'baseline':>10} {'naive4':>10} {'nsep+pc':>10}")
    print(f"  {'-'*55}")
    for r in results:
        label = f"p={r['n_paragraphs']}, pos={r['needle_position']:.0%}"
        scores = {}
        for m in METHODS:
            scores[m] = "FOUND" if r["methods"][m]["found"] else "MISS"
        print(f"  {label:<25} {scores['baseline']:>10} {scores['naive4']:>10} {scores['nsep+pchan4']:>10}")

# Aggregate
for model_name, results in all_results.items():
    print(f"\n  {model_name} -- Retrieval Rate:")
    for m in METHODS:
        found = sum(1 for r in results if r["methods"][m]["found"])
        total = len(results)
        print(f"    {m:>14}: {found}/{total} ({found/total:.0%})")

print(f"\n{'='*60}")
print("JSON OUTPUT:")
print(f"{'='*60}")
output = {
    "experiment": "poc_needle_in_haystack",
    "timestamp": datetime.now().isoformat(),
    "results": all_results,
}
print(json.dumps(output, indent=2, ensure_ascii=False))
