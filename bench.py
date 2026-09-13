#!/usr/bin/env python3
# bench_bge.py - bge-m3 dense embedding throughput bench on torch/ROCm.
# Recreates the methodology from session 9c33af53: real ~400-token arXiv
# chunks, bf16, SDPA, batches 32/64/128, CLS pooling, warm GPU, wall-clock
# chunks/s. Original numbers (R9700 gfx1201, torch 2.13.0+rocm7.1):
#   batch 32: 162  batch 64: 147  batch 128: 131
# iGPU 8060S (gfx1151): batch 32: 27.6  batch 64: 23.5  batch 128: 22.4
import sys, time, json, torch
from transformers import AutoTokenizer, AutoModel

dev = "cuda"  # ROCm presents as cuda in torch
print(f"torch {torch.__version__}  hip {torch.version.hip}  available={torch.cuda.is_available()}", flush=True)
for i in range(torch.cuda.device_count()):
    print(f"  device {i}: {torch.cuda.get_device_name(i)}  arch={torch.cuda.get_device_properties(i).gcnArchName}", flush=True)

idx = int(sys.argv[1]) if len(sys.argv) > 1 else 0
torch.cuda.set_device(idx)
tok = AutoTokenizer.from_pretrained("BAAI/bge-m3")
model = AutoModel.from_pretrained("BAAI/bge-m3", dtype=torch.bfloat16, attn_implementation="sdpa").to(dev).eval()
print(f"loaded on {torch.cuda.get_device_name(idx)}  attn={model.config._attn_implementation}", flush=True)

# build chunks from local corpus if available
import glob
text = ""
for f in sorted(glob.glob("/mnt/data2/corpus/arxiv/*.txt")):
    try: text += open(f, errors="ignore").read() + "\n\n"
    except Exception: pass
    if len(text) > 900000: break
if text:
    chunks = [text[i:i+1500] for i in range(0, min(len(text), 1500*256), 1500)][:256]
    print(f"{len(chunks)} chunks from corpus, avg {sum(map(len,chunks))//len(chunks)} chars", flush=True)
else:
    chunks = ["benchmark filler text " * 80] * 256
    print(f"{len(chunks)} synthetic chunks", flush=True)

@torch.inference_mode()
def embed(batch):
    enc = tok(batch, padding=True, truncation=True, max_length=8192, return_tensors="pt").to(dev)
    out = model(**enc).last_hidden_state[:, 0]          # CLS pooling, bge-m3 dense
    return torch.nn.functional.normalize(out, dim=-1)

embed(chunks[:8]); torch.cuda.synchronize()               # warm-up
for bs in (32, 64, 128):
    t0 = time.time(); n = 0
    for i in range(0, 256, bs):
        n += embed(chunks[i:i+bs]).shape[0]
    torch.cuda.synchronize(); dt = time.time() - t0
    print(f"  batch={bs:<3}  {n} chunks in {dt:5.2f}s  = {n/dt:6.1f} chunks/s", flush=True)
