# TEI-compatible bge-m3 embeddings on AMD GPUs

<p align="center"><img src="logo.svg" width="120" alt="bge-m3 on ROCm"/></p>

TEI-compatible embedding server for [bge-m3](https://huggingface.co/BAAI/bge-m3)
on AMD RDNA GPUs. Drop-in replacement for
[text-embeddings-inference](https://github.com/huggingface/text-embeddings-inference)
on hardware TEI doesn't support.

## Why

[TEI](https://github.com/huggingface/text-embeddings-inference)'s
[ROCm support](https://huggingface.co/docs/text-embeddings-inference/en/amd_gpu)
is experimental and only tested on Instinct cards (MI200/MI300). On consumer
RDNA (gfx1201, gfx1151) the candle backend never initializes; the Python
fallback reduces to PyTorch behind the router anyway. This is that setup
without the router — measured 114 ch/s direct vs 82 through the TEI router
on the same GPU.

## Throughput

~400-token chunks, bf16, SDPA, batch 32:

| GPU | arch | engine | chunks/s |
|-----|------|--------|---------:|
| Radeon AI PRO R9700 | gfx1201 | **this** | **162** |
| Radeon AI PRO R9700 | gfx1201 | llama.cpp HIP | ~44 |
| Strix Halo 8060S iGPU | gfx1151 | **this** | **27.6** |
| Strix Halo 8060S iGPU | gfx1151 | ollama | ~18 |
| RTX 6000 Blackwell | CUDA | TEI | ~220 |

On the R9700, torch is **3.7× llama.cpp** on the same silicon (llama.cpp's
HIP path saturates at ~44 regardless of batch or flash attention). ~1.4×
slower than an RTX 6000, at roughly **10× lower price**. Comparison holds
for this workload shape only (single-model, batch-32, ~400-token chunks).

## Setup

```bash
python3 -m venv venv
./venv/bin/pip install torch==2.13.0 --index-url https://download.pytorch.org/whl/rocm7.1
./venv/bin/pip install transformers
./venv/bin/python bench.py
```

## GPU notes

**gfx1151 (Strix Halo / Phoenix APUs)**: the torch rocm7.1 wheel's bundled
`libhsa-runtime64.so` segfaults at the first kernel launch (`segfault at
0x34`, `libhsa-runtime64.so+0x5a70`). Preload the system ROCm runtime:

```
LD_PRELOAD=/opt/rocm/lib/libhsa-runtime64.so
```

`LD_LIBRARY_PATH` doesn't help — torch's RPATH wins.

**gfx1201 (RDNA4 discrete)**: runs natively. Do **not** use the LD_PRELOAD
above — it costs ~15% throughput.

## Server

```bash
BGE_PORT=8110 BGE_BIND=0.0.0.0 ./venv/bin/python3 server.py
```

- `POST /embed` — `{"inputs": [...], "truncate": true}` → `[[...], ...]`
- `GET /health` — `{"status":"ok"}`

Env: `BGE_PORT` (8110), `BGE_MAX_BATCH` (32), `BGE_BIND` (127.0.0.1).

Architecture: HTTP threads → merge-whole-only batcher → tokenizer thread →
GPU worker. Tokenization runs off the GPU's critical path. See `server.py`
for why splitting batches was a bug.

## Systemd

```ini
[Unit]
Description=bge-m3 embedder on GPU

[Service]
Environment=ROCR_VISIBLE_DEVICES=0
Environment=BGE_PORT=8110
Environment=LD_PRELOAD=/opt/rocm/lib/libhsa-runtime64.so  # gfx1151 only
ExecStart=%h/bge-torch-tei/venv/bin/python3 %h/bge-torch-tei/server.py
Restart=on-failure

[Install]
WantedBy=default.target
```

`ROCR_VISIBLE_DEVICES` counts **bound** devices. If one GPU failed to probe,
the other shifts to index 0.
