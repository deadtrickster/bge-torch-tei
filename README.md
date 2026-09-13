# bge-torch-tei

bge-m3 dense embedder on AMD ROCm GPUs via PyTorch, speaking the TEI `/embed`
protocol so it slots into an [embed-fanout](https://github.com/) pool as a
`tei@` backend. Deployed on two Strix Halo boxes (lubuntu3 + lubuntu2) feeding
a RAGFlow arXiv ingestion pipeline.

## Measured throughput

| GPU | arch | chunks/s @ batch 32 | notes |
|-----|------|--------------------:|-------|
| R9700 (32GB, USB4 dock) | gfx1201 | 162 bench / 114 sustained | torch bf16+SDPA; batch 32 sweet spot (falls past it from padding) |
| iGPU 8060S (Strix Halo) | gfx1151 | 27.6 | same config; GTT-backed |
| ollama (replaced) | gfx1151 | ~18 | llama.cpp HIP path |

## Requirements

- Python 3.12+ with `torch==2.13.0 --index-url https://download.pytorch.org/whl/rocm7.1`
  and `transformers`
- HF cache with `BAAI/bge-m3` (model + tokenizer)
- Kernel amdgpu/KFD access (`/dev/kfd`)

### GPU-specific notes

- **gfx1151 (iGPU)**: the torch rocm7.1 wheel's bundled `libhsa-runtime64.so`
  null-derefs on gfx1151 at the first kernel launch (segfault at 0x34,
  `libhsa-runtime64.so+0x5a70`). You MUST preload the system ROCm 10 runtime:
  ```
  Environment=LD_PRELOAD=/opt/rocm/lib/libhsa-runtime64.so
  ```
- **gfx1201 (R9700)**: runs natively. Do NOT use the LD_PRELOAD there — it
  costs ~15% throughput.

## Server

```bash
BGE_PORT=8110 BGE_BIND=0.0.0.0 python3 server.py
```

Endpoints:
- `POST /embed` — `{"inputs": [...], "truncate": true}` → `[[...], ...]` (TEI protocol)
- `GET /health` — `{"status":"ok"}` when model is loaded

Environment: `BGE_PORT` (8110), `BGE_MAX_BATCH` (32), `BGE_BIND` (127.0.0.1).

Architecture: HTTP handler threads → work queue → tokenizer thread (merges
whole requests up to `BGE_MAX_BATCH`, never splits) → GPU worker (pre-tokenized
tensors only, so tokenization never idles the card).

## Systemd units

Two instances on a dual-GPU box:

```ini
# bge-torch.service (iGPU, port 8110)
[Service]
Environment=ROCR_VISIBLE_DEVICES=0
Environment=BGE_PORT=8110
Environment=LD_PRELOAD=/opt/rocm/lib/libhsa-runtime64.so
ExecStart=%h/tei-rocm/venv/bin/python3 %h/Projects/bge-torch-tei/server.py

# bge-torch-dgpu.service (R9700, port 8111)
[Service]
Environment=ROCR_VISIBLE_DEVICES=1
Environment=BGE_PORT=8111
ExecStart=%h/tei-rocm/venv/bin/python3 %h/Projects/bge-torch-tei/server.py
```

Device index (`ROCR_VISIBLE_DEVICES`) counts BOUND devices: with an unbound
iGPU the R9700 enumerates as index 0.

If a qwen LLM server shares the dGPU, add mutual exclusion:

```ini
[Unit]
Conflicts=qwen-server.service
After=qwen-server.service
```

## Bench

```bash
python3 bench.py [device_index]
```

Real ~400-token chunks from the local arXiv corpus, bf16+SDPA, batches
32/64/128.

## Fleet deployment

The embed-fanout proxy (ollama→TEI protocol bridge on :11434) takes these
as `tei@host:port` backends. Current fleet pool: iGPU (:8110) + R9700
(:8111) + lubuntu2's R9700 (:8111 on 192.168.1.78) = ~194 chunks/s burst.
