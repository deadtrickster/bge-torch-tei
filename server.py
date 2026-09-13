#!/usr/bin/env python3
# bge-m3 dense embedder on AMD ROCm GPUs, speaking the TEI /embed protocol.
#
# Measured (~400-token chunks, bf16, SDPA, batch 32):
#   gfx1201 (Radeon AI PRO R9700): 162 chunks/s bench / 114 sustained
#   gfx1151 (Strix Halo iGPU):     27.6 chunks/s
#
# GPU-specific requirements:
#   gfx1151: MUST preload system ROCm 10 HSA runtime - the one bundled
#     in the torch rocm7.1 wheel null-derefs on gfx1151 at the first kernel
#     launch (segfault at 0x34, libhsa-runtime64.so+0x5a70).
#     Unit sets: LD_PRELOAD=/opt/rocm/lib/libhsa-runtime64.so
#   gfx1201: runs natively, no preload (it costs ~15% there).
#
# Environment:
#   BGE_PORT      listen port (default 8110)
#   BGE_MAX_BATCH merge ceiling (default 32; throughput FALLS past it
#                 due to longest-chunk padding)
#   BGE_BIND      bind address (default 127.0.0.1; set 0.0.0.0 to serve LAN)
import json, os, queue, threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import torch
from transformers import AutoTokenizer, AutoModel

PORT = int(os.environ.get("BGE_PORT", 8110))
MAX_BATCH = int(os.environ.get("BGE_MAX_BATCH", 32))
BIND = os.environ.get("BGE_BIND", "127.0.0.1")
MAX_LEN = 8192           # bge-m3 ceiling; TEI clients send "truncate": true
MODEL_READY = threading.Event()

dev = "cuda"
tok = AutoTokenizer.from_pretrained("BAAI/bge-m3")
model = AutoModel.from_pretrained("BAAI/bge-m3", dtype=torch.bfloat16,
                                  attn_implementation="sdpa").to(dev).eval()
MODEL_READY.set()
print(f"bge-torch-tei: model ready on {torch.cuda.get_device_name(0)}", flush=True)

@torch.inference_mode()
def embed(enc):
    out = model(**enc).last_hidden_state[:, 0]      # CLS pooling, bge-m3 dense
    return torch.nn.functional.normalize(out, dim=-1)

def tokenize(inputs):
    # Tokenization is pure CPU (HF tokenizers, GIL-bound) and was on the GPU
    # worker's critical path: the R9700 benched 162 ch/s with tokenization paid
    # outside the timed loop, but the wrapper delivered 114 - the GPU idled
    # while the worker tokenized every batch. Pre-tokenized batches now move
    # through a second queue; the GPU worker only moves tensors.
    return tok(inputs, padding=True, truncation=True, max_length=MAX_LEN,
               return_tensors="pt")

# ---- micro-batcher: tokenize thread + GPU worker, merge queued up to MAX_BATCH ----
work = queue.Queue()        # items: (inputs:list, reply_queue)
tokenized = queue.Queue()   # items: (pending, enc)

def tokenizer_thread():
    while True:
        first = work.get()
        inputs = list(first[0]); pending = [first]
        # Merge only what fits WHOLE. The old take/put-back split path was a
        # count bug: a split request's handler did one rq.get() but received
        # two puts (first slice now, leftover slice later, never read), so the
        # caller got a PARTIAL batch - fewer vectors than inputs sent.
        while len(inputs) < MAX_BATCH:
            try:
                item = work.get_nowait()
                if len(item[0]) > MAX_BATCH - len(inputs):
                    work.put(item)          # does not fit whole - back it goes
                    break
                pending.append(item)
                inputs.extend(item[0])
            except queue.Empty:
                break
        try:
            tokenized.put((pending, tokenize(inputs)))
        except Exception as e:
            for _, rq in pending: rq.put(("error", str(e)))

def gpu_worker():
    while True:
        pending, enc = tokenized.get()
        try:
            vecs = embed(enc.to(dev)).float().cpu().tolist()
        except Exception as e:
            for _, rq in pending: rq.put(("error", str(e)))
            continue
        off = 0
        for src, rq in pending:
            rq.put(("ok", vecs[off:off + len(src)]))
            off += len(src)

threading.Thread(target=tokenizer_thread, daemon=True).start()
threading.Thread(target=gpu_worker, daemon=True).start()

class H(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    # Default listen backlog is 5; concurrent clients burst past it and
    # connections reset.
    request_queue_size = 128

    def log_message(self, *a): pass

    def _send(self, code, body, ctype="application/json"):
        b = body if isinstance(body, bytes) else body.encode()
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(b)))
        self.end_headers()
        self.wfile.write(b)

    def do_GET(self):
        if self.path.startswith("/health"):
            if MODEL_READY.is_set():
                self._send(200, '{"status":"ok"}')
            else:
                self._send(503, '{"status":"loading"}')
        else:
            self._send(404, '{"error":"not found"}')

    def do_POST(self):
        if not self.path.startswith("/embed"):
            self._send(404, '{"error":"not found"}'); return
        try:
            n = int(self.headers.get("Content-Length", 0))
            d = json.loads(self.rfile.read(n))
        except Exception as e:
            self._send(400, json.dumps({"error": f"bad request: {e}"})); return
        v = d.get("inputs")
        if v is None:
            self._send(400, '{"error":"inputs required"}'); return
        if isinstance(v, str): v = [v]
        if not v:
            self._send(400, '{"error":"empty inputs"}'); return
        rq = queue.Queue()
        work.put((v, rq))
        status, payload = rq.get()
        if status == "error":
            self._send(500, json.dumps({"error": payload})); return
        self._send(200, json.dumps(payload))

srv = ThreadingHTTPServer((BIND, PORT), H)
print(f"bge-torch-tei: listening on {BIND}:{PORT}", flush=True)
srv.serve_forever()
