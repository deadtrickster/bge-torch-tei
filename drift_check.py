#!/usr/bin/env python3
"""Verify every embedding backend still produces the SAME vector as a fixed probe.

WHY: several things can silently change what a backend produces without
erroring:

  * attention implementation switched (flash / SDPA / eager)
  * the runtime silently selecting a different compute library when VRAM
    is occupied at load time
  * different silicon — a CUDA card and a ROCm card never produce
    bit-identical results and no configuration makes them
  * a model re-pull that is not byte-identical

None of these fail loudly. They produce plausible vectors from a slightly
different function, and once mixed into an index there is no way to tell
which chunk came from which backend. The only cheap defence is to keep
asserting that every backend agrees with a fixed probe, and to alert the
moment one does not.

The probe deliberately includes a LONG input: divergence is monotonic in
length and essentially invisible on short text (~1e-7), so a short-only
check would pass while a real difference sat at ~1e-3 on the chunks that
carry the most content.

IMPORTANT: drift magnitudes do not identify causes. Any small numeric
difference grows with input length, so flash-attention, silicon change,
and model re-pull all produce the same length-monotonic fingerprint.
Diagnose the cause with controls on the suspect host (sweep the flag,
read `library=` from the live runner, compare against a THIRD backend),
never from the shape of the numbers.

Exit 0 = all agree. Exit 1 = a backend ANSWERED and DISAGREED (real drift).
Exit 2 = a backend did not answer — unverified, not drifted. The distinction
matters: exit 1 says retire the backend, exit 2 says try again later.
"""
import json
import math
import os
import sys
import urllib.request

# Comma-separated host:port list. Defaults to localhost.
# Supports both ollama (/api/embed) and TEI (/embed) backends.
BACKENDS = [b for b in os.environ.get(
    "EMBED_BACKENDS", "127.0.0.1:11434").split(",") if b.strip()]
# 1e-9 is far above float-epsilon agreement (2.2e-16 measured between
# healthy same-config peers) and far below the smallest real mechanism
# (~1.4e-7 for attention-implementation differences on short text).
TOL = float(os.environ.get("EMBED_DRIFT_TOL", "1e-9"))

PROBES = {
    "short": "quantum entanglement",
    "typical": "The decoherence of superconducting qubits arises from coupling to environmental "
               "degrees of freedom, including two-level systems in the oxide layer. " * 6,
    "long": "superconducting qubit decoherence and error mitigation in transmon devices. " * 160,
}


def emb(host, text):
    req = urllib.request.Request(
        f"http://{host}/api/embed",
        data=json.dumps({"model": "bge-m3", "input": [text], "truncate": True}).encode(),
        headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=300) as r:
        return json.load(r)["embeddings"][0]
