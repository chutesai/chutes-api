#!/usr/bin/env python
"""
debug_server_attestation.py

Reproduce a server's attestation flow against the attestation proxy and surface
the REAL GPU-evidence verification error that verify_server() throws away.

Why this exists
---------------
In api/server/util.py::verify_gpu_evidence the CLI is run as a subprocess with no
stdout/stderr capture, and any non-zero exit is re-wrapped as the generic
GpuEvidenceError("Encountered an unexpected exception verifying GPU evidence.").
So the pod logs only ever show that opaque message. This script hits the proxy
the same way the API does (validator-signed GET https://<ip>:30443/server/attest),
pulls the TDX quote + nvtrust GPU evidence, and then verifies the GPU evidence
with FULL output so you can see what nv-attest actually complains about.

Run this INSIDE an API pod (needs VALIDATOR_SEED / VALIDATOR_SS58 creds and
network reachability to the server IP on port 30443). It reuses the API's own
TeeServerClient for request signing.

Usage:
    uv run scripts/debug_server_attestation.py --ip 154.59.156.9
    uv run scripts/debug_server_attestation.py --ip 154.59.156.9 --server-id e21ada91-...
    uv run scripts/debug_server_attestation.py --ip 154.59.156.9 --nonce <64-hex>
    uv run scripts/debug_server_attestation.py --ip 154.59.156.9 --outdir /tmp/attest-debug

The GPU verification runs in the nv-attest venv (default /app/nv-attest/bin/python,
override with --nv-python). It runs TWO ways:
  1. exactly what the API does: `chutes-nvattest --nonce N --evidence FILE` with
     stdout/stderr captured (the API discards these).
  2. a verbose direct SDK run (DEBUG logging, full traceback, per-GPU claims).
"""

import argparse
import asyncio
import json
import os
import secrets
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

# Make `import api...` work when run via `uv run` from repo root.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from api.server.client import TeeServerClient  # noqa: E402
from api.server.quote import RuntimeTdxQuote  # noqa: E402
from api.server.util import extract_report_data  # noqa: E402

VERBOSE_VERIFIER = Path(__file__).resolve().parent / "_nv_verify_verbose.py"


def _hr(title: str) -> None:
    print(f"\n{'=' * 78}\n{title}\n{'=' * 78}", flush=True)


async def fetch_evidence(ip: str, nonce: str):
    """
    Fetch evidence via the API's own signed request path, but with full HTTP
    visibility (status + body on failure) instead of the swallowed GetEvidenceError.
    """
    server = SimpleNamespace(ip=ip, server_id="debug")
    client = TeeServerClient(server)  # type: ignore[arg-type]
    url = f"{client._url}/server/attest"
    headers, _ = client._sign_request(purpose="attest")

    _hr(f"1. GET {url}?nonce={nonce}")
    for k, v in headers.items():
        print(f"  {k}: {v}")

    async with client._attestation_session() as session:
        # Override the session's raise_for_status so we can read the body on a
        # non-200 (that payload is usually the actual proxy error).
        async with session.get(
            url, headers=headers, params={"nonce": nonce}, raise_for_status=False
        ) as resp:
            status = resp.status
            body = await resp.text()

    print(f"\n  HTTP {status}")
    if status != 200:
        print(f"  body:\n{body[:4000]}")
        raise SystemExit(f"proxy returned HTTP {status}; cannot verify GPU evidence")

    data = json.loads(body)
    return data


def summarize_quote(data: dict, nonce: str) -> None:
    _hr("2. TDX quote sanity check")
    try:
        quote = RuntimeTdxQuote.from_base64(data["tdx_quote"])
        q_nonce, cert_hash = extract_report_data(quote)
        print(f"  quote parsed OK, {len(quote.raw_bytes)} bytes")
        print(f"  report_data nonce : {q_nonce}")
        print(f"  expected nonce    : {nonce}")
        print(f"  nonce match       : {q_nonce == nonce}")
        print(f"  cert public-key hash in quote: {cert_hash}")
    except Exception as exc:
        print(f"  could not parse/inspect quote: {type(exc).__name__}: {exc}")


def write_evidence(data: dict, outdir: Path) -> Path:
    gpu_evidence = json.loads(data["nvtrust_evidence"])
    outdir.mkdir(parents=True, exist_ok=True)
    ev_path = outdir / "gpu_evidence.json"
    ev_path.write_text(json.dumps(gpu_evidence))
    print(f"\n  wrote {len(gpu_evidence)} GPU evidence entries -> {ev_path}")
    return ev_path


def reproduce_cli(nonce: str, ev_path: Path) -> None:
    """Exactly what verify_gpu_evidence() runs, but capturing the output it discards."""
    _hr("3. Reproduce API subprocess: `chutes-nvattest --nonce ... --evidence ...`")
    cmd = ["chutes-nvattest", "--nonce", nonce, "--evidence", str(ev_path)]
    print(f"  $ {' '.join(cmd)}")
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
    except FileNotFoundError:
        print("  chutes-nvattest not on PATH (are you in an API pod?)")
        return
    print(f"\n  return code: {proc.returncode}  (API treats != 0 as failure)")
    if proc.stdout.strip():
        print(f"\n  --- stdout ---\n{proc.stdout}")
    if proc.stderr.strip():
        print(f"\n  --- stderr ---\n{proc.stderr}")


def verbose_verify(nonce: str, ev_path: Path, nv_python: str) -> None:
    """Run the verbose direct-SDK verifier in the nv-attest venv for the real reason."""
    _hr("4. Verbose direct SDK verification (DEBUG logging + traceback + claims)")
    if not Path(nv_python).exists():
        print(f"  nv-attest interpreter not found at {nv_python}")
        print(f"  re-run with --nv-python pointing at the nv-attest venv python")
        return
    cmd = [nv_python, str(VERBOSE_VERIFIER), "--nonce", nonce, "--evidence", str(ev_path)]
    print(f"  $ {' '.join(cmd)}\n")
    subprocess.run(cmd)


async def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ip", required=True, help="Server IP (attestation proxy on :30443)")
    ap.add_argument("--server-id", default=None, help="Optional, for your reference only")
    ap.add_argument("--nonce", default=None, help="64-hex nonce; default: random 32 bytes")
    ap.add_argument("--nv-python", default="/app/nv-attest/bin/python", help="nv-attest venv python")
    ap.add_argument("--outdir", default=None, help="Where to save evidence (default: scratch temp)")
    args = ap.parse_args()

    nonce = args.nonce or secrets.token_hex(32)
    outdir = Path(args.outdir) if args.outdir else Path("/tmp/attest-debug") / args.ip

    print(f"server_id : {args.server_id}")
    print(f"ip        : {args.ip}")
    print(f"nonce     : {nonce}")
    print(f"outdir    : {outdir}")

    data = await fetch_evidence(args.ip, nonce)
    summarize_quote(data, nonce)
    ev_path = write_evidence(data, outdir)
    reproduce_cli(nonce, ev_path)
    verbose_verify(nonce, ev_path, args.nv_python)

    _hr("Done")
    print("If step 3/4 show a real failure reason, that's what the API is masking.")


if __name__ == "__main__":
    asyncio.run(main())
