#!/usr/bin/env python
"""
_nv_verify_verbose.py

Verbose GPU-evidence verification, run INSIDE the nv-attest venv
(/app/nv-attest/bin/python). Mirrors chutes_nvattest.verifier.NvVerifier but with
DEBUG logging, a full traceback on exception, and — crucially — a decode of the
per-GPU detailed NRAS claims so you can see EXACTLY which GPU and which sub-check
failed (e.g. measres, driver/vbios RIM fetch, measurement match).

Background: NRAS can return HTTP 200 with a valid signature yet
x-nvidia-overall-att-result=false. The overall token only lists per-GPU DIGEST
pointers; the real pass/fail booleans live in the detached per-GPU tokens, which
this script decodes and prints, flagging any field that is not True/"success".

Invoked by scripts/debug_server_attestation.py; can also be run standalone:
    /app/nv-attest/bin/python scripts/_nv_verify_verbose.py \
        --nonce <64-hex> --evidence /path/to/gpu_evidence.json
"""

import argparse
import json
import logging
import sys
import traceback

# Turn on everything the SDK / local verifier emit so the real reason shows up.
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
    stream=sys.stdout,
)
for name in ("nv_attestation_sdk", "verifier", "sdk", "gpu_verifier"):
    logging.getLogger(name).setLevel(logging.DEBUG)

from nv_attestation_sdk.attestation import Attestation, Devices, Environment  # noqa: E402
from nv_attestation_sdk.utils import claim_utils  # noqa: E402


def _is_bad(key: str, value) -> bool:
    """A claim is 'bad' if it's an explicit False, or a *res field that isn't success."""
    if value is False:
        return True
    if key == "measres" and value not in ("success", None):
        return True
    return False


def dump_per_gpu_claims(token: str) -> None:
    """
    Decode and print the detached per-GPU claims. Token structure:
        [claims_version, {"REMOTE_GPU_CLAIMS": [[type, overall_jwt], {GPU-N: jwt}]}]
    The per-GPU jwt holds the real x-nvidia-* / measres pass-fail booleans.
    """
    if not token:
        print("\n(no token available to decode — attestation may have raised before NRAS)")
        return
    try:
        arr = json.loads(token)
        remote = arr[1]["REMOTE_GPU_CLAIMS"]
        overall = claim_utils.decode_jwt(remote[0][1])
        gpu_map = remote[1]
    except Exception:
        print("\ncould not parse token structure:")
        traceback.print_exc()
        return

    print("\n==== overall ====")
    print(f"  x-nvidia-overall-att-result: {overall.get('x-nvidia-overall-att-result')}")

    # gpu -> (driver_version, vbios_version, measres, [failing checks])
    summary = {}

    print("\n==== per-GPU detailed claims (failing checks flagged with >>>) ====")
    for gpu, jwt_str in gpu_map.items():
        try:
            claims = claim_utils.decode_jwt(jwt_str)
        except Exception:
            print(f"\n  {gpu}: could not decode detailed token")
            traceback.print_exc()
            continue
        failures = [
            k for k in claims if (k.startswith("x-nvidia") or k == "measres") and _is_bad(k, claims[k])
        ]
        driver = claims.get("x-nvidia-gpu-driver-version", "?")
        vbios = claims.get("x-nvidia-gpu-vbios-version", "?")
        measres = claims.get("measres", "?")
        summary[gpu] = (driver, vbios, measres, failures)

        marker = "FAIL" if failures else "ok"
        # Firmware versions up front — divergence between cards is a common root cause.
        print(f"\n  {gpu}  [{marker}]  driver={driver}  vbios={vbios}")
        for k in sorted(claims):
            if not (k.startswith("x-nvidia") or k == "measres"):
                continue
            flag = ">>> " if _is_bad(k, claims[k]) else "    "
            print(f"    {flag}{k}: {claims[k]}")
        if failures:
            print(f"    ^ failing checks on {gpu}: {failures}")

    _print_firmware_summary(summary)


def _print_firmware_summary(summary: dict) -> None:
    """
    Compact table of driver/VBIOS/measres per GPU, flagging any card whose firmware
    diverges from the majority — a mismatched card is a common cause of measres:fail.
    """
    if not summary:
        return
    from collections import Counter

    drivers = Counter(v[0] for v in summary.values())
    vbioses = Counter(v[1] for v in summary.values())
    common_driver = drivers.most_common(1)[0][0]
    common_vbios = vbioses.most_common(1)[0][0]

    print("\n==== firmware summary (per GPU) ====")
    print(f"  {'GPU':<7} {'driver':<14} {'vbios':<20} {'measres':<8} note")
    for gpu, (driver, vbios, measres, failures) in summary.items():
        notes = []
        if driver != common_driver:
            notes.append(f"driver != {common_driver}")
        if vbios != common_vbios:
            notes.append(f"vbios != {common_vbios}")
        note = "  <<< DIVERGENT: " + ", ".join(notes) if notes else ""
        print(f"  {gpu:<7} {driver:<14} {vbios:<20} {measres:<8}{note}")

    if len(drivers) > 1 or len(vbioses) > 1:
        print(
            "\n  NOTE: GPUs are NOT uniform. A card with different firmware/driver than the"
            "\n  rest will mismatch the reference and fail measres. Reflash/align, then re-run."
        )
    else:
        print(
            f"\n  All GPUs uniform (driver={common_driver}, vbios={common_vbios}). "
            "measres:fail here\n  is a measurement mismatch against the reference, not a version skew."
        )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--nonce", required=True)
    ap.add_argument("--evidence", required=True)
    args = ap.parse_args()

    with open(args.evidence) as fh:
        evidence = json.load(fh)
    print(f"loaded {len(evidence)} GPU evidence entries; nonce={args.nonce}\n")

    client = Attestation()
    client.set_name("chutes-debug")  # get_token() is keyed by client name; set it so we can read claims
    client.add_verifier(Devices.GPU, Environment.REMOTE, "", "")
    client.set_nonce(args.nonce)

    result = None
    try:
        result = client.attest(evidence)
    except Exception:
        print("\n*** attest() RAISED — this is the API's 'unexpected exception' path ***")
        traceback.print_exc()

    print(f"\n==== attest() result: {result} ====")

    try:
        dump_per_gpu_claims(client.get_token())
    except Exception:
        print("\ncould not dump per-GPU claims:")
        traceback.print_exc()

    return 0 if result else 1


if __name__ == "__main__":
    raise SystemExit(main())
