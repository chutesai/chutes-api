"""
Unit tests for the Intel TDX Module Identity TCB-resolution algorithm in
``api.server.quote.resolve_tdx_tcb_status``.

Context: dcap-qvl (through at least 0.5.x) compares all 16 TDX TCB components at
the platform level instead of skipping the two module-governed bytes, so
newer-generation TDX hosts (e.g. B200, FMSPC 00A06D080000) whose SEAM module SVN
restarts low fail with "No matching TCB level found" even though Intel reports
them UpToDate. ``resolve_tdx_tcb_status`` reimplements Intel's algorithm over the
Intel-signed TCB Info so we resolve the correct verdict.

Fixtures ``tests/assets/tdx_tcb_*.json`` are the real, unmodified responses from
Intel's live API (GET /tdx/certification/v4/tcb?fmspc=<FMSPC>).
"""

import json
from pathlib import Path

import pytest

from api.server.exceptions import InvalidQuoteError
from api.server.quote import resolve_tdx_tcb_status


ZERO_MRSIGNER = "00" * 48
ZERO_SEAM_ATTRS = "00" * 8


def _tcb_info(fmspc: str) -> dict:
    path = Path(f"tests/assets/tdx_tcb_{fmspc}.json")
    return json.loads(path.read_text())["tcbInfo"]


def _pad16(values: list[int]) -> list[int]:
    return (values + [0] * 16)[:16]


# --- B200 (FMSPC 00A06D080000): the real failing case -----------------------
# These are the exact values parsed from the real failing quote
# (tests/assets/b200_boot.quote) plus Intel's live TCB Info:
#   TEE_TCB_SVN        = [3, 3, 4, 0, ...]   (module version=3, isvsvn=3)
#   PCK SGX components = [4, 4, 2, 2, 4, 1, 0, 2, ...], PCESVN = 13
#   Platform Level 1   : tdxtcbcomponents = [5, 0, 2, ...] UpToDate @ PCESVN 13
#   tdxModuleIdentities: TDX_03 @ isvsvn 3 = UpToDate
B200_FMSPC = "00A06D080000"
B200_TEE_TCB_SVN = _pad16([3, 3, 4])
B200_SGX_COMPONENTS = _pad16([4, 4, 2, 2, 4, 1, 0, 2])
B200_PCE_SVN = 13


def test_b200_resolves_uptodate():
    """The real B200 host resolves to UpToDate (platform Level 1 + TDX_03)."""
    status, advisory_ids = resolve_tdx_tcb_status(
        tcb_info=_tcb_info(B200_FMSPC),
        tee_tcb_svn=B200_TEE_TCB_SVN,
        sgx_tcb_components=B200_SGX_COMPONENTS,
        pce_svn=B200_PCE_SVN,
        mr_signer_seam=ZERO_MRSIGNER,
        seam_attributes=ZERO_SEAM_ATTRS,
    )
    assert status == "UpToDate"
    assert advisory_ids == []


def test_index_skip_is_what_unblocks_b200():
    """
    The fix hinges on skipping tdxtcbcomponents[0..1] when tee_tcb_svn[1] > 0.
    Index 0 is 3 (quote) vs 5 (platform); comparing it (as dcap-qvl does) would
    reject the host. Forcing module version 0 disables the skip and must fail
    closed, proving the skip is load-bearing and legacy hosts stay strict.
    """
    legacy_like = _pad16([3, 0, 4])  # tee_tcb_svn[1] == 0 -> no skip, no module
    with pytest.raises(InvalidQuoteError, match="No matching platform TCB level"):
        resolve_tdx_tcb_status(
            tcb_info=_tcb_info(B200_FMSPC),
            tee_tcb_svn=legacy_like,
            sgx_tcb_components=B200_SGX_COMPONENTS,
            pce_svn=B200_PCE_SVN,
            mr_signer_seam=ZERO_MRSIGNER,
            seam_attributes=ZERO_SEAM_ATTRS,
        )


def test_module_out_of_date_downgrades_final_status():
    """
    A TDX_01 module at isvsvn 4 is OutOfDate per Intel even though the platform
    level is UpToDate; final status is the worse of the two, with advisories.
    """
    status, advisory_ids = resolve_tdx_tcb_status(
        tcb_info=_tcb_info(B200_FMSPC),
        tee_tcb_svn=_pad16([4, 1, 4]),  # TDX_01, isvsvn 4 -> OutOfDate
        sgx_tcb_components=B200_SGX_COMPONENTS,
        pce_svn=B200_PCE_SVN,
        mr_signer_seam=ZERO_MRSIGNER,
        seam_attributes=ZERO_SEAM_ATTRS,
    )
    assert status == "OutOfDate"
    assert "INTEL-SA-01036" in advisory_ids


def test_unsupported_module_version_fails_closed():
    with pytest.raises(InvalidQuoteError, match="Unsupported TDX module version"):
        resolve_tdx_tcb_status(
            tcb_info=_tcb_info(B200_FMSPC),
            tee_tcb_svn=_pad16([3, 99, 4]),  # no TDX_63 identity
            sgx_tcb_components=B200_SGX_COMPONENTS,
            pce_svn=B200_PCE_SVN,
            mr_signer_seam=ZERO_MRSIGNER,
            seam_attributes=ZERO_SEAM_ATTRS,
        )


def test_mrsigner_mismatch_fails_closed():
    with pytest.raises(InvalidQuoteError, match="MRSIGNER"):
        resolve_tdx_tcb_status(
            tcb_info=_tcb_info(B200_FMSPC),
            tee_tcb_svn=B200_TEE_TCB_SVN,
            sgx_tcb_components=B200_SGX_COMPONENTS,
            pce_svn=B200_PCE_SVN,
            mr_signer_seam="11" * 48,  # not the expected module MRSIGNER
            seam_attributes=ZERO_SEAM_ATTRS,
        )


def test_seam_attributes_mismatch_fails_closed():
    with pytest.raises(InvalidQuoteError, match="SEAMATTRIBUTES"):
        resolve_tdx_tcb_status(
            tcb_info=_tcb_info(B200_FMSPC),
            tee_tcb_svn=B200_TEE_TCB_SVN,
            sgx_tcb_components=B200_SGX_COMPONENTS,
            pce_svn=B200_PCE_SVN,
            mr_signer_seam=ZERO_MRSIGNER,
            seam_attributes="0100000000000000",  # bit set, masked expected is 0
        )


def test_sgx_components_below_level_fails_closed():
    with pytest.raises(InvalidQuoteError, match="No matching platform TCB level"):
        resolve_tdx_tcb_status(
            tcb_info=_tcb_info(B200_FMSPC),
            tee_tcb_svn=B200_TEE_TCB_SVN,
            sgx_tcb_components=_pad16([0]),  # below Level 1 sgxtcbcomponents
            pce_svn=B200_PCE_SVN,
            mr_signer_seam=ZERO_MRSIGNER,
            seam_attributes=ZERO_SEAM_ATTRS,
        )


def test_pcesvn_below_level_fails_closed():
    with pytest.raises(InvalidQuoteError, match="No matching platform TCB level"):
        resolve_tdx_tcb_status(
            tcb_info=_tcb_info(B200_FMSPC),
            tee_tcb_svn=B200_TEE_TCB_SVN,
            sgx_tcb_components=B200_SGX_COMPONENTS,
            pce_svn=0,  # below every level's pcesvn
            mr_signer_seam=ZERO_MRSIGNER,
            seam_attributes=ZERO_SEAM_ATTRS,
        )


# --- Legacy TDX_01 host (FMSPC 90C06F000000): no-regression -----------------
# Real values parsed from tests/assets/quote.bin (a genuine TDX_01 host that
# dcap-qvl already verifies UpToDate):
#   TEE_TCB_SVN        = [6, 1, 3, 0, ...]   (module version=1, isvsvn=6)
#   PCK SGX components = [3, 3, 2, 2, 4, 1, 0, 5, ...], PCESVN = 13
LEGACY_FMSPC = "90C06F000000"


def test_legacy_tdx01_host_resolves_uptodate():
    """Legacy TDX_01 host resolves UpToDate, matching dcap-qvl's own verdict."""
    status, advisory_ids = resolve_tdx_tcb_status(
        tcb_info=_tcb_info(LEGACY_FMSPC),
        tee_tcb_svn=_pad16([6, 1, 3]),
        sgx_tcb_components=_pad16([3, 3, 2, 2, 4, 1, 0, 5]),
        pce_svn=13,
        mr_signer_seam=ZERO_MRSIGNER,
        seam_attributes=ZERO_SEAM_ATTRS,
    )
    assert status == "UpToDate"
    assert advisory_ids == []


def test_bytes_and_hex_inputs_are_equivalent():
    """mr_signer_seam / seam_attributes accept both bytes and hex strings."""
    common = dict(
        tcb_info=_tcb_info(B200_FMSPC),
        tee_tcb_svn=B200_TEE_TCB_SVN,
        sgx_tcb_components=B200_SGX_COMPONENTS,
        pce_svn=B200_PCE_SVN,
    )
    as_hex = resolve_tdx_tcb_status(
        mr_signer_seam=ZERO_MRSIGNER, seam_attributes=ZERO_SEAM_ATTRS, **common
    )
    as_bytes = resolve_tdx_tcb_status(mr_signer_seam=bytes(48), seam_attributes=bytes(8), **common)
    assert as_hex == as_bytes == ("UpToDate", [])


# --- End-to-end regression against the real signed B200 quote ---------------
# tests/assets/b200_boot.quote is the real quote that dcap-qvl 0.3.12 rejected
# with "No matching TCB level found". This exercises the full path: dcap-qvl
# verifies signatures/certs, raises on the TCB match, and our fallback resolves
# UpToDate. Requires fetching live collateral, so it skips when offline.
@pytest.mark.asyncio
async def test_real_b200_quote_verifies_end_to_end():
    quote_path = Path("tests/assets/b200_boot.quote")
    if not quote_path.exists():
        pytest.skip("b200_boot.quote fixture not present")

    import base64

    from dcap_qvl import get_collateral, PHALA_PCCS_URL

    from api.server.quote import BootTdxQuote
    from api.server.util import verify_quote_signature

    raw = bytes(base64.b64decode(quote_path.read_text().strip()))

    # Probe connectivity so the test skips (not fails) when collateral can't be
    # fetched; verify_quote_signature masks fetch errors as InvalidQuoteError.
    try:
        await get_collateral(PHALA_PCCS_URL, raw)
    except Exception as e:  # pragma: no cover - network dependent
        pytest.skip(f"collateral fetch unavailable: {e}")

    quote = BootTdxQuote.from_bytes(raw)
    result = await verify_quote_signature(quote)

    assert result.status == "UpToDate"
    assert result.is_valid is True
    assert result.debug_enabled is False
