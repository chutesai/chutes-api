from abc import ABC, abstractmethod
import pybase64 as base64
import binascii
from dataclasses import dataclass
from datetime import datetime, timezone
import json
import struct
from typing import Any, Dict, List, Optional
from dcap_qvl import VerifiedReport
from loguru import logger

from api.config import TeeMeasurementConfig
from api.server.exceptions import InvalidQuoteError

# RTMR dict keys; use these when building or iterating over rtmrs.
RTMR0 = "rtmr0"
RTMR1 = "rtmr1"
RTMR2 = "rtmr2"
RTMR3 = "rtmr3"
RTMR_KEYS = (RTMR0, RTMR1, RTMR2, RTMR3)


@dataclass
class TdxQuote(ABC):
    """
    Parsed TDX quote with extracted measurements.
    """

    version: int
    att_key_type: int
    tee_type: int
    mrtd: str
    rtmr0: str
    rtmr1: str
    rtmr2: str
    rtmr3: str
    report_data: Optional[str]
    user_data: str
    platform_id: str
    raw_quote_size: int
    parsed_at: str
    raw_bytes: bytes

    @property
    def rtmrs(self) -> Dict[str, str]:
        """Get RTMRs as a dictionary."""
        return {
            RTMR0: self.rtmr0,
            RTMR1: self.rtmr1,
            RTMR2: self.rtmr2,
            RTMR3: self.rtmr3,
        }

    @property
    @abstractmethod
    def quote_type(self): ...

    def matches_measurement(self, config: TeeMeasurementConfig) -> bool:
        """
        Return True if this quote's measurements match the given measurement config.

        Compares MRTD (case-insensitive) and the appropriate RTMR set for
        this quote's type ("boot" -> config.boot_rtmrs, "runtime" -> config.runtime_rtmrs).
        """
        if self.mrtd.upper() != config.mrtd.upper():
            return False
        expected = config.boot_rtmrs if self.quote_type == "boot" else config.runtime_rtmrs
        if not expected:
            return False
        for rtmr_name, expected_value in expected.items():
            actual = self.rtmrs.get(rtmr_name.lower()) or self.rtmrs.get(rtmr_name)
            if not actual or actual.upper() != expected_value.upper():
                return False
        return True

    @classmethod
    def from_base64(cls, quote_base64: str) -> "TdxQuote":
        try:
            quote_bytes = base64.b64decode(quote_base64)
            return cls.from_bytes(quote_bytes)
        except binascii.Error:
            raise InvalidQuoteError("Invalid base64 quote.")

    @classmethod
    def from_bytes(cls, quote_bytes: bytes) -> "TdxQuote":
        """
        Parse TDX quote using manual byte parsing based on TDX quote structure.

        Args:
            quote_bytes: Raw quote bytes

        Returns:
            TdxQuote object with parsed data

        Raises:
            InvalidQuoteError: If parsing fails
        """
        try:
            # Validate minimum size (header + TD report = 48 + 584)
            if len(quote_bytes) < 632:
                raise InvalidQuoteError(f"Quote too short: {len(quote_bytes)} bytes")

            # Parse header (48 bytes, little-endian)
            header_format = "<HHI16s20s"  # uint16 version, uint16 att_key_type, uint32 tee_type, 16s QE Vendor ID, 20s user_data
            header = struct.unpack_from(header_format, quote_bytes, 0)
            version, att_key_type, tee_type, qe_vendor_id, header_user_data = header

            # Validate header
            if version not in (4, 5):
                raise InvalidQuoteError(f"Invalid quote version: {version} (expected 4 or 5)")
            if tee_type != 0x81:
                raise InvalidQuoteError(f"Invalid TEE type: {tee_type:08x} (expected 0x81 for TDX)")
            if att_key_type not in (2, 3):  # ECDSA-256 or ECDSA-384
                raise InvalidQuoteError(f"Invalid attestation key type: {att_key_type}")

            # Extract platform identifier (first 16 bytes of user_data)
            platform_id = header_user_data[:16].hex().upper()
            user_data = header_user_data.hex().upper()

            # TD report starts at offset 48
            td_report = quote_bytes[48:632]  # Explicitly limit to 584 bytes for TD report

            # Extract fields using offsets from Intel TDX specification
            mrtd = td_report[136:184].hex().upper()
            rtmr0 = td_report[328:376].hex().upper()
            rtmr1 = td_report[376:424].hex().upper()
            rtmr2 = td_report[424:472].hex().upper()
            rtmr3 = td_report[472:520].hex().upper()
            report_data = td_report[520:584].hex().upper()

            # Create TdxQuote object
            quote = cls(
                version=version,
                att_key_type=att_key_type,
                tee_type=tee_type,
                mrtd=mrtd,
                rtmr0=rtmr0,
                rtmr1=rtmr1,
                rtmr2=rtmr2,
                rtmr3=rtmr3,
                report_data=report_data,  # TD report's report_data (nonce)
                user_data=user_data,  # Header's user_data
                platform_id=platform_id,  # First 16 bytes of user_data
                raw_quote_size=len(quote_bytes),
                raw_bytes=quote_bytes,
                parsed_at=datetime.now(timezone.utc).isoformat(),
            )

            logger.success(
                f"Successfully parsed TDX quote: MRTD={quote.mrtd[:16]}..., Platform ID={quote.platform_id[:16]}..."
            )
            return quote

        except struct.error as e:
            # Parse failure of attacker-supplied bytes: log the raw struct error server-side
            # and return only the generic default message to the client.
            logger.warning(f"Failed to parse TDX quote: {e}")
            raise InvalidQuoteError()

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary format for compatibility."""
        return {
            "quote_version": str(self.version),
            "mrtd": self.mrtd,
            "rtmrs": self.rtmrs,
            "report_data": self.report_data,
            "user_data": self.user_data,
            "platform_id": self.platform_id,
            "raw_quote_size": self.raw_quote_size,
            "parsed_at": self.parsed_at,
            "header": {
                "version": self.version,
                "att_key_type": self.att_key_type,
                "tee_type": f"0x{self.tee_type:02x}",
            },
        }


class BootTdxQuote(TdxQuote):
    @property
    def quote_type(self):
        return "boot"


class RuntimeTdxQuote(TdxQuote):
    @property
    def quote_type(self):
        return "runtime"


# Intel TDX: debug attribute is bit 0 of td_attributes (Linux kernel TDX_ATTR_DEBUG_BIT).
# When set, the TD is in debug mode; we reject such quotes.
TDX_ATTR_DEBUG_BIT = 0


@dataclass
class TdxVerificationResult:
    """
    Parsed verification report: raw fields from the report with computed properties.

    Store status, advisory_ids, td_attributes, and measurements; is_valid and
    debug_enabled are derived from this raw data.
    """

    mrtd: str
    rtmr0: str
    rtmr1: str
    rtmr2: str
    rtmr3: str
    user_data: Optional[str]
    parsed_at: datetime
    status: str
    advisory_ids: List[str]
    td_attributes: str

    @property
    def rtmrs(self) -> Dict[str, str]:
        """Get RTMRs as a dictionary."""
        return {
            RTMR0: self.rtmr0,
            RTMR1: self.rtmr1,
            RTMR2: self.rtmr2,
            RTMR3: self.rtmr3,
        }

    @property
    def debug_enabled(self) -> bool:
        """True if the TD has debug mode enabled (bit 0 set in td_attributes).

        td_attributes is a hex-encoded byte array in little-endian (memory)
        order as emitted by dcap_qvl's JSON serialization, so we must
        reconstruct the integer with LE byte order before testing bits.
        """
        if not self.td_attributes:
            return True  # treat missing as unsafe
        try:
            value = int.from_bytes(bytes.fromhex(self.td_attributes), byteorder="little")
            return bool(value & (1 << TDX_ATTR_DEBUG_BIT))
        except (ValueError, TypeError):
            return True  # treat unparseable as unsafe

    @property
    def is_valid(self) -> bool:
        """
        True only if TCB status is UpToDate, no Intel advisories (INTEL-SA-*)
        apply, and TD debug mode is disabled. We reject on any advisory so a host
        flagged by a security advisory can never verify.
        """
        return self.status == "UpToDate" and not self.advisory_ids and not self.debug_enabled

    @classmethod
    def from_report(cls, verified_report: VerifiedReport) -> "TdxVerificationResult":
        _json = json.loads(verified_report.to_json())
        _report = _json.get("report", {}).get("TD10", {})
        status = _json.get("status", "Unknown")
        advisory_ids = _json.get("advisory_ids") or []
        td_attributes = _report.get("td_attributes", "")

        result = cls(
            mrtd=_report.get("mr_td", ""),
            rtmr0=_report.get("rt_mr0", ""),
            rtmr1=_report.get("rt_mr1", ""),
            rtmr2=_report.get("rt_mr2", ""),
            rtmr3=_report.get("rt_mr3", ""),
            user_data=_report.get("report_data"),
            parsed_at=datetime.now(timezone.utc),
            status=status,
            advisory_ids=advisory_ids,
            td_attributes=td_attributes,
        )

        return result

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary format for compatibility."""
        return {
            "mrtd": self.mrtd,
            "rtmrs": self.rtmrs,
            "user_data": self.user_data,
            "parsed_at": self.parsed_at,
            "is_valid": self.is_valid,
        }

    @classmethod
    def from_fields(
        cls,
        *,
        mr_td: Any,
        rt_mr0: Any,
        rt_mr1: Any,
        rt_mr2: Any,
        rt_mr3: Any,
        report_data: Any,
        td_attributes: Any,
        status: str,
        advisory_ids: Optional[List[str]] = None,
    ) -> "TdxVerificationResult":
        """
        Build a result from parsed TD report fields plus an externally resolved
        TCB ``status`` (used by the module-identity fallback in ``api.server.util``).
        Field values may be ``bytes`` or hex ``str``; normalized to lowercase hex.
        """
        return cls(
            mrtd=_hex(mr_td),
            rtmr0=_hex(rt_mr0),
            rtmr1=_hex(rt_mr1),
            rtmr2=_hex(rt_mr2),
            rtmr3=_hex(rt_mr3),
            user_data=_hex(report_data),
            parsed_at=datetime.now(timezone.utc),
            status=status,
            advisory_ids=advisory_ids or [],
            td_attributes=_hex(td_attributes),
        )


def _hex(value: Any) -> str:
    """Normalize bytes/bytearray/hex-str to a lowercase hex string."""
    if isinstance(value, (bytes, bytearray)):
        return value.hex()
    if isinstance(value, str):
        return value.lower()
    raise InvalidQuoteError(f"Cannot interpret {type(value).__name__} as hex bytes")


def _as_bytes(value: Any) -> bytes:
    """Normalize bytes/bytearray/hex-str to bytes."""
    if isinstance(value, (bytes, bytearray)):
        return bytes(value)
    if isinstance(value, str):
        try:
            return bytes.fromhex(value)
        except ValueError:
            raise InvalidQuoteError("Invalid hex value in TDX module field")
    raise InvalidQuoteError(f"Cannot interpret {type(value).__name__} as bytes")


# Intel TDX TCB status severity, least-current wins when merging platform and
# TDX-module statuses. Mirrors dcap-qvl's TcbStatus::severity ordering so this
# fallback yields the same verdict the (fixed) library would.
_TCB_STATUS_SEVERITY = {
    "UpToDate": 0,
    "SWHardeningNeeded": 1,
    "ConfigurationNeeded": 2,
    "ConfigurationAndSWHardeningNeeded": 3,
    "OutOfDate": 4,
    "OutOfDateConfigurationNeeded": 5,
    "Revoked": 6,
}


def _merge_tcb_status(
    a: "tuple[str, List[str]]", b: "Optional[tuple[str, List[str]]]"
) -> "tuple[str, List[str]]":
    """Take the worse (least-current) of two statuses and union advisory IDs."""
    if b is None:
        return a
    a_status, a_adv = a
    b_status, b_adv = b
    # Unknown statuses sort as most-severe so we never silently upgrade.
    unknown = max(_TCB_STATUS_SEVERITY.values()) + 1
    a_sev = _TCB_STATUS_SEVERITY.get(a_status, unknown)
    b_sev = _TCB_STATUS_SEVERITY.get(b_status, unknown)
    status = b_status if b_sev > a_sev else a_status
    advisory_ids = list(a_adv)
    for adv in b_adv:
        if adv not in advisory_ids:
            advisory_ids.append(adv)
    return status, advisory_ids


def _match_platform_tcb_level(
    tcb_info: Dict[str, Any],
    tee_tcb_svn: List[int],
    sgx_tcb_components: List[int],
    pce_svn: int,
) -> "tuple[str, List[str]]":
    """
    Select the platform TCB level per Intel's DCAP algorithm: the platform must
    meet PCESVN and every SGX and TDX component. Per Intel's TDX TCB mapping,
    ``tee_tcb_svn`` bytes 0-1 are the module SVN/version (governed by
    ``tdxModuleIdentities``), so they are skipped here when ``tee_tcb_svn[1] > 0``.
    Fail-closed if no level matches.
    """
    # Intel: compare tdx components 0..15 when tee_tcb_svn[1] == 0, else 2..15.
    tdx_start = 2 if tee_tcb_svn[1] > 0 else 0
    for level in tcb_info.get("tcbLevels", []):
        tcb = level["tcb"]
        if pce_svn < tcb["pcesvn"]:
            continue
        sgx_components = [c["svn"] for c in tcb.get("sgxtcbcomponents", [])]
        if len(sgx_components) != len(sgx_tcb_components):
            raise InvalidQuoteError("SGX TCB component count mismatch in TCB Info")
        if any(a < b for a, b in zip(sgx_tcb_components, sgx_components)):
            continue
        tdx_components = [c["svn"] for c in tcb.get("tdxtcbcomponents", [])]
        if len(tdx_components) != len(tee_tcb_svn):
            raise InvalidQuoteError("TDX TCB component count mismatch in TCB Info")
        if any(a < b for a, b in zip(tee_tcb_svn[tdx_start:], tdx_components[tdx_start:])):
            continue
        return level["tcbStatus"], level.get("advisoryIDs", []) or []
    raise InvalidQuoteError("No matching platform TCB level found")


def _verify_tdx_module(
    tcb_info: Dict[str, Any],
    tee_tcb_svn: List[int],
    mr_signer_seam: Any,
    seam_attributes: Any,
) -> "Optional[tuple[str, List[str]]]":
    """
    Verify the TDX (SEAM) module identity and derive its TCB status.

    Selects identity ``TDX_{version:02X}`` (version = ``tee_tcb_svn[1]``), checks
    MR_SIGNER_SEAM and masked SEAMATTRIBUTES, then reads the status of the highest
    module level whose ``isvsvn`` <= the module ISVSVN (``tee_tcb_svn[0]``).
    Returns ``None`` when there is no module status to merge; fail-closed on
    mismatch.
    """
    if tcb_info.get("id") != "TDX" or tcb_info.get("version", 0) < 3:
        return None

    base = tcb_info.get("tdxModule")
    if base is None:
        raise InvalidQuoteError("TDX TCB Info is missing tdxModule field")

    module_version = tee_tcb_svn[1]
    module_isvsvn = tee_tcb_svn[0]

    expected_mrsigner = base["mrsigner"]
    expected_attributes = base["attributes"]
    attributes_mask = base["attributesMask"]
    identity_levels: Optional[List[Dict[str, Any]]] = None

    identities = tcb_info.get("tdxModuleIdentities") or []
    if module_version > 0 and identities:
        wanted_id = f"TDX_{module_version:02X}"
        identity = next(
            (i for i in identities if i.get("id", "").upper() == wanted_id.upper()),
            None,
        )
        if identity is None:
            raise InvalidQuoteError(
                f"Unsupported TDX module version: no identity '{wanted_id}' in TCB Info"
            )
        expected_mrsigner = identity["mrsigner"]
        expected_attributes = identity["attributes"]
        attributes_mask = identity["attributesMask"]
        identity_levels = identity.get("tcbLevels", [])

    # MR_SIGNER_SEAM must equal the expected module MRSIGNER.
    if _as_bytes(expected_mrsigner) != _as_bytes(mr_signer_seam):
        raise InvalidQuoteError("TDX module MRSIGNER mismatch")

    # Masked SEAMATTRIBUTES must match; bits outside the mask must be zero.
    exp = _as_bytes(expected_attributes)
    mask = _as_bytes(attributes_mask)
    act = _as_bytes(seam_attributes)
    if not (len(exp) == len(mask) == len(act)):
        raise InvalidQuoteError("TDX module SEAMATTRIBUTES length mismatch")
    for e, m, a in zip(exp, mask, act):
        if (e & m) != (a & m) or (a & (~m & 0xFF)) != 0:
            raise InvalidQuoteError("TDX module SEAMATTRIBUTES mismatch")

    if identity_levels is not None:
        for level in identity_levels:
            if module_isvsvn >= level["tcb"]["isvsvn"]:
                return level["tcbStatus"], level.get("advisoryIDs", []) or []
        raise InvalidQuoteError(
            f"TDX module ISVSVN {module_isvsvn} below minimum in TDX module TCB levels"
        )
    return None


def resolve_tdx_tcb_status(
    tcb_info: Dict[str, Any],
    tee_tcb_svn: List[int],
    sgx_tcb_components: List[int],
    pce_svn: int,
    mr_signer_seam: Any,
    seam_attributes: Any,
) -> "tuple[str, List[str]]":
    """
    Resolve a TDX quote's overall TCB status from the Intel-signed ``tcb_info``
    using Intel's TDX Module Identity algorithm.

    Needed because dcap-qvl compares all 16 TDX TCB components at the platform
    level instead of skipping the two module-governed bytes (Phala dcap-qvl
    issue #124), so current newer-generation TDX hosts fail to match a level.
    Invoked only after dcap-qvl has verified all signatures.

    Returns ``(status, advisory_ids)`` = the least-current of the matched
    platform level and the TDX module identity. Fail-closed (raises
    ``InvalidQuoteError``) if nothing matches or the module can't be verified.
    """
    if len(tee_tcb_svn) < 2:
        raise InvalidQuoteError("TEE_TCB_SVN too short to resolve TDX TCB status")
    platform = _match_platform_tcb_level(tcb_info, tee_tcb_svn, sgx_tcb_components, pce_svn)
    module = _verify_tdx_module(tcb_info, tee_tcb_svn, mr_signer_seam, seam_attributes)
    return _merge_tcb_status(platform, module)
