from abc import ABC, abstractmethod
import base64
import binascii
from dataclasses import dataclass
from datetime import datetime, timezone
import json
import struct
from typing import Any, Dict, Optional
from dcap_qvl import VerifiedReport
from loguru import logger

from api.server.exceptions import InvalidQuoteError


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
            "rtmr0": self.rtmr0,
            "rtmr1": self.rtmr1,
            "rtmr2": self.rtmr2,
            "rtmr3": self.rtmr3,
        }

    @property
    @abstractmethod
    def quote_type(self): ...

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
                user_data=user_data,      # Header's user_data
                platform_id=platform_id,  # First 16 bytes of user_data
                raw_quote_size=len(quote_bytes),
                raw_bytes=quote_bytes,
                parsed_at=datetime.now(timezone.utc).isoformat(),
            )

            logger.success(f"Successfully parsed TDX quote: MRTD={quote.mrtd[:16]}..., Platform ID={quote.platform_id[:16]}...")
            return quote

        except struct.error as e:
            raise InvalidQuoteError(f"Failed to parse quote: {str(e)}")

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


@dataclass
class TdxVerificationResult:
    """
    Parsed TDX quote with extracted measurements.
    """

    mrtd: str
    rtmr0: str
    rtmr1: str
    rtmr2: str
    rtmr3: str
    user_data: Optional[str]
    parsed_at: datetime
    is_valid: bool

    @property
    def rtmrs(self) -> Dict[str, str]:
        """Get RTMRs as a dictionary."""
        return {
            "rtmr0": self.rtmr0,
            "rtmr1": self.rtmr1,
            "rtmr2": self.rtmr2,
            "rtmr3": self.rtmr3,
        }

    @classmethod
    def from_report(cls, verified_report: VerifiedReport) -> "TdxQuote":
        _json = json.loads(verified_report.to_json())
        _report = _json.get("report").get("TD10")
        return TdxVerificationResult(
            mrtd=_report.get("mr_td"),
            rtmr0=_report.get("rt_mr0"),
            rtmr1=_report.get("rt_mr1"),
            rtmr2=_report.get("rt_mr2"),
            rtmr3=_report.get("rt_mr3"),
            user_data=_report.get("report_data"),
            parsed_at=datetime.now(timezone.utc),
            is_valid=bool(verified_report.status == "UpToDate"),
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary format for compatibility."""
        return {
            "mrtd": self.mrtd,
            "rtmrs": self.rtmrs,
            "user_data": self.user_data,
            "parsed_at": self.parsed_at,
            "is_valid": self.is_valid,
        }
