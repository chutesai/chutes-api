"""Errors raised by the external HTTP transport layer."""


class ExternalTransportError(Exception):
    """Base class for transport failures."""


class ProfileError(ExternalTransportError, ValueError):
    """An endpoint or artifact profile is invalid."""


class RequestRejectedError(ExternalTransportError, ValueError):
    """A request was rejected before it reached the network."""


class RedirectRejectedError(RequestRejectedError):
    """An upstream redirect violated the configured redirect policy."""


class UpstreamTimeoutError(ExternalTransportError, TimeoutError):
    """The upstream request exceeded a configured timeout."""


class UpstreamConnectionError(ExternalTransportError, ConnectionError):
    """The transport could not establish or maintain an upstream connection."""


class ResponseTooLargeError(ExternalTransportError):
    """An upstream response exceeded its configured byte limit."""


class StreamProtocolError(ExternalTransportError):
    """An upstream streaming response did not satisfy the configured protocol."""


class WebSocketLimitError(ExternalTransportError):
    """A WebSocket message, idle period, or session exceeded its configured limit."""
