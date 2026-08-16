"""JSON-RPC transport between the UI process and the engine process."""

from svc_engine.rpc.protocol import (
    PROTOCOL_VERSION,
    Request,
    Response,
    decode_request,
    decode_response,
    encode,
)
from svc_engine.rpc.server import Server, serve_stdio

__all__ = [
    "PROTOCOL_VERSION",
    "Request",
    "Response",
    "Server",
    "decode_request",
    "decode_response",
    "encode",
    "serve_stdio",
]
