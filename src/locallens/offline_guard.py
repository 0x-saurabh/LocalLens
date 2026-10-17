"""Hard guarantee that nothing leaves the machine.

Call enable() at startup in every mode except `locallens setup`. Any attempt to
connect a socket to a non-loopback address raises immediately.
"""

from __future__ import annotations

import ipaddress
import os
import socket

_LOOPBACK_NAMES = {"localhost", "127.0.0.1", "::1", "0.0.0.0"}
_original_connect = socket.socket.connect
_original_connect_ex = socket.socket.connect_ex
_enabled = False


class NetworkBlockedError(RuntimeError):
    pass


def _is_loopback(address) -> bool:
    if not isinstance(address, tuple):
        return True  # unix sockets etc.
    host = address[0]
    if host in _LOOPBACK_NAMES:
        return True
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False  # a hostname: resolving it is itself a network action


def _guarded_connect(self, address):
    if not _is_loopback(address):
        raise NetworkBlockedError(f"LocalLens offline guard blocked connection to {address!r}")
    return _original_connect(self, address)


def _guarded_connect_ex(self, address):
    if not _is_loopback(address):
        raise NetworkBlockedError(f"LocalLens offline guard blocked connection to {address!r}")
    return _original_connect_ex(self, address)


def enable() -> None:
    global _enabled
    if _enabled:
        return
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    os.environ.setdefault("OLLAMA_HOST", "127.0.0.1:11434")
    socket.socket.connect = _guarded_connect          # type: ignore[method-assign]
    socket.socket.connect_ex = _guarded_connect_ex    # type: ignore[method-assign]
    _enabled = True


def disable() -> None:
    """Only for `locallens setup` (model download) and tests."""
    global _enabled
    socket.socket.connect = _original_connect         # type: ignore[method-assign]
    socket.socket.connect_ex = _original_connect_ex   # type: ignore[method-assign]
    _enabled = False


def is_enabled() -> bool:
    return _enabled
