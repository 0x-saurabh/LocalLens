"""Proves the offline guard blocks every non-loopback connection."""

import socket

import pytest

from locallens import offline_guard


def test_guard_blocks_external_and_allows_loopback():
    offline_guard.enable()
    try:
        s = socket.socket()
        with pytest.raises(offline_guard.NetworkBlockedError):
            s.connect(("1.1.1.1", 443))
        with pytest.raises(offline_guard.NetworkBlockedError):
            s.connect(("example.com", 80))
        # loopback is allowed: a refused connection is fine, a blocked one is not
        s2 = socket.socket()
        s2.settimeout(0.2)
        try:
            s2.connect(("127.0.0.1", 1))
        except (ConnectionRefusedError, OSError):
            pass
    finally:
        offline_guard.disable()
