"""Messages capteurs signés : compteur + 5 mesures (float32) + HMAC-SHA256 tronqué à 8 octets.

Le nœud capteurs signe chaque message avec une clé partagée. L'Edge vérifie (1) l'étiquette, en temps constant,
et (2) que le compteur est strictement croissant : un message rejoué porte un ancien compteur.
"""
from __future__ import annotations

import hashlib
import hmac
import struct

FMT = "<I5f"           # compteur (uint32) + 5 mesures float32
TAG_LEN = 8            # octets d'étiquette (64 bits)


def pack(counter: int, values) -> bytes:
    return struct.pack(FMT, counter, *[float(v) for v in values])


def sign(key: bytes, counter: int, values) -> tuple[bytes, bytes]:
    body = pack(counter, values)
    return body, hmac.new(key, body, hashlib.sha256).digest()[:TAG_LEN]


def decode(body: bytes):
    c, *vals = struct.unpack(FMT, body)
    return c, vals


class Verifier:
    """Couche 1 de l'Edge : authenticité + anti-rejeu."""

    def __init__(self, key: bytes):
        self.key = key
        self.last = -1

    def check(self, body: bytes, tag: bytes) -> bool:
        good = hmac.compare_digest(hmac.new(self.key, body, hashlib.sha256).digest()[:TAG_LEN], tag)
        if not good:
            return False
        c, _ = decode(body)
        if c <= self.last:
            return False          # compteur ancien ou répété : rejeu
        self.last = c
        return True
