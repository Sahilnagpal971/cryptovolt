"""
Security utilities: password hashing + lightweight JWT (HS256)

This project is a research/prototype; keep dependencies minimal by using
stdlib-based PBKDF2 for password storage and a small JWT implementation.
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
import os
import time
from typing import Any, Dict, Optional


PBKDF2_ALGO = "sha256"
PBKDF2_ITERATIONS = 200_000
PBKDF2_SALT_BYTES = 16


def _b64url_encode(raw: bytes) -> str:
    return base64.urlsafe_b64encode(raw).rstrip(b"=").decode("ascii")


def _b64url_decode(data: str) -> bytes:
    padding = "=" * ((4 - (len(data) % 4)) % 4)
    return base64.urlsafe_b64decode((data + padding).encode("ascii"))


def hash_password(password: str) -> str:
    """
    Hash password using PBKDF2-HMAC-SHA256.

    Format: pbkdf2_sha256$<iterations>$<salt_b64>$<hash_b64>
    """
    if not password:
        raise ValueError("Password must not be empty")

    salt = os.urandom(PBKDF2_SALT_BYTES)
    digest = hashlib.pbkdf2_hmac(
        PBKDF2_ALGO,
        password.encode("utf-8"),
        salt,
        PBKDF2_ITERATIONS,
    )
    return "pbkdf2_sha256${}${}${}".format(
        PBKDF2_ITERATIONS,
        _b64url_encode(salt),
        _b64url_encode(digest),
    )


def verify_password(password: str, stored_hash: str) -> bool:
    if not stored_hash or not password:
        return False

    if stored_hash.startswith("pbkdf2_sha256$"):
        try:
            _, iterations_s, salt_b64, digest_b64 = stored_hash.split("$", 3)
            iterations = int(iterations_s)
            salt = _b64url_decode(salt_b64)
            expected = _b64url_decode(digest_b64)
        except Exception:
            return False

        candidate = hashlib.pbkdf2_hmac(
            PBKDF2_ALGO,
            password.encode("utf-8"),
            salt,
            iterations,
        )
        return hmac.compare_digest(candidate, expected)

    # Best-effort support for existing bcrypt hashes (e.g., legacy seed data).
    if stored_hash.startswith("$2a$") or stored_hash.startswith("$2b$") or stored_hash.startswith("$2y$"):
        try:
            import bcrypt  # type: ignore
        except Exception:
            return False
        try:
            return bcrypt.checkpw(password.encode("utf-8"), stored_hash.encode("utf-8"))
        except Exception:
            return False

    return False


def jwt_encode(payload: Dict[str, Any], secret_key: str) -> str:
    header = {"alg": "HS256", "typ": "JWT"}
    header_b64 = _b64url_encode(json.dumps(header, separators=(",", ":")).encode("utf-8"))
    payload_b64 = _b64url_encode(json.dumps(payload, separators=(",", ":")).encode("utf-8"))
    signing_input = f"{header_b64}.{payload_b64}".encode("ascii")
    signature = hmac.new(secret_key.encode("utf-8"), signing_input, hashlib.sha256).digest()
    sig_b64 = _b64url_encode(signature)
    return f"{header_b64}.{payload_b64}.{sig_b64}"


def jwt_decode(token: str, secret_key: str) -> Dict[str, Any]:
    parts = token.split(".")
    if len(parts) != 3:
        raise ValueError("Invalid token format")

    header_b64, payload_b64, sig_b64 = parts
    signing_input = f"{header_b64}.{payload_b64}".encode("ascii")
    expected_sig = hmac.new(secret_key.encode("utf-8"), signing_input, hashlib.sha256).digest()
    actual_sig = _b64url_decode(sig_b64)
    if not hmac.compare_digest(expected_sig, actual_sig):
        raise ValueError("Invalid token signature")

    payload = json.loads(_b64url_decode(payload_b64).decode("utf-8"))

    exp = payload.get("exp")
    if exp is not None and int(exp) < int(time.time()):
        raise ValueError("Token expired")

    return payload


def create_access_token(subject: str, secret_key: str, expires_minutes: int = 30, extra_claims: Optional[Dict[str, Any]] = None) -> str:
    now = int(time.time())
    payload: Dict[str, Any] = {
        "sub": subject,
        "iat": now,
        "exp": now + int(expires_minutes) * 60,
    }
    if extra_claims:
        payload.update(extra_claims)
    return jwt_encode(payload, secret_key)

