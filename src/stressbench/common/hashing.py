"""Deterministic payload hashing for raw-message integrity tracking."""

import hashlib
import json


def payload_hash(payload: dict) -> str:
    """Compute a deterministic SHA-256 hex digest of a JSON-serialisable dict.

    Keys are sorted and separators are compact to ensure identical output for
    semantically equal payloads regardless of insertion order.

    Args:
        payload: Any JSON-serialisable dictionary.

    Returns:
        A 64-character lowercase hex string.
    """
    blob = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()
