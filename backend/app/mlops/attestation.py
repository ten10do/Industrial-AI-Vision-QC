"""Trusted-pipeline attestation.

The registry stores three facts a caller must never be able to assert about
itself: the evaluation metrics, the domain validation verdict, and the
artifact hash. All three arrive here, and each is checked against evidence the
server computes or verifies on its own:

* the payload is HMAC-signed with the pipeline secret, inside a time window;
* the artifact SHA256 is recomputed from the artifact bytes;
* the domain validation verdict must reference an eval report whose hash the
  server also recomputes.

Anything the server cannot verify stays unverified, and an unverified input
blocks promotion (see promotion_policy.provenance_requirements).
"""

from __future__ import annotations

import hashlib
import hmac
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..config import get_settings, project_root

MAX_ARTIFACT_BYTES = 4 * 1024 * 1024 * 1024  # refuse to hash absurd inputs


def canonical_json(payload: Any) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def sha256_hex(payload: str | bytes) -> str:
    data = payload.encode("utf-8") if isinstance(payload, str) else payload
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    size = 0
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            size += len(chunk)
            if size > MAX_ARTIFACT_BYTES:
                raise ValueError(f"artifact too large to hash: {path}")
            h.update(chunk)
    return h.hexdigest()


def resolve_artifact(uri: str | None) -> Path | None:
    """Resolve a registry artifact_uri to a local file.

    Only relative URIs inside the project root, or absolute paths, resolve.
    Remote schemes (s3://, gs://, http://) return None: the server cannot
    verify them, so they stay unverified rather than being trusted.
    """
    if not uri:
        return None
    if "://" in uri:
        return None
    p = Path(uri)
    if not p.is_absolute():
        p = project_root() / p
    try:
        resolved = p.resolve()
    except OSError:
        return None
    root = project_root().resolve()
    if resolved != root and root not in resolved.parents:
        return None  # path escape attempt
    return resolved if resolved.is_file() else None


@dataclass(frozen=True)
class Verification:
    verified: bool
    status: str  # "verified" | "mismatch" | "unverifiable" | "absent"
    detail: str

    def to_dict(self) -> dict:
        return {"verified": self.verified, "status": self.status, "detail": self.detail}


def verify_artifact_hash(artifact_uri: str | None, expected: str | None) -> Verification:
    if not expected:
        return Verification(False, "absent", "no artifact hash on record")
    path = resolve_artifact(artifact_uri)
    if path is None:
        return Verification(False, "unverifiable", f"artifact not resolvable locally: {artifact_uri!r}")
    try:
        actual = sha256_file(path)
    except (OSError, ValueError) as exc:
        return Verification(False, "unverifiable", f"artifact unreadable: {exc}")
    if hmac.compare_digest(actual, str(expected).strip().lower()):
        return Verification(True, "verified", f"sha256 recomputed from {path.name}")
    return Verification(False, "mismatch", f"sha256 mismatch: expected {expected}, got {actual}")


def verify_domain_evidence(evidence: dict | None, required_domain: str) -> Verification:
    """Check the eval report behind a domain_validated=true claim."""
    if not evidence:
        return Verification(False, "absent", "no domain evidence on record")
    domain = str(evidence.get("domain") or "")
    if domain != required_domain:
        return Verification(False, "mismatch", f"evidence domain {domain!r} != required {required_domain!r}")
    expected = evidence.get("eval_report_sha256")
    if not expected:
        return Verification(False, "absent", "domain evidence carries no eval report hash")
    path = resolve_artifact(str(evidence.get("eval_report_uri") or ""))
    if path is None:
        return Verification(False, "unverifiable", "eval report not resolvable locally")
    try:
        actual = sha256_file(path)
    except (OSError, ValueError) as exc:
        return Verification(False, "unverifiable", f"eval report unreadable: {exc}")
    if hmac.compare_digest(actual, str(expected).strip().lower()):
        return Verification(True, "verified", f"eval report {path.name} hash matches")
    return Verification(False, "mismatch", f"eval report sha256 mismatch: expected {expected}, got {actual}")


# ---- signature ----

SIGNATURE_HEADER = "X-Attestation-Signature"
TIMESTAMP_HEADER = "X-Attestation-Timestamp"


@dataclass(frozen=True)
class SignatureCheck:
    ok: bool
    reason: str
    digest: str | None = None


ATTESTATION_FIELDS = (
    "artifact_sha256",
    "metrics",
    "domain_validated",
    "domain_evidence",
)


def attestation_payload(
    *,
    model_name: str,
    model_version: str,
    training_run_id: str | None,
    body: dict,
) -> dict:
    """The exact object the pipeline signs.

    Built from the RAW request body, not from the parsed pydantic model: a
    client signs what it sends, and any server-side normalization (added
    default fields, int -> float coercion) would otherwise make a legitimate
    signature unverifiable for reasons the client cannot predict. Both sides
    run json.loads -> sorted json.dumps, so numbers serialize identically.
    """
    return {
        "schema_version": "ivqc_model_attestation_v1",
        "model_name": model_name,
        "model_version": model_version,
        "training_run_id": training_run_id,
        **{field: body.get(field) for field in ATTESTATION_FIELDS},
    }


def sign_attestation(secret: str, payload: dict, timestamp: int | None = None) -> tuple[str, int]:
    ts = int(timestamp if timestamp is not None else time.time())
    message = f"{ts}.{canonical_json(payload)}"
    return hmac.new(secret.encode("utf-8"), message.encode("utf-8"), hashlib.sha256).hexdigest(), ts


# ---- evaluation evidence attestation ----
#
# Evaluation reports are much larger than the metric/domain attestation above,
# so the signature covers the report's digest rather than the report body: the
# pipeline signs the identity of the evidence, and the server recomputes the
# digest from the bytes it actually received. A report that was edited in
# flight changes its digest and stops matching the signature.

EVALUATION_ATTESTATION_SCHEMA = "ivqc_evaluation_attestation_v1"


def evaluation_attestation_payload(
    *,
    model_name: str,
    model_version: str,
    report_sha256: str,
) -> dict:
    """The exact object the pipeline signs when submitting evaluation evidence."""
    return {
        "schema_version": EVALUATION_ATTESTATION_SCHEMA,
        "model_name": model_name,
        "model_version": model_version,
        "report_sha256": report_sha256,
    }


def verify_attestation_signature(
    secret: str,
    payload: dict,
    signature: str | None,
    timestamp: str | None,
    *,
    max_skew_seconds: int | None = None,
    now: int | None = None,
) -> SignatureCheck:
    """Verify an HMAC attestation. Fails closed on a missing secret."""
    if not secret:
        return SignatureCheck(False, "attestation_not_configured: IVQC_PIPELINE_HMAC_SECRET is empty")
    if not signature or not timestamp:
        return SignatureCheck(False, "attestation_signature_missing")
    try:
        ts = int(timestamp)
    except (TypeError, ValueError):
        return SignatureCheck(False, "attestation_timestamp_invalid")
    skew = max_skew_seconds if max_skew_seconds is not None else get_settings().attestation_max_skew_seconds
    current = int(now if now is not None else time.time())
    if abs(current - ts) > skew:
        return SignatureCheck(False, f"attestation_timestamp_out_of_window: skew {abs(current - ts)}s > {skew}s")
    expected, _ = sign_attestation(secret, payload, ts)
    digest = sha256_hex(canonical_json(payload))
    if not hmac.compare_digest(expected, str(signature).strip().lower()):
        return SignatureCheck(False, "attestation_signature_mismatch")
    return SignatureCheck(True, "verified", digest=digest)
