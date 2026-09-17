"""Storage abstraction for inspection images (Phase 4D).

The backend saves the raw uploaded image when the inspection is created and
serves it back to the browser so the dashboard can draw Bounding Boxes from the
Vision Contract. The local filesystem provider is the default; the interface
is kept small so a MinIO (S3) provider can be dropped in later without touching
the inspection flow.

Traceability contract (P0): a stored artifact is written atomically (temp file
+ os.replace) and returns the real URI, SHA256 and media type. Callers must
persist those fields, never the original upload filename, so the database
always points at bytes that actually exist on disk.
"""

from __future__ import annotations

import hashlib
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from ..config import get_settings, project_root

_IMAGE_MAGIC: dict[str, str] = {
    b"\xff\xd8\xff": "image/jpeg",
    b"\x89PNG\r\n\x1a\n": "image/png",
    b"BM": "image/bmp",
}


@dataclass(frozen=True)
class StoredArtifact:
    """Metadata for a persisted image. uri is the real on-disk location,
    sha256 the content digest, media_type the detected (not client-claimed)
    media type."""

    uri: str
    sha256: str
    media_type: str


def detect_media_type(data: bytes) -> str:
    """Detect media type from magic bytes; the client's claimed content type
    is never trusted for the stored record."""
    for magic, media_type in _IMAGE_MAGIC.items():
        if data.startswith(magic):
            return media_type
    return "application/octet-stream"


def sha256_of(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


class StorageService(Protocol):
    def save(self, inspection_id: str, data: bytes) -> StoredArtifact:
        """Atomically persist raw image bytes. Returns real URI + digest +
        detected media type. Raises OSError on failure (caller must fail
        closed: never continue to inference without a stored image)."""

    def save_anomaly_map(self, inspection_id: str, data: bytes) -> Path:
        """Persist the anomaly heatmap PNG for an inspection (Phase 6)."""

    def load(self, inspection_id: str) -> bytes | None:
        """Return stored bytes or None when the image is missing."""

    def load_anomaly_map(self, inspection_id: str) -> bytes | None:
        """Return the anomaly heatmap PNG bytes or None."""

    def path_for(self, inspection_id: str) -> Path | None:
        """Return the stored path or None when missing."""


class LocalStorageProvider:
    def __init__(self, root: Path) -> None:
        self.root = root
        self.root.mkdir(parents=True, exist_ok=True)

    def _path(self, inspection_id: str) -> Path:
        return self.root / f"{inspection_id}.jpg"

    def _anomaly_path(self, inspection_id: str) -> Path:
        return self.root / f"{inspection_id}-anomaly.png"

    def _uri(self, path: Path) -> str:
        """Real on-disk location as a project-relative URI when possible,
        absolute otherwise."""
        try:
            return path.resolve().relative_to(project_root()).as_posix()
        except ValueError:
            return path.resolve().as_posix()

    @staticmethod
    def _atomic_write(path: Path, data: bytes) -> None:
        """Write via temp file + os.replace so a crash mid-write never leaves
        a truncated artifact at the final path."""
        path.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent))
        try:
            with os.fdopen(fd, "wb") as handle:
                handle.write(data)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(tmp, path)
        finally:
            if os.path.exists(tmp):
                os.unlink(tmp)

    def save(self, inspection_id: str, data: bytes) -> StoredArtifact:
        path = self._path(inspection_id)
        self._atomic_write(path, data)
        return StoredArtifact(uri=self._uri(path), sha256=sha256_of(data), media_type=detect_media_type(data))

    def save_anomaly_map(self, inspection_id: str, data: bytes) -> Path:
        path = self._anomaly_path(inspection_id)
        self._atomic_write(path, data)
        return path

    def load(self, inspection_id: str) -> bytes | None:
        path = self._path(inspection_id)
        if path.exists():
            return path.read_bytes()
        return None

    def load_anomaly_map(self, inspection_id: str) -> bytes | None:
        path = self._anomaly_path(inspection_id)
        if path.exists():
            return path.read_bytes()
        return None

    def path_for(self, inspection_id: str) -> Path | None:
        path = self._path(inspection_id)
        return path if path.exists() else None


_storage: StorageService | None = None


def get_storage() -> StorageService:
    global _storage
    if _storage is None:
        root = Path(get_settings().storage_dir)
        _storage = LocalStorageProvider(root)
    return _storage


def reset_storage() -> None:
    global _storage
    _storage = None
