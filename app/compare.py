"""Compare two published ZIP diagnostic packages by decompressed content.

Rules:
  * entries are keyed by their exact ZIP-internal path; file *contents*
    (decompressed) decide equality — compression method, compression level
    and timestamps are deliberately ignored;
  * directory entries never appear in the diff, but they do count toward
    the entry limit and the duplicate-path check;
  * a package is rejected as a whole (no partial results) when it has
    duplicate paths, encrypted entries, unreadable/corrupt entries, more
    than MAX_ENTRIES entries, or a decompressed payload over
    MAX_UNCOMPRESSED bytes;
  * before an archive is even opened, the published file on disk must
    still match the size and SHA-256 registered at upload time.
"""
from __future__ import annotations

import hashlib
import zipfile
from pathlib import Path

MAX_ENTRIES = 2000
MAX_UNCOMPRESSED = 256 * 1024 * 1024  # 256 MiB per package, decompressed
READ_BUF = 1024 * 1024

BASELINE = "baseline"
REVIEW = "review"


class CompareError(Exception):
    """A problem that aborts the whole comparison (no partial diff)."""

    def __init__(self, code: str, message: str, package: str | None = None):
        super().__init__(message)
        self.code = code
        self.package = package

    def payload(self) -> dict:
        return {"code": self.code, "message": str(self), "package": self.package}


def verify_artifact(path: Path, size: int, sha256: str, which: str) -> None:
    """The published file must still match what was registered at upload."""
    if not path.exists():
        raise CompareError(
            "artifact_missing", "published file is missing on disk", which
        )
    actual_size = path.stat().st_size
    if actual_size != size:
        raise CompareError(
            "integrity_mismatch",
            f"size mismatch: registered {size} bytes, file on disk has"
            f" {actual_size} bytes",
            which,
        )
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        while True:
            buf = fh.read(READ_BUF)
            if not buf:
                break
            digest.update(buf)
    actual = digest.hexdigest()
    if actual != sha256:
        raise CompareError(
            "integrity_mismatch",
            f"sha256 mismatch: registered {sha256}, file on disk hashes"
            f" to {actual}",
            which,
        )


def _hash_entry(
    zf: zipfile.ZipFile, info: zipfile.ZipInfo, which: str
) -> tuple[str, int]:
    """Stream one entry through a hasher.

    Returns (hex_digest, uncompressed_size). Any decompression failure —
    bad CRC, truncated data, unknown compression method — aborts the
    whole comparison.
    """
    digest = hashlib.sha256()
    total = 0
    try:
        with zf.open(info, "r") as fh:
            while True:
                buf = fh.read(READ_BUF)
                if not buf:
                    break
                total += len(buf)
                if total > MAX_UNCOMPRESSED:
                    raise CompareError(
                        "too_large",
                        f"entry {info.filename!r} alone decompresses beyond"
                        f" {MAX_UNCOMPRESSED} bytes",
                        which,
                    )
                digest.update(buf)
    except CompareError:
        raise
    except NotImplementedError as exc:
        raise CompareError(
            "unsupported_compression",
            f"entry {info.filename!r}: {exc}",
            which,
        )
    except Exception as exc:
        # BadZipFile (bad CRC, truncated stream), RuntimeError, EOFError,
        # zlib/lzma/bz2 errors — all mean the archive cannot be trusted.
        raise CompareError(
            "corrupt_entry",
            f"entry {info.filename!r} cannot be decompressed: {exc}",
            which,
        )
    return digest.hexdigest(), total


def extract_index(path: Path, which: str) -> dict[str, str]:
    """Map ZIP-internal path -> sha256 of decompressed content.

    Validates the archive first; any rule violation raises CompareError
    and no index is returned.
    """
    try:
        zf = zipfile.ZipFile(path)
    except Exception as exc:
        raise CompareError(
            "bad_zip", f"not a readable ZIP archive: {exc}", which
        )
    with zf:
        try:
            infos = zf.infolist()
        except Exception as exc:
            raise CompareError(
                "bad_zip", f"central directory is unreadable: {exc}", which
            )
        if len(infos) > MAX_ENTRIES:
            raise CompareError(
                "too_many_entries",
                f"{len(infos)} entries exceed the limit of {MAX_ENTRIES}",
                which,
            )
        seen: set[str] = set()
        for info in infos:
            if info.filename in seen:
                raise CompareError(
                    "duplicate_path",
                    f"path {info.filename!r} appears more than once",
                    which,
                )
            seen.add(info.filename)
            if info.flag_bits & 0x1:
                raise CompareError(
                    "encrypted",
                    f"entry {info.filename!r} is encrypted",
                    which,
                )
        declared = sum(i.file_size for i in infos if not i.is_dir())
        if declared > MAX_UNCOMPRESSED:
            raise CompareError(
                "too_large",
                f"declared decompressed size {declared} bytes exceeds"
                f" {MAX_UNCOMPRESSED} bytes",
                which,
            )
        index: dict[str, str] = {}
        total = 0
        for info in infos:
            if info.is_dir():
                continue
            hex_digest, size = _hash_entry(zf, info, which)
            total += size
            if total > MAX_UNCOMPRESSED:
                raise CompareError(
                    "too_large",
                    f"decompressed payload exceeds {MAX_UNCOMPRESSED} bytes",
                    which,
                )
            index[info.filename] = hex_digest
        return index


def diff_indexes(baseline: dict[str, str], review: dict[str, str]) -> dict:
    base_paths = set(baseline)
    rev_paths = set(review)
    added = sorted(rev_paths - base_paths)
    deleted = sorted(base_paths - rev_paths)
    changed = sorted(
        p for p in base_paths & rev_paths if baseline[p] != review[p]
    )
    unchanged = len(base_paths & rev_paths) - len(changed)
    return {
        "added": added,
        "deleted": deleted,
        "changed": changed,
        "unchanged_count": unchanged,
        "counts": {
            "added": len(added),
            "deleted": len(deleted),
            "changed": len(changed),
        },
    }


def compare_archives(baseline_path: Path, review_path: Path) -> dict:
    """Full comparison of two ZIP files; raises CompareError on any defect."""
    baseline = extract_index(baseline_path, BASELINE)
    review = extract_index(review_path, REVIEW)
    return diff_indexes(baseline, review)
