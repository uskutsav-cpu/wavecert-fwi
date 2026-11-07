"""Small, deterministic provenance records for reproducible experiments."""

from __future__ import annotations

import hashlib
import importlib.metadata
import os
import platform
import subprocess
import sys
from pathlib import Path


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def source_file_manifest(root: str | Path) -> dict[str, str]:
    """Identify implementation, runner, CI, and configuration bytes exactly."""
    root = Path(root)
    files = [root / "pyproject.toml"]
    files += sorted((root / "src").rglob("*.py"))
    files += sorted(p for p in (root / "configs").rglob("*") if p.is_file())
    files += sorted(p for p in (root / "scripts").rglob("*") if p.suffix in {".py", ".sh"})
    files += sorted((root / ".github" / "workflows").glob("*.yml"))
    files += sorted(root.glob("requirements*.txt"))
    return {
        p.relative_to(root).as_posix(): sha256_file(p) for p in sorted(set(files)) if p.is_file()
    }


def source_digest(root: str | Path) -> str:
    """Hash the complete relevant file manifest, including uncommitted edits."""
    digest = hashlib.sha256()
    for name, checksum in source_file_manifest(root).items():
        digest.update(name.encode())
        digest.update(b"\0")
        digest.update(checksum.encode())
        digest.update(b"\0")
    return digest.hexdigest()


def environment_record(root: str | Path) -> dict:
    root = Path(root).resolve()
    try:
        revision = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=root, text=True, stderr=subprocess.DEVNULL
        ).strip()
        dirty = bool(
            subprocess.check_output(["git", "status", "--porcelain"], cwd=root, text=True).strip()
        )
    except (OSError, subprocess.CalledProcessError):
        revision, dirty = None, None
    packages = {}
    for name in (
        "numpy",
        "scipy",
        "torch",
        "matplotlib",
        "pandas",
        "pytest",
        "h5py",
        "hdf5plugin",
        "pyarrow",
        "devito",
    ):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    return {
        "git_revision": revision,
        "working_tree_dirty": dirty,
        "source_sha256": source_digest(root),
        "source_hash_schema": "sha256-of-path-and-file-sha256-v2",
        "source_files": source_file_manifest(root),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "packages": packages,
        "thread_environment": {
            key: os.environ.get(key)
            for key in (
                "OMP_NUM_THREADS",
                "MKL_NUM_THREADS",
                "OPENBLAS_NUM_THREADS",
                "VECLIB_MAXIMUM_THREADS",
                "NUMEXPR_NUM_THREADS",
            )
        },
        "torch_threads": sys.modules["torch"].get_num_threads() if "torch" in sys.modules else None,
    }
