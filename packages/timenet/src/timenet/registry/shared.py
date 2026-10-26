"""Local cache completeness."""

from pathlib import Path

from timenet.manifest import Manifest


def complete_copy(root: Path, manifest: Manifest) -> bool:
    """Return whether a release has its manifest and every declared file, each with its recorded size."""
    return (root / "manifest.json").is_file() and all(
        (file := root / part.path).is_file() and file.stat().st_size == part.size for part in manifest.files.all_files()
    )
