"""Regenerate ``files.csv``, the pinned list of 3W source files, from a local 3W checkout.

Run it only when moving the connector to a new 3W commit, then update ``SOURCE_REVISION`` in
``connector.py`` and the dataset version in ``dataset.yaml``::

    git clone --filter=blob:none --sparse --no-checkout https://github.com/petrobras/3W.git 3w
    cd 3w && git sparse-checkout set dataset && git checkout <commit>
    uv run python generate_manifest.py <path-to-3w-checkout>
"""

import csv
import hashlib
from pathlib import Path
import sys

from timenet_connectors.datasets.petrobras.three_w.connector import MANIFEST


def main(checkout: Path) -> None:
    """Hash every instance file of a 3W checkout into the manifest.

    Args:
        checkout: The root of a 3W checkout at the pinned commit.
    """
    files = sorted(checkout.glob("dataset/*/*.parquet"))
    with MANIFEST.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(["path", "size", "sha256"])
        for path in files:
            with path.open("rb") as data:
                digest = hashlib.file_digest(data, "sha256").hexdigest()
            writer.writerow([path.relative_to(checkout).as_posix(), path.stat().st_size, digest])
    print(f"wrote {len(files)} entries to {MANIFEST}")


if __name__ == "__main__":
    main(Path(sys.argv[1]))
