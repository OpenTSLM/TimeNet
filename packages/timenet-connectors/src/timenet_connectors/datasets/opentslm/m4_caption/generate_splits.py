"""Write splits.json.xz, the frozen train, validation and test split of m4-caption.

The release archive stores no split. This script makes an 80/10/10 split with StratifiedSplitter,
stratified by M4 frequency, so that validation and test each hold a tenth of every frequency. The
result depends on the version of scikit-learn. So the script ran once and wrote the validation and
test series to splits.json.xz, and the connector reads that file. The script is kept here for
reference.

Run it once from the root of the repository. The argument is the path to the release archive,
M4TimeSeriesCaptionDatasetV02.zip::

    uv run python -m timenet_connectors.datasets.opentslm.m4_caption.generate_splits <archive>
"""

import hashlib
import json
import lzma
from pathlib import Path
import sys
import tempfile
import zipfile

from timenet.types import Split
from timenet_connectors.datasets.opentslm.m4_caption.connector import (
    M4_CAPTION_SHA256,
    RELEASE_FOLDER,
    SPLITS_PATH,
    M4CaptionReader,
)
from timenet_connectors.splitting import StratifiedSplitter


SEED = 42


def frequency_letter(m4_id: str) -> str:
    """Return the letter that starts an M4 id, such as M in M1. The letter names the frequency.

    Returns:
        The letter.
    """
    return m4_id[0]


def m4_id_order(m4_id: str) -> tuple[str, int]:
    """Sort M4 ids by frequency letter, then by number: M2 before M10.

    Returns:
        The sort key.
    """
    return m4_id[0], int(m4_id[1:])


def main() -> None:
    """Split the release archive that the first argument names, and write splits.json.xz.

    Raises:
        SystemExit: If the archive is not the pinned release.
    """
    archive_path = Path(sys.argv[1])
    source_sha256 = hashlib.sha256(archive_path.read_bytes()).hexdigest()
    if source_sha256 != M4_CAPTION_SHA256:
        raise SystemExit(f"{archive_path} has the SHA-256 {source_sha256}, not the pinned {M4_CAPTION_SHA256}")
    with tempfile.TemporaryDirectory() as directory, zipfile.ZipFile(archive_path) as archive:
        archive.extractall(directory)
        reader = M4CaptionReader(directory=Path(directory) / RELEASE_FOLDER, splits_path=SPLITS_PATH)
        m4_ids = reader.read_m4_ids()

    splitter = StratifiedSplitter[str](
        train_folds=8, validation_folds=1, test_folds=1, stratify_by=frequency_letter, seed=SEED
    )
    splits = splitter.split(m4_ids)
    content = {
        "source_sha256": source_sha256,
        "validation": sorted(splits[Split.VALIDATION], key=m4_id_order),
        "test": sorted(splits[Split.TEST], key=m4_id_order),
    }
    SPLITS_PATH.write_bytes(lzma.compress((json.dumps(content, indent=2) + "\n").encode()))


if __name__ == "__main__":
    main()
