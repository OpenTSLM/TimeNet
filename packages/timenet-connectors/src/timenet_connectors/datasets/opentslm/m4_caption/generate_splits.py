"""The script that made splits.json.xz, the train, validation and test split of m4-caption.

We ran this script once, offline, and committed its output. The connector reads splits.json.xz and
does not run this script, so you do not need to run it. It is here only to show how the split was
made. A new run can give another split, because the result depends on the version of scikit-learn.

The release archive stores no split. The script makes an 80/10/10 split with StratifiedSplitter,
stratified by M4 frequency. M4 has 90 pairs of series with equal values, so the splitter groups the
series by their values and each pair stays in one split.

To run it from the root of the repository, give the path to M4TimeSeriesCaptionDatasetV02.zip::

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
    M4CaptionSeries,
    M4Frequency,
)
from timenet_connectors.splitting import StratifiedSplitter


SEED = 42


def frequency(series: M4CaptionSeries) -> M4Frequency:
    """Return the frequency of a series, which the split stratifies by.

    Returns:
        The frequency.
    """
    return series.frequency


def values_digest(series: M4CaptionSeries) -> str:
    """Return the SHA-256 digest of the values of a series. Series with the same values form one group.

    Returns:
        The digest.
    """
    return hashlib.sha256(series.values.tobytes()).hexdigest()


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
        captioned = reader.read_series()

    splitter = StratifiedSplitter[M4CaptionSeries](
        train_folds=8,
        validation_folds=1,
        test_folds=1,
        stratify_by=frequency,
        group_by=values_digest,
        seed=SEED,
    )
    splits = splitter.split(captioned)
    content = {
        "source_sha256": source_sha256,
        "validation": sorted((series.m4_id for series in splits[Split.VALIDATION]), key=m4_id_order),
        "test": sorted((series.m4_id for series in splits[Split.TEST]), key=m4_id_order),
    }
    SPLITS_PATH.write_bytes(lzma.compress((json.dumps(content, indent=2) + "\n").encode()))


if __name__ == "__main__":
    main()
