# /// script
# requires-python = ">=3.12"
# dependencies = [
#     # The versions that the OpenTSLM uv.lock pins. The split depends on datasets and numpy.
#     "datasets==4.4.1",
#     "numpy==2.3.5",
#     "pandas==2.3.3",
# ]
# ///
"""Write splits.json.xz, the frozen train, validation and test split of m4-caption.

The release does not store the split. OpenTSLM's loader makes it at load time with the datasets library
and a fixed seed, so the split depends on the versions of datasets and NumPy. Instead of pinning these
libraries in the connector, this script ran the calls of the loader once, with the library versions of
the OpenTSLM uv.lock, and wrote the validation and test series to splits.json.xz. The connector reads
that file. xz keeps the file small, and Python reads it with the lzma module of the standard library.
The script is here for reference, so you can see how the file was made.

Run it once with the release archive, from this directory::

    uv run generate_splits.py <path to M4TimeSeriesCaptionDatasetV02.zip>
"""

import hashlib
import json
import lzma
from pathlib import Path
import sys
import zipfile


FREQUENCIES = ["Daily", "Hourly", "Monthly", "Quarterly", "Weekly", "Yearly"]
SPLITS_PATH = Path(__file__).with_name("splits.json.xz")


def load_samples(archive: zipfile.ZipFile) -> list[dict[str, str]]:
    """Read the samples in the order of load_all_m4_data and create_combined_dataset.

    Returns:
        One sample with the release id and the frequency for each series.
    """
    # Discovery imports every module beside the connectors, so the script imports its own dependencies
    # only when it runs.
    import pandas as pd  # ruff: ignore[import-outside-top-level]

    all_samples = []
    for frequency in FREQUENCIES:
        series_df = pd.read_csv(archive.open(f"M4TimeSeriesCaptionDataset/m4_series_{frequency}.csv"))
        captions_df = pd.read_csv(archive.open(f"M4TimeSeriesCaptionDataset/m4_captions_{frequency}.csv"))
        common_ids = set(series_df["id"]) & set(captions_df["id"])
        series_df = series_df[series_df["id"].isin(common_ids)].reset_index(drop=True)
        captions_df = captions_df[captions_df["id"].isin(common_ids)].reset_index(drop=True)
        merged_df = series_df.merge(captions_df, on="id", how="inner")
        for _, row in merged_df.iterrows():
            all_samples.append({"id": row["id"], "frequency": frequency})
    return all_samples


def m4_id_order(m4_id: str) -> tuple[str, int]:
    """Sort M4 ids by frequency letter, then by number: M2 before M10.

    Returns:
        The sort key.
    """
    return m4_id[0], int(m4_id[1:])


def main() -> None:
    """Split the release archive that the first argument names, and write splits.json.xz."""
    from datasets import Dataset  # ruff: ignore[import-outside-top-level]  # ty: ignore[unresolved-import]

    archive_path = Path(sys.argv[1])
    with zipfile.ZipFile(archive_path) as archive:
        all_samples = load_samples(archive)

    # The calls of create_combined_dataset, with their names and settings.
    full_dataset = Dataset.from_list(all_samples)
    test_frac = 0.1
    val_frac = 0.1
    seed = 42
    train_val, test = full_dataset.train_test_split(test_size=test_frac, seed=seed).values()
    val_frac_adj = val_frac / (1.0 - test_frac)
    _, val = train_val.train_test_split(test_size=val_frac_adj, seed=seed + 1).values()

    # The file names each series by its M4 id: the release id without its "series-" prefix.
    splits = {
        "source_sha256": hashlib.sha256(archive_path.read_bytes()).hexdigest(),
        "validation": sorted((sample_id.removeprefix("series-") for sample_id in val["id"]), key=m4_id_order),
        "test": sorted((sample_id.removeprefix("series-") for sample_id in test["id"]), key=m4_id_order),
    }
    SPLITS_PATH.write_bytes(lzma.compress((json.dumps(splits, indent=2) + "\n").encode()))


if __name__ == "__main__":
    main()
