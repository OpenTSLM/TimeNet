# /// script
# requires-python = ">=3.12"
# dependencies = [
#     # The versions that the OpenTSLM uv.lock pins. The split depends on datasets and numpy.
#     "datasets==4.4.1",
#     "numpy==2.3.5",
#     "pandas==2.3.3",
# ]
# ///
"""Print OPENTSLM_SPLIT_SHA256, the digest of the train, validation and test split of m4-caption.

The release does not store the split. OpenTSLM's loader makes it at load time with the datasets library
and a fixed seed. The connector computes the same split with NumPy and checks it against the digest that
this script prints. This script runs the calls of OpenTSLM's loader with the library versions of the
OpenTSLM uv.lock, so the digest comes from OpenTSLM's own code path and not from the connector.

Run it once with the extracted release folder, from this directory::

    uv run generate_splits.py <path to M4TimeSeriesCaptionDataset>
"""

import hashlib
import json
from pathlib import Path
import sys


FREQUENCIES = ["Daily", "Hourly", "Monthly", "Quarterly", "Weekly", "Yearly"]


def load_samples(release: Path) -> list[dict[str, str]]:
    """Read the samples in the order of load_all_m4_data and create_combined_dataset.

    Returns:
        One sample with the release id and the frequency for each series.
    """
    # Discovery imports every module beside the connectors, so the script imports its own dependencies
    # only when it runs.
    import pandas as pd  # ruff: ignore[import-outside-top-level]

    all_samples = []
    for frequency in FREQUENCIES:
        series_df = pd.read_csv(release / f"m4_series_{frequency}.csv")
        captions_df = pd.read_csv(release / f"m4_captions_{frequency}.csv")
        common_ids = set(series_df["id"]) & set(captions_df["id"])
        series_df = series_df[series_df["id"].isin(common_ids)].reset_index(drop=True)
        captions_df = captions_df[captions_df["id"].isin(common_ids)].reset_index(drop=True)
        merged_df = series_df.merge(captions_df, on="id", how="inner")
        for _, row in merged_df.iterrows():
            # The loader skips a row whose series does not parse, so parse each one. The split depends
            # only on the number and order of the rows, so the samples here hold no values.
            json.loads(row["series"])
            all_samples.append({"id": row["id"], "frequency": frequency})
    return all_samples


def main() -> None:
    """Split the release in the folder that the first argument names, and print the digest."""
    from datasets import Dataset  # ruff: ignore[import-outside-top-level]  # ty: ignore[unresolved-import]

    all_samples = load_samples(Path(sys.argv[1]))

    # The calls of create_combined_dataset, with their names and settings.
    full_dataset = Dataset.from_list(all_samples)
    test_frac = 0.1
    val_frac = 0.1
    seed = 42
    train_val, test = full_dataset.train_test_split(test_size=test_frac, seed=seed).values()
    val_frac_adj = val_frac / (1.0 - test_frac)
    train, val = train_val.train_test_split(test_size=val_frac_adj, seed=seed + 1).values()

    # The digest of OpenTSLMSplit.digest in connector.py: one "<m4_id>\t<split>\n" line per series, in
    # the order of the loader. The M4 id is the release id without its "series-" prefix.
    split_of = dict.fromkeys(train["id"], "train")
    split_of |= dict.fromkeys(val["id"], "validation")
    split_of |= dict.fromkeys(test["id"], "test")
    lines = "".join(f"{sample['id'].removeprefix('series-')}\t{split_of[sample['id']]}\n" for sample in all_samples)
    print(f"train {len(train)}, validation {len(val)}, test {len(test)}")
    print(f'OPENTSLM_SPLIT_SHA256 = "{hashlib.sha256(lines.encode()).hexdigest()}"')


if __name__ == "__main__":
    main()
