# /// script
# requires-python = ">=3.12"
# dependencies = [
#     # The versions that the OpenTSLM uv.lock pins. The split depends on scikit-learn.
#     "numpy==2.3.5",
#     "pandas==2.3.3",
#     "scikit-learn==1.7.2",
# ]
# ///
"""Write splits.json, the frozen train, validation and test split of sleep-cot.

sleep-cot does not store the train/val/test splits explicitly. Rather they give a random seed and
allow scikit to do the split. This is hard to reproduce, because it means the official split is
dependent on a particular scikit version. Instead of pinning scikit in the connector, we rather
generated the split indices once with this script and store them in the accompanying splits.json.
This script is included, but really just for reference, so you get to see how the proverbial
sausage got made.

Run it once with the release CSV, from this directory::

    uv run generate_splits.py <path to sleep_cot.csv>
"""

import hashlib
import json
from pathlib import Path
import sys

import pandas as pd
from sklearn.model_selection import train_test_split


SPLITS_PATH = Path(__file__).with_name("splits.json")


def main() -> None:
    """Split the CSV that the first argument names, and write splits.json."""
    csv_path = Path(sys.argv[1])
    df = pd.read_csv(csv_path)

    # The calls of load_sleepedf_cot_splits, with its names and settings.
    test_frac = 0.1
    val_frac = 0.1
    seed = 42
    train_val_df, test_df = train_test_split(df, test_size=test_frac, random_state=seed, stratify=df["label"])
    val_frac_adj = val_frac / (1.0 - test_frac)
    _, val_df = train_test_split(
        train_val_df, test_size=val_frac_adj, random_state=seed + 1, stratify=train_val_df["label"]
    )

    # read_csv numbers the rows from 0 in file order, so an index label is a row position.
    splits = {
        "source_sha256": hashlib.sha256(csv_path.read_bytes()).hexdigest(),
        "validation": sorted(int(position) for position in val_df.index),
        "test": sorted(int(position) for position in test_df.index),
    }
    # One position per line, so a diff of a regenerated file shows each row that changed split.
    SPLITS_PATH.write_text(json.dumps(splits, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
