# /// script
# requires-python = ">=3.12"
# dependencies = [
#     # The versions that the OpenTSLM uv.lock pins. The split depends on scikit-learn.
#     "numpy==2.3.5",
#     "pandas==2.3.3",
#     "scikit-learn==1.7.2",
# ]
# ///
"""Write splits.json, the split that OpenTSLM's loader gives each row of the SleepEDF-CoT CSV.

The release has no split column. OpenTSLM's ``load_sleepedf_cot_splits``, in
``src/opentslm/time_series_datasets/sleep/sleepedf_cot_loader.py``, splits the rows each time it
loads them. This script repeats its calls with the library versions that OpenTSLM pins, and writes
the positions of the validation and the test rows. The connector reads splits.json, so its split
does not depend on the installed scikit-learn. The connector does not run this script.

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
    SPLITS_PATH.write_text(json.dumps(splits) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
