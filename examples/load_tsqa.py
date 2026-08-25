"""Load a built TimeF dataset and print a summary (no pandas or extras needed).

Build the dataset first, then run this script::

    uv run timenet-build build "chengsenwang/tsqa"
    uv run python examples/load_tsqa.py

``TimeNet().load(...)`` returns a ``TimeFDataset``; ``describe()`` prints its identity, counts, per-spec
columns, and a sample preview. Series values load lazily as Arrow arrays only when you ask for them. Pass
a dataset id as the first argument to inspect another dataset.
"""

import sys

from timenet.client import TimeNet


def main() -> None:
    """Load the dataset, print its describe() summary, and show one raw series slice."""
    dataset_id = sys.argv[1] if len(sys.argv) > 1 else "chengsenwang/tsqa"
    dataset = TimeNet().load(dataset_id)
    dataset.describe()

    if dataset.samples:
        values = dataset.samples[0].time_series[0].to_numpy()
        print(f"\nsample[0] first channel, first 5 values: {values[:5]}")


if __name__ == "__main__":
    main()
