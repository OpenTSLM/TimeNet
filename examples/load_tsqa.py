"""Load a built TimeF dataset and print a summary (no pandas or extras needed).

Build the dataset into an explicit local registry, then run this script::

    uv run timenet-build build "chengsenwang/tsqa" --out .timenet-registry
    uv run python examples/load_tsqa.py

``TimeNet(registry).load(...)`` returns a ``TimeFDataset``. ``describe()`` prints its identity,
counts, per-spec columns, and a record preview. Signal values load lazily as Arrow arrays only when
you ask for them. Pass a dataset ID as the first argument and a registry path as the second argument.
"""

from pathlib import Path
import sys

from timenet.client import TimeNet


def main() -> None:
    """Load the dataset, print its describe() summary, and show one raw series slice."""
    arguments = sys.argv[1:]
    dataset_id = arguments[0] if arguments else "chengsenwang/tsqa"
    registry = Path(arguments[1]) if len(arguments) > 1 else Path(".timenet-registry")
    dataset = TimeNet(registry).load(dataset_id)
    dataset.describe()

    if dataset.records:
        values = dataset.records[0].signals[0].to_numpy()
        print(f"\nrecord[0] first signal, first 5 values: {values[:5]}")


if __name__ == "__main__":
    main()
