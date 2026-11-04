"""Load a local TimeF dataset and print its summary and a sample.

Build the dataset into a local registry first, then run this script::

    uv run timenet-build build chengsenwang/tsqa --local-registry .timenet-registry
    uv run python examples/load_tsqa.py

Pass another dataset ID and registry path as the first and second arguments.
"""

from pathlib import Path
import sys

from timenet import TimeNet


def main() -> None:
    """Print a local dataset's summary and the first five values of its first signal."""
    arguments = sys.argv[1:]
    dataset_id = arguments[0] if arguments else "chengsenwang/tsqa"
    registry = Path(arguments[1]) if len(arguments) > 1 else Path(".timenet-registry")
    with TimeNet(sources=[registry]).open(dataset_id) as reader:
        dataset = reader.read()
        dataset.describe()

        if dataset.records:
            values = dataset.records[0].signals[0].to_numpy()
            print(f"\nrecord[0] first signal, first 5 values: {values[:5]}")


if __name__ == "__main__":
    main()
