"""Build a dataset when needed, then show its train and test splits and task inputs.

Pass a short dataset name or its full id::

    uv run python examples/dataset_splits.py hearts
    uv run python examples/dataset_splits.py leochen085/slip

The script uses ``~/data/timenet`` as its registry. It reuses a converted version there, or runs the
dataset's connector first when none exists. ``get_train()`` and ``get_test()`` return the tasks of
one split, and ``get_all()`` every task whatever its split. For each split the script shows the
first few tasks of each kind, with the records they read: a record's sources and, per source, its
signals with their lengths.
"""

import argparse
import collections
from datetime import UTC, datetime
from itertools import islice
from pathlib import Path
import time

from timenet.client import TimeNet
from timenet.dataset import Record, TimeFDataset
from timenet.errors import TimeNetDatasetNotFoundError
from timenet.types import Task
from timenet_connectors.discovery import has_connector


TOP = 3  # tasks shown per task kind and split
SOURCES = 3  # sources shown per input record
WIDTH = 90  # characters of a prompt or target shown per task
REGISTRY = Path("~/data/timenet").expanduser()
DATASETS = {
    "arfbench": "datadog/arfbench",
    "chatts": "chattsrepo/chatts-training-dataset",
    "chatts-training-dataset": "chattsrepo/chatts-training-dataset",
    "ecg-qa-cot": "physionet/ecg-qa-cot",
    "hearts": "yang-ai-lab/hearts",
    "hello-world": "timenet/hello-world",
    "openrca": "microsoft/openrca",
    "opensqa": "bash-lab/opensqa",
    "sleep-edfx": "physionet/sleep-edfx",
    "slip": "leochen085/slip",
    "test-mean": "timenet/test-mean",
    "tsqa": "chengsenwang/tsqa",
    "verbalts": "seqml/verbalts",
}


def _clip(text: str) -> str:
    """Cut text to ``WIDTH`` characters, on one line.

    Returns:
        The clipped text.
    """
    text = text.replace("\n", " ⏎ ")
    return text if len(text) <= WIDTH else text[:WIDTH] + "…"


def _kind(task: Task) -> str:
    """Name a task by its type and the input kinds it declares, such as ``AnswerTask[image, text]``.

    Returns:
        The kind.
    """
    return f"{type(task).__name__}[{', '.join(sorted(kind.value for kind in task.resolved_input_modalities))}]"


def _describe_task(task: Task) -> list[str]:
    """Describe a task: its prompt, its answer, its annotations, and the records it reads.

    A task with inline ``targets`` answers with them; one without answers by reference to the
    annotations it names in ``target_annotations``.

    Returns:
        The lines.
    """
    if task.targets is not None:
        answer = ", ".join(str(target) for target in task.targets)
    else:
        answer = " | ".join(str(annotation.value) for annotation in task.target_annotations)
    lines = [f"{task.id} ({task.split.value if task.split else 'no split'})"]
    if task.prompt:
        lines.append(f"  prompt: {_clip(task.prompt)}")
    lines.append(f"  answer: {_clip(answer)}")
    if task.annotations:
        facts = ", ".join(f"{annotation.key}={annotation.value!r}" for annotation in task.annotations)
        lines.append(f"  annotations: {_clip(facts)}")
    for record in task.inputs:
        lines.extend(_describe_record(record))
    return lines


def _describe_record(record: Record) -> list[str]:
    """Describe an input record: when it starts, and its sources with their signals and lengths.

    Returns:
        The lines.
    """
    started = record.start_time.timestamp
    if started is None:
        when = "no wall time"
    elif isinstance(started, int):
        when = datetime.fromtimestamp(started / 1_000_000, UTC).isoformat()
    else:
        when = started.isoformat()
    lines = [f"  reads {record.id}: {len(record.sources)} sources, {len(record.signals)} signals, starts {when}"]
    for source in record.sources[:SOURCES]:
        signals = ", ".join(f"{signal.name} ({signal.n_values} values)" for signal in source.signals)
        lines.append(f"    source {source.name!r}: {_clip(signals)}")
    if len(record.sources) > SOURCES:
        lines.append(f"    … {len(record.sources) - SOURCES} more sources")
    return lines


def _dataset_id(name: str) -> str:
    """Resolve a short dataset name or validate a full dataset id.

    Args:
        name: A short alias such as ``hearts`` or a full id such as ``yang-ai-lab/hearts``.

    Returns:
        The full dataset id.

    Raises:
        ValueError: If the name does not identify a connector in this checkout.
    """
    dataset_id = DATASETS.get(name, name)
    if has_connector(dataset_id):
        return dataset_id
    choices = ", ".join(sorted(DATASETS))
    raise ValueError(f"unknown dataset {name!r}; use a full dataset id or one of: {choices}")


def _load(dataset_id: str) -> TimeFDataset:
    """Load a converted dataset, building it in the shared registry when absent.

    Returns:
        The loaded dataset.
    """
    client = TimeNet(REGISTRY)
    try:
        client.get(dataset_id)
    except TimeNetDatasetNotFoundError:
        print(f"No converted {dataset_id} found in {REGISTRY}; running its connector")
        return client.load(dataset_id)
    print(f"Using converted {dataset_id} from {REGISTRY}")
    return client.load(dataset_id, auto_build=False)


def main() -> None:
    """Load the dataset, time the split accessors, and print the first tasks of each kind per split."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", help="short name (for example hearts) or full dataset id")
    arguments = parser.parse_args()
    try:
        dataset_id = _dataset_id(arguments.dataset)
    except ValueError as exc:
        parser.error(str(exc))
    print(f"Reading {dataset_id} from registry at {REGISTRY}")
    started = time.perf_counter()
    dataset = _load(dataset_id)
    print(f"Loading took {time.perf_counter() - started:.1f} seconds")

    splits = {}
    for name, select in (("train", dataset.get_train), ("test", dataset.get_test)):
        started = time.perf_counter()
        splits[name] = select()
        print(f"dataset.get_{name}() took {time.perf_counter() - started:.2f} seconds")

    for name, tasks in splits.items():
        kinds = collections.Counter(_kind(task) for task in tasks)
        print(f"\n== {name}: {len(tasks):,} tasks, {dict(kinds)}")
        for kind in kinds:
            for task in islice((task for task in tasks if _kind(task) == kind), TOP):
                print("\n".join(_describe_task(task)))


if __name__ == "__main__":
    main()
