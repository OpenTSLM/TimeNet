"""Load a built dataset and show its train and test splits, and what the tasks read.

Build the dataset into an explicit local registry, then run this script::

    uv run timenet-build build "datadog/arfbench" --out .timenet-registry
    uv run python examples/dataset_splits.py datadog/arfbench .timenet-registry

The first argument is the dataset id, the second the registry path. ``get_train()`` and
``get_test()`` return the tasks of one split, and ``get_all()`` every task whatever its split. For
each split the script shows the first few tasks of each kind, with the records they read: a
record's sources and, per source, its signals with their lengths. In ARFBench a record is a
metric, a source one of its tag groups, and a signal that group's series at one rollup interval;
a chart task reads image records instead.
"""

import collections
from datetime import UTC, datetime
from itertools import islice
from pathlib import Path
import sys
import time

from timenet.client import TimeNet
from timenet.dataset import Record
from timenet.types import Task


TOP = 3  # tasks shown per task kind and split
SOURCES = 3  # sources shown per input record
WIDTH = 90  # characters of a prompt or target shown per task


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


def main() -> None:
    """Load the dataset, time the split accessors, and print the first tasks of each kind per split."""
    arguments = sys.argv[1:]
    dataset_id = arguments[0] if arguments else "datadog/arfbench"
    registry = Path(arguments[1]) if len(arguments) > 1 else Path(".timenet-registry")
    print(f"Reading {dataset_id} from registry at {registry}")
    started = time.perf_counter()
    dataset = TimeNet(registry).load(dataset_id)
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
