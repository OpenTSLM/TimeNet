"""The HEARTS connector: the released frozen test cases of a health time-series agent benchmark.

The public artifact is the benchmark's frozen test cases, one Python pickle per case in a flat
``<corpus>/<task>/<index>.pkl`` tree over five upstream corpora. Each kept case becomes one record
carrying the signals the reference harness hands its agent, plus one task holding the question and
the ground truth.

``download`` fetches the twenty-one task directories in scope at the pinned revision, then checks
the tree still holds the test cases this connector was written against. ``convert`` reads each
payload once, reads the shapes off it, and attaches loaders that re-read the file later, so no
values plane is ever held in memory. The tasks stream: they are read off a second walk of the tree,
after ``convert`` returns.
"""

from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from timenet.connectors import BaseConnector
from timenet.dataset import TimeFDataset
from timenet.errors import TimeFFormatError, TimeNetDownloadError
from timenet.types import Annotation, Task
from timenet_connectors.datasets.yang_ai_lab.hearts.pickles import load_payload
from timenet_connectors.datasets.yang_ai_lab.hearts.series import series_for
from timenet_connectors.datasets.yang_ai_lab.hearts.tasks import (
    TASK_DEFINITIONS,
    TASK_TYPES,
    TaskDefinition,
    build_task,
    name_vocabulary,
    option_annotations,
    plain,
)


HF_REPO = "yang-ai-lab/HEARTS"
HF_REVISION = "7c18df521ae36cbc6b61e17782f1ac08dc378ea1"
"""A commit, never a branch: a branch would let the same connector code read different bytes."""

_ID_PREFIX = "hearts"
"""Every id this connector writes is built on this, record ids and vocabulary ids alike."""

_PROVENANCE = f"{_ID_PREFIX}:provenance"
_AGENT_INPUT = f"{_ID_PREFIX}:agent_input"
_ANSWER_KEY = "GT"

_NORMALIZED_DESCRIPTION = (
    "The release ships these values min-max scaled to the unit interval and does not publish the "
    "scaling constants, so the physical unit is not recoverable."
)


@dataclass(frozen=True)
class HeartsSource:
    """A handle to the downloaded test-case tree, so ``convert`` opens files rather than holding them."""

    root: Path
    """The directory holding the ``<corpus>/<task>/<index>.pkl`` tree."""


def _walk(root: Path) -> Iterator[tuple[TaskDefinition, int, Path]]:
    """Walk the test cases in canonical order: corpus, then task, then case index.

    Args:
        root: The directory holding the downloaded tree.

    Yields:
        Each test case with the definition of its task directory and its index.

    Raises:
        TimeFFormatError: If a file in a task directory is not named after its index.
    """
    for definition in TASK_DEFINITIONS:
        directory = root / definition.source / definition.task
        if not directory.is_dir():
            continue
        try:
            paths = sorted(directory.glob("*.pkl"), key=lambda path: int(path.stem))
        except ValueError as exc:
            raise TimeFFormatError(
                f"HEARTS {definition.directory} holds a file whose name is not a test-case index"
            ) from exc
        for path in paths:
            yield definition, int(path.stem), path


def _check_case_counts(root: Path) -> None:
    """Check the fetched tree publishes the test cases this connector was written against.

    Args:
        root: The directory holding the downloaded tree.

    Raises:
        TimeNetDownloadError: If a task directory in scope does not hold the number of test cases
            the pinned revision published when this connector was written.
    """
    for definition in TASK_DEFINITIONS:
        found = len(list((root / definition.source / definition.task).glob("*.pkl")))
        if found != definition.n_items:
            raise TimeNetDownloadError(
                f"HEARTS {definition.directory} holds {found} test cases at revision {HF_REVISION}, "
                f"but this connector was written against {definition.n_items}"
            )


def _subject_ids(source: str, payload: dict[str, Any]) -> tuple[str, ...]:
    """Return the subjects a test case is about, namespaced by their corpus.

    Args:
        source: The corpus the payload came from.
        payload: The test case.

    Returns:
        The subject ids, empty when the payload names none.
    """
    for key in ("subject_id", "speaker_id"):
        if key in payload:
            return (f"{source}-{payload[key]}",)
    pair = [payload[key] for key in ("A_subject", "B_subject") if key in payload]
    return tuple(f"{source}-{subject}" for subject in pair)


def _record_id(definition: TaskDefinition, index: int) -> str:
    """Build the id of the record one test case becomes.

    The record loop and the task stream both walk the tree, so both read the id from here.

    Args:
        definition: The task directory's definition.
        index: The test-case index.

    Returns:
        The record id, built from the release's own directory and file names.
    """
    return f"{_ID_PREFIX}-{definition.source}-{definition.task}-{index:02d}"


def _agent_input_id(record_id: str, key: str) -> str:
    """Build the id of one agent-input annotation.

    The record carries the annotation and a streamed task names it, so both read the id from here.

    Args:
        record_id: The owning record's id.
        key: The payload key the annotation holds.

    Returns:
        The annotation id.
    """
    return f"{record_id}-{key}"


def _input_annotation_ids(definition: TaskDefinition, record_id: str) -> tuple[str, ...]:
    """Name the annotations the reference harness shows its agent, the answer vocabulary first.

    Args:
        definition: The task directory's definition.
        record_id: The owning record's id.

    Returns:
        The annotation ids: the directory's registered vocabulary when it has one, then the
        agent-input annotations the record carries.
    """
    inputs = tuple(_agent_input_id(record_id, key) for key in definition.input_keys)
    if definition.options:
        return (name_vocabulary(_ID_PREFIX, definition.task), *inputs)
    return inputs


def _annotations_for(
    definition: TaskDefinition, index: int, payload: dict[str, Any], record_id: str
) -> list[Annotation]:
    """Build a test case's annotations, provenance first and agent input after.

    Args:
        definition: The task directory's definition.
        index: The test-case index.
        payload: The test case.
        record_id: The owning record's id.

    Returns:
        The annotations to attach to the record.
    """
    annotations = [
        Annotation(key="hearts_source", value=definition.source, source=_PROVENANCE, id=f"{record_id}-source"),
        Annotation(key="hearts_task", value=definition.task, source=_PROVENANCE, id=f"{record_id}-task"),
        Annotation(key="testcase_idx", value=index, source=_PROVENANCE, id=f"{record_id}-idx"),
    ]
    if definition.source == "harespod":
        annotations.append(
            Annotation(
                key="values_normalized",
                value=True,
                description=_NORMALIZED_DESCRIPTION,
                source=_PROVENANCE,
                id=f"{record_id}-normalized",
            )
        )
    quality = (payload.get("data") or {}).get("quality") if definition.source == "coswara" else None
    if quality is not None:
        annotations.append(
            Annotation(key="audio_quality", value=int(quality), source=_PROVENANCE, id=f"{record_id}-quality")
        )
    for key in definition.input_keys:
        annotations.append(
            Annotation(key=key, value=plain(payload[key]), source=_AGENT_INPUT, id=_agent_input_id(record_id, key))
        )
    return annotations


def _iter_tasks(source: HeartsSource) -> Iterator[Task]:
    """Yield the task of every test case, in the order ``convert`` built the records.

    This walks the tree again and reads each payload for its answer. A pickle states nothing until
    the whole object is rebuilt, so an answer cannot be reached without its file. The writer reads
    the stream after ``convert`` returns, and can read it more than once, so this keeps nothing
    between calls.

    Args:
        source: The handle to the downloaded tree.

    Yields:
        One task per test case, carrying its own record id.

    Raises:
        TimeFFormatError: If a file is not named after its index, or an answer does not fit the
            task type its directory states.
    """  # noqa: DOC502 (raised by _walk and build_task, not directly here)
    for definition, index, path in _walk(source.root):
        payload = load_payload(str(path))
        record_id = _record_id(definition, index)
        yield build_task(
            definition, record_id, payload[_ANSWER_KEY], _input_annotation_ids(definition, record_id), _ID_PREFIX
        )


class HeartsConnector(BaseConnector[HeartsSource]):
    """Connector for the HEARTS released test cases (Hub repo ``yang-ai-lab/HEARTS``)."""

    def download(self, cache_dir: Path) -> list[HeartsSource]:  # noqa: PLR6301 (the base declares it)
        """Fetch the in-scope test cases at the pinned revision.

        The nine out-of-scope task directories are not fetched. What arrives is checked against
        the case counts this connector was written against, before ``convert`` walks it.

        Args:
            cache_dir: The directory downloads land in.

        Returns:
            A single-element list holding the handle to the downloaded tree.

        Raises:
            ImportError: If ``huggingface_hub``, declared in this connector's ``requirements.txt``,
                is not installed.
            TimeNetDownloadError: If a task directory does not hold the case count this connector
                was written against.
        """  # noqa: DOC502 (TimeNetDownloadError comes from _check_case_counts, not directly here)
        # discovery.available() imports every connector module to read its CONNECTOR, and
        # huggingface_hub is declared in this connector's requirements.txt rather than by the
        # package. A module-level import would break dataset listing for every connector in an
        # environment without it.
        try:
            from huggingface_hub import snapshot_download  # noqa: PLC0415 (see the comment above)
        except ImportError as exc:
            raise ImportError(
                f"reading {HF_REPO!r} needs huggingface_hub, declared in this connector's "
                "requirements.txt. Run the build without --no-isolation, or install it yourself"
            ) from exc

        root = Path(
            snapshot_download(
                HF_REPO,
                repo_type="dataset",
                revision=HF_REVISION,
                cache_dir=str(cache_dir),
                allow_patterns=[f"{definition.directory}/*.pkl" for definition in TASK_DEFINITIONS],
            )
        )
        _check_case_counts(root)
        return [HeartsSource(root=root)]

    def convert(self, raw_refs: list[HeartsSource]) -> TimeFDataset:
        """Build one record per test case and stream one task per case.

        Each payload is read once here, to learn its signals, lengths and time axes. The values
        themselves stay on disk behind per-signal loaders. The tasks come from
        :func:`_iter_tasks`, which walks the tree again, so a task carries its own record id and
        none reaches ``Record.task_ids``.

        Args:
            raw_refs: The single-element list from :meth:`download`.

        Returns:
            The dataset: one record per test case, plus the task stream.

        Raises:
            TimeFFormatError: If the tree holds no test case in scope.
        """
        source = raw_refs[0]
        dataset = TimeFDataset(metadata=self.metadata())
        dataset.register_annotations(option_annotations(_ID_PREFIX))
        records = 0
        for definition, index, path in _walk(source.root):
            payload = load_payload(str(path))
            record_id = _record_id(definition, index)
            record = dataset.add_record(
                time_series=series_for(definition.source, path, payload, record_id),
                subject_ids=_subject_ids(definition.source, payload),
                record_id=record_id,
            )
            record.add_annotations(_annotations_for(definition, index, payload, record_id))
            records += 1
        if not records:
            raise TimeFFormatError(f"HEARTS tree at {source.root} holds no test case this connector converts")
        dataset.set_task_stream(TASK_TYPES, lambda: _iter_tasks(source))
        return dataset


CONNECTOR = HeartsConnector
