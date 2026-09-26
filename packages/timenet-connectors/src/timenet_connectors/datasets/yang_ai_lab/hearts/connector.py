"""Convert all thirty released HEARTS task directories into TimeF."""

from collections.abc import Iterator
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from typing import Any

from timenet.connectors import BaseConnector
from timenet.dataset import Record, Signal, Source, TimeFDataset
from timenet.dataset.axis import RegularAxis
from timenet.errors import TimeFFormatError, TimeNetDownloadError
from timenet.types import Annotation, ForecastingTask, Task, TimeSeriesSpec, TSEditingTask, ureg
from timenet_connectors.datasets.yang_ai_lab.hearts.pickles import load_payload
from timenet_connectors.datasets.yang_ai_lab.hearts.series import source_for
from timenet_connectors.datasets.yang_ai_lab.hearts.tasks import (
    TASK_DEFINITIONS,
    TaskDefinition,
    annotation_value,
    build_task,
    name_vocabulary,
    option_annotations,
)
from timenet_connectors.sources.huggingface_hub import hub_snapshot


HF_REPO = "yang-ai-lab/HEARTS"
HF_REVISION = "7c18df521ae36cbc6b61e17782f1ac08dc378ea1"
"""A commit, never a branch: a branch would let the same connector code read different bytes."""

_ID_PREFIX = "hearts"
"""Every id this connector writes is built on this, record ids and vocabulary ids alike."""

_PROVENANCE = f"{_ID_PREFIX}:provenance"
_AGENT_INPUT = f"{_ID_PREFIX}:agent_input"
_ANSWER_KEY = "GT"
_TARGET_LENGTH = 30
_TARGET_SPEC = TimeSeriesSpec(
    spec_type="cgm_target",
    name="Held-out interstitial glucose",
    unit_value=ureg.milligram / ureg.deciliter,
    dtype="float64",
)

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

    Args:
        definition: The task directory's definition.
        index: The test-case index.

    Returns:
        The record id, built from the release's own directory and file names.
    """
    return f"{_ID_PREFIX}-{definition.source}-{definition.task}-{index:02d}"


def _input_annotations(
    definition: TaskDefinition, record: Record, vocabularies: dict[str, Annotation]
) -> tuple[Annotation, ...]:
    """Collect the annotation occurrences the reference harness shows its agent, the vocabulary first.

    A task names occurrences, not content, so the vocabulary comes from the dataset's own
    attachment and each agent input from the record's. The record attaches its agent inputs in
    ``input_keys`` order, so they are read back in that order.

    Args:
        definition: The task directory's definition.
        record: The record the test case became, with its annotations attached.
        vocabularies: The dataset's answer-vocabulary occurrences, keyed by content id.

    Returns:
        The occurrences: the directory's vocabulary when it has one, then the agent inputs.
    """
    inputs = tuple(annotation for annotation in record.annotations if annotation.source == _AGENT_INPUT)
    if definition.options:
        return (vocabularies[name_vocabulary(_ID_PREFIX, definition.task)], *inputs)
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
            Annotation(key=key, value=annotation_value(payload[key]), source=_AGENT_INPUT, id=f"{record_id}-{key}")
        )
    return annotations


def _target_record(case_id: str, answer: Any, origin: Any) -> Record:
    """Keep a held-out CGM sequence separate from the task's input record.

    Args:
        case_id: Stable identifier of the source test case.
        answer: The thirty held-out glucose values.
        origin: The source time of the first held-out reading.

    Returns:
        A record whose signal contains only the answer sequence.

    Raises:
        TimeFFormatError: If the answer is not thirty numeric readings.
    """
    if (
        not isinstance(answer, list)
        or len(answer) != _TARGET_LENGTH
        or any(not isinstance(value, int | float) or isinstance(value, bool) for value in answer)
    ):
        raise TimeFFormatError(f"HEARTS {case_id} needs thirty numeric CGM target readings")
    target_id = f"{case_id}-target"
    signal = Signal.from_values(
        [float(value) for value in answer],
        spec=_TARGET_SPEC,
        name="held_out_cgm",
        time_axis=RegularAxis(period_us=Fraction(60_000_000)),
        id=f"{target_id}-cgm",
        source_id=target_id,
    )
    record = Record(
        record_id=target_id,
        sources=(Source(id=f"{target_id}-source", name="HEARTS held-out answer", signals=(signal,)),),
    )
    record.annotate(Annotation(key="target_time_origin", value=annotation_value(origin), id=f"{target_id}-origin"))
    return record


class HeartsConnector(BaseConnector[HeartsSource]):
    """Connector for the HEARTS released test cases (Hub repo ``yang-ai-lab/HEARTS``)."""

    values_backend = "zarr"

    def download(self, cache_dir: Path) -> list[HeartsSource]:  # noqa: PLR6301 (the base declares it)
        """Fetch all released test cases at the pinned revision.

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
        root = hub_snapshot(
            HF_REPO,
            HF_REVISION,
            cache_dir,
            tuple(f"{definition.directory}/*.pkl" for definition in TASK_DEFINITIONS),
        )
        _check_case_counts(root)
        return [HeartsSource(root=root)]

    def convert(self, raw_refs: list[HeartsSource]) -> TimeFDataset:
        """Build the records and tasks from each test case in one pass.

        Args:
            raw_refs: The single-element list from :meth:`download`.

        Returns:
            The dataset with input and held-out target records and all tasks.

        Raises:
            TimeFFormatError: If the tree holds no test case in scope.
        """
        source = raw_refs[0]
        dataset = TimeFDataset(metadata=self.metadata())
        vocabularies = {annotation.id: dataset.annotate(annotation) for annotation in option_annotations(_ID_PREFIX)}
        tasks: list[Task] = []
        for definition, index, path in _walk(source.root):
            payload = load_payload(str(path))
            record_id = _record_id(definition, index)
            if definition.no_record:
                inputs = [vocabularies[name_vocabulary(_ID_PREFIX, definition.task)]]
                for key in definition.input_keys:
                    inputs.append(
                        dataset.annotate(
                            Annotation(
                                key=key,
                                value=annotation_value(payload[key]),
                                source=_AGENT_INPUT,
                                id=f"{record_id}-{key}",
                            )
                        )
                    )
                record = None
            else:
                record = Record(
                    record_id=record_id,
                    sources=(
                        source_for(
                            definition.source,
                            path,
                            payload,
                            record_id,
                            series_keys=definition.series_keys,
                            images=definition.images,
                        ),
                    ),
                    subject_ids=_subject_ids(definition.source, payload),
                )
                dataset.add_record(record=record)
                record.add_annotations(_annotations_for(definition, index, payload, record_id))
                inputs = list(_input_annotations(definition, record, vocabularies))
            target_record = None
            if definition.task_type in {ForecastingTask, TSEditingTask}:
                origin_key = "meal_time" if definition.task_type is ForecastingTask else "mask_start"
                target_record = _target_record(record_id, payload[_ANSWER_KEY], payload[origin_key])
                dataset.add_record(record=target_record)
            tasks.append(
                build_task(
                    definition,
                    record,
                    payload[_ANSWER_KEY],
                    tuple(inputs),
                    _ID_PREFIX,
                    record_id,
                    target_record=target_record,
                )
            )
        if not tasks:
            raise TimeFFormatError(f"HEARTS tree at {source.root} holds no test case this connector converts")
        dataset.add_tasks(tasks=tasks)
        return dataset


CONNECTOR = HeartsConnector
