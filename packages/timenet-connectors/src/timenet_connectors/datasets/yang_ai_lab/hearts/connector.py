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
"""Pinned source revision for reproducible builds."""

_ID_PREFIX = "hearts"
"""Prefix for record and annotation IDs."""

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
    """Path to the downloaded test cases."""

    root: Path
    """The directory holding the ``<corpus>/<task>/<index>.pkl`` tree."""


def _walk(root: Path) -> Iterator[tuple[TaskDefinition, int, Path]]:
    """Yield test cases in task-definition order, sorted by case index.

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
    """Check that each directory contains the expected number of cases.

    Raises:
        TimeNetDownloadError: If a task directory has an unexpected case count.
    """
    for definition in TASK_DEFINITIONS:
        found = len(list((root / definition.source / definition.task).glob("*.pkl")))
        if found != definition.n_items:
            raise TimeNetDownloadError(
                f"HEARTS {definition.directory} holds {found} test cases at revision {HF_REVISION}, "
                f"but this connector was written against {definition.n_items}"
            )


def _subject_ids(source: str, payload: dict[str, Any]) -> tuple[str, ...]:
    """Return subject IDs with a corpus prefix.

    Returns:
        The subject ids, empty when the payload names none.
    """
    for key in ("subject_id", "speaker_id"):
        if key in payload:
            return (f"{source}-{payload[key]}",)
    pair = [payload[key] for key in ("A_subject", "B_subject") if key in payload]
    return tuple(f"{source}-{subject}" for subject in pair)


def _record_id(definition: TaskDefinition, index: int) -> str:
    """Build a stable record ID from the source directory and case index.

    Returns:
        The record id, built from the release's own directory and file names.
    """
    return f"{_ID_PREFIX}-{definition.source}-{definition.task}-{index:02d}"


def _input_annotations(
    definition: TaskDefinition, record: Record, vocabularies: dict[str, Annotation]
) -> tuple[Annotation, ...]:
    """Collect attached input annotations, with the answer vocabulary first.

    Returns:
        The vocabulary and model-input annotation occurrences.
    """
    inputs = tuple(annotation for annotation in record.annotations if annotation.source == _AGENT_INPUT)
    if definition.options:
        return (vocabularies[name_vocabulary(_ID_PREFIX, definition.task)], *inputs)
    return inputs


def _annotations_for(
    definition: TaskDefinition, index: int, payload: dict[str, Any], record_id: str
) -> list[Annotation]:
    """Build provenance and model-input annotations.

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
    """Create a separate record for the thirty held-out glucose readings.

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
        """Download all test cases from the pinned revision.

        Returns:
            A single-element list holding the handle to the downloaded tree.

        Raises:
            ImportError: If the Hugging Face Hub dependency is missing.
            TimeNetDownloadError: If a task directory has an unexpected case count.
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
        """Create the dataset's records, annotations, and tasks.

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
