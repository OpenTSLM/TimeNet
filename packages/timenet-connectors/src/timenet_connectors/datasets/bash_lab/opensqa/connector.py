"""Convert OpenSQA into shared IMU records with caption and QA tasks."""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from fractions import Fraction
import hashlib
import json
from pathlib import Path

import pyarrow as pa

from timenet.composition import BuildContext
from timenet.connectors import BaseConnector
from timenet.dataset import Record, RegularAxis, Signal, Source, TimeFDataset
from timenet.errors import TimeFFormatError, TimeFValidationError, TimeNetDownloadError
from timenet.types import Annotation, AnswerTask, InputModality, Split, TimeSeriesSpec
from timenet_connectors.datasets.bash_lab.opensqa.parser import (
    OpenSqaFile,
    OpenSqaSource,
    ParsedRow,
    iter_rows,
    qa_pairs,
    row_at,
)
from timenet_connectors.datasets.bash_lab.opensqa.release import FILES, REPOSITORY, REVISION
from timenet_connectors.sources.huggingface_hub import hub_snapshot


_AXIS = RegularAxis.from_rate_hz(Fraction(10))
_ACCELEROMETER_SPEC = TimeSeriesSpec(
    spec_type="opensqa_accelerometer",
    name="OpenSQA accelerometer axis",
    unit_value=None,
    dtype="float64",
)
_GYROSCOPE_SPEC = TimeSeriesSpec(
    spec_type="opensqa_gyroscope",
    name="OpenSQA gyroscope axis",
    unit_value=None,
    dtype="float64",
)
_SENSORS = (
    ("gyroscope", _GYROSCOPE_SPEC),
    ("accelerometer", _ACCELEROMETER_SPEC),
)
_AXES = ("x", "y", "z")


class OpenSqaConnector(BaseConnector[OpenSqaSource]):
    """Connector for the pinned BASH Lab OpenSQA release."""

    def download(self, cache_dir: Path) -> list[OpenSqaSource]:  # noqa: PLR6301 - BaseConnector override
        """Fetch the nine training JSONL files at the pinned revision.

        Args:
            cache_dir: Hugging Face cache directory.

        Returns:
            One source containing the validated release paths.

        Raises:
            TimeNetDownloadError: If the snapshot lacks an expected file.
        """
        patterns = tuple(source.path for source in FILES)
        root = hub_snapshot(REPOSITORY, REVISION, cache_dir, patterns)
        files = tuple(OpenSqaFile(source, root / source.path) for source in FILES)
        missing = [source.release.path for source in files if not source.path.is_file()]
        if missing:
            raise TimeNetDownloadError(f"{REPOSITORY}@{REVISION} is missing OpenSQA files: {', '.join(missing)}")
        return [OpenSqaSource(files=files, revision=REVISION)]

    def convert(self, raw_refs: list[OpenSqaSource], context: BuildContext | None = None) -> TimeFDataset:  # noqa: ARG002
        """Build shared lazy sensor records and a rereadable AnswerTask stream.

        Args:
            raw_refs: The single pinned source returned by :meth:`download`.

        Returns:
            The populated OpenSQA dataset.

        Raises:
            TimeFValidationError: If the connector receives anything other than one source.
            TimeFFormatError: If source row counts or paired sensor windows disagree.
        """
        if len(raw_refs) != 1:
            raise TimeFValidationError(f"OpenSQA convert expects one source, got {len(raw_refs)}")
        source = raw_refs[0]
        if source.revision != REVISION:
            raise TimeFValidationError(f"OpenSQA convert expects revision {REVISION}, got {source.revision}")
        dataset = TimeFDataset(metadata=self.metadata())
        records: dict[tuple[str, int], Record] = {}
        signatures: dict[tuple[str, int], bytes] = {}
        for source_file in source.files:
            rows = 0
            for index, offset, row in iter_rows(source_file):
                key = (source_file.release.corpus, index)
                signature = _sensor_signature(row)
                if key in records:
                    if signatures[key] != signature:
                        raise TimeFFormatError(
                            f"{source_file.path}:{index + 1}: sensor window differs from the paired "
                            f"OpenSQA version for {key[0]} row {index}"
                        )
                else:
                    record = _record(source_file, index, offset)
                    records[key] = record
                    signatures[key] = signature
                    dataset.add_record(record=record)
                rows += 1
            if rows != source_file.release.rows:
                raise TimeFFormatError(f"{source_file.path}: expected {source_file.release.rows} rows, found {rows}")
        dataset.set_task_stream(
            [AnswerTask],
            lambda: _tasks(source.files, records),
        )
        return dataset


def _record(source: OpenSqaFile, index: int, offset: int) -> Record:
    record_id = _record_id(source.release.corpus, index)
    signals = tuple(
        Signal.from_loader(
            id=f"{record_id}-{sensor}-{axis_name}",
            name=f"{sensor}_{axis_name}",
            spec=spec,
            time_axis=_AXIS,
            n_values=60,
            source_id=record_id,
            loader=lambda path=source.path, offset=offset, index=index, sensor=sensor, axis=axis: _values(
                path,
                offset,
                index,
                sensor,
                axis,
            ),
        )
        for sensor, spec in _SENSORS
        for axis, axis_name in enumerate(_AXES)
    )
    record = Record(
        record_id=record_id,
        sources=(Source(id=f"{record_id}-imu", name=f"{source.release.corpus} IMU", signals=signals),),
        metadata={"corpus": source.release.corpus, "source_row": index},
    )
    record.annotate(
        Annotation(
            id=f"{record_id}-corpus",
            key="source_corpus",
            value=source.release.corpus,
            description="OpenSQA source activity-recognition corpus.",
        )
    )
    return record


def _values(path: Path, offset: int, index: int, sensor: str, axis: int) -> pa.Array:
    row = row_at(path, offset, index)
    samples = row.gyroscope if sensor == "gyroscope" else row.accelerometer
    return pa.array((sample[axis] for sample in samples), type=pa.float64())


def _tasks(
    files: tuple[OpenSqaFile, ...],
    records: Mapping[tuple[str, int], Record],
) -> Iterator[AnswerTask]:
    for source in files:
        qa_count = 0
        row_count = 0
        for index, _offset, row in iter_rows(source):
            record = records.get((source.release.corpus, index))
            if record is None:
                raise TimeFFormatError(
                    f"{source.path}:{index + 1}: no converted record for {source.release.corpus} row {index}"
                )
            task_root = f"opensqa-{source.release.corpus}-{source.release.version}-{index:06d}"
            common_metadata: dict[str, object] = {
                "corpus": source.release.corpus,
                "source_version": source.release.version,
                "source_row": index,
                "activity_summary": row.summary,
            }
            yield AnswerTask(
                id=f"{task_root}-caption",
                inputs=(record,),
                targets=(row.caption,),
                split=Split.TRAIN,
                input_modalities=frozenset({InputModality.TIME_SERIES}),
                metadata={**common_metadata, "task_kind": "caption"},
            )
            pairs = qa_pairs(row.assistant, maximum=source.release.max_qa_pairs)
            for pair_index, (question, answer) in enumerate(pairs, start=1):
                yield AnswerTask(
                    id=f"{task_root}-qa-{pair_index:02d}",
                    inputs=(record,),
                    prompt=question,
                    targets=(answer,),
                    split=Split.TRAIN,
                    input_modalities=frozenset({InputModality.TIME_SERIES, InputModality.TEXT}),
                    metadata={
                        **common_metadata,
                        "task_kind": "question_answer",
                        "qa_index": pair_index,
                    },
                )
            qa_count += len(pairs)
            row_count += 1
        if row_count != source.release.rows:
            raise TimeFFormatError(f"{source.path}: expected {source.release.rows} task rows, found {row_count}")
        if qa_count != source.release.qa_tasks:
            raise TimeFFormatError(
                f"{source.path}: expected {source.release.qa_tasks} complete QA pairs, found {qa_count}"
            )


def _record_id(corpus: str, index: int) -> str:
    return f"opensqa-{corpus}-{index:06d}"


def _sensor_signature(row: ParsedRow) -> bytes:
    payload = json.dumps(
        (row.gyroscope, row.accelerometer),
        ensure_ascii=True,
        separators=(",", ":"),
    ).encode()
    return hashlib.sha256(payload).digest()


CONNECTOR = OpenSqaConnector
