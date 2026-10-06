"""Convert the pinned ChatTS training JSONL release into TimeF.

Each row pairs ordered, synthetic time series with an input prompt and output
text. The ``<ts><ts/>`` markers in the prompt refer to the series in array
order. The release has no timestamps or physical units, so the signals use an
ordinal axis and an unknown value unit.
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from functools import lru_cache
from pathlib import Path
from typing import Annotated, Self

import pyarrow as pa
from pydantic import BaseModel, ConfigDict, Field, model_validator
from pydantic.dataclasses import dataclass

from timenet.composition import BuildContext
from timenet.connectors import BaseConnector
from timenet.dataset import OrdinalAxis, Record, Signal, Source, TimeFDataset
from timenet.errors import TimeFFormatError, TimeNetDownloadError
from timenet.types import Annotation, AnswerTask, Split, TimeSeriesSpec
from timenet_connectors.sources.huggingface_hub import hub_snapshot


REPO = "ChatTSRepo/ChatTS-Training-Dataset"
REVISION = "1e2ae56ae0b858d8f7b45ab935ed4346f98df83d"
CONFIGS = ("align_256", "align_random", "dev", "ift", "sft")
MARKER = "<ts><ts/>"
SPEC = TimeSeriesSpec(spec_type="chatts_series", name="ChatTS synthetic series", unit_value=None, dtype="float64")
AXIS = OrdinalAxis()


@dataclass(frozen=True, slots=True)
class ChatTSFile:
    """One source config's pinned JSONL file."""

    config: str
    path: Path


_Series = Annotated[tuple[float, ...], Field(min_length=1)]


class _Row(BaseModel):
    """One decoded ChatTS input, output, and ordered series."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    prompt: str = Field(alias="input")
    answer: str = Field(alias="output")
    series: Annotated[tuple[_Series, ...], Field(alias="timeseries", min_length=1)]

    @model_validator(mode="after")
    def _matching_markers(self) -> Self:
        markers = self.prompt.count(MARKER)
        if markers != len(self.series):
            raise ValueError(f"input has {markers} time-series markers for {len(self.series)} series")
        return self


def _parse_row(line: bytes) -> _Row:
    """Decode and validate one source line.

    Returns:
        The input text, output text, and series in source order.
    """
    return _Row.model_validate_json(line)


def _rows(source: ChatTSFile) -> Iterator[tuple[int, int, _Row]]:
    """Yield row index, byte offset, and parsed row without retaining earlier rows."""
    with source.path.open("rb") as handle:
        index = 0
        while True:
            offset = handle.tell()
            line = handle.readline()
            if not line:
                break
            yield index, offset, _parse_row(line)
            index += 1


@lru_cache(maxsize=1)
def _row_at(path: Path, offset: int) -> _Row:
    """Read one row for adjacent signal loaders without holding the full release.

    Returns:
        The parsed row at the byte offset.

    Raises:
        TimeFFormatError: If the offset has no valid ChatTS row.
    """
    with path.open("rb") as handle:
        handle.seek(offset)
        line = handle.readline()
    if not line:
        raise TimeFFormatError(f"{path}: no ChatTS row at byte offset {offset}")
    # The byte offset identifies the row even when the file has no source row ids.
    return _parse_row(line)


def _values(path: Path, offset: int, row_index: int, signal_index: int) -> pa.Array:
    """Load one signal's unmodified values as float64.

    Returns:
        The source values as an Arrow array.

    Raises:
        TimeFFormatError: If the source values cannot form a non-null float64 array.
    """
    values = _row_at(path, offset).series[signal_index]
    try:
        array = pa.array(values, type=pa.float64())
    except (pa.ArrowException, TypeError, ValueError) as exc:
        raise TimeFFormatError(f"{path}:{row_index + 1}: invalid values for signal {signal_index}") from exc
    if array.null_count:
        raise TimeFFormatError(f"{path}:{row_index + 1}: null values for signal {signal_index}")
    return array


def _record_id(config: str, index: int) -> str:
    """Name a source row stably across builds.

    Returns:
        The config and row index joined in one stable identifier.
    """
    return f"chatts-{config}-{index:06d}"


def _tasks(records: Sequence[Record], files: Sequence[ChatTSFile]) -> Iterator[AnswerTask]:
    """Reread source text and yield one task for each record in source order.

    Yields:
        An answer task for each source row.

    Raises:
        TimeFFormatError: If a source row no longer matches its record.
    """
    source_rows = ((source.config, index, row) for source in files for index, _offset, row in _rows(source))
    count = 0
    for config, index, row in source_rows:
        if count >= len(records):
            raise TimeFFormatError(f"{config} row {index}: source has more rows than the converted dataset")
        record = records[count]
        record_id = _record_id(config, index)
        if record.id != record_id:
            raise TimeFFormatError(f"{config} row {index}: expected record {record_id}, got {record.id}")
        yield AnswerTask(
            id=f"{record_id}-answer",
            inputs=(record,),
            prompt=row.prompt,
            targets=(row.answer,),
            split=Split.VALIDATION if config == "dev" else Split.TRAIN,
        )
        count += 1
    if count != len(records):
        raise TimeFFormatError(f"ChatTS source has {count} rows, but the converted dataset has {len(records)} records")


class ChatTSConnector(BaseConnector[ChatTSFile]):
    """Connector for the five configs of the ChatTS training release."""

    def download(self, cache_dir: Path) -> list[ChatTSFile]:  # noqa: PLR6301 - BaseConnector override
        """Fetch the five original JSONL files at the pinned source revision.

        Args:
            cache_dir: Hugging Face cache directory.

        Returns:
            One file reference per config, in stable order.

        Raises:
            TimeNetDownloadError: If a source config file is absent.
        """
        patterns = tuple(f"{config}/train.jsonl" for config in CONFIGS)
        root = hub_snapshot(REPO, REVISION, cache_dir, patterns)
        files = [ChatTSFile(config, root / f"{config}/train.jsonl") for config in CONFIGS]
        missing = [source.config for source in files if not source.path.is_file()]
        if missing:
            raise TimeNetDownloadError(f"{REPO}@{REVISION} is missing ChatTS configs: {', '.join(missing)}")
        return files

    def convert(self, raw_refs: list[ChatTSFile], context: BuildContext | None = None) -> TimeFDataset:  # noqa: ARG002
        """Build lazy ordinal records and a rereadable text-answer task stream.

        Args:
            raw_refs: The five pinned source files from :meth:`download`.

        Returns:
            The populated dataset.

        Raises:
            TimeFFormatError: If the required source configs are absent or out of order.
        """
        if tuple(source.config for source in raw_refs) != CONFIGS:
            raise TimeFFormatError(f"ChatTS requires config files in this order: {CONFIGS!r}")
        dataset = TimeFDataset(metadata=self.metadata())
        for source in raw_refs:
            for index, offset, row in _rows(source):
                record_id = _record_id(source.config, index)
                signals = tuple(
                    Signal.from_loader(
                        id=f"{record_id}-s{signal_index:02d}",
                        name=f"s{signal_index:02d}",
                        spec=SPEC,
                        time_axis=AXIS,
                        n_values=len(values),
                        source_id=record_id,
                        loader=lambda path=source.path, offset=offset, row_index=index, signal_index=signal_index: (
                            _values(path, offset, row_index, signal_index)
                        ),
                    )
                    for signal_index, values in enumerate(row.series)
                )
                record = Record(
                    record_id=record_id,
                    sources=(Source(id=f"{record_id}-source", name="ChatTS synthetic series", signals=signals),),
                )
                dataset.add_record(record=record)
                record.annotate(Annotation(key="source_config", value=source.config))
        dataset.set_task_stream([AnswerTask], lambda: _tasks(dataset.records, raw_refs))
        return dataset


CONNECTOR = ChatTSConnector
