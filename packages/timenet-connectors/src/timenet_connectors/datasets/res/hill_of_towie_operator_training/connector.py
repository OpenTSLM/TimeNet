"""Convert the selected Hill of Towie operator-training splits into TimeF."""

from collections.abc import Mapping
from datetime import datetime, timedelta
from fractions import Fraction
from functools import lru_cache, partial
import gzip
import json
from pathlib import Path
from typing import TypedDict

import pyarrow as pa
from pydantic import JsonValue

from timenet.connectors import BaseConnector
from timenet.dataset import Record, Signal, Source, TimeFDataset
from timenet.dataset.axis import RegularAxis
from timenet.errors import TimeNetConnectorError
from timenet.types import (
    AnswerTask,
    ClassificationTask,
    LabelSet,
    MissingReason,
    MissingValues,
    Split,
    TimeOrigin,
    TimeSeriesSpec,
    ureg,
)
from timenet_connectors.download import Artifact, download_files


_SOURCE = "https://raw.githubusercontent.com/ZMakschZ/TimeNet/e47bbbc7316e3eaf752bee265e133c5fdc8d4ed2"
_SHA256 = {
    Split.TRAIN: "7873ab4dd9370e81a62ef8c9addff7c0153aa6c6c35e1ad2f4fcea8540672475",
    Split.VALIDATION: "524a45948f49a0a0f419b3bf36b7ec249441250fa983b72515bb097ec89647e5",
    Split.TEST: "4095235e1fa0d99644c7fa5c5c95d889ca82bc93392bc3a3534461f4c2df62d3",
}
_EXPECTED_COUNTS = {Split.TRAIN: 9709, Split.VALIDATION: 172, Split.TEST: 371}
_STEP = timedelta(minutes=10)
_AXIS = RegularAxis.from_rate_hz(Fraction(1, 600))
_MISSING = MissingValues(reason=MissingReason.NOT_RECORDED, description="Reading absent from the model input.")


class _Series(TypedDict):
    name: str
    values: list[float | None]


class _Example(TypedDict):
    series: list[_Series]
    prompt: str
    options: list[str]
    answer: str
    category: str


class _Row(TypedDict):
    stage: str
    cutoff: str
    turbine: str
    steps: int
    label: str
    example: _Example


@lru_cache(maxsize=26)
def _spec(media_name: str) -> TimeSeriesSpec:
    key, separator, display = media_name.partition(": ")
    name, unit_separator, unit = display.removesuffix(")").rpartition(" (")
    if not (key and separator and name and unit_separator and unit and display.endswith(")")):
        raise TimeNetConnectorError(f"Invalid SCADA series name: {media_name!r}")
    return TimeSeriesSpec(spec_type=key, name=name, unit_value=ureg.Unit(unit), dtype="float64", missing=_MISSING)


def _signal(series: _Series, steps: int) -> Signal:
    values = series["values"]
    if len(values) != steps:
        raise TimeNetConnectorError(f"{series['name']}: expected {steps} readings, got {len(values)}")
    return Signal.from_loader(
        spec=_spec(series["name"]),
        name=series["name"],
        time_axis=_AXIS,
        n_values=steps,
        null_count=values.count(None),
        loader=partial(pa.array, values, type=pa.float64()),
    )


def _task(dataset: TimeFDataset, row: _Row, split: Split) -> AnswerTask | ClassificationTask:
    example = row["example"]
    steps = row["steps"]
    cutoff = datetime.fromisoformat(row["cutoff"])
    if steps not in {36, 1008} or cutoff.utcoffset() != timedelta(0) or not example["series"]:
        raise TimeNetConnectorError("Expected a six-hour or seven-day SCADA window with a UTC cutoff and signals.")
    channels: list[JsonValue] = [series["name"] for series in example["series"]]
    record = dataset.add_record(
        record=Record(
            start_time=TimeOrigin(cutoff - (steps - 1) * _STEP),
            sources=(Source(name="SCADA", signals=tuple(_signal(series, steps) for series in example["series"])),),
            metadata={"turbine_id": row["turbine"], "cutoff_utc": row["cutoff"], "channel_order": channels},
        )
    )
    options = example["options"]
    metadata = {"category": example["category"], "stage": row["stage"], "label": row["label"]}
    if options and example["answer"] == row["label"]:
        return ClassificationTask(
            inputs=(record,),
            targets=(example["answer"],),
            prompt=example["prompt"],
            split=split,
            metadata=metadata,
            label_set=LabelSet(options, ordered=True),
        )
    answer_options: list[JsonValue] = list(options)
    return AnswerTask(
        inputs=(record,),
        targets=(example["answer"],),
        prompt=example["prompt"],
        split=split,
        metadata={**metadata, "options": answer_options},
    )


class HillOfTowieOperatorTrainingConnector(BaseConnector[Mapping[Split, Path]]):
    """Load the exact split exports for the audited operator-response run."""

    # ruff: ignore[no-self-use] - override for fixed release files
    async def download_async(self, cache_dir: Path) -> Mapping[Split, Path]:
        """Download and verify the three pinned split archives."""
        artifacts = [
            Artifact(f"{_SOURCE}/{split}.jsonl.gz", cache_dir / f"{split}.jsonl.gz", sha256=digest)
            for split, digest in _SHA256.items()
        ]
        return dict(zip(_SHA256, await download_files(artifacts), strict=True))

    def convert(self, raw: Mapping[Split, Path], /) -> TimeFDataset:
        """Preserve the selected windows and supervision in their released order."""
        dataset = TimeFDataset(metadata=self.metadata())
        for split, expected in _EXPECTED_COUNTS.items():
            line_number = 0
            try:
                with gzip.open(raw[split], "rt", encoding="utf-8") as stream:
                    for line_number, line in enumerate(stream, 1):
                        if line_number > expected:
                            raise TimeNetConnectorError(f"Hill of Towie {split} split has more than {expected} rows.")
                        row: _Row = json.loads(line)
                        dataset.add_task(task=_task(dataset, row, split))
            except (OSError, EOFError, KeyError, TypeError, ValueError) as exc:
                raise TimeNetConnectorError(
                    f"Could not read Hill of Towie {split} split near row {line_number}: {exc}"
                ) from exc
            if line_number != expected:
                raise TimeNetConnectorError(f"Hill of Towie {split} split has {line_number} rows, expected {expected}.")
        return dataset


CONNECTOR = HillOfTowieOperatorTrainingConnector
