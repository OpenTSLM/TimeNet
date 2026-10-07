"""Preserve the original CGMacros 1.0.0 recordings independently of downstream benchmarks."""

import asyncio
import csv
from fractions import Fraction
import functools
import hashlib
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import pyarrow as pa
from pydantic import BaseModel, ConfigDict, Field, FiniteFloat

from timenet.composition import BuildContext
from timenet.connectors import BaseConnector
from timenet.dataset import IrregularAxis, Record, RegularAxis, Signal, Source, TimeFDataset
from timenet.errors import TimeFFormatError
from timenet.types import Annotation, TimePoint, TimeSeriesSpec, ureg
from timenet_connectors.download import ensure_archive, find_dir_containing
from timenet_connectors.sources.images import image_signal
from timenet_connectors.time_axes import axis_for_offsets


if TYPE_CHECKING:
    from numpy.typing import ArrayLike, NDArray
    from pandas import DataFrame


URL = "https://physionet-open.s3.amazonaws.com/cgmacros/1.0.0/CGMacros_dateshifted365.zip"
SHA256 = "05c8b0e6f1a2757050aced55ce4bf6ab2ac9b30f2fd8ca193056812d9c621d4d"
CHANNELS = {
    "Libre GL": ("cgmacros_libre_glucose", "mg/dL"),
    "Dexcom GL": ("cgmacros_dexcom_glucose", "mg/dL"),
    "HR": ("cgmacros_heart_rate", "bpm"),
    "Calories (Activity)": ("cgmacros_activity_calories", "kilocalorie"),
    # The published column is METs multiplied by ten; retain its original scale explicitly.
    "METs": ("cgmacros_mets_times_ten", "dimensionless"),
    "Intensity": ("cgmacros_activity_intensity", "dimensionless"),
    "Steps": ("cgmacros_steps", "dimensionless"),
    "RecordIndex": ("cgmacros_record_index", "dimensionless"),
}


class Meal(BaseModel):
    """A source meal event, with missing nutrition values preserved."""

    model_config = ConfigDict(populate_by_name=True, extra="forbid")
    timestamp: str = Field(alias="Timestamp")
    meal_type: str = Field(alias="Meal Type")
    calories: FiniteFloat | None = Field(default=None, alias="Calories")
    carbs: FiniteFloat | None = Field(default=None, alias="Carbs")
    protein: FiniteFloat | None = Field(default=None, alias="Protein")
    fat: FiniteFloat | None = Field(default=None, alias="Fat")
    fiber: FiniteFloat | None = Field(default=None, alias="Fiber")
    sugar: FiniteFloat | None = Field(default=None, alias="Sugar")
    amount_consumed: FiniteFloat | None = Field(default=None, alias="Amount Consumed")
    image_path: str | None = Field(default=None, alias="Image path")


class CGMacrosConnector(BaseConnector[Path]):
    """Read original participant CSVs, supplementary tables, and meal photographs."""

    values_backend = "zarr"

    def download(self, cache_dir: Path) -> list[Path]:  # noqa: PLR6301
        """Fetch the checksummed original release.

        Returns:
            The extracted dataset root.
        """
        root = asyncio.run(ensure_archive(URL, cache_dir, sha256=SHA256))
        return [find_dir_containing(root, "bio.csv")]

    def convert(self, raw_refs: list[Path], context: BuildContext | None = None) -> TimeFDataset:  # noqa: ARG002
        """Build taskless participant and photograph records with lazy values.

        Returns:
            The standalone original dataset.

        Raises:
            TimeFFormatError: If no participant CSVs are present.
        """
        root = raw_refs[0]
        paths = sorted(root.glob("CGMacros-*/CGMacros-*.csv"))
        if not paths:
            raise TimeFFormatError(f"CGMacros contains no participant recordings under {root}")
        dataset = TimeFDataset(metadata=self.metadata())
        supplementary = _supplementary(root)
        for path in paths:
            dataset.add_record(record=_record(path, supplementary.get(int(path.stem.rsplit("-", 1)[1]), {})))
            for photo in sorted((path.parent / "photos").glob("*.jpg")):
                dataset.add_record(record=_photo_record(photo, path.stem))
        return dataset


def _photo_record(photo: Path, subject_id: str) -> Record:
    """Build a photograph record for a participant.

    Returns:
        The photograph record.
    """
    record_id = f"cgmacros-photo-{subject_id}-{photo.stem}"
    signal = image_signal(photo, signal_id=f"{record_id}-image", name="photograph")
    return Record(
        record_id=record_id,
        subject_ids=(subject_id,),
        sources=(Source(id=f"{record_id}-source", name="photograph", signals=(signal,)),),
        metadata={
            "subject_id": subject_id,
            "filename": photo.name,
            "sha256": hashlib.sha256(photo.read_bytes()).hexdigest(),
        },
    )


def _timestamps_us(timestamps: "ArrayLike") -> "NDArray[np.int64]":
    """Convert source timestamps to integer microseconds.

    Returns:
        Timestamps in microseconds, preserving the source's local time.
    """
    return np.asarray(timestamps, dtype="datetime64[us]").astype(np.int64)


def _channel_signal(path: Path, column: str, axis: RegularAxis | IrregularAxis, offsets: "NDArray[np.int64]") -> Signal:
    """Build a channel signal with lazy measurement loading.

    Returns:
        The measurement signal.
    """
    spec_type, unit = CHANNELS[column]
    return Signal.from_loader(
        id=f"cgmacros-{path.stem}-{spec_type}",
        name=column,
        spec=TimeSeriesSpec(
            spec_type=spec_type,
            name=column,
            unit_value=ureg.Unit(unit),
            dtype="float64",
            nullable=True,
        ),
        time_axis=axis,
        n_values=len(offsets),
        time_offsets_loader=(lambda: pa.array(offsets)) if isinstance(axis, IrregularAxis) else None,
        loader=lambda: pa.array(_measurements(path)[column], type=pa.float64(), from_pandas=True),
        source_id=path.stem,
    )


def _read_meals(frame: "DataFrame") -> list[Meal]:
    """Parse meal events, preserving missing nutrition values.

    Returns:
        The source meal events.
    """
    meals = []
    for row in frame.loc[frame["Meal Type"] != ""].to_dict("records"):
        for key, value in row.items():
            if isinstance(value, str) and not value:
                row[key] = None
        meals.append(Meal.model_validate(row))
    return meals


def _record(path: Path, supplementary: dict[str, Any]) -> Record:
    from pandas import read_csv  # noqa: PLC0415 - connector dependency

    # Only timing and event metadata are read here. Measurement columns stay behind lazy loaders.
    frame = read_csv(
        path,
        usecols=lambda column: column.strip() not in CHANNELS and not column.startswith("Unnamed:"),
        keep_default_na=False,
    )
    frame.columns = frame.columns.str.strip()
    columns = read_csv(path, nrows=0).columns
    times = _timestamps_us(frame["Timestamp"].to_numpy())
    offsets = times - times[0]
    axis = axis_for_offsets(offsets, period_us=Fraction(60_000_000))
    record_id = f"cgmacros-{path.stem}"
    signals = tuple(_channel_signal(path, column, axis, offsets) for column in CHANNELS if column in columns)
    meals = _read_meals(frame)
    record = Record(
        record_id=record_id,
        subject_ids=(path.stem,),
        sources=(Source(id=f"{record_id}-sensors", name="sensors", signals=signals),),
        metadata={
            "subject_id": path.stem,
            "recording_start_local": str(frame["Timestamp"].iloc[0]),
            "supplementary": supplementary,
            "meals": [meal.model_dump(mode="json", by_alias=True) for meal in meals],
        },
    )
    for meal in meals:
        offset = int(_timestamps_us([meal.timestamp])[0]) - int(times[0])
        record.annotate(Annotation(key="meal_type", value=meal.meal_type, span=TimePoint.micros(offset)))
    return record


@functools.lru_cache(maxsize=2)
def _measurements(path: Path) -> Any:
    from pandas import read_csv  # noqa: PLC0415 - connector dependency

    return read_csv(path, usecols=lambda column: column in CHANNELS, dtype="float64")


def _supplementary(root: Path) -> dict[int, dict[str, Any]]:
    result: dict[int, dict[str, Any]] = {}
    for name in ("bio", "microbes", "gut_health_test"):
        with (root / f"{name}.csv").open(encoding="utf-8-sig", newline="") as handle:
            reader = csv.reader(handle)
            columns = next(reader)
            # Keep columns and cells separately: bio.csv repeats the header 'Time (t)'.
            for cells in reader:
                result.setdefault(int(cells[0]), {})[name] = {"columns": columns[1:], "values": cells[1:]}
    return result


CONNECTOR = CGMacrosConnector
