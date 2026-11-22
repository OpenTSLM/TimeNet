"""TimeNet connector for the signed Hydrocron coastal snapshot."""

from collections import defaultdict
from collections.abc import Iterator
import csv
from dataclasses import dataclass
from datetime import datetime, timedelta
import gzip
from hashlib import file_digest
import json
from pathlib import Path
from typing import TypedDict

import numpy as np

from timenet.connectors import BaseConnector
from timenet.dataset import Record, Signal, Source, TimeFDataset
from timenet.errors import TimeFFormatError, TimeNetConnectorError
from timenet.types import (
    Annotation,
    AnswerTask,
    ForecastingTask,
    Split,
    Task,
    TimeInterval,
    TimeOrigin,
    TimeSeriesSpec,
    ureg,
)
from timenet_connectors.download import Artifact, download_files

from .coastal import (
    COASTAL_ENDPOINT,
    FIELDS,
    SEALED_REGIONS,
    Issue,
    Pass,
    astronomical_phase,
    coastal_issues,
    read_pass,
)


PACKAGE = Path(__file__).parent
SNAPSHOT = "coastal_hydrocron_v2_320reaches.csv.gz"
REACH_INDEX = "coastal_reaches_v2.json"
RELEASE_FILES = (SNAPSHOT, REACH_INDEX)
CHECKSUMS = "coastal_checksums_v2.json"
RELEASE_BASE = "https://github.com/wienkers/revir/releases/download/swot-river-slope-v0.2.0"
SEALED_TEST_SPLIT = (2025, SEALED_REGIONS, Split.TEST)
TRAIN_REGIONS = frozenset({1, 6})
VALIDATION_REGIONS = frozenset({2, 7, 8, 9})
FALL_UNIT = ureg.decimeter / ureg.kilometer  # 1 dm/km = 1 m per 10 km.
FEATURES = {
    "wse_m": ureg.meter,
    "wse_u_m": ureg.meter,
    "fall_m_per_10km": FALL_UNIT,
    "fall_u_m_per_10km": FALL_UNIT,
    "fall_z": ureg.dimensionless,
    "reach_q": ureg.dimensionless,
}
SPECS = {
    name: TimeSeriesSpec(spec_type=f"swot_{name}", name=name, unit_value=unit, dtype="float64")
    for name, unit in FEATURES.items()
}
TARGET_NAMES = ("fall_m_per_10km", "reversal_next_pass", "reversal_sig_next_pass", "steepening_gt_0p20")
LOW_GRADIENT_LIMIT = 0.20
TRAIN_YEAR = 2023
VERIFICATION = "the first daily representative whose timestamp is 2 to 21 days after issue"
DAILY_RULE = " Each UTC day's representative is the retained pass with the smallest slope uncertainty."


@dataclass(frozen=True, slots=True)
class Reach:
    """SWORD reach information frozen with the Hydrocron release."""

    reach_id: int
    length_m: float
    lon: float
    lat: float
    year: int
    stratum: str

    @property
    def split(self) -> Split:
        """The supervised split assigned by the fixed regional design."""
        return Split.TRAIN if self.year == TRAIN_YEAR else Split.VALIDATION


class _Inputs(TypedDict):
    inputs: tuple[Record, ...]
    input_annotations: tuple[Annotation, ...]
    scope: TimeInterval
    split: Split


def _verify(path: Path, pin: str) -> None:
    try:
        with path.open("rb") as handle:
            matches = file_digest(handle, "sha256").hexdigest() == pin
    except OSError as exc:
        raise TimeNetConnectorError(f"Could not read {path.name}") from exc
    if not matches:
        raise TimeFFormatError(f"SHA-256 mismatch for {path.name}")


def _pins() -> dict[str, str]:
    return json.loads((PACKAGE / CHECKSUMS).read_text(encoding="utf-8"))


def _read_reaches(lookup: Path) -> dict[int, Reach]:
    try:
        rows = json.loads(lookup.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise TimeNetConnectorError(f"Could not parse {lookup.name}") from exc
    if not isinstance(rows, list):
        raise TimeNetConnectorError(f"Expected a list of reaches in {lookup.name}")
    reaches: dict[int, Reach] = {}
    for index, row in enumerate(rows, start=1):
        try:
            reach = int(row["reach_id"])
            phase = int(row["phase"])
            descriptor = Reach(reach, row["length_m"], row["lon"], row["lat"], TRAIN_YEAR + phase, row["stratum"])
        except (KeyError, TypeError, ValueError) as exc:
            raise TimeNetConnectorError(f"Invalid reach entry {index} in {lookup.name}") from exc
        allowed = TRAIN_REGIONS if phase == 0 else VALIDATION_REGIONS if phase == 1 else frozenset()
        if reach // 10**10 not in allowed or reach in reaches:
            raise TimeNetConnectorError(f"Invalid, duplicate or sealed reach at entry {index} in {lookup.name}")
        reaches[reach] = descriptor
    return reaches


def _read_release(snapshot: Path, lookup: Path) -> Iterator[tuple[Reach, list[Pass]]]:
    pins = _pins()
    _verify(snapshot, pins[SNAPSHOT])
    _verify(lookup, pins[REACH_INDEX])
    reaches = _read_reaches(lookup)
    passes: dict[int, list[Pass]] = defaultdict(list)
    previous: dict[int, datetime] = {}
    try:
        with gzip.open(snapshot, "rt", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            if tuple(reader.fieldnames or ()) != FIELDS:
                raise TimeNetConnectorError(f"Unexpected columns in {snapshot.name}")
            for row in reader:
                try:
                    reach = int(row["reach_id"])
                    visit = read_pass(row, reach) if reach in reaches else None
                except (KeyError, TypeError, ValueError) as exc:
                    raise TimeNetConnectorError(
                        f"Invalid observation at row {reader.line_num} in {snapshot.name}"
                    ) from exc
                if visit is None or (reach in previous and previous[reach] >= visit.time):
                    raise TimeNetConnectorError(
                        f"Invalid, off-split or out-of-order observation at row {reader.line_num} in {snapshot.name}"
                    )
                previous[reach] = visit.time
                passes[reach].append(visit)
    except (OSError, UnicodeError, EOFError, csv.Error) as exc:
        raise TimeNetConnectorError(f"Could not parse {snapshot.name}") from exc
    for reach in reaches.values():
        if reach.reach_id not in passes:
            raise TimeNetConnectorError(f"No observations for reach {reach.reach_id} in {snapshot.name}")
        yield reach, passes[reach.reach_id]


def _utc(at: datetime) -> str:
    return at.isoformat().replace("+00:00", "Z")


def _minutes_utc(at: datetime) -> str:
    return _utc(at.replace(second=0, microsecond=0))


def _evidence(reach: Reach, case: Issue) -> str:
    latest = case.history[-1]
    phase = astronomical_phase(latest.time, reach.lon)
    fall = latest.fall
    regime = (
        "adverse measured surface gradient"
        if fall < 0
        else "low gradient"
        if fall <= LOW_GRADIENT_LIMIT
        else "positive grade"
    )
    minimum = (
        f"The 21-day minimum was {case.minimum.fall:+.3f} m per 10 km at {_utc(case.minimum.time)}."
        if case.minimum is not None
        else "No pass in the 21 days before issue."
    )
    return (
        f"At {_utc(latest.time)} the latest signed downstream fall was {fall:+.3f} ± {latest.fall_u:.3f} "
        f"m per 10 km (z = {latest.z:+.2f}, {regime}). The median of four nonnegative visits "
        f"between {_utc(case.reference.time)} and {_utc([p for p in case.history if p.fall >= 0][-1].time)} "
        f"was {case.baseline:.3f} m per 10 km. {minimum} At the latest pass, "
        f"mean astronomical M2/S2/K1/O1 phases were "
        f"{phase['M2']:.0f}/{phase['S2']:.0f}/{phase['K1']:.0f}/{phase['O1']:.0f} degrees "
        f"with spring index {phase['spring_index']:.2f} (0 neap, 1 spring)."
    )


def record_and_tasks(reach: Reach, case: Issue) -> tuple[Record, tuple[Task, ...]]:
    """Make a history-only Record and four scoped tasks with task-only outcome values."""
    origin = case.history[0].time
    one_microsecond = timedelta(microseconds=1)
    offsets = np.asarray([(p.time - origin) // one_microsecond for p in case.history], dtype=np.int64)
    issue_offset = (case.at - origin) // one_microsecond
    values = (
        [p.wse for p in case.history],
        [p.wse_u for p in case.history],
        [p.fall for p in case.history],
        [p.fall_u for p in case.history],
        [p.z for p in case.history],
        [float(p.reach_q) for p in case.history],
    )
    signals = tuple(
        Signal.from_irregular(np.asarray(data, dtype=np.float64), time_offsets_us=offsets, spec=SPECS[name], name=name)
        for name, data in zip(FEATURES, values, strict=True)
    )
    record = Record(
        start_time=TimeOrigin(origin),
        sources=(Source(name="central_reach", signals=signals),),
        annotations=(
            Annotation("issue_utc", value=_utc(case.at)),
            Annotation("issue_offset_us", value=issue_offset, unit="microsecond"),
            Annotation("length_m", value=reach.length_m, unit="meter"),
            Annotation("longitude", value=reach.lon, unit="degree"),
            Annotation("latitude", value=reach.lat, unit="degree"),
            Annotation("stratum", value=reach.stratum),
            Annotation("baseline_nonneg_m_per_10km", value=case.baseline, unit=FALL_UNIT),
        ),
        metadata={
            "central_reach_id": reach.reach_id,
            "endpoint": COASTAL_ENDPOINT,
            "availability": "acquisition timestamp proxy",
        },
    )
    scope = TimeInterval(start_us=0, end_us=int(offsets[-1]) + 1, time_series=signals)
    shared: _Inputs = {
        "inputs": (record,),
        "input_annotations": record.annotations,
        "scope": scope,
        "split": reach.split,
    }
    schedule = (
        f" First nominal eligible overpass estimated for {_minutes_utc(case.scheduled)} (UTC)."
        if case.scheduled is not None
        else ""
    )
    evidence = _evidence(reach, case)
    forecast = ForecastingTask(
        **shared,
        metadata={
            "task": "forecast_coastal",
            "target_names": list(TARGET_NAMES),
            "verification_utc": _utc(case.verified.time),
        },
        prompt=f"Forecast the fall, reversal flags and steepening at {VERIFICATION}." + DAILY_RULE + schedule,
        targets=(case.verified.fall, int(case.reversal), int(case.reversal_sig), int(case.steepening)),
    )
    reversal = AnswerTask(
        **shared,
        metadata={"task": "forecast_reversal_qa"},
        prompt=f"Will {VERIFICATION} show negative downstream fall at reach {reach.reach_id}?" + DAILY_RULE + schedule,
        targets=("Yes" if case.reversal else "No",),
        rationale=evidence,
    )
    steepening = AnswerTask(
        **shared,
        metadata={"task": "forecast_steepening_qa"},
        prompt=f"Will {VERIFICATION} show a downstream fall at reach {reach.reach_id} more than "
        "0.20 m per 10 km above the four-visit baseline?" + DAILY_RULE + schedule,
        targets=("Yes" if case.steepening else "No",),
        rationale=evidence,
    )
    caption = AnswerTask(
        **shared,
        metadata={"task": "caption_pre_issue_evidence"},
        prompt="Describe the dated pre-issue hydraulic and astronomical evidence.",
        targets=(evidence,),
    )
    return record, (forecast, reversal, steepening, caption)


class SwotRiverSlopeConnector(BaseConnector[list[Path]]):
    """Build coastal reversal and steepening tasks from a versioned Hydrocron snapshot."""

    async def download_async(self, cache_dir: Path) -> list[Path]:  # ruff: ignore[no-self-use]
        """Fetch the pinned snapshot and reach index from the versioned data release."""
        pins = _pins()
        return await download_files(
            Artifact(f"{RELEASE_BASE}/{name}", cache_dir / name, sha256=pins[name]) for name in RELEASE_FILES
        )

    def convert(self, raw_refs: list[Path], /) -> TimeFDataset:
        """Convert the complete verified 2023-2024 snapshot into TimeF forecast issues."""
        if len(raw_refs) != len(RELEASE_FILES):
            raise TimeNetConnectorError("Expected the Hydrocron snapshot and SWORD reach index.")
        snapshot, reach_index = raw_refs
        dataset = TimeFDataset(metadata=self.metadata())
        for reach, passes in _read_release(snapshot, reach_index):
            for case in coastal_issues(passes, reach.year):
                record, tasks = record_and_tasks(reach, case)
                dataset.add_record(record=record)
                dataset.add_tasks(tasks=tasks)
        return dataset


CONNECTOR = SwotRiverSlopeConnector
