"""Build admission-level MIMIC-IV samples from authorized local CSV files.

Each sample is one hospital admission. It contains hourly ICU vital signs on sparse, admission-relative
time axes. Each main discharge summary is an unprompted :class:`~timenet.types.AnswerTask` on its
admission. The connector does not impute a missing vital sign and does not assign a timezone to the
deidentified source timestamps.

Set ``MIMIC_IV_ROOT`` to a downloaded MIMIC-IV 3.1 directory. Set ``MIMIC_IV_NOTE_ROOT`` to a downloaded
MIMIC-IV-Note 2.2 directory. The connector reads the compressed CSV files with DuckDB. It writes a
compact Parquet cache before it constructs lazy TimeF series. The source files stay outside the cache.
"""

from collections import defaultdict
from collections.abc import Iterator, Mapping
from dataclasses import dataclass
import functools
import importlib
import json
import os
from pathlib import Path
from typing import Any, ClassVar

import pyarrow as pa
import pyarrow.parquet as pq

from timenet.connectors import BaseConnector
from timenet.dataset import TimeFDataset, TimeSeries
from timenet.dataset.axis import IrregularAxis
from timenet.errors import TimeFFormatError, TimeFValidationError, TimeNetBuildError
from timenet.types import Annotation, AnswerTask, DataSource, TimeInterval, TimeSeriesSpec, ureg


MIMIC_IV_ROOT_ENV = "MIMIC_IV_ROOT"
MIMIC_IV_NOTE_ROOT_ENV = "MIMIC_IV_NOTE_ROOT"

_EXTRACTION_VERSION = 1
_SOURCE = DataSource(data_source_type="electronic-health-record", name="MIMIC-IV 3.1", provider="PhysioNet")

_HEART_RATE = TimeSeriesSpec(
    spec_type="heart_rate",
    name="Heart Rate",
    unit_value=ureg.bpm,
    data_source=_SOURCE,
)
_BLOOD_PRESSURE = TimeSeriesSpec(
    spec_type="blood_pressure",
    name="Blood Pressure",
    unit_value=ureg.mmHg,
    data_source=_SOURCE,
)
_OXYGEN_SATURATION = TimeSeriesSpec(
    spec_type="oxygen_saturation",
    name="Peripheral Oxygen Saturation",
    unit_value=ureg.percent,
    data_source=_SOURCE,
)
_RESPIRATORY_RATE = TimeSeriesSpec(
    spec_type="respiratory_rate",
    name="Respiratory Rate",
    unit_value=ureg.brpm,
    data_source=_SOURCE,
)
_BODY_TEMPERATURE = TimeSeriesSpec(
    spec_type="body_temperature",
    name="Body Temperature",
    unit_value=ureg.degree_Celsius,
    data_source=_SOURCE,
)

_SERIES_LAYOUT: dict[tuple[str, str], TimeSeriesSpec] = {
    ("blood_pressure", "diastolic"): _BLOOD_PRESSURE,
    ("blood_pressure", "mean"): _BLOOD_PRESSURE,
    ("blood_pressure", "systolic"): _BLOOD_PRESSURE,
    ("body_temperature", "temperature"): _BODY_TEMPERATURE,
    ("heart_rate", "heart-rate"): _HEART_RATE,
    ("oxygen_saturation", "spo2"): _OXYGEN_SATURATION,
    ("respiratory_rate", "respiratory-rate"): _RESPIRATORY_RATE,
}

_ADMISSIONS_COLUMNS = {
    "subject_id": "BIGINT",
    "hadm_id": "BIGINT",
    "admittime": "TIMESTAMP",
    "dischtime": "TIMESTAMP",
    "deathtime": "TIMESTAMP",
    "admission_type": "VARCHAR",
    "admit_provider_id": "VARCHAR",
    "admission_location": "VARCHAR",
    "discharge_location": "VARCHAR",
    "insurance": "VARCHAR",
    "language": "VARCHAR",
    "marital_status": "VARCHAR",
    "race": "VARCHAR",
    "edregtime": "TIMESTAMP",
    "edouttime": "TIMESTAMP",
    "hospital_expire_flag": "SMALLINT",
}
_ICUSTAYS_COLUMNS = {
    "subject_id": "BIGINT",
    "hadm_id": "BIGINT",
    "stay_id": "BIGINT",
    "first_careunit": "VARCHAR",
    "last_careunit": "VARCHAR",
    "intime": "TIMESTAMP",
    "outtime": "TIMESTAMP",
    "los": "DOUBLE",
}
_CHARTEVENTS_COLUMNS = {
    "subject_id": "BIGINT",
    "hadm_id": "BIGINT",
    "stay_id": "BIGINT",
    "caregiver_id": "BIGINT",
    "charttime": "TIMESTAMP",
    "storetime": "TIMESTAMP",
    "itemid": "INTEGER",
    "value": "VARCHAR",
    "valuenum": "DOUBLE",
    "valueuom": "VARCHAR",
    "warning": "SMALLINT",
}
_DISCHARGE_COLUMNS = {
    "note_id": "VARCHAR",
    "subject_id": "BIGINT",
    "hadm_id": "BIGINT",
    "note_type": "VARCHAR",
    "note_seq": "INTEGER",
    "charttime": "TIMESTAMP",
    "storetime": "TIMESTAMP",
    "text": "VARCHAR",
}

_HOURLY_VITALS_SQL = """
CREATE TEMP TABLE hourly_vitals AS
WITH normalized AS (
    SELECT
        ce.hadm_id,
        ce.charttime,
        CASE
            WHEN ce.itemid = 220045 THEN 'heart_rate'
            WHEN ce.itemid IN (220179, 220050, 225309, 220180, 220051, 225310, 220052, 220181, 225312)
                THEN 'blood_pressure'
            WHEN ce.itemid = 220277 THEN 'oxygen_saturation'
            WHEN ce.itemid IN (220210, 224690) THEN 'respiratory_rate'
            WHEN ce.itemid IN (223761, 223762) THEN 'body_temperature'
        END AS spec_type,
        CASE
            WHEN ce.itemid = 220045 THEN 'heart-rate'
            WHEN ce.itemid IN (220179, 220050, 225309) THEN 'systolic'
            WHEN ce.itemid IN (220180, 220051, 225310) THEN 'diastolic'
            WHEN ce.itemid IN (220052, 220181, 225312) THEN 'mean'
            WHEN ce.itemid = 220277 THEN 'spo2'
            WHEN ce.itemid IN (220210, 224690) THEN 'respiratory-rate'
            WHEN ce.itemid IN (223761, 223762) THEN 'temperature'
        END AS channel,
        CASE
            WHEN ce.itemid = 220045 AND ce.valuenum > 0 AND ce.valuenum < 300 THEN ce.valuenum
            WHEN ce.itemid IN (220179, 220050, 225309) AND ce.valuenum > 0 AND ce.valuenum < 400
                THEN ce.valuenum
            WHEN ce.itemid IN (220180, 220051, 225310, 220052, 220181, 225312)
                AND ce.valuenum > 0 AND ce.valuenum < 300 THEN ce.valuenum
            WHEN ce.itemid IN (220210, 224690) AND ce.valuenum > 0 AND ce.valuenum < 70 THEN ce.valuenum
            WHEN ce.itemid = 220277 AND ce.valuenum > 0 AND ce.valuenum <= 100 THEN ce.valuenum
            WHEN ce.itemid = 223761 AND ce.valuenum > 70 AND ce.valuenum < 120
                THEN (ce.valuenum - 32) / 1.8
            WHEN ce.itemid = 223762 AND ce.valuenum > 10 AND ce.valuenum < 50 THEN ce.valuenum
        END AS normalized_value
    FROM chartevents_raw AS ce
    WHERE ce.hadm_id IS NOT NULL
        AND ce.stay_id IS NOT NULL
        AND ce.valuenum IS NOT NULL
        AND ce.itemid IN (
            220045, 220179, 220050, 225309, 220180, 220051, 225310,
            220052, 220181, 225312, 220210, 224690, 220277, 223761, 223762
        )
), charttime_vitals AS (
    SELECT hadm_id, charttime, spec_type, channel, avg(normalized_value) AS normalized_value
    FROM normalized
    WHERE normalized_value IS NOT NULL
    GROUP BY hadm_id, charttime, spec_type, channel
), binned AS (
    SELECT
        vital.hadm_id,
        vital.spec_type,
        vital.channel,
        CAST(floor(date_diff('second', admission.admittime, vital.charttime) / 3600) AS BIGINT) AS hour_index,
        CAST(avg(vital.normalized_value) AS FLOAT) AS value
    FROM charttime_vitals AS vital
    JOIN admissions_raw AS admission USING (hadm_id)
    WHERE vital.charttime >= admission.admittime
        AND vital.charttime < admission.dischtime
    GROUP BY vital.hadm_id, vital.spec_type, vital.channel, hour_index
)
SELECT * FROM binned
"""


@dataclass(frozen=True)
class MimicRawFiles:
    """Paths to the authorized source files."""

    admissions: Path
    icustays: Path
    chartevents: Path
    discharge: Path

    def items(self) -> tuple[tuple[str, Path], ...]:
        """Return stable names and paths for the source fingerprint.

        Returns:
            The source file entries in stable order.
        """
        return (
            ("admissions", self.admissions),
            ("icustays", self.icustays),
            ("chartevents", self.chartevents),
            ("discharge", self.discharge),
        )


@dataclass(frozen=True)
class MimicSource:
    """Authorized source paths and a location for the prepared cache."""

    raw_files: MimicRawFiles
    prepared_dir: Path


@dataclass(frozen=True)
class MimicPreparedSource:
    """Paths to the prepared local Parquet cache."""

    admissions_path: Path
    vitals_path: Path
    notes_path: Path


@dataclass(frozen=True)
class MimicSeriesRef:
    """A lightweight reference to one nested vital-sign row in Parquet."""

    hadm_id: int
    spec_type: str
    channel: str
    n_values: int
    first_offset_us: int
    last_offset_us: int
    path: Path
    row_group: int
    row_index: int


@functools.lru_cache(maxsize=8)
def _read_series_row_group(path: str, row_group: int) -> pa.Table:
    """Read and cache one prepared row group.

    Args:
        path: The prepared vital-sign Parquet path.
        row_group: The row-group index.

    Returns:
        The nested values and time offsets in the row group.
    """
    return pq.ParquetFile(path).read_row_group(row_group, columns=["values", "time_offsets_us"]).combine_chunks()


@dataclass(frozen=True)
class _PreparedListLoader:
    """Load one nested list from a prepared vital-sign row."""

    path: Path
    row_group: int
    row_index: int
    column: str

    def __call__(self) -> pa.Array:
        """Return the primitive values from the selected nested list.

        Returns:
            The Arrow values for one TimeF series.

        Raises:
            TimeFFormatError: If the prepared row contains a null or malformed list.
        """
        table = _read_series_row_group(str(self.path), self.row_group)
        scalar = table.column(self.column)[self.row_index]
        if not scalar.is_valid or not isinstance(scalar, pa.ListScalar | pa.LargeListScalar):
            raise TimeFFormatError(
                f"prepared MIMIC row {self.row_group}:{self.row_index} has an invalid {self.column!r} list"
            )
        return scalar.values


class MimicIvExtVitalsNotesConnector(BaseConnector[MimicSource]):
    """Connector for hourly MIMIC-IV vital signs and MIMIC-IV-Note discharge summaries."""

    REQUIRED_FILES: ClassVar[tuple[tuple[str, str, str], ...]] = (
        ("admissions", MIMIC_IV_ROOT_ENV, "hosp/admissions.csv.gz"),
        ("icustays", MIMIC_IV_ROOT_ENV, "icu/icustays.csv.gz"),
        ("chartevents", MIMIC_IV_ROOT_ENV, "icu/chartevents.csv.gz"),
        ("discharge", MIMIC_IV_NOTE_ROOT_ENV, "note/discharge.csv.gz"),
    )

    def download(self, cache_dir: Path) -> list[MimicSource]:
        """Discover authorized MIMIC CSV files and reserve a local cache path.

        The method never downloads credentialed data. It reads the source roots from the environment
        and validates the required files. :meth:`convert` does the expensive DuckDB scan.

        Args:
            cache_dir: The connector cache directory.

        Returns:
            One handle to the source files and prepared-cache directory.
        """
        return [MimicSource(raw_files=self._raw_files_from_environment(), prepared_dir=cache_dir / "prepared")]

    def convert(self, raw_refs: list[MimicSource]) -> TimeFDataset:
        """Build admission samples with lazy vital-sign values and streamed discharge tasks.

        Args:
            raw_refs: The single source from :meth:`download`.

        Returns:
            The admission-level TimeF dataset.

        Raises:
            TimeFValidationError: If ``raw_refs`` does not contain exactly one source.
            TimeFFormatError: If a prepared vital-sign row has an unknown channel.
        """
        if len(raw_refs) != 1:
            raise TimeFValidationError(f"MIMIC conversion needs one source, got {len(raw_refs)}")
        source = _prepare_cache(raw_refs[0])
        refs_by_admission: dict[int, list[MimicSeriesRef]] = defaultdict(list)
        for ref in _iter_series_refs(source.vitals_path):
            if (ref.spec_type, ref.channel) not in _SERIES_LAYOUT:
                raise TimeFFormatError(f"prepared MIMIC data has unknown series {ref.spec_type!r}/{ref.channel!r}")
            refs_by_admission[ref.hadm_id].append(ref)

        dataset = TimeFDataset(metadata=self.metadata())
        for row in pq.ParquetFile(source.admissions_path).iter_batches(batch_size=1_024):
            for admission in row.to_pylist():
                hadm_id = int(admission["hadm_id"])
                refs = refs_by_admission.get(hadm_id)
                if not refs:
                    raise TimeFFormatError(f"prepared admission {hadm_id} has no vital-sign series")
                sample_id = _sample_id(hadm_id)
                sample = dataset.add_sample(
                    time_series=tuple(_time_series(ref) for ref in refs),
                    subject_ids=(f"mimic-subject-{int(admission['subject_id'])}",),
                    sample_id=sample_id,
                    time_span=TimeInterval.micros(0, int(admission["duration_us"])),
                )
                annotations = [
                    Annotation(
                        key="admission_type",
                        value=str(admission["admission_type"]),
                        id=f"{sample_id}-admission-type",
                    )
                ]
                if admission["first_careunit"]:
                    annotations.append(
                        Annotation(
                            key="first_careunit",
                            value=str(admission["first_careunit"]),
                            id=f"{sample_id}-first-careunit",
                        )
                    )
                sample.add_annotations(annotations)

        dataset.set_task_stream([AnswerTask], lambda: _iter_note_tasks(source.notes_path))
        return dataset

    @classmethod
    def _raw_files_from_environment(cls) -> MimicRawFiles:
        """Resolve the required local source files from the environment.

        Returns:
            The four authorized source file paths.

        Raises:
            TimeNetBuildError: If a source-root variable is unset or a required file is missing.
        """
        roots: dict[str, Path] = {}
        for variable in (MIMIC_IV_ROOT_ENV, MIMIC_IV_NOTE_ROOT_ENV):
            value = os.environ.get(variable)
            if not value:
                raise TimeNetBuildError(
                    f"{variable} is not set. Set it to the extracted authorized PhysioNet dataset directory"
                )
            roots[variable] = Path(value).expanduser()

        paths: dict[str, Path] = {}
        for name, variable, relative in cls.REQUIRED_FILES:
            path = roots[variable] / relative
            if not path.is_file():
                raise TimeNetBuildError(f"{variable} does not contain the required file {relative!r}: {path}")
            paths[name] = path
        return MimicRawFiles(
            admissions=paths["admissions"],
            icustays=paths["icustays"],
            chartevents=paths["chartevents"],
            discharge=paths["discharge"],
        )


def _prepare_cache(source: MimicSource) -> MimicPreparedSource:
    """Prepare or reuse compact Parquet files for one source.

    Returns:
        The paths to the prepared files.
    """
    source.prepared_dir.mkdir(parents=True, exist_ok=True)
    prepared = MimicPreparedSource(
        admissions_path=source.prepared_dir / "admissions.parquet",
        vitals_path=source.prepared_dir / "vitals.parquet",
        notes_path=source.prepared_dir / "notes.parquet",
    )
    fingerprint = _source_fingerprint(source.raw_files)
    marker = source.prepared_dir / "complete.json"
    if not _cache_is_current(prepared, marker, fingerprint):
        _prepare_source(source.raw_files, prepared, marker, fingerprint)
        _read_series_row_group.cache_clear()
    return prepared


def _time_series(ref: MimicSeriesRef) -> TimeSeries:
    """Build one lazy vital-sign series from its prepared row reference.

    Args:
        ref: The prepared row reference.

    Returns:
        A lazy irregular TimeF series.
    """
    return TimeSeries(
        spec=_SERIES_LAYOUT[ref.spec_type, ref.channel],
        channel=ref.channel,
        time_axis=IrregularAxis(first_us=ref.first_offset_us, last_us=ref.last_offset_us),
        loader=_PreparedListLoader(ref.path, ref.row_group, ref.row_index, "values"),
        time_offsets_loader=_PreparedListLoader(ref.path, ref.row_group, ref.row_index, "time_offsets_us"),
        source_id=_sample_id(ref.hadm_id),
        time_series_id=f"mimic-hadm-{ref.hadm_id}-{ref.spec_type}-{ref.channel}",
        n_values=ref.n_values,
    )


def _iter_series_refs(path: Path) -> Iterator[MimicSeriesRef]:
    """Yield row references without reading the nested vital-sign values.

    Args:
        path: The prepared vital-sign Parquet file.

    Yields:
        One lightweight reference per vital-sign channel and admission.
    """
    parquet = pq.ParquetFile(path)
    metadata_columns = [
        "hadm_id",
        "spec_type",
        "channel",
        "n_values",
        "first_offset_us",
        "last_offset_us",
    ]
    for row_group in range(parquet.num_row_groups):
        table = parquet.read_row_group(row_group, columns=metadata_columns)
        for row_index, row in enumerate(table.to_pylist()):
            yield MimicSeriesRef(
                hadm_id=int(row["hadm_id"]),
                spec_type=str(row["spec_type"]),
                channel=str(row["channel"]),
                n_values=int(row["n_values"]),
                first_offset_us=int(row["first_offset_us"]),
                last_offset_us=int(row["last_offset_us"]),
                path=path,
                row_group=row_group,
                row_index=row_index,
            )


def _iter_note_tasks(path: Path) -> Iterator[AnswerTask]:
    """Stream one caption task per main discharge summary.

    Args:
        path: The prepared note Parquet file.

    Yields:
        One unprompted answer task per note.
    """
    columns = ["note_id", "hadm_id", "text"]
    for batch in pq.ParquetFile(path).iter_batches(batch_size=512, columns=columns):
        for row in batch.to_pylist():
            yield AnswerTask(
                id=f"mimic-note-{row['note_id']}",
                sample_ids=(_sample_id(int(row["hadm_id"])),),
                target=str(row["text"]),
            )


def _sample_id(hadm_id: int) -> str:
    return f"mimic-hadm-{hadm_id}"


def _duckdb() -> Any:
    """Import DuckDB lazily and state how to install it when it is missing.

    Returns:
        The imported DuckDB module.

    Raises:
        ImportError: If the connector dependency is not installed.
    """
    try:
        return importlib.import_module("duckdb")
    except ModuleNotFoundError as exc:
        raise ImportError(
            "preparing MIMIC needs duckdb, declared in this connector's requirements.txt. "
            "Run the build without --no-isolation, or install it yourself"
        ) from exc


def _source_fingerprint(files: MimicRawFiles) -> dict[str, object]:
    """Build a cache fingerprint from the extraction version and source file metadata.

    Args:
        files: The source file paths.

    Returns:
        JSON-compatible fingerprint data.
    """
    sources = {}
    for name, path in files.items():
        stat = path.stat()
        sources[name] = {
            "path": str(path.resolve()),
            "size": stat.st_size,
            "mtime_ns": stat.st_mtime_ns,
        }
    return {"extraction_version": _EXTRACTION_VERSION, "sources": sources}


def _cache_is_current(source: MimicPreparedSource, marker: Path, fingerprint: Mapping[str, object]) -> bool:
    """Return true when all prepared files match the current source fingerprint."""
    if not all(path.is_file() for path in (source.admissions_path, source.vitals_path, source.notes_path)):
        return False
    try:
        return json.loads(marker.read_text(encoding="utf-8")) == fingerprint
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return False


def _prepare_source(
    raw: MimicRawFiles,
    source: MimicPreparedSource,
    marker: Path,
    fingerprint: Mapping[str, object],
) -> None:
    """Scan the local source once and atomically write the prepared cache.

    Args:
        raw: The authorized MIMIC source files.
        source: The final prepared file paths.
        marker: The completion marker path.
        fingerprint: The source fingerprint stored in the marker.

    Raises:
        TimeNetBuildError: If the source join produces no eligible admissions.
    """
    outputs = (source.admissions_path, source.vitals_path, source.notes_path)
    parts = tuple(path.with_suffix(path.suffix + ".part") for path in outputs)
    marker_part = marker.with_suffix(".json.part")
    for path in (*parts, marker_part):
        path.unlink(missing_ok=True)

    duckdb = _duckdb()
    connection = duckdb.connect(":memory:")
    try:
        temp_dir = marker.parent / "duckdb-temp"
        temp_dir.mkdir(parents=True, exist_ok=True)
        connection.execute(f"SET temp_directory = {_sql_literal(str(temp_dir))}")
        _create_source_views(connection, raw)
        connection.execute(_HOURLY_VITALS_SQL)
        connection.execute(
            """
            CREATE TEMP TABLE cohort AS
            SELECT DISTINCT vital.hadm_id
            FROM hourly_vitals AS vital
            JOIN discharge_raw AS note USING (hadm_id)
            JOIN admissions_raw AS admission USING (hadm_id)
            WHERE note.note_type = 'DS'
                AND trim(note.text) <> ''
                AND admission.dischtime > admission.admittime
            """
        )
        cohort_size = int(connection.execute("SELECT count(*) FROM cohort").fetchone()[0])
        if cohort_size == 0:
            raise TimeNetBuildError(
                "the MIMIC source join produced no admissions with vital signs and a main discharge summary"
            )

        _copy_query(connection, _ADMISSIONS_EXPORT_SQL, parts[0])
        _copy_query(connection, _VITALS_EXPORT_SQL, parts[1])
        _copy_query(connection, _NOTES_EXPORT_SQL, parts[2])
    finally:
        connection.close()

    for part, output in zip(parts, outputs, strict=True):
        part.replace(output)
    marker_part.write_text(json.dumps(fingerprint, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    marker_part.replace(marker)


def _create_source_views(connection: Any, raw: MimicRawFiles) -> None:
    """Create typed DuckDB views over the compressed source CSV files."""
    definitions = (
        ("admissions_raw", raw.admissions, _ADMISSIONS_COLUMNS),
        ("icustays_raw", raw.icustays, _ICUSTAYS_COLUMNS),
        ("chartevents_raw", raw.chartevents, _CHARTEVENTS_COLUMNS),
        ("discharge_raw", raw.discharge, _DISCHARGE_COLUMNS),
    )
    for view, path, columns in definitions:
        connection.execute(
            # Identifiers and paths come from connector-owned definitions.
            f"CREATE TEMP VIEW {view} AS SELECT * FROM {_read_csv_expression(path, columns)}"  # noqa: S608
        )


def _read_csv_expression(path: Path, columns: Mapping[str, str]) -> str:
    """Build a typed DuckDB CSV expression for a trusted local path.

    Returns:
        A DuckDB ``read_csv`` expression with an explicit schema.
    """
    fields = ", ".join(f"{_sql_literal(name)}: {_sql_literal(dtype)}" for name, dtype in columns.items())
    return (
        f"read_csv({_sql_literal(str(path))}, header = true, auto_detect = false, "
        f"columns = {{{fields}}}, compression = 'gzip', nullstr = '')"
    )


def _copy_query(connection: Any, query: str, destination: Path) -> None:
    """Write one query result to a prepared Parquet part file."""
    connection.execute(f"COPY ({query}) TO {_sql_literal(str(destination))} (FORMAT PARQUET, COMPRESSION ZSTD)")


def _sql_literal(value: str) -> str:
    """Quote one trusted string as a DuckDB SQL literal.

    Returns:
        The escaped SQL string literal.
    """
    return "'" + value.replace("'", "''") + "'"


_ADMISSIONS_EXPORT_SQL = """
WITH first_icu AS (
    SELECT hadm_id, arg_min(first_careunit, intime) AS first_careunit
    FROM icustays_raw
    WHERE hadm_id IS NOT NULL
    GROUP BY hadm_id
)
SELECT
    admission.hadm_id,
    admission.subject_id,
    admission.admission_type,
    first_icu.first_careunit,
    date_diff('microsecond', admission.admittime, admission.dischtime) AS duration_us
FROM admissions_raw AS admission
JOIN cohort USING (hadm_id)
LEFT JOIN first_icu USING (hadm_id)
ORDER BY admission.hadm_id
"""

_VITALS_EXPORT_SQL = """
SELECT
    vital.hadm_id,
    vital.spec_type,
    vital.channel,
    CAST(count(*) AS INTEGER) AS n_values,
    min(vital.hour_index) * 3600000000 AS first_offset_us,
    max(vital.hour_index) * 3600000000 AS last_offset_us,
    list(vital.hour_index * 3600000000 ORDER BY vital.hour_index) AS time_offsets_us,
    list(vital.value ORDER BY vital.hour_index) AS values
FROM hourly_vitals AS vital
JOIN cohort USING (hadm_id)
GROUP BY vital.hadm_id, vital.spec_type, vital.channel
ORDER BY vital.spec_type, vital.hadm_id, vital.channel
"""

_NOTES_EXPORT_SQL = """
SELECT note.note_id, note.hadm_id, note.text
FROM discharge_raw AS note
JOIN cohort USING (hadm_id)
WHERE note.note_type = 'DS' AND trim(note.text) <> ''
ORDER BY note.hadm_id, note.note_seq, note.note_id
"""


CONNECTOR = MimicIvExtVitalsNotesConnector
