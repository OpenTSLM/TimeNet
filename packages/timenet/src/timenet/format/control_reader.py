"""Hydrate the TimeF object hierarchy from ``control.duckdb``."""

from collections import defaultdict
from collections.abc import Callable, Iterable
from fractions import Fraction
import json
from pathlib import Path
from typing import Any

import duckdb
import pyarrow as pa

from timenet.dataset import IrregularAxis, OrdinalAxis, Record, RegularAxis, Signal, Source
from timenet.errors import TimeFFormatError
from timenet.format.duckdb import check_control_schema, connect_control
from timenet.types import Annotation, DataSource, TimeInterval, TimePoint, TimeSeriesSpec, ureg


ValueLoader = Callable[[str], pa.Array]


def _decode_json(value: str | None, *, default: Any = None) -> Any:
    """Decode one DuckDB JSON value.

    Returns:
        The decoded value, or ``default`` for SQL ``NULL``.

    Raises:
        TimeFFormatError: If stored JSON is malformed.
    """
    if value is None:
        return default
    try:
        return json.loads(value)
    except (TypeError, json.JSONDecodeError) as exc:
        raise TimeFFormatError(f"control.duckdb contains invalid JSON: {value!r}") from exc


def _missing_values(signal_id: str) -> pa.Array:
    """Fail when metadata-only hydration is asked to load values.

    Raises:
        TimeFFormatError: Always, because no values loader was configured.
    """
    raise TimeFFormatError(f"no values loader is configured for signal {signal_id!r}")


class DuckDBControlReader:
    """Keep one read-only DuckDB connection and hydrate requested records from it."""

    def __init__(self, path: Path, *, value_loader: ValueLoader | None = None) -> None:
        """Open and validate an immutable control database.

        Args:
            path: Local path to ``control.duckdb``.
            value_loader: Lazy values-plane resolver keyed by signal ID.

        Raises:
            TimeFFormatError: If the database cannot be opened or has an unsupported schema.
        """
        self.path = Path(path)
        try:
            self.connection = connect_control(self.path, read_only=True)
            check_control_schema(self.connection)
        except duckdb.Error as exc:
            raise TimeFFormatError(f"could not open control database {self.path}: {exc}") from exc
        self._value_loader = value_loader or _missing_values

    def close(self) -> None:
        """Close the reader's DuckDB connection."""
        self.connection.close()

    def __enter__(self) -> "DuckDBControlReader":
        """Return this open reader.

        Returns:
            This reader.
        """
        return self

    def __exit__(self, *_: object) -> None:
        """Close the connection when leaving a context manager."""
        self.close()

    def read_records(  # noqa: PLR0914 - hydration keeps related row sets together
        self, record_ids: Iterable[str] | None = None
    ) -> tuple[Record, ...]:
        """Hydrate complete recursive records while leaving signal values lazy.

        Args:
            record_ids: Requested IDs in result order, or ``None`` for every record.

        Returns:
            The hydrated records.

        Raises:
            TimeFFormatError: If a requested record is absent or the stored hierarchy is inconsistent.
        """
        requested = None if record_ids is None else tuple(record_ids)
        rows = self.connection.execute(
            """SELECT record_id, start_time_us, time_span_start_us, time_span_end_us, metadata
               FROM records ORDER BY record_id"""
        ).fetchall()
        by_id = {row[0]: row for row in rows}
        order = tuple(by_id) if requested is None else requested
        missing = [record_id for record_id in order if record_id not in by_id]
        if missing:
            raise TimeFFormatError(f"control.duckdb has no record(s) {missing}")

        source_rows = self.connection.execute(
            "SELECT source_id, record_id, parent_source_id, name, metadata FROM sources ORDER BY source_id"
        ).fetchall()
        signal_rows = self.connection.execute(
            """SELECT signal_id, source_id, name, axis_id, spec_type, spec_name, unit, dtype,
                      categories, value_shape, dimension_names, nullable, data_source, n_values, metadata
               FROM signals ORDER BY signal_id"""
        ).fetchall()
        annotations = self._read_annotations()
        axes = self._read_axes()

        signals_by_source: dict[str, list[Signal]] = defaultdict(list)
        for row in signal_rows:
            signal_id, source_id, name, axis_id = row[:4]
            axis = axes.get(axis_id)
            if axis is None:
                raise TimeFFormatError(f"signal {signal_id!r} refers to missing axis {axis_id!r}")
            spec = self._spec_from_row(row)
            offsets_loader = self._offsets_loader(axis_id) if isinstance(axis, IrregularAxis) else None
            signals_by_source[source_id].append(
                Signal(
                    time_series_id=signal_id,
                    signal=name,
                    spec=spec,
                    time_axis=axis,
                    n_values=row[13],
                    loader=lambda signal_id=signal_id: self._value_loader(signal_id),
                    time_offsets_loader=offsets_loader,
                    annotations=annotations.get(("Signal", signal_id), ()),
                    metadata=_decode_json(row[14], default={}),
                )
            )

        source_data = {row[0]: row for row in source_rows}
        children: dict[str | None, list[str]] = defaultdict(list)
        roots_by_record: dict[str, list[str]] = defaultdict(list)
        for source_id, record_id, parent_id, *_ in source_rows:
            if parent_id is None:
                roots_by_record[record_id].append(source_id)
            else:
                children[parent_id].append(source_id)

        active: set[str] = set()

        def hydrate_source(source_id: str, record_id: str) -> Source:
            if source_id in active:
                raise TimeFFormatError(f"source hierarchy contains a cycle at {source_id!r}")
            row = source_data.get(source_id)
            if row is None:
                raise TimeFFormatError(f"source hierarchy refers to missing source {source_id!r}")
            if row[1] != record_id:
                raise TimeFFormatError(f"source {source_id!r} crosses record boundaries")
            active.add(source_id)
            source = Source(
                id=source_id,
                name=row[3],
                sources=tuple(hydrate_source(child, record_id) for child in children[source_id]),
                signals=tuple(signals_by_source[source_id]),
                annotations=annotations.get(("Source", source_id), ()),
                metadata=_decode_json(row[4], default={}),
            )
            active.remove(source_id)
            return source

        records: list[Record] = []
        for record_id in order:
            row = by_id[record_id]
            span = None
            if row[2] is not None:
                span = TimeInterval(start_us=row[2], end_us=row[3])
            records.append(
                Record(
                    record_id=record_id,
                    sources=tuple(hydrate_source(source_id, record_id) for source_id in roots_by_record[record_id]),
                    annotations=annotations.get(("Record", record_id), ()),
                    start_time=row[1],
                    time_span=span,
                    metadata=_decode_json(row[4], default={}),
                )
            )
        return tuple(records)

    def _read_axes(self) -> dict[str, RegularAxis | IrregularAxis | OrdinalAxis]:
        """Hydrate each shared axis exactly once.

        Returns:
            Axes keyed by their stored IDs.

        Raises:
            TimeFFormatError: If an axis has an unknown type.
        """
        axes: dict[str, RegularAxis | IrregularAxis | OrdinalAxis] = {}
        rows = self.connection.execute(
            """SELECT axis_id, axis_type, period_numerator_us, period_denominator,
                      origin_us, first_us, last_us FROM axes"""
        ).fetchall()
        for axis_id, axis_type, numerator, denominator, origin, first, last in rows:
            if axis_type == "regular":
                axes[axis_id] = RegularAxis(
                    axis_id=axis_id,
                    period_us=Fraction(numerator, denominator),
                    start_index=origin,
                )
            elif axis_type == "irregular":
                axes[axis_id] = IrregularAxis(axis_id=axis_id, first_us=first, last_us=last)
            elif axis_type == "ordinal":
                axes[axis_id] = OrdinalAxis(axis_id=axis_id)
            else:
                raise TimeFFormatError(f"axis {axis_id!r} has unknown type {axis_type!r}")
        return axes

    def _offsets_loader(self, axis_id: str) -> Callable[[], pa.Array]:
        """Return a lazy loader over one shared irregular axis.

        Returns:
            A callable that reads the axis offsets as Arrow int64 values.
        """

        def load() -> pa.Array:
            rows = self.connection.execute(
                "SELECT offset_us FROM axis_offsets WHERE axis_id = ? ORDER BY position", [axis_id]
            ).fetchall()
            return pa.array((row[0] for row in rows), type=pa.int64())

        return load

    @staticmethod
    def _spec_from_row(row: tuple[Any, ...]) -> TimeSeriesSpec:
        """Reconstruct an inline signal specification.

        Returns:
            The typed specification.
        """
        source_data = _decode_json(row[12])
        source = None if source_data is None else DataSource(**source_data)
        return TimeSeriesSpec(
            spec_type=row[4],
            name=row[5],
            unit_value=ureg.Unit(row[6]),
            dtype=row[7],
            categories=tuple(_decode_json(row[8], default=[])),
            value_shape=tuple(_decode_json(row[9], default=[])),
            dimension_names=tuple(_decode_json(row[10], default=[])),
            nullable=row[11],
            data_source=source,
        )

    def _read_annotations(self) -> dict[tuple[str, str], tuple[Annotation, ...]]:
        """Hydrate annotation content and occurrences for hierarchy objects.

        Returns:
            Attached occurrences keyed by object type and object ID.

        Raises:
            TimeFFormatError: If an occurrence has an unknown span type.
        """
        grouped: dict[tuple[str, str], list[Annotation]] = defaultdict(list)
        rows = self.connection.execute(
            """SELECT o.occurrence_id, o.object_type, o.object_id, o.span_type,
                      o.start_us, o.end_us, o.signal_ids, o.provenance, o.confidence,
                      o.metadata, c.content_id, c.name, c.value, c.unit, c.metadata
               FROM annotation_occurrences o
               JOIN annotation_contents c USING (content_id)
               ORDER BY o.object_type, o.object_id, o.occurrence_id"""
        ).fetchall()
        for row in rows:
            signal_ids = _decode_json(row[6])
            scope = None if signal_ids is None else tuple(signal_ids)
            span = None
            if row[3] == "point":
                span = TimePoint(start_us=row[4], time_series_ids=scope)
            elif row[3] == "interval":
                span = TimeInterval(start_us=row[4], end_us=row[5], time_series_ids=scope)
            elif row[3] != "static":
                raise TimeFFormatError(f"annotation occurrence {row[0]!r} has unknown span type {row[3]!r}")
            content_metadata = _decode_json(row[14], default={})
            description = content_metadata.pop("description", None)
            grouped[row[1], row[2]].append(
                Annotation(
                    occurrence_id=row[0],
                    id=row[10],
                    key=row[11],
                    value=_decode_json(row[12]),
                    unit=row[13],
                    description=description,
                    metadata=content_metadata,
                    span=span,
                    source=_decode_json(row[7]),
                    confidence=row[8],
                    occurrence_metadata=_decode_json(row[9], default={}),
                )
            )
        return {key: tuple(value) for key, value in grouped.items()}
