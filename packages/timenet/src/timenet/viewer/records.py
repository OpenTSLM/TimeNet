"""Bounded record projections and explicit scalar signal windows."""

from collections.abc import Generator
from contextlib import contextmanager
from dataclasses import dataclass
import json
import math
from typing import Any

from fastapi import FastAPI, Query
from fastapi.responses import Response
from pydantic import BaseModel, StrictFloat, StrictInt

from timenet.dataset import Record, Signal, Source
from timenet.dataset.axis import AxisType, IrregularAxis, RegularAxis
from timenet.errors import TimeFFormatError, TimeFValidationError
from timenet.json import JsonMapping
from timenet.manifest import Manifest
from timenet.reader import TimeFReader
from timenet.registry import BaseRegistry
from timenet.types.clock import unix_us
from timenet.types.text import read_value
from timenet.viewer.errors import fail, registry_errors, resolve_version
from timenet.viewer.models import AnnotationDetail, SpanDetail, project_annotation, project_span, public_id


@dataclass(frozen=True)
class RecordLimits:
    """Initial operation caps, configurable by the server owner."""

    hierarchy: int = 5_000
    annotations: int = 10_000
    window: int = 2_000
    response_bytes: int = 2 * 1024 * 1024


class AxisDetail(BaseModel):
    """Axis scalar facts, without reading irregular offsets."""

    kind: AxisType
    period_numerator_us: int | None = None
    period_denominator: int | None = None
    start_index: int | None = None
    offset_us: int | None = None
    first_us: int | None = None
    last_us: int | None = None


class SpecDetail(BaseModel):
    """Explicit signal spec facts used for metadata and plotting."""

    spec_type: str
    name: str
    dtype: str
    unit_value: str | None
    value_shape: list[int]
    dimension_names: list[str]
    nullable: bool
    categories: list[str]


class SignalDetail(BaseModel):
    """Signal structure and plotting capability, with lazy values."""

    id: str
    source_id: str
    name: str
    n_values: int
    metadata: JsonMapping
    annotations: list[AnnotationDetail]
    spec: SpecDetail
    axis: AxisDetail
    plot_supported: bool
    plot_reason: str | None


class SourceDetail(BaseModel):
    """One flat source entry; parent keys reconstruct the tree.

    Sources carry no ID, so each is keyed by its position path within the record, such as ``0.1``
    for the second child of the first root source.
    """

    id: str
    name: str
    parent_source_id: str | None
    metadata: JsonMapping
    annotations: list[AnnotationDetail]


class RecordDetail(BaseModel):
    """Record facts and flat hierarchy, without hydrating tasks."""

    id: str
    start_time: int | None
    time_span: SpanDetail | None
    metadata: JsonMapping
    annotations: list[AnnotationDetail]
    sources: list[SourceDetail]
    signals: list[SignalDetail]


class SignalWindow(BaseModel):
    """A half-open step window with JSON-safe gaps."""

    record_id: str
    signal_id: str
    start: int
    stop: int
    dtype: str
    values: list[StrictInt | StrictFloat | None]
    nonfinite_steps: list[int]


def _cap(count: int, maximum: int, operation: str, label: str) -> None:
    if count > maximum:
        fail(413, "operation_limit", f"{label} exceeds the configured limit.", operation, maximum)


class _Budget:
    """Encode JSON incrementally with bounded string chunks and a fixed byte budget."""

    def __init__(self, maximum: int, operation: str) -> None:
        self.maximum = maximum
        self.operation = operation
        self.data = bytearray()

    def add(self, chunk: str) -> None:
        encoded = chunk.encode("utf-8")
        _cap(len(self.data) + len(encoded), self.maximum, self.operation, "JSON response")
        self.data.extend(encoded)

    def encode(self, value: Any) -> None:
        if isinstance(value, str):
            self.add('"')
            for start in range(0, len(value), 1024):
                self.add(json.dumps(value[start : start + 1024], ensure_ascii=True)[1:-1])
            self.add('"')
        elif isinstance(value, dict):
            self.add("{")
            for index, (key, item) in enumerate(value.items()):
                if index:
                    self.add(",")
                self.encode(key)
                self.add(":")
                self.encode(item)
            self.add("}")
        elif isinstance(value, (list, tuple)):
            self.add("[")
            for index, item in enumerate(value):
                if index:
                    self.add(",")
                self.encode(item)
            self.add("]")
        else:
            self.add(json.dumps(value, allow_nan=False))


def _response(model: BaseModel, limits: RecordLimits, operation: str) -> Response:
    budget = _Budget(limits.response_bytes, operation)
    budget.encode(model.model_dump(mode="json"))
    return Response(bytes(budget.data), media_type="application/json")


@contextmanager
def _record(reader: TimeFReader, record_id: int, operation: str) -> Generator[Record]:
    iterator = reader.iter_records(record_ids=[record_id], with_annotations=True)
    try:
        try:
            record = next(iterator)
        except StopIteration:
            fail(404, "not_found", "Record not found.", operation)
        except TimeFValidationError as exc:
            # The reader raises this for an unknown ID, but also when a stored record fails
            # validation. Only the first is a 404; the second is a corrupt artifact.
            if record_id not in reader.record_ids():
                fail(404, "not_found", "Record not found.", operation)
            raise TimeFFormatError(f"record {record_id!r} failed validation") from exc
        yield record
    finally:
        if isinstance(iterator, Generator):
            iterator.close()


def _children(sources: tuple[Source, ...], parent: str | None) -> list[tuple[Source, str, str | None]]:
    """Return sources keyed by position path, in reverse name order for a depth-first stack."""
    keyed = [(source, str(index) if parent is None else f"{parent}.{index}") for index, source in enumerate(sources)]
    return [(source, key, parent) for source, key in sorted(keyed, key=lambda item: item[0].name, reverse=True)]


def _hierarchy(record: Record, limits: RecordLimits, operation: str) -> list[tuple[Source, str, str | None]]:
    result: list[tuple[Source, str, str | None]] = []
    pending = _children(record.sources, None)
    count = 0
    annotations = len(record.annotations)
    while pending:
        source, key, parent = pending.pop()
        count += 1 + len(source.signals)
        annotations += len(source.annotations) + sum(len(signal.annotations) for signal in source.signals)
        _cap(count, limits.hierarchy, operation, "Record hierarchy")
        _cap(annotations, limits.annotations, operation, "Attached annotations")
        result.append((source, key, parent))
        pending.extend(_children(source.sources, key))
    _cap(annotations, limits.annotations, operation, "Attached annotations")
    # A window response holds no record fields, so only record details measure them.
    if operation == "record":
        _preflight(record, result, limits, operation)
    return result


def _preflight(
    record: Record, hierarchy: list[tuple[Source, str, str | None]], limits: RecordLimits, operation: str
) -> None:
    """Bound variable-size fields before Pydantic copies or projects them."""
    budget = _Budget(limits.response_bytes, operation)
    for obj in [
        record,
        *(source for source, _, _ in hierarchy),
        *(signal for source, _, _ in hierarchy for signal in source.signals),
    ]:
        budget.encode(None if isinstance(obj, Source) else public_id(obj.id))
        if isinstance(obj, (Source, Signal)):
            budget.encode(obj.name)
        if isinstance(obj, Signal):
            budget.encode(obj.spec.name)
            budget.encode(obj.spec.spec_type)
            budget.encode(None if obj.spec.unit_value is None else str(obj.spec.unit_value))
            budget.encode(obj.spec.categories)
            budget.encode(obj.spec.dimension_names)
        budget.encode(obj.metadata)
        for annotation in obj.annotations:
            for text in (annotation.key, annotation.description, annotation.source):
                budget.encode(text)
            budget.encode(None if annotation.unit is None else str(annotation.unit))
            if annotation.span is not None and annotation.span.time_series is not None:
                budget.encode([public_id(signal.id) for signal in annotation.span.time_series])
            budget.encode(read_value(annotation.value))
            budget.encode(annotation.metadata)
            budget.encode(annotation.occurrence_metadata)


def _reason(signal: Signal) -> str | None:
    if signal.spec.value_shape or not signal.spec.dtype.startswith(("int", "uint", "float")):
        return "Plotting supports scalar integer and floating-point signals only."
    return None


def _signal(signal: Signal, source_id: str) -> SignalDetail:
    axis = signal.time_axis
    facts = AxisDetail(kind=axis.axis_type)
    if isinstance(axis, RegularAxis):
        facts.period_numerator_us = axis.period_us.numerator
        facts.period_denominator = axis.period_us.denominator
        facts.start_index = axis.start_index
        facts.offset_us = axis.offset_us
        if signal.n_values:
            facts.first_us = axis.time_offset_us(0)
            facts.last_us = axis.time_offset_us(signal.n_values - 1)
    elif isinstance(axis, IrregularAxis):
        facts.first_us, facts.last_us = axis.first_us, axis.last_us
    reason = _reason(signal)
    return SignalDetail(
        id=public_id(signal.id),
        source_id=source_id,
        name=signal.name,
        n_values=signal.n_values,
        metadata=signal.metadata,
        annotations=[project_annotation(a) for a in signal.annotations],
        spec=SpecDetail(
            spec_type=signal.spec.spec_type,
            name=signal.spec.name,
            dtype=signal.spec.dtype,
            unit_value=None if signal.spec.unit_value is None else str(signal.spec.unit_value),
            value_shape=list(signal.spec.value_shape),
            dimension_names=list(signal.spec.dimension_names),
            nullable=signal.spec.nullable,
            categories=list(signal.spec.categories),
        ),
        axis=facts,
        plot_supported=reason is None,
        plot_reason=reason,
    )


def install_record_routes(
    app: FastAPI, registry: BaseRegistry, initial_manifest: Manifest | None, limits: RecordLimits
) -> None:
    """Install authenticated record and signal endpoints with request-owned readers."""

    @app.get("/api/v1/record")
    def record(dataset_id: str, version: str, record_id: int = Query(..., ge=1)) -> Response:
        """Return one annotated record's structure while leaving all values lazy."""
        operation = "record"
        version = resolve_version(dataset_id, version, initial_manifest, operation)
        with (
            registry_errors(operation),
            TimeFReader(registry.open_version(dataset_id, version)) as reader,
            _record(reader, record_id, operation) as item,
        ):
            hierarchy = _hierarchy(item, limits, operation)
            detail = RecordDetail(
                id=public_id(item.id),
                start_time=None if item.start_time.timestamp is None else unix_us(item.start_time.timestamp),
                time_span=project_span(item.time_span),
                metadata=item.metadata,
                annotations=[project_annotation(a) for a in item.annotations],
                sources=[
                    SourceDetail(
                        id=key,
                        name=s.name,
                        parent_source_id=p,
                        metadata=s.metadata,
                        annotations=[project_annotation(a) for a in s.annotations],
                    )
                    for s, key, p in hierarchy
                ],
                signals=[
                    _signal(signal, key)
                    for s, key, _ in hierarchy
                    for signal in sorted(s.signals, key=lambda v: (v.name, v.id))
                ],
            )
            return _response(detail, limits, operation)

    @app.get("/api/v1/signal-window")
    def window(  # noqa: PLR0913 - explicit HTTP window identity and bounds
        *,
        dataset_id: str,
        version: str,
        start: int = Query(..., ge=0),
        stop: int = Query(..., ge=1),
        record_id: int = Query(..., ge=1),
        signal_id: int = Query(..., ge=1),
    ) -> Response:
        """Read exactly one eligible signal's explicit half-open step window.

        Returns:
            A bounded JSON window response.
        """
        operation = "signal-window"
        version = resolve_version(dataset_id, version, initial_manifest, operation)
        if stop <= start:
            fail(422, "invalid_parameters", "Require 0 <= start < stop.", operation)
        _cap(stop - start, limits.window, operation, "Signal window")
        with (
            registry_errors(operation),
            TimeFReader(registry.open_version(dataset_id, version)) as reader,
            _record(reader, record_id, operation) as item,
        ):
            hierarchy = _hierarchy(item, limits, operation)
            selected = next((s for source, _, _ in hierarchy for s in source.signals if s.id == signal_id), None)
            if selected is None:
                fail(404, "not_found", "Signal not found in this record.", operation)
            if stop > selected.n_values:
                fail(422, "invalid_parameters", "Window exceeds the signal length.", operation)
            reason = _reason(selected)
            if reason:
                fail(422, "unsupported_plot", reason, operation)
            return _response(_window(selected, record_id, start, stop, operation), limits, operation)


def _window(selected: Signal, record_id: int, start: int, stop: int, operation: str) -> SignalWindow:
    """Project a scalar window, retaining integer precision and nonfinite gaps.

    Returns:
        Values and absolute indices of nonfinite gaps.

    Raises:
        TimeFFormatError: If storage returns the wrong number of steps.
    """
    array = selected.read_steps(start, stop)
    if len(array) != stop - start:
        raise TimeFFormatError("incorrect window length")
    values: list[int | float | None] = []
    nonfinite: list[int] = []
    for index, scalar in enumerate(array):
        value = scalar.as_py()
        if value is not None and (isinstance(value, bool) or not isinstance(value, (int, float))):
            raise TimeFFormatError("non-numeric scalar in numeric signal")
        if isinstance(value, int) and abs(value) > 2**53 - 1:
            fail(422, "unsupported_plot", "Observed integer exceeds JavaScript safe integer range.", operation)
        if isinstance(value, float) and not math.isfinite(value):
            value = None
            nonfinite.append(start + index)
        values.append(value)
    return SignalWindow(
        record_id=public_id(record_id),
        signal_id=public_id(selected.id),
        start=start,
        stop=stop,
        dtype=selected.spec.dtype,
        values=values,
        nonfinite_steps=nonfinite,
    )
