"""Convert VerbalTS windows and captions to TimeF records and generation tasks.

Each window becomes a record. Each caption becomes a task in the window's split.
Values use memory-mapped arrays, and tasks stream from the caption arrays.
"""

from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from enum import StrEnum
from fractions import Fraction
import functools
import json
import operator
from pathlib import Path
import tempfile

import numpy as np
import pyarrow as pa

from timenet.connectors import BaseConnector
from timenet.dataset import Record, Signal, Source, TimeFDataset
from timenet.dataset.axis import OrdinalAxis, RegularAxis, TimeAxis
from timenet.errors import TimeFFormatError, TimeNetDownloadError
from timenet.types import Annotation, InputModality, Split, TimeOrigin, TimeSeriesSpec, TSGenerationTask, ureg
from timenet_connectors.datasets.seqml.verbalts.files import COMPONENTS, FILES, check_drive_body
from timenet_connectors.datasets.seqml.verbalts.release import CHANNELS, CODEBOOKS, STRIDE, VARIABLE, Attribute


_VALUES_DIMS = 3
_PLANE_DIMS = 2

SPLIT_BY_NAME: dict[str, Split] = {"train": Split.TRAIN, "valid": Split.VALIDATION, "test": Split.TEST}
"""Map release split names to TimeF splits."""

# The release gives no instrument units, so all converted values are dimensionless.
# See the dataset appendix: https://proceedings.mlr.press/v267/gu25a.html.
_SYNTHETIC = TimeSeriesSpec(
    spec_type="synthetic",
    name="Synthetic generated variable",
    unit_value=ureg.dimensionless,
    dtype="float64",
)
_WEATHER = TimeSeriesSpec(
    spec_type="weather",
    name="Jena weather variable (z-scored)",
    unit_value=ureg.dimensionless,
    dtype="float64",
)
_POSE = TimeSeriesSpec(
    spec_type="pose",
    name="Body-joint coordinate (z-scored)",
    unit_value=ureg.dimensionless,
    dtype="float64",
)
_POWER = TimeSeriesSpec(
    spec_type="power",
    name="ETTm1 variable (z-scored)",
    unit_value=ureg.dimensionless,
    dtype="float64",
)
_TRAFFIC = TimeSeriesSpec(
    spec_type="traffic",
    name="Istanbul traffic index variable (z-scored)",
    unit_value=ureg.dimensionless,
    dtype="float64",
)

SPEC_BY_COMPONENT: dict[str, TimeSeriesSpec] = {
    "synthetic_u": _SYNTHETIC,
    "synthetic_m": _SYNTHETIC,
    "Weather": _WEATHER,
    "BlindWays": _POSE,
    "ETTm1": _POWER,
    "istanbul_traffic": _TRAFFIC,
}
"""Map each component to its signal spec."""

AXIS_BY_COMPONENT: dict[str, TimeAxis] = {
    "synthetic_u": OrdinalAxis(),
    "synthetic_m": OrdinalAxis(),
    "Weather": RegularAxis.from_rate_hz(Fraction(1, 600)),
    "BlindWays": OrdinalAxis(),
    "ETTm1": RegularAxis.from_rate_hz(Fraction(1, 900)),
    "istanbul_traffic": RegularAxis.from_rate_hz(Fraction(1, 600)),
}
"""Map components to axes without absolute timestamps.

BlindWays and the synthetic sets have no known cadence. ETTm1 uses the
15-minute cadence of its original CSV.
"""

_ID_PREFIX = "verbalts"


class VerbalTsKey(StrEnum):
    """Keys for window provenance annotations."""

    COMPONENT = "component"
    """The component name used by the release."""


def record_id(component: str, split: str, row: int) -> str:
    """Build a window ID with a padded row number for source sort order.

    Args:
        component: Component name.
        split: Split name.
        row: Window row within the split.

    Returns:
        The window ID.
    """
    return f"{_ID_PREFIX}-{component}-{split}-{row:05d}"


@dataclass(frozen=True)
class VerbalTsComponent:
    """Paths to one downloaded component."""

    name: str
    folder: Path

    def npy(self, split: str, kind: str) -> Path:
        """Return the path to one split array.

        Args:
            split: Split name.
            kind: Array kind: ``ts``, ``attrs_idx``, or ``text_caps``.

        Returns:
            The array path.
        """
        return self.folder / f"{split}_{kind}.npy"

    @property
    def meta_path(self) -> Path:
        """Return the path to this component's ``meta.json``.

        Returns:
            The metadata path.
        """
        return self.folder / "meta.json"


@functools.lru_cache(maxsize=24)
def _open_npy(path: str) -> np.ndarray:
    """Cache a memory-mapped NPY array by path.

    Args:
        path: Path to the NPY file.

    Returns:
        The mapped array.
    """
    return np.load(path, mmap_mode="r")


def _plane(path: Path, n_rows: int, n_columns: int | None = None) -> np.ndarray:
    """Map a caption or attribute array and check its row and column counts.

    Args:
        path: Path to the array.
        n_rows: Expected number of windows.
        n_columns: Expected column count, if known.

    Returns:
        The mapped array.

    Raises:
        TimeFFormatError: If the array shape does not match the split.
    """
    plane = _open_npy(str(path))
    if plane.ndim != _PLANE_DIMS or plane.shape[0] != n_rows:
        raise TimeFFormatError(
            f"{path.name} holds an array of shape {plane.shape}, expected ({n_rows}, n) to match its values file"
        )
    if n_columns is not None and plane.shape[1] != n_columns:
        raise TimeFFormatError(
            f"{path.name} holds {plane.shape[1]} columns, but meta.json names {n_columns} attributes"
        )
    return plane


def _values_loader(path: str, row: int, signal: int) -> Callable[[], pa.Array]:
    """Return a loader for one signal of a window.

    Args:
        path: Path to the values array.
        row: Window row.
        signal: Signal column.

    Returns:
        A loader that returns the signal as an Arrow array.
    """

    def load() -> pa.Array:
        """Load the selected signal.

        Returns:
            The signal values as an Arrow array.
        """
        window = _open_npy(path)[row, :, signal]
        return pa.array(np.ascontiguousarray(window, dtype=np.float64))

    return load


class VerbalTsConnector(BaseConnector[VerbalTsComponent]):
    """Convert the six VerbalTS components from Google Drive."""

    def download(self, cache_dir: Path) -> list[VerbalTsComponent]:  # noqa: PLR6301 - BaseConnector override
        """Download the pinned files and return a path for each component.

        Cached and new files must match their pinned SHA-256 digests.

        Args:
            cache_dir: Directory for downloaded files.

        Returns:
            Component paths in release order.

        Raises:
            TimeNetDownloadError: If a file download fails.
        """
        import gdown  # noqa: PLC0415  # ty: ignore[unresolved-import] - optional dependency

        cache_dir.mkdir(parents=True, exist_ok=True)
        for component, name, file_id, n_bytes, digest in FILES:
            destination = cache_dir / component / name
            if destination.is_file():
                check_drive_body(destination, n_bytes, digest)
                continue
            destination.parent.mkdir(parents=True, exist_ok=True)
            with tempfile.TemporaryDirectory(prefix="verbalts-", dir=cache_dir) as temporary:
                staged = Path(temporary) / name
                try:
                    result = gdown.download(id=file_id, output=str(staged), quiet=True)
                except Exception as exc:
                    raise TimeNetDownloadError(f"could not download VerbalTS {component}/{name}: {exc}") from exc
                if result is None:
                    raise TimeNetDownloadError(f"gdown did not download VerbalTS {component}/{name}")
                check_drive_body(staged, n_bytes, digest)
                staged.replace(destination)
        return [VerbalTsComponent(name=component, folder=cache_dir / component) for component in COMPONENTS]

    def convert(self, raw_refs: list[VerbalTsComponent]) -> TimeFDataset:
        """Build window records and stream one generation task per caption.

        Args:
            raw_refs: Paths to downloaded components.

        Returns:
            The converted dataset.
        """
        dataset = TimeFDataset(metadata=self.metadata())
        records: dict[str, Record] = {}
        for component in raw_refs:
            attributes = _attributes(component)
            _add_vocabularies(dataset, component, attributes)
            records.update(_add_component(dataset, component, attributes))
        dataset.set_task_stream([TSGenerationTask], lambda: _iter_tasks(raw_refs, records))
        return dataset


def _add_vocabularies(dataset: TimeFDataset, component: VerbalTsComponent, attributes: tuple[Attribute, ...]) -> None:
    """Add each attribute's full label list as a dataset annotation.

    The list includes unused values. Its index is the release code.
    Integer values appear as strings in the list.

    Args:
        dataset: Dataset to annotate.
        component: Source component.
        attributes: Attributes in column order.
    """
    prefix = component.name.lower()
    for attribute in attributes:
        vocabulary = attribute.vocabulary
        if vocabulary is None:
            continue
        annotation = Annotation(
            key=f"{prefix}_{attribute.key}_labels",
            value=[str(item) for item in vocabulary],
            description=(
                f"Every value the {attribute.key!r} annotation takes on {component.name} windows, in the "
                f"release's code order, so a value's index is its code. Includes values no window carries. "
                f"An integer value is spelled out as a string here."
            ),
            id=f"{_ID_PREFIX}-labels-{prefix}-{attribute.key}",
        )
        dataset.register_annotations((annotation,))
        dataset.annotate(annotation)


@dataclass(frozen=True)
class _SplitArrays:
    """Mapped arrays and window dimensions for one split."""

    values_path: Path
    n_steps: int
    n_signals: int
    codes: np.ndarray
    """Attribute codes by window."""
    captions: np.ndarray
    """Captions by window."""
    origin: TimeOrigin | None
    """Shared split origin, or ``None`` for independent windows."""


def _add_component(
    dataset: TimeFDataset,
    component: VerbalTsComponent,
    attributes: tuple[Attribute, ...],
) -> dict[str, Record]:
    """Add a component's windows and return its records by ID.

    Args:
        dataset: Dataset to update.
        component: Source component.
        attributes: Attributes in column order.

    Returns:
        The component's records by ID.
    """
    records: dict[str, Record] = {}
    labels = _Labels(component, attributes)
    for split in SPLIT_BY_NAME:
        arrays = _split_arrays(component, split, len(attributes))
        starts = _window_starts(component, arrays, attributes) if component.name in STRIDE else None
        for row in range(arrays.codes.shape[0]):
            start_index = 0 if starts is None else starts[row]
            record = _window_record(component, split, row, arrays, labels, start_index=start_index)
            dataset.add_record(record=record)
            records[record.record_id] = record
    return records


class _Labels:
    """Reuse one annotation object for each attribute code.

    The writer stores the shared label once and records each window's use of it.
    """

    def __init__(self, component: VerbalTsComponent, attributes: tuple[Attribute, ...]) -> None:
        """Prepare annotation caches for one component.

        Args:
            component: Source component.
            attributes: Attributes in column order.
        """
        self.attributes = attributes
        self.component = Annotation(
            key=VerbalTsKey.COMPONENT,
            value=component.name,
            id=f"{_ID_PREFIX}-component-{component.name.lower()}",
        )
        self._source = component
        self._by_code: tuple[dict[int, Annotation], ...] = tuple({} for _ in attributes)

    def of(self, column: int, code: int) -> Annotation:
        """Return the cached annotation for a column and code.

        Args:
            column: Attribute column.
            code: Attribute code.

        Returns:
            The shared annotation.
        """
        cache = self._by_code[column]
        annotation = cache.get(code)
        if annotation is None:
            attribute = self.attributes[column]
            annotation = Annotation(
                key=attribute.key,
                value=_label(self._source, attribute, code),
                description=attribute.description,
                id=f"{_ID_PREFIX}-label-{self._source.name.lower()}-{attribute.key}-{code}",
            )
            cache[code] = annotation
        return annotation


def _split_arrays(component: VerbalTsComponent, split: str, n_attributes: int) -> _SplitArrays:
    """Map and check a split's values, attributes, and captions.

    Args:
        component: Source component.
        split: Split name.
        n_attributes: Expected attribute count.

    Returns:
        The mapped split arrays.

    Raises:
        TimeFFormatError: If the values array has the wrong shape or type.
    """
    values_path = component.npy(split, "ts")
    values = _open_npy(str(values_path))
    if values.ndim != _VALUES_DIMS:
        raise TimeFFormatError(
            f"{values_path.name} holds a {values.ndim}-dimensional array; the release stores "
            f"(n_windows, n_steps, n_signals)"
        )
    if values.dtype != np.float64:
        raise TimeFFormatError(f"{values_path.name} holds {values.dtype} values; the release stores float64")
    n_rows, n_steps, n_signals = values.shape
    return _SplitArrays(
        values_path=values_path,
        n_steps=n_steps,
        n_signals=n_signals,
        codes=_plane(component.npy(split, "attrs_idx"), n_rows, n_attributes),
        captions=_plane(component.npy(split, "text_caps"), n_rows),
        origin=TimeOrigin() if component.name in STRIDE else None,
    )


def _window_starts(
    component: VerbalTsComponent, arrays: _SplitArrays, attributes: tuple[Attribute, ...]
) -> tuple[int, ...]:
    """Find each window's step on its variable's split timeline.

    Each window advances by at most one stride. The final window can move by
    fewer steps or repeat. Reject windows with no valid overlap.

    Args:
        component: Source component.
        arrays: Mapped split arrays.
        attributes: Attributes in column order.

    Returns:
        Start steps in window row order.

    Raises:
        TimeFFormatError: If consecutive windows have no valid overlap.
    """
    stride = STRIDE[component.name]
    values = _open_npy(str(arrays.values_path))
    column = next(column for column, attribute in enumerate(attributes) if attribute.key == VARIABLE)
    previous: dict[int, tuple[int, int]] = {}  # variable code -> (last row, start step)
    starts: list[int] = []
    for row, code in enumerate(int(code) for code in arrays.codes[:, column]):
        if code not in previous:
            start = 0
        else:
            last_row, last_start = previous[code]
            shift = next(
                (
                    shift
                    for shift in range(stride, -1, -1)
                    if np.array_equal(values[row, : arrays.n_steps - shift], values[last_row, shift:])
                ),
                None,
            )
            if shift is None:
                raise TimeFFormatError(
                    f"{arrays.values_path.name} of {component.name}: window {row} does not continue "
                    f"window {last_row} of the same variable by any shift of up to {stride} steps"
                )
            start = last_start + shift
        previous[code] = (row, start)
        starts.append(start)
    return tuple(starts)


def _window_record(  # noqa: PLR0913 - one row needs its split, arrays, labels, and position
    component: VerbalTsComponent,
    split: str,
    row: int,
    arrays: _SplitArrays,
    labels: _Labels,
    *,
    start_index: int,
) -> Record:
    """Build a window record with lazy signals and shared annotations.

    Args:
        component: Source component.
        split: Split name.
        row: Window row.
        arrays: Mapped split arrays.
        labels: Shared annotation cache.
        start_index: Window start step, or zero for independent windows.

    Returns:
        The annotated record.
    """
    identifier = record_id(component.name, split, row)
    annotations = tuple(labels.of(column, int(code)) for column, code in enumerate(arrays.codes[row]))
    values = tuple(annotation.value for annotation in annotations)
    names = _channel_names(component, arrays, row, labels.attributes, values)
    axis = _axis(AXIS_BY_COMPONENT[component.name], start_index)
    signals = tuple(
        Signal.from_loader(
            spec=SPEC_BY_COMPONENT[component.name],
            name=name,
            time_axis=axis,
            loader=_values_loader(str(arrays.values_path), row, signal),
            source_id=identifier,
            id=f"{identifier}-{signal:02d}",
            n_values=arrays.n_steps,
        )
        for signal, name in enumerate(names)
    )
    record = Record(
        record_id=identifier,
        sources=(Source(id=f"{identifier}-source", name=f"VerbalTS {component.name}", signals=signals),),
        start_time=arrays.origin or TimeOrigin(),
    )
    record.add_annotations((labels.component, *annotations))
    return record


def _axis(axis: TimeAxis, start_index: int) -> TimeAxis:
    """Place a window on a regular axis at its start step.

    Args:
        axis: Component axis.
        start_index: Window start step.

    Returns:
        The window's axis.

    Raises:
        TimeFFormatError: If the axis has no cadence for a nonzero start step.
    """
    if start_index == 0:
        return axis
    if not isinstance(axis, RegularAxis):
        raise TimeFFormatError(f"a window cannot start at step {start_index} of an axis without a cadence")
    return axis.at_index(start_index)


def _channel_names(
    component: VerbalTsComponent,
    arrays: _SplitArrays,
    row: int,
    attributes: tuple[Attribute, ...],
    values: tuple[object, ...],
) -> tuple[str, ...]:
    """Return channel names in array order.

    Single-channel windows use their ``variable`` label, checked against the caption.

    Args:
        component: Source component.
        arrays: Mapped split arrays.
        row: Window row.
        attributes: Attributes in column order.
        values: Attribute values in column order.

    Returns:
        Channel names in array order.

    Raises:
        TimeFFormatError: If the signal count differs from the named channels.
    """
    names = CHANNELS.get(component.name)
    if names is None:
        column = next(column for column, attribute in enumerate(attributes) if attribute.key == VARIABLE)
        names = (str(values[column]),)
        _check_caption_names_variable(component, row, str(arrays.captions[row, 0]), names[0])
    if len(names) != arrays.n_signals:
        raise TimeFFormatError(
            f"{arrays.values_path.name} of {component.name} holds {arrays.n_signals} channels per "
            f"window, but the release names {len(names)}"
        )
    return names


def _label(component: VerbalTsComponent, attribute: Attribute, code: int) -> str | int:
    """Return an attribute's label, or its integer code if it has no labels.

    Args:
        component: Source component, used in error messages.
        attribute: Attribute codebook entry.
        code: Code to decode.

    Returns:
        The label or integer code.

    Raises:
        TimeFFormatError: If the code has no label.
    """
    if attribute.labels is None:
        return code
    if not 0 <= code < len(attribute.labels):
        raise TimeFFormatError(
            f"{component.name} holds code {code} for {attribute.key!r}, but its codebook labels only "
            f"codes 0 to {len(attribute.labels) - 1}"
        )
    return attribute.labels[code]


def _check_caption_names_variable(component: VerbalTsComponent, row: int, caption: str, name: str) -> None:
    """Check that the caption's first line ends with the channel name.

    Args:
        component: Source component.
        row: Window row.
        caption: First caption for the window.
        name: Channel name from the variable code.

    Raises:
        TimeFFormatError: If the names differ.
    """
    first_line = caption.split("\n", 1)[0]
    if not first_line.endswith(f" {name}."):
        raise TimeFFormatError(
            f"{component.name} window {row} carries var_id {name!r}, but its caption opens with {first_line!r}"
        )


def _attributes(component: VerbalTsComponent) -> tuple[Attribute, ...]:
    """Match ``meta.json`` to the pinned codebook and return column order.

    Args:
        component: Source component.

    Returns:
        Attributes in column order.

    Raises:
        TimeFFormatError: If names or code counts differ from the codebook.
    """
    names, counts = _read_meta(component)
    codebook = CODEBOOKS[component.name]
    if names != tuple(codebook):
        raise TimeFFormatError(
            f"{component.name}/meta.json states the attributes {list(names)}, but the codebook "
            f"describes {list(codebook)}"
        )
    for name, count in zip(names, counts, strict=True):
        attribute = codebook[name]
        if attribute.labels is not None and count != len(attribute.labels):
            raise TimeFFormatError(
                f"{component.name}/meta.json declares {count} codes for {name!r}, but the codebook "
                f"labels {len(attribute.labels)}"
            )
    return tuple(codebook.values())


def _read_meta(component: VerbalTsComponent) -> tuple[tuple[str, ...], tuple[int, ...]]:
    """Read attribute names and code counts from ``meta.json``.

    Args:
        component: Source component.

    Returns:
        Attribute names and their code counts.

    Raises:
        TimeFFormatError: If the metadata is invalid.
    """
    try:
        meta = json.loads(component.meta_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise TimeFFormatError(f"{component.name}/meta.json is not valid JSON: {error}") from error
    if not isinstance(meta, dict):
        raise TimeFFormatError(f"{component.name}/meta.json holds a JSON {type(meta).__name__}, not an object")
    absent = [key for key in ("attr_list", "attr_n_ops") if key not in meta]
    if absent:
        raise TimeFFormatError(f"{component.name}/meta.json states no {' and no '.join(absent)}")
    try:
        names = tuple(str(name) for name in meta["attr_list"])
        counts = tuple(operator.index(count) for count in meta["attr_n_ops"])
    except (TypeError, ValueError) as error:
        raise TimeFFormatError(
            f"{component.name}/meta.json holds an attr_list or an attr_n_ops this connector cannot read: {error}"
        ) from error
    if len(names) != len(counts):
        raise TimeFFormatError(
            f"{component.name}/meta.json lists {len(names)} attributes but {len(counts)} option counts"
        )
    return names, counts


def _iter_tasks(components: Sequence[VerbalTsComponent], records: Mapping[str, Record]) -> Iterator[TSGenerationTask]:
    """Yield one text-to-series task per caption in component and split order.

    Each call reads the caption arrays again. Tasks use no input record.

    Args:
        components: Source components in release order.
        records: Window records by ID.

    Yields:
        A task for each caption.
    """
    for component in components:
        for split, partition in SPLIT_BY_NAME.items():
            n_rows = _open_npy(str(component.npy(split, "ts"))).shape[0]
            captions = _plane(component.npy(split, "text_caps"), n_rows)
            for row in range(n_rows):
                identifier = record_id(component.name, split, row)
                for caption_index, caption in enumerate(captions[row]):
                    yield TSGenerationTask(
                        id=f"{identifier}-caption-{caption_index}",
                        prompt=str(caption),
                        targets=(records[identifier],),
                        split=partition,
                        input_modalities=frozenset({InputModality.TEXT, InputModality.NO_INPUT}),
                    )


CONNECTOR = VerbalTsConnector
