"""The Original side of the VerbalTS row: the release's own NPY arrays, read directly.

The canonical item is one window with its caption set, which is the same item the TimeF side
delivers. A window lives at row ``r`` of ``{component}/{split}_ts.npy``, an
``(n_windows, n_steps, n_signals)`` float64 array, and the release ships 18 of those over six
components and three splits.

The release ships a torch ``Dataset`` and no pandas path at all. So the torch lane follows the
release, and the four pandas cells of this row are ours: the paper declares them as ours, the same
way it declares the ARFBench row. What a practitioner writes for pandas is what is here, one row
per window with one column per signal, built straight off the array.

Two variants of the same read, recorded as ``loader_variant``:

``as_shipped``
    ``np.load(path)``, which is what the release's own ``Dataset`` does: it reads a split's whole
    array in ``__init__`` and indexes the resident array in ``__getitem__``. A split is read once
    and held for the life of the handle, because that is the bargain the eager loader offers.
    Dropping it between blocks would re-read a 284 MB file at every block boundary, which nobody
    writes.
``lazy_capable``
    ``np.load(path, mmap_mode="r")``, a real and cheap lazy read for NPY: it reads the 128-byte
    header and maps the body, so a window costs the pages that window sits on. Nothing else about
    the read changes.

Both variants slice and convert identically after the load, so the gap between them is the bytes
each one pulled off disk rather than two different pipelines. Both also learn the window count from
the 18 NPY headers, because the enumeration is what the item registry gates the two sides on and it
has to be the same list either way.

Signal names come from :func:`~timenet_connectors.datasets.seqml.verbalts.connector.signal_names`,
the connector's own naming rule, rather than a second copy of it here. The names are the release's
labels: the Jena weather header, the BlindWays joint numbering and the ETTm1 and istanbul_traffic
column names. For those last two a data value picks the name, the window's own ``var_id`` attribute
code, so this lane reads ``{split}_attrs_idx.npy`` for them and for nothing else. A second copy of
that table here could drift from the one the artifact was built with, and then the parity gate could
not tell a real difference from a typo.

The values are float64 and z-scored per variable, and the release never published the constants. So
the units are dimensionless and the physical signal names are labels only.
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from dataclasses import dataclass, field
from fractions import Fraction
import importlib
from pathlib import Path
from typing import Any, NamedTuple

import numpy as np

from benchmarks.paper.errors import LaneError
from benchmarks.paper.lanes import Buffer, DeliveredItem, array_buffers, array_frame, as_buffer
from benchmarks.paper.record import AS_SHIPPED, LAZY_CAPABLE, ORIGINAL, PANDAS, TORCH


DATASET = "seqml/verbalts"
"""The dataset id both representations of this row carry."""

CONNECTOR_MODULE = "timenet_connectors.datasets.seqml.verbalts.connector"
"""Where the record id and the signal naming rule are written down, for both sides of the row."""

VALUES_KIND = "ts"
ATTRIBUTES_KIND = "attrs_idx"
CAPTIONS_KIND = "text_caps"
KINDS: tuple[str, ...] = (VALUES_KIND, ATTRIBUTES_KIND, CAPTIONS_KIND)
"""The three arrays a split ships. A file is ``{component}/{split}_{kind}.npy``."""

META_JSON = "meta.json"
"""Per component, the attribute names and their option counts."""

VALUES_NDIM = 3
"""Dimensions of a values file: windows, steps and signals."""

WINDOW_NDIM = VALUES_NDIM - 1
"""Dimensions of one window taken out of a values file: steps and signals."""

ATTRIBUTES_NDIM = 2
"""Dimensions of an attribute file: windows and attribute codes."""

VAR_ID_COMPONENTS: frozenset[str] = frozenset({"ETTm1", "istanbul_traffic"})
"""The components whose signal name comes out of the data rather than off the header.

A window of either one holds a single signal cut from one column of the upstream table, and its
first attribute code says which column. So naming that signal needs the attribute array, and
getting it wrong names the right values after the wrong variable.
"""

PERIOD_US_BY_COMPONENT: dict[str, Fraction | None] = {
    "synthetic_u": None,
    "synthetic_m": None,
    "Weather": Fraction(600_000_000),
    "BlindWays": Fraction(1_000_000, 60),
    "ETTm1": Fraction(900_000_000),
    "istanbul_traffic": Fraction(600_000_000),
}
"""Microseconds between two steps of a window, per component, or ``None`` for no clock at all.

No component ships a timestamp. The four real-world cadences come from the upstream releases the
components were cut from, which is where the connector takes them from too, and the two synthetic
sets have none: they sit on an ordinal axis on both sides.
"""

N_WINDOWS = 104_628
"""Windows the release holds, over the six components and their three splits."""

TIER_A_BYTES = 799_997_660
"""Apparent bytes of the release: 54 NPY arrays and 6 ``meta.json``.

The captions are in scope even though a values read never opens them. They are what makes a window
an item, and the TimeF artifact this row compares against carries all 130,828 of them as generation
tasks. A Tier A that dropped them would compare two different corpora.
"""


class Window(NamedTuple):
    """One window of one split, addressed the way the release stores it.

    Attributes:
        item_id: The canonical item id, equal to the TimeF record id.
        component: Which of the six components it belongs to.
        split: ``train``, ``valid`` or ``test``.
        row: The window's index inside its split's arrays.
    """

    item_id: str
    component: str
    split: str
    row: int

    def values_file(self) -> str:
        """Return the source file this window's values sit in.

        Returns:
            The path of the values array, relative to the release directory.
        """
        return f"{self.component}/{self.split}_{VALUES_KIND}.npy"


def windows(root: Path) -> tuple[Window, ...]:
    """Walk the release's 18 values files and return one entry per window.

    Only the NPY headers are read, so this costs 18 opens rather than 560 MB. The result is sorted
    by item id, which is the order the TimeF artifact stores its records in, so ordinal ``k`` is the
    same window on both sides of the row. Inside one component and split that order is the release's
    own row order, because the id pads the row index to five digits.

    Args:
        root: The release directory, holding one folder per component.

    Returns:
        The windows, in canonical enumeration order.

    Raises:
        LaneError: If a values file is missing, or does not hold a three-dimensional array.
    """
    # Lane dependencies, imported here rather than at module import.
    from timenet_connectors.datasets.seqml.verbalts.connector import record_id  # noqa: PLC0415
    from timenet_connectors.datasets.seqml.verbalts.files import COMPONENTS, SPLITS  # noqa: PLC0415

    found: list[Window] = []
    for component in COMPONENTS:
        for split in SPLITS:
            path = root / component / f"{split}_{VALUES_KIND}.npy"
            n_rows = _window_count(path)
            found.extend(Window(record_id(component, split, row), component, split, row) for row in range(n_rows))
    if not found:
        raise LaneError(f"{root} holds no VerbalTS window, so this lane has nothing to enumerate")
    return tuple(sorted(found, key=lambda window: window.item_id))


def item_ids(root: Path) -> tuple[str, ...]:
    """Return the canonical enumeration: one id per window, in stored order.

    The ids are the TimeF record ids, so the item registry can gate the two sides against each other.

    Args:
        root: The release directory.

    Returns:
        The item ids.
    """
    return tuple(window.item_id for window in windows(root))


def tier_a_bytes(root: Path) -> int:
    """Return the bytes of the release this row is measured against.

    Args:
        root: The release directory.

    Returns:
        The apparent size of the 54 arrays and the 6 ``meta.json``.

    Raises:
        LaneError: If one of the 60 files is not there.
    """
    from timenet_connectors.datasets.seqml.verbalts.files import COMPONENTS, SPLITS  # noqa: PLC0415

    names = [META_JSON, *(f"{split}_{kind}.npy" for split in SPLITS for kind in KINDS)]
    try:
        return sum((root / component / name).stat().st_size for component in COMPONENTS for name in names)
    except OSError as error:
        raise LaneError(f"could not size the VerbalTS release under {root}: {error}") from error


def time_axis_of(component: str) -> Any:
    """Return the axis every window of one component sits on.

    Args:
        component: The component the window belongs to.

    Returns:
        A :class:`~timenet.dataset.axis.RegularAxis` at the component's cadence, or an
        :class:`~timenet.dataset.axis.OrdinalAxis` for a component with no clock.

    Raises:
        LaneError: If the component is not one of the six.
    """
    # Lane dependencies, imported here rather than at module import.
    from timenet.dataset.axis import OrdinalAxis, RegularAxis  # noqa: PLC0415

    if component not in PERIOD_US_BY_COMPONENT:
        raise LaneError(f"{component!r} is not a VerbalTS component, so this lane has no cadence for it")
    period = PERIOD_US_BY_COMPONENT[component]
    return OrdinalAxis() if period is None else RegularAxis(period_us=period)


@dataclass
class VerbalTsHandle:
    """An open release directory, delivering windows as one consumer's items.

    Attributes:
        root: The release directory.
        windows: The canonical enumeration, one entry per ordinal.
        consumer: ``pandas`` or ``torch``.
        loader_variant: ``as_shipped`` or ``lazy_capable``.
        arrays: The arrays this handle has opened, keyed by path.
        names: Signal names already resolved, keyed by component, ``var_id`` and signal count.
    """

    root: Path
    windows: tuple[Window, ...]
    consumer: str
    loader_variant: str
    arrays: dict[str, np.ndarray] = field(default_factory=dict)
    names: dict[tuple[str, int, int], tuple[str, ...]] = field(default_factory=dict)

    def n_items(self) -> int:
        """Return how many canonical items this lane holds.

        Returns:
            The item count.
        """
        return len(self.windows)

    def deliver(self, ordinals: Sequence[int]) -> Iterator[DeliveredItem]:
        """Read one request's worth of windows and convert them to the consumer's type.

        Args:
            ordinals: Canonical ordinals to read.

        Yields:
            One item per ordinal, in the order asked for.
        """
        for ordinal in ordinals:
            window = self._window(ordinal)
            values = self._array(window.component, window.split, VALUES_KIND)[window.row]
            yield DeliveredItem(ordinal=ordinal, item_id=window.item_id, buffers=self._buffers(window, values))

    def group_of(self, ordinal: int) -> object:
        """Return the source file this item came from.

        Returns:
            The values file, which is the Original side's storage group.
        """
        return self._window(ordinal).values_file()

    def close(self) -> None:
        """Drop the arrays and the resolved names the handle was holding."""
        self.arrays.clear()
        self.names.clear()

    def _window(self, ordinal: int) -> Window:
        """Return the window at one ordinal.

        Returns:
            The window.

        Raises:
            LaneError: If the ordinal is outside the enumeration.
        """
        if not 0 <= ordinal < len(self.windows):
            raise LaneError(f"a request asked for an ordinal outside the {len(self.windows)} items")
        return self.windows[ordinal]

    def _array(self, component: str, split: str, kind: str) -> np.ndarray:
        """Open one array, reusing it once this handle has opened it.

        Returns:
            The whole array under ``as_shipped``, or a memory map under ``lazy_capable``.

        Raises:
            LaneError: If the variant is unknown, or the file cannot be read.
        """
        path = self.root / component / f"{split}_{kind}.npy"
        key = str(path)
        held = self.arrays.get(key)
        if held is not None:
            return held
        if self.loader_variant == AS_SHIPPED:
            mmap_mode = None
        elif self.loader_variant == LAZY_CAPABLE:
            mmap_mode = "r"
        else:
            raise LaneError(f"the VerbalTS lane reads {AS_SHIPPED} or {LAZY_CAPABLE}, got {self.loader_variant!r}")
        try:
            array = np.load(path, mmap_mode=mmap_mode)
        except (OSError, ValueError) as error:
            raise LaneError(f"could not read the VerbalTS array {path}: {error}") from error
        self.arrays[key] = array
        return array

    def _signal_names(self, window: Window, n_signals: int) -> tuple[str, ...]:
        """Name every signal of one window, resolving each distinct shape once.

        Returns:
            One name per signal.
        """
        # A lane dependency, imported per call rather than at module import.
        from timenet_connectors.datasets.seqml.verbalts.connector import signal_names  # noqa: PLC0415

        var_id = self._var_id(window) if window.component in VAR_ID_COMPONENTS else -1
        key = (window.component, var_id, n_signals)
        held = self.names.get(key)
        if held is not None:
            return held
        codes = (var_id,) if var_id >= 0 else ()
        resolved = signal_names(window.component, codes, n_signals)
        self.names[key] = resolved
        return resolved

    def _var_id(self, window: Window) -> int:
        """Return the attribute code that picks a single-signal window's variable.

        Returns:
            The window's first attribute code.

        Raises:
            LaneError: If the attribute array holds no code for this window.
        """
        attributes = self._array(window.component, window.split, ATTRIBUTES_KIND)
        if attributes.ndim != ATTRIBUTES_NDIM or window.row >= len(attributes) or attributes.shape[1] < 1:
            raise LaneError(
                f"{window.component}/{window.split}_{ATTRIBUTES_KIND}.npy has shape {attributes.shape}, "
                f"which carries no var_id for window {window.row}"
            )
        return int(attributes[window.row, 0])

    def _buffers(self, window: Window, values: np.ndarray) -> tuple[Buffer, ...]:
        """Build the consumer's own type and return its value planes.

        Returns:
            One buffer per value plane, which the checksum walks inside the timed region.

        Raises:
            LaneError: If the window is not a two-dimensional slab, or the consumer is not one this
                lane implements.
        """
        if values.ndim != WINDOW_NDIM:
            raise LaneError(
                f"window {window.item_id} is {values.ndim}-dimensional; a VerbalTS values file holds "
                f"(windows, steps, signals)"
            )
        _, n_signals = values.shape
        if self.consumer == PANDAS:
            axis = time_axis_of(window.component)
            frame = array_frame(
                window.item_id,
                [
                    (name, axis, np.array(values[:, signal], dtype=np.float64))
                    for signal, name in enumerate(self._signal_names(window, n_signals))
                ],
            )
            return array_buffers(frame)
        if self.consumer == TORCH:
            import torch  # noqa: PLC0415 - as above

            return tuple(
                as_buffer(torch.from_numpy(np.array(values[:, signal], dtype=np.float64)).numpy())
                for signal in range(n_signals)
            )
        raise LaneError(f"the VerbalTS lane has no {self.consumer!r} consumer; it implements {PANDAS} and {TORCH}")


@dataclass(frozen=True)
class VerbalTsLane:
    """One Original lane over the VerbalTS release: a directory, a consumer and a read variant.

    Attributes:
        consumer: ``pandas`` or ``torch``.
        root: The release directory, holding one folder per component.
        tier_a_bytes: Tier A bytes of this release, which size the blocks.
        loader_variant: ``as_shipped`` or ``lazy_capable``.
        dataset: Dataset id.
        name: The lane name that reaches the cell record.
    """

    consumer: str
    root: Path
    tier_a_bytes: int
    loader_variant: str = LAZY_CAPABLE
    dataset: str = DATASET
    name: str = ""
    representation: str = ORIGINAL

    def __post_init__(self) -> None:
        """Refuse a lane that cannot be measured.

        Raises:
            LaneError: If the consumer or the variant is unknown, or Tier A bytes are not positive.
        """
        if self.consumer not in {PANDAS, TORCH}:
            raise LaneError(f"the VerbalTS lane implements {PANDAS} and {TORCH}, got {self.consumer!r}")
        if self.loader_variant not in {AS_SHIPPED, LAZY_CAPABLE}:
            raise LaneError(f"the VerbalTS lane reads {AS_SHIPPED} or {LAZY_CAPABLE}, got {self.loader_variant!r}")
        if self.tier_a_bytes < 1:
            raise LaneError(
                f"the VerbalTS lane needs positive Tier A bytes to size its blocks, got {self.tier_a_bytes}"
            )
        if not self.name:
            object.__setattr__(self, "name", f"{self.dataset}/{ORIGINAL}/{self.consumer}/{self.loader_variant}")

    def preload(self) -> None:
        """Import everything this lane reads with, without touching the release directory.

        The enumeration and the signal naming always need the connector module, and the consumer
        decides the delivered type. Both are imported on first use in the normal path, which is right
        for a timed read but wrong for the memory baseline: torch alone costs a few hundred megabytes
        before it touches data. This pulls them forward so the baseline can be taken after them.
        """
        modules = [CONNECTOR_MODULE, "timenet.dataset.axis"]
        modules.extend(("pandas", "timenet.pandas") if self.consumer == PANDAS else ("torch",))
        for module in modules:
            importlib.import_module(module)

    def open(self) -> VerbalTsHandle:
        """Open the release: read the 18 values headers and enumerate the windows.

        No values page is touched here, under either variant. That is the honest split for the
        first-item cell: opening costs the headers, and the first delivery costs whatever the variant
        has to read to reach one window.

        Returns:
            The open handle.

        Raises:
            LaneError: If the release directory is not there.
        """
        if not self.root.is_dir():
            raise LaneError(f"lane {self.name} could not open {self.root}: it is not a directory")
        return VerbalTsHandle(
            root=self.root,
            windows=windows(self.root),
            consumer=self.consumer,
            loader_variant=self.loader_variant,
        )


def _window_count(path: Path) -> int:
    """Read one values file's header and return how many windows it holds.

    ``np.load`` with ``mmap_mode`` reads the 128-byte header and maps the body, so no page of the
    values is read. Both variants enumerate this way, because the enumeration is the list the item
    registry gates the two sides on rather than part of the values read.

    Args:
        path: The ``{split}_ts.npy`` file.

    Returns:
        The window count.

    Raises:
        LaneError: If the file is missing, or does not hold a three-dimensional array.
    """
    try:
        shape = np.load(path, mmap_mode="r").shape
    except (OSError, ValueError) as error:
        raise LaneError(f"could not read the VerbalTS values header of {path}: {error}") from error
    if len(shape) != VALUES_NDIM:
        raise LaneError(f"{path} holds a {len(shape)}-D array; a VerbalTS values file is (windows, steps, signals)")
    return int(shape[0])
