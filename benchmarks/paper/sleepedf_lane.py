"""The Original side of the lane protocol for Sleep-EDF Expanded, in three loader variants.

The release ships 197 ``*-PSG.edf`` recordings with a ``*-Hypnogram.edf`` scoring beside each one,
and no Python loader. The canonical item is one scored 30 s epoch in the six stages the release
writes, so a lane cuts the epochs out of the hypnograms and reads the matching window of every
signal.

**The parser is edfio on both sides.** The TimeF artifact was written by the Sleep-EDF connector,
which parses with ``edfio`` at native rates, so the Original lane parses with ``edfio`` at native
rates too and shares :func:`~timenet_connectors.bases.edf.reader.convert_digital_to_physical` with
it. The two sides then carry the same physical float32 values and differ only in the container.
Reading the Original side through MNE instead would upsample the four 1 Hz signals of a cassette
recording to 100 Hz, which is 99% interpolation, a 100x value inflation and a 68 GiB working set,
and none of that is a fact about EDF.

Three loader variants, each recorded as ``loader_variant``:

``as_shipped``
    ``edfio.read_edf(path, lazy_load_data=False)``: the whole recording is decoded into memory
    before the first epoch comes out. No loader ships with this release, so this variant is ours,
    and it stands for the whole-recording read that every published Sleep-EDF pipeline performs.

``lazy_capable``
    ``edfio.read_edf(path, lazy_load_data=True)`` plus ``get_digital_slice``, which memory-maps the
    data records and reads only the records one epoch covers. edfio picks this itself for a path
    argument, so the variant costs a caller nothing, and saying so is part of the result.

``reference_pipeline``
    PyHealth's ``SleepStagingSleepEDF`` call sequence over MNE, at 100 Hz across every signal. It
    is the pipeline practitioners run, not a like-for-like container read, and it is reported as
    such. MNE crops a scoring to the signals, so an epoch that overruns them produces no row here;
    the handle counts those in ``pipeline_dropped`` and delivers the item empty, so the item set
    stays the canonical one. The call sequence is run over MNE rather than through PyHealth's own
    dataset class, so nothing here writes the LitData cache PyHealth builds on a first pass.
    :meth:`SleepEdfxLane.derived_caches` still names it, so a campaign can declare it and clear it
    rather than measure a second run against chunks a first one left behind.

An epoch that runs past the end of its own signals is kept, because the connector kept it: it warns
with ``SpanOutsideWindowWarning`` and writes the task anyway. 509 of the 458,161 items are in that
state and 504 of them hold no samples at all, so both sides deliver the item and let it carry the
values the signals hold, which for those 504 is nothing.
"""

from __future__ import annotations

from collections import OrderedDict
from collections.abc import Iterator, Sequence
from dataclasses import dataclass, field
from decimal import Decimal
from fractions import Fraction
import importlib
from pathlib import Path
from typing import Any
import warnings

import numpy as np

from benchmarks.paper.derived import DerivedCache, pyhealth_cache
from benchmarks.paper.errors import LaneError
from benchmarks.paper.items import EpochGrid, ItemRef, ScoredRun, epoch_grid_items
from benchmarks.paper.lanes import Buffer, DeliveredItem, array_buffers, array_frame, as_buffer
from benchmarks.paper.record import AS_SHIPPED, LAZY_CAPABLE, ORIGINAL, PANDAS, TORCH


DATASET = "physionet/sleep-edfx"
"""The dataset id the cell record carries."""

ID_PREFIX = "sleep-edfx"
"""The prefix the connector writes in front of every id, so both sides name the same record."""

RELEASE_DIRNAME = "sleep-edf-database-expanded-1.0.0"
"""The directory the release archive extracts to."""

STUDIES = ("sleep-cassette", "sleep-telemetry")
"""The two studies, in the order the connector walks them. It fixes the enumeration order."""

US_PER_S = 1_000_000

EPOCH_US = 30 * US_PER_S
"""One scored epoch, from the 1968 Rechtschaffen and Kales manual."""

SCORED_LABELS = frozenset(
    {
        "Sleep stage W",
        "Sleep stage 1",
        "Sleep stage 2",
        "Sleep stage 3",
        "Sleep stage 4",
        "Sleep stage R",
    }
)
"""The six stages an item is cut for. ``Sleep stage ?`` and ``Movement time`` produce none."""

SLEEP_EDFX_GRID = EpochGrid(key="sleep_stage", epoch_us=EPOCH_US, keep_labels=SCORED_LABELS)
"""How the run-length scoring is cut, shared with whatever registers the item spec."""

STAGE_VOCABULARY = f"{ID_PREFIX}-vocabulary-sleep_stage"
"""The target schema the connector writes on an epoch task, which no other task carries.

It is how the TimeF side finds the same epochs in the artifact: the recordings are records there
and the scored windows are tasks, so the epoch enumeration comes off the tasks that score against
this vocabulary."""

N_ITEMS = 458_161
"""What the release yields: 483,419 epochs, less 25,047 unscored and 211 movement epochs."""

REFERENCE_PIPELINE = "reference_pipeline"
"""The third loader variant. It is a training pipeline, not a container read."""

LOADER_VARIANTS = (AS_SHIPPED, LAZY_CAPABLE, REFERENCE_PIPELINE)

PYHEALTH_EVENT_ID = {
    "Sleep stage W": 0,
    "Sleep stage 1": 1,
    "Sleep stage 2": 2,
    "Sleep stage 3": 3,
    "Sleep stage 4": 4,
    "Sleep stage R": 5,
}
"""The six stages PyHealth maps, copied from ``pyhealth.tasks.sleep_staging_v2``."""

_OPEN_RECORDINGS = {AS_SHIPPED: 1, LAZY_CAPABLE: 8, REFERENCE_PIPELINE: 1}
"""How many recordings a variant keeps open.

An eager recording is 48 MB of decoded int16 and a reference-pipeline one is 445 MB of float64, so
those hold one. A lazy recording is a memory map of the data records and costs almost no resident
bytes, so those hold a few and stop reopening a file the next block asks for again.
"""


def release_root(cache_dir: Path) -> Path:
    """Return the extracted release directory under a download cache.

    Args:
        cache_dir: The connector's cache directory for this dataset.

    Returns:
        The directory that holds the two study directories and the two subject sheets.
    """
    return cache_dir / RELEASE_DIRNAME


def psg_paths(root: Path) -> dict[str, Path]:
    """Return each record's PSG file, keyed by the record id the connector writes.

    Args:
        root: The extracted release directory.

    Returns:
        Record id to ``*-PSG.edf`` path, in study order and then filename order.

    Raises:
        LaneError: If a study directory is missing or holds no recording.
    """
    found: dict[str, Path] = {}
    for study in STUDIES:
        study_dir = root / study
        if not study_dir.is_dir():
            raise LaneError(f"the Sleep-EDF release at {root} has no {study!r} directory")
        recordings = sorted(study_dir.glob("*-PSG.edf"))
        if not recordings:
            raise LaneError(f"{study_dir} holds no *-PSG.edf recording")
        for path in recordings:
            found[f"{ID_PREFIX}-{path.stem.removesuffix('-PSG')}"] = path
    return found


def hypnogram_path(psg_path: Path) -> Path:
    """Return the scoring that belongs to one recording.

    A hypnogram name ends with the initial of the technician who scored it, which the PSG name does
    not predict, so this matches on the seven characters the two names share.

    Args:
        psg_path: The ``*-PSG.edf`` file.

    Returns:
        The ``*-Hypnogram.edf`` file beside it.

    Raises:
        LaneError: If the directory holds no scoring for this recording, or more than one.
    """
    matches = sorted(psg_path.parent.glob(f"{psg_path.name[:7]}?-Hypnogram.edf"))
    if len(matches) != 1:
        found = ", ".join(match.name for match in matches) or "none"
        raise LaneError(f"expected one scoring for {psg_path.name}, found {found}")
    return matches[0]


def scored_runs(root: Path) -> tuple[ScoredRun, ...]:
    """Read the 197 hypnograms and return the run-length scoring they hold.

    The release stores a stretch of equal epochs as one EDF+ annotation, so this returns one run
    per annotation rather than one per epoch. Every label is returned, including the two the grid
    drops, because the grid is what decides.

    Args:
        root: The extracted release directory.

    Returns:
        One run per scored annotation, in study order, then filename order, then file order.
    """
    runs: list[ScoredRun] = []
    for record_id, path in psg_paths(root).items():
        for onset, duration, label in _read_annotations(hypnogram_path(path)):
            runs.append(ScoredRun(record_id=record_id, label=label, start_us=onset, end_us=onset + duration))
    return tuple(runs)


def tier_a_bytes(root: Path) -> int:
    """Return the bytes this lane opens: every PSG recording and the scoring beside it.

    Derived from the tree the campaign points at, because the block size comes out of this number
    and an apparent size is a property of the extracted release, not a constant to carry around.

    Args:
        root: The extracted release directory.

    Returns:
        The apparent size of those files.

    Raises:
        LaneError: If a file is missing.
    """
    try:
        return sum(path.stat().st_size + hypnogram_path(path).stat().st_size for path in psg_paths(root).values())
    except OSError as error:
        raise LaneError(f"could not size the Sleep-EDF raw files: {error}") from error


def scored_epochs(root: Path) -> tuple[ItemRef, ...]:
    """Enumerate the canonical items of Sleep-EDF from the raw release.

    The signal ends are deliberately not passed to :func:`epoch_grid_items`. The connector keeps an
    epoch whose window overruns its signals, warns, and writes the task, so dropping those here
    would put this side 509 items below the artifact.

    Args:
        root: The extracted release directory.

    Returns:
        One reference per scored epoch, in the connector's task order.

    Raises:
        LaneError: If the walk does not produce the count the release is known to hold.
    """
    refs = tuple(epoch_grid_items(scored_runs(root), SLEEP_EDFX_GRID, {})())
    if len(refs) != N_ITEMS:
        raise LaneError(f"the release at {root} enumerated {len(refs)} epochs, and Sleep-EDF holds {N_ITEMS}")
    return refs


@dataclass(frozen=True)
class _Signal:
    """One signal of one recording, with everything an epoch slice needs.

    Attributes:
        time_series_id: The id the connector writes, so both sides label the same column.
        signal: The label the header states.
        rate_hz: The native rate. Nothing here resamples.
        period_us: Microseconds per sample, as an exact fraction.
        edf_signal: The ``edfio.EdfSignal`` the slice is read from.
        calibration: The four header values that turn counts into physical units.
    """

    time_series_id: str
    signal: str
    rate_hz: Fraction
    period_us: Fraction
    edf_signal: Any
    calibration: tuple[float, float, float, float]


@dataclass(frozen=True)
class _Recording:
    """One open recording: its signals, and where its signals stop.

    Attributes:
        record_id: The record the recording is stored as.
        signals: Its signals, in header order.
        signal_end_us: Where the last recorded sample ends, in microseconds from the first.
    """

    record_id: str
    signals: tuple[_Signal, ...]
    signal_end_us: int


_PipelineRun = tuple[np.ndarray, dict[int, int], tuple[str, ...], Fraction]
"""What one reference-pipeline run holds: epochs, onset to row, signal labels, sample period."""


@dataclass
class SleepEdfxHandle:
    """An open Sleep-EDF release, delivering one scored epoch as one consumer's item.

    The handle keeps a small number of recordings open and evicts the least recently used one, so a
    pass over a corpus larger than memory holds one recording rather than the release.

    Attributes:
        refs: The canonical enumeration, one reference per ordinal.
        paths: Record id to PSG path, resolved when the lane opened.
        consumer: ``pandas`` or ``torch``.
        loader_variant: One of :data:`LOADER_VARIANTS`.
        max_open: How many recordings stay open at once.
        pipeline_dropped: Items the reference pipeline does not produce. Zero for the other two.
    """

    refs: tuple[ItemRef, ...]
    paths: dict[str, Path]
    consumer: str
    loader_variant: str
    max_open: int
    pipeline_dropped: int = 0
    _open: OrderedDict[str, Any] = field(default_factory=OrderedDict, repr=False)

    def n_items(self) -> int:
        """Return how many canonical items this lane holds.

        Returns:
            The item count.
        """
        return len(self.refs)

    def deliver(self, ordinals: Sequence[int]) -> Iterator[DeliveredItem]:
        """Read one request's worth of epochs and convert them to the consumer's type.

        Args:
            ordinals: Canonical ordinals to read, in the order the plan asked for them.

        Yields:
            One item per ordinal.

        Raises:
            LaneError: If an ordinal is outside the enumeration.
        """
        for ordinal in ordinals:
            try:
                ref = self.refs[ordinal]
            except IndexError as error:
                raise LaneError(f"a request asked for an ordinal outside the {len(self.refs)} items") from error
            yield DeliveredItem(ordinal=ordinal, item_id=ref.item_id, buffers=self._buffers(ref))

    def group_of(self, ordinal: int) -> object:
        """Return the source file this item came from.

        Returns:
            The PSG filename, which is the Original side's storage group.
        """
        return self.paths[self.refs[ordinal].record_id].name

    def close(self) -> None:
        """Drop every open recording."""
        self._open.clear()

    def _buffers(self, ref: ItemRef) -> tuple[Buffer, ...]:
        """Read one epoch and return the value planes of the consumer's own type.

        Returns:
            One buffer per value plane, which the checksum walks inside the timed region.

        Raises:
            LaneError: If the item carries no window, or the consumer is not one this lane
                implements.
        """
        if ref.start_step is None or ref.stop_step is None:
            raise LaneError(f"item {ref.item_id} carries no epoch window, so no lane can read it")
        if self.loader_variant == REFERENCE_PIPELINE:
            return self._pipeline_buffers(ref, ref.start_step)
        recording = self._recording(ref.record_id)
        planes = _epoch_planes(recording, ref.start_step, ref.stop_step)
        return _to_consumer(self.consumer, ref.record_id, planes)

    def _recording(self, record_id: str) -> _Recording:
        """Return an open recording, opening it and evicting another when it is not open yet.

        Returns:
            The open recording.

        Raises:
            LaneError: If the release holds no recording for that record.
        """
        held = self._open.get(record_id)
        if held is not None:
            self._open.move_to_end(record_id)
            return held
        path = self.paths.get(record_id)
        if path is None:
            raise LaneError(f"the release holds no recording for record {record_id!r}")
        opened = _open_recording(record_id, path, lazy=self.loader_variant != AS_SHIPPED)
        self._admit(record_id, opened)
        return opened

    def _pipeline_buffers(self, ref: ItemRef, start_us: int) -> tuple[Buffer, ...]:
        """Read one epoch out of the reference pipeline's epoch array.

        Args:
            ref: The item to read.
            start_us: Where its epoch starts, which is how the pipeline's rows are matched.

        Returns:
            The value planes, or an empty tuple for an epoch the pipeline cropped away.

        Raises:
            LaneError: If the release holds no recording for that record.
        """
        held = self._open.get(ref.record_id)
        if held is None:
            path = self.paths.get(ref.record_id)
            if path is None:
                raise LaneError(f"the release holds no recording for record {ref.record_id!r}")
            held = _run_reference_pipeline(ref.record_id, path, hypnogram_path(path))
            self._admit(ref.record_id, held)
        else:
            self._open.move_to_end(ref.record_id)
        epochs, row_of, signals, period_us = held
        row = row_of.get(start_us)
        if row is None:
            self.pipeline_dropped += 1
            return ()
        first = start_us * period_us.denominator // period_us.numerator
        planes = tuple(
            (f"{ref.record_id}-{signal}", signal, period_us, first, epochs[row, index])
            for index, signal in enumerate(signals)
        )
        return _to_consumer(self.consumer, ref.record_id, planes)

    def _admit(self, record_id: str, opened: Any) -> None:
        """Hold one more recording open, evicting the least recently used one over the limit."""
        self._open[record_id] = opened
        while len(self._open) > self.max_open:
            self._open.popitem(last=False)


@dataclass(frozen=True)
class SleepEdfxLane:
    """One Original lane over the Sleep-EDF release: a release directory, items and a consumer.

    Attributes:
        consumer: ``pandas`` or ``torch``.
        root: The extracted release directory.
        items: The canonical enumeration, one reference per ordinal.
        tier_a_bytes: Tier A bytes of the raw release, which sizes the blocks.
        loader_variant: One of :data:`LOADER_VARIANTS`.
        name: The lane name that reaches the cell record.
        max_open: How many recordings stay open at once. ``0`` takes the variant's own default.
        dataset: Dataset id.
    """

    consumer: str
    root: Path
    items: tuple[ItemRef, ...]
    tier_a_bytes: int
    loader_variant: str = LAZY_CAPABLE
    name: str = ""
    max_open: int = 0
    dataset: str = DATASET
    representation: str = ORIGINAL

    def __post_init__(self) -> None:
        """Refuse a lane that cannot be measured.

        Raises:
            LaneError: If the consumer or the loader variant is unknown, the enumeration is empty,
                or Tier A bytes are not positive.
        """
        if self.consumer not in {PANDAS, TORCH}:
            raise LaneError(f"the Sleep-EDF lane implements {PANDAS} and {TORCH}, got {self.consumer!r}")
        if self.loader_variant not in LOADER_VARIANTS:
            raise LaneError(f"loader_variant must be one of {list(LOADER_VARIANTS)}, got {self.loader_variant!r}")
        if not self.items:
            raise LaneError(f"the Sleep-EDF lane for {self.dataset} has no items to enumerate")
        if self.tier_a_bytes < 1:
            raise LaneError(f"the Sleep-EDF lane for {self.dataset} needs positive Tier A bytes to size its blocks")
        if not self.name:
            object.__setattr__(self, "name", f"{self.dataset}/{ORIGINAL}/{self.consumer}/{self.loader_variant}")

    def derived_caches(self) -> tuple[DerivedCache, ...]:
        """Return the on-disk caches this lane's loader can write.

        The two edfio variants write none: edfio decodes into memory and leaves nothing behind. The
        reference pipeline names PyHealth, whose LitData chunks turn a second pass into a different
        measurement, so it is named here even though this lane runs the call sequence over MNE and
        never builds it.

        Returns:
            One cache for the reference pipeline, and none for the other two variants.
        """
        return (pyhealth_cache(),) if self.loader_variant == REFERENCE_PIPELINE else ()

    def preload(self) -> None:
        """Import everything this lane reads with, without listing the release directory.

        edfio reads the annotations in every variant and the signals in the two edfio ones, the
        reference pipeline reads its signals through MNE, and the consumer decides the delivered
        type. All of them are imported on first use in the normal path, which is right for a timed
        read but wrong for the memory baseline. This pulls them forward so the baseline can be taken
        after them.

        Raises:
            LaneError: If the reference pipeline's MNE is not installed. It lives in that lane's own
                environment, not the shared one.
        """
        modules = ["edfio", "timenet.dataset.axis", "timenet_connectors.bases.edf.reader"]
        modules.extend(("pandas", "timenet.pandas") if self.consumer == PANDAS else ("torch",))
        for module in modules:
            importlib.import_module(module)
        if self.loader_variant == REFERENCE_PIPELINE:
            try:
                importlib.import_module("mne")
            except ImportError as error:
                raise LaneError(f"the {REFERENCE_PIPELINE} variant needs mne installed: {error}") from error

    def open(self) -> SleepEdfxHandle:
        """Resolve the release's recordings and return a handle that has read no signal yet.

        A directory of EDF files carries no manifest, so this lists the two study directories.
        That is the Original side's cheap half of the first-item cell, and the expensive half is
        the first delivery.

        Returns:
            The open handle.

        Raises:
            LaneError: If the release directory cannot be walked.
        """
        try:
            paths = psg_paths(self.root)
        except OSError as error:
            raise LaneError(f"lane {self.name} could not open {self.root}: {error}") from error
        return SleepEdfxHandle(
            refs=self.items,
            paths=paths,
            consumer=self.consumer,
            loader_variant=self.loader_variant,
            max_open=self.max_open or _OPEN_RECORDINGS[self.loader_variant],
        )


Plane = tuple[str, str, Fraction, int, np.ndarray]
"""One signal's slice of one epoch: its series id, label, sample period, first index and values."""


def _epoch_planes(recording: _Recording, start_us: int, stop_us: int) -> tuple[Plane, ...]:
    """Cut one epoch out of every signal of a recording, at native rates.

    The window is clipped to where the signals stop. The connector kept an epoch that overruns
    them, so this returns the samples that exist rather than dropping the item or padding it.

    Returns:
        One plane per signal, in header order.
    """
    limit = min(stop_us, recording.signal_end_us)
    planes: list[Plane] = []
    for signal in recording.signals:
        first = _sample_index(start_us, signal.rate_hz)
        last = max(first, _sample_index(limit, signal.rate_hz))
        if last == first:
            values = np.empty(0, dtype=np.float32)
        else:
            digital = signal.edf_signal.get_digital_slice(start_us / US_PER_S, limit / US_PER_S)
            values = _to_physical(digital, signal.calibration)
        planes.append((signal.time_series_id, signal.signal, signal.period_us, first, values))
    return tuple(planes)


def _sample_index(when_us: int, rate_hz: Fraction) -> int:
    """Return the index of the sample at a microsecond offset.

    Returns:
        The index, rounded down.
    """
    return int(when_us * rate_hz // US_PER_S)


def _to_physical(digital: np.ndarray, calibration: tuple[float, float, float, float]) -> np.ndarray:
    """Turn stored counts into the physical unit the header names.

    This is the connector's own conversion, so both sides of the row carry the same float32 values.

    Returns:
        The values in physical units, as float32.
    """
    # Imported per call site, not at module import, so the harness stays importable without the
    # connectors package installed.
    from timenet_connectors.bases.edf.reader import convert_digital_to_physical  # noqa: PLC0415

    digital_min, digital_max, physical_min, physical_max = calibration
    return convert_digital_to_physical(
        digital,
        digital_min=digital_min,
        digital_max=digital_max,
        physical_min=physical_min,
        physical_max=physical_max,
    )


def _to_consumer(consumer: str, record_id: str, planes: tuple[Plane, ...]) -> tuple[Buffer, ...]:
    """Turn one epoch's signal slices into the consumer's own type.

    Returns:
        One buffer per value plane of that type.

    Raises:
        LaneError: If the consumer is not one this lane implements.
    """
    if consumer == PANDAS:
        return array_buffers(epoch_frame(record_id, planes))
    if consumer == TORCH:
        return tuple(as_buffer(tensor.numpy()) for tensor in epoch_tensors(planes))
    raise LaneError(f"the Sleep-EDF lane has no {consumer!r} consumer; it implements {PANDAS} and {TORCH}")


def epoch_frame(record_id: str, planes: tuple[Plane, ...]) -> Any:
    """Return one epoch as one row, one column per signal, each cell that signal's whole slice.

    A cassette recording runs three signals at 100 Hz beside four at 1 Hz, so an epoch's slices have
    two lengths and no wide table of values can hold both. Array cells can: a cell is a slice as
    read. This is the shape :func:`timenet.pandas.record_array_frame` builds off a record, so the
    two sides of the row hold the same frame.

    A signal's clock rides as a :class:`~timenet.dataset.axis.RegularAxis` started at the epoch's
    own first sample. That is exact at every rate, and it costs two numbers rather than one int64
    per value.

    Args:
        record_id: The record the epoch was cut from.
        planes: One slice per signal, from :func:`_epoch_planes`.

    Returns:
        A one-row frame: ``record_id``, ``time_axis``, then one column per signal.
    """
    # Imported per call, not at module import, so the torch lane does not need either.
    from timenet.dataset.axis import RegularAxis  # noqa: PLC0415

    return array_frame(
        record_id,
        [
            (signal, RegularAxis(period_us=period_us, start_index=first), values)
            for _, signal, period_us, first, values in planes
        ],
    )


def epoch_tensors(planes: tuple[Plane, ...]) -> tuple[Any, ...]:
    """Return one epoch as one tensor per signal.

    The path is memory map to float32 numpy to tensor, with no frame in between, which is what
    :func:`timenet.torch.record_item` returns on the other side.

    Args:
        planes: One slice per signal, from :func:`_epoch_planes`.

    Returns:
        One tensor per signal, in header order.
    """
    # Imported per call, not at module import, so the pandas lane does not need torch.
    import torch  # noqa: PLC0415

    return tuple(torch.from_numpy(np.ascontiguousarray(plane[4])) for plane in planes)


def _open_recording(record_id: str, path: Path, *, lazy: bool) -> _Recording:
    """Open one PSG file and describe its signals.

    ``lazy`` is the whole difference between the two edfio variants: it memory-maps the data
    records instead of decoding all of them, and every later slice reads the records it needs.

    Args:
        record_id: The record this recording is stored as.
        path: The ``*-PSG.edf`` file.
        lazy: Read the data records on demand rather than up front.

    Returns:
        The open recording.

    Raises:
        LaneError: If the file cannot be read as EDF.
    """
    # Imported per call, not at module import, so the harness stays importable without edfio.
    import edfio  # noqa: PLC0415

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            edf = edfio.read_edf(path, lazy_load_data=lazy)
    except Exception as error:
        raise LaneError(f"{path}: cannot be read as EDF: {error}") from error
    record_duration = Fraction(str(edf.data_record_duration))
    signals = tuple(
        _Signal(
            time_series_id=f"{record_id}-{signal.label}",
            signal=signal.label,
            rate_hz=Fraction(int(signal.samples_per_data_record)) / record_duration,
            period_us=Fraction(US_PER_S) * record_duration / int(signal.samples_per_data_record),
            edf_signal=signal,
            calibration=(signal.digital_min, signal.digital_max, signal.physical_min, signal.physical_max),
        )
        for signal in edf.signals
    )
    return _Recording(
        record_id=record_id,
        signals=signals,
        signal_end_us=int(int(edf.num_data_records) * record_duration * US_PER_S),
    )


def _read_annotations(path: Path) -> tuple[tuple[int, int, str], ...]:
    """Read the labelled entries of one EDF+ scoring, in file order.

    An EDF+ file states its onsets as text, so they are parsed with :class:`~decimal.Decimal`.
    A float would move them off the 30 s grid.

    Args:
        path: The ``*-Hypnogram.edf`` file.

    Returns:
        One ``(onset, duration, label)`` triple in microseconds per labelled entry.

    Raises:
        LaneError: If the file cannot be read, or a time is not a whole number of microseconds.
    """
    # Imported per call, not at module import, so the harness stays importable without edfio.
    import edfio  # noqa: PLC0415

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            edf = edfio.read_edf(path, lazy_load_data=True)
    except Exception as error:
        raise LaneError(f"{path}: cannot be read as EDF: {error}") from error
    return tuple(
        (_whole_microseconds(path, entry.onset), _whole_microseconds(path, entry.duration or 0), entry.text)
        for entry in edf.annotations
        if entry.text
    )


def _whole_microseconds(path: Path, seconds: float) -> int:
    """Convert a time an EDF+ file states in seconds into whole microseconds.

    Returns:
        The value in microseconds.

    Raises:
        LaneError: If the value is not a whole number of microseconds.
    """
    total = Decimal(str(seconds)) * US_PER_S
    if total != total.to_integral_value():
        raise LaneError(f"{path}: states a time of {seconds} s, which is not a whole number of microseconds")
    return int(total)


def _run_reference_pipeline(record_id: str, psg_path: Path, hypnogram: Path) -> _PipelineRun:
    """Run PyHealth's sleep-staging call sequence over one recording.

    The calls and their arguments are ``pyhealth.tasks.sleep_staging_v2.SleepStagingSleepEDF``.
    MNE reads every signal at the file's highest rate, so a cassette recording comes back with its
    four 1 Hz signals interpolated to 100 Hz, and the epoch array is float64.

    MNE crops a scoring to the signals, so an epoch that overruns them produces no row here. The
    handle counts those and delivers the item empty rather than changing the item set.

    Args:
        record_id: The record this recording is stored as.
        psg_path: The ``*-PSG.edf`` file.
        hypnogram: The ``*-Hypnogram.edf`` file beside it.

    Returns:
        The epoch array, a map from epoch onset in microseconds to its row, the signal labels and
        the sample period MNE resampled every signal onto.

    Raises:
        LaneError: If MNE is not installed, or the recording cannot be run through the pipeline.
    """
    # Imported per call: MNE lives in the reference lane's own environment, not the shared one.
    try:
        import mne  # noqa: PLC0415  # ty: ignore[unresolved-import]
    except ImportError as error:
        raise LaneError(f"the {REFERENCE_PIPELINE} variant needs mne installed: {error}") from error

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            data = mne.io.read_raw_edf(
                psg_path, stim_signal="Event marker", infer_types=True, preload=True, verbose="error"
            )
            data.set_annotations(mne.read_annotations(hypnogram), emit_warning=False)
            events, _ = mne.events_from_annotations(data, event_id=PYHEALTH_EVENT_ID, chunk_duration=30.0)
            epochs = mne.Epochs(
                data,
                events,
                PYHEALTH_EVENT_ID,
                tmin=0.0,
                tmax=30.0 - 1.0 / data.info["sfreq"],
                baseline=None,
                preload=True,
            )
            signals = epochs.get_data(copy=False)
    except Exception as error:
        raise LaneError(f"{psg_path}: the {REFERENCE_PIPELINE} variant failed on {record_id}: {error}") from error
    period_us = Fraction(US_PER_S) / Fraction(str(data.info["sfreq"]))
    row_of = {int(onset * period_us): row for row, onset in enumerate(epochs.events[:, 0])}
    return signals, row_of, tuple(epochs.ch_names), period_us
