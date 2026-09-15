"""The cell record: one measured cell of the paper matrix, stored as one JSON file.

A cell is one point of the campaign grid, keyed by dataset, representation, consumer, metric and
loader variant. It holds the reported value, the raw per-repetition samples behind it, the facts that
make the value interpretable (item count, block size, Tier A bytes, byte rate) and the state the
machine was in when it was measured. The aggregator gates on those states, so they are part of the
record rather than of a campaign log next to it.

Files live at ``results/matrix/<dataset slug>/<run id>.json``.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from dataclasses import dataclass, replace
from datetime import UTC, datetime
import hashlib
import json
import math
from pathlib import Path
import statistics
from typing import Any, NamedTuple

from benchmarks.paper.errors import CellRecordError


SCHEMA_VERSION = 1

ORIGINAL = "original"
TIMEF = "timef"
CONTROL = "control"
REPRESENTATIONS = (ORIGINAL, TIMEF, CONTROL)

PANDAS = "pandas"
TORCH = "torch"
NO_CONSUMER = "none"
CONSUMERS = (PANDAS, TORCH, NO_CONSUMER)

STORAGE_BYTES = "storage_bytes"
FIRST_ITEM_S = "first_item_s"
FULL_READ_S = "full_read_s"
BLOCK_SHUFFLED_ITEMS_PER_S = "block_shuffled_items_per_s"
PEAK_MEMORY_BYTES = "peak_memory_bytes"
METRICS = (STORAGE_BYTES, FIRST_ITEM_S, FULL_READ_S, BLOCK_SHUFFLED_ITEMS_PER_S, PEAK_MEMORY_BYTES)

BUDGET_COMPLETED = "completed"
BUDGET_DID_NOT_COMPLETE = "did_not_complete"
BUDGET_NOT_ATTEMPTED = "not_attempted"
BUDGET_STATUSES = (BUDGET_COMPLETED, BUDGET_DID_NOT_COMPLETE, BUDGET_NOT_ATTEMPTED)
"""A read that completes at no budget the machine can offer is a result, not a missing cell."""

AS_SHIPPED = "as_shipped"
LAZY_CAPABLE = "lazy_capable"
NATIVE = "native"

OK = "ok"
FAILED = "failed"
EXCLUDED = "excluded"
STATUSES = (OK, FAILED, EXCLUDED)

COLD = "cold"
WARM = "warm"
CACHE_STATES = (COLD, WARM)

CACHE_ABSENT = "absent"
CACHE_PREBUILT = "prebuilt"
CACHE_FORBIDDEN = "forbidden"
DERIVED_CACHE_STATES = (CACHE_ABSENT, CACHE_PREBUILT, CACHE_FORBIDDEN)

MATRIX_DIRNAME = "matrix"


def dataset_slug(dataset: str) -> str:
    """Return the directory name for a dataset id.

    Dataset ids are ``org/name``, which cannot be a single path component.

    Returns:
        The id with slashes replaced by double underscores.
    """
    return dataset.replace("/", "__")


class CellKey(NamedTuple):
    """The identity of one cell. Two records with the same key are a campaign error."""

    dataset: str
    representation: str
    consumer: str
    metric: str
    loader_variant: str

    def as_slug(self) -> str:
        """Return a filesystem-safe rendering of the key.

        Returns:
            The five parts joined by double underscores.
        """
        return "__".join(
            (dataset_slug(self.dataset), self.representation, self.consumer, self.metric, self.loader_variant)
        )


@dataclass(frozen=True, kw_only=True)
class RunProvenance:
    """The conditions one observation was measured under.

    Attributes:
        git_commit: Full commit hash of the timenet checkout that ran the measurement.
        git_dirty: Whether that checkout had uncommitted changes.
        timenet_version: Installed ``timenet`` distribution version.
        env_digest: Digest of the resolved lane environment, one per lane by design.
        machine: Machine description printed in the table caption.
        os_version: Operating system version printed in the table caption.
        resident_set_cap_bytes: The declared resident-set cap the full read ran under.
    """

    git_commit: str
    git_dirty: bool
    timenet_version: str
    env_digest: str
    machine: str
    os_version: str
    resident_set_cap_bytes: int

    def as_dict(self) -> dict[str, Any]:
        """Return the JSON form.

        Returns:
            A plain mapping of every field.
        """
        return {
            "git_commit": self.git_commit,
            "git_dirty": self.git_dirty,
            "timenet_version": self.timenet_version,
            "env_digest": self.env_digest,
            "machine": self.machine,
            "os_version": self.os_version,
            "resident_set_cap_bytes": self.resident_set_cap_bytes,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> RunProvenance:
        """Rebuild a provenance block from its JSON form.

        Returns:
            The parsed block.

        Raises:
            CellRecordError: If a field is missing or has the wrong type.
        """
        try:
            return cls(
                git_commit=str(payload["git_commit"]),
                git_dirty=bool(payload["git_dirty"]),
                timenet_version=str(payload["timenet_version"]),
                env_digest=str(payload["env_digest"]),
                machine=str(payload["machine"]),
                os_version=str(payload["os_version"]),
                resident_set_cap_bytes=int(payload["resident_set_cap_bytes"]),
            )
        except (KeyError, TypeError, ValueError) as error:
            raise CellRecordError(f"provenance block is malformed: {error}") from error


@dataclass(frozen=True, kw_only=True)
class AllocationProfile:
    """What a read cost in memory, taken from the allocators rather than from the resident set.

    Every byte count here is above the lane's own import baseline, which is recorded beside them so
    both halves are auditable. Without that subtraction the columns compare interpreters instead of
    formats: torch costs a few hundred megabytes before it touches data.

    The three allocators are kept apart as well as summed. "Where did the memory go" is an appendix
    question, and one number cannot answer it.

    Attributes:
        peak_bytes: Summed high-water mark. ``None`` when the full read never completed.
        arrow_bytes: Arrow pool high-water. ``None`` on a lane that never imported pyarrow.
        python_bytes: ``tracemalloc`` peak, which covers NumPy buffers too.
        torch_bytes: Torch allocator peak. ``None`` on the pandas lane and on a CPU-only lane.
        baseline_bytes: What the allocators held after imports and before anything was opened.
        scaling_points: ``(n_items, peak_bytes)`` at each measured fraction of the corpus.
        scaling_slope: Fitted bytes per item. Near zero is bounded, large is corpus-resident.
        scaling_r2: Fit quality, so a bad fit cannot be read as bounded.
        min_budget_bytes: Smallest memory ceiling at which the full read completed.
        min_budget_status: ``completed``, ``did_not_complete`` or ``not_attempted``.
    """

    peak_bytes: float | None
    arrow_bytes: float | None
    python_bytes: float | None
    torch_bytes: float | None
    baseline_bytes: float
    scaling_points: tuple[tuple[int, float], ...]
    scaling_slope: float | None
    scaling_r2: float | None
    min_budget_bytes: int | None
    min_budget_status: str

    def __post_init__(self) -> None:
        """Refuse a profile that contradicts itself.

        Raises:
            CellRecordError: If a byte count is negative, the budget status and its value disagree,
                the fit is half-present, or the scaling points do not climb.
        """
        for name, value in (
            ("mem_peak_bytes", self.peak_bytes),
            ("mem_peak_arrow_bytes", self.arrow_bytes),
            ("mem_peak_python_bytes", self.python_bytes),
            ("mem_peak_torch_bytes", self.torch_bytes),
            ("mem_baseline_bytes", self.baseline_bytes),
        ):
            if value is not None and (not math.isfinite(value) or value < 0.0):
                raise CellRecordError(f"{name} must be a finite byte count, got {value!r}")
        if self.min_budget_status not in BUDGET_STATUSES:
            raise CellRecordError(
                f"mem_min_budget_status must be one of {list(BUDGET_STATUSES)}, got {self.min_budget_status!r}"
            )
        completed = self.min_budget_status == BUDGET_COMPLETED
        if completed != (self.min_budget_bytes is not None):
            raise CellRecordError(
                f"a {self.min_budget_status} budget search and a budget of {self.min_budget_bytes!r} disagree"
            )
        if self.min_budget_bytes is not None and self.min_budget_bytes < 1:
            raise CellRecordError(f"mem_min_budget_bytes must be positive, got {self.min_budget_bytes}")
        if (self.scaling_slope is None) != (self.scaling_r2 is None):
            raise CellRecordError("a scaling fit carries both a slope and an r2, or neither")
        if self.scaling_r2 is not None and not 0.0 <= self.scaling_r2 <= 1.0:
            raise CellRecordError(f"mem_scaling_r2 must be between 0 and 1, got {self.scaling_r2}")
        counts = [count for count, _ in self.scaling_points]
        if counts != sorted(set(counts)):
            raise CellRecordError(f"mem_scaling_points must climb through distinct item counts, got {counts}")

    @property
    def bounded(self) -> bool:
        """Return whether the peak stayed inside one item's worth of the corpus.

        Returns:
            True when the fit exists and its slope is under one byte per item.
        """
        return self.scaling_slope is not None and self.scaling_slope < 1.0

    def as_dict(self) -> dict[str, Any]:
        """Return the JSON form, flat, under the ``mem_`` names the spec fixed.

        Returns:
            A plain mapping of every field.
        """
        return {
            "mem_peak_bytes": self.peak_bytes,
            "mem_peak_arrow_bytes": self.arrow_bytes,
            "mem_peak_python_bytes": self.python_bytes,
            "mem_peak_torch_bytes": self.torch_bytes,
            "mem_baseline_bytes": self.baseline_bytes,
            "mem_scaling_points": [[count, peak] for count, peak in self.scaling_points],
            "mem_scaling_slope": self.scaling_slope,
            "mem_scaling_r2": self.scaling_r2,
            "mem_min_budget_bytes": self.min_budget_bytes,
            "mem_min_budget_status": self.min_budget_status,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> AllocationProfile | None:
        """Rebuild a profile from its JSON form.

        Returns:
            The parsed profile, or ``None`` when the payload holds no memory fields at all.

        Raises:
            CellRecordError: If the fields are present but malformed.
        """
        if "mem_min_budget_status" not in payload:
            return None
        try:
            return cls(
                peak_bytes=_optional_float(payload["mem_peak_bytes"]),
                arrow_bytes=_optional_float(payload["mem_peak_arrow_bytes"]),
                python_bytes=_optional_float(payload["mem_peak_python_bytes"]),
                torch_bytes=_optional_float(payload["mem_peak_torch_bytes"]),
                baseline_bytes=float(payload["mem_baseline_bytes"]),
                scaling_points=tuple((int(count), float(peak)) for count, peak in payload["mem_scaling_points"]),
                scaling_slope=_optional_float(payload["mem_scaling_slope"]),
                scaling_r2=_optional_float(payload["mem_scaling_r2"]),
                min_budget_bytes=None
                if payload["mem_min_budget_bytes"] is None
                else int(payload["mem_min_budget_bytes"]),
                min_budget_status=str(payload["mem_min_budget_status"]),
            )
        except CellRecordError:
            raise
        except (KeyError, TypeError, ValueError) as error:
            raise CellRecordError(f"allocation profile is malformed: {error}") from error


@dataclass(frozen=True, kw_only=True)
class CellRecord:
    """One measured cell, with the samples behind its value.

    Attributes:
        run_id: Unique id of this run, and the file stem it is written to.
        dataset: Dataset id, ``org/name``.
        representation: ``original``, ``timef`` or ``control``.
        consumer: ``pandas``, ``torch``, or ``none`` for storage.
        lane: The lane that produced the number, for example ``sleep-edfx/original/pandas/edfio``.
        loader_variant: ``as_shipped``, ``lazy_capable`` or ``native``.
        metric: One of :data:`METRICS`.
        value: The reported statistic, the median of ``samples``. ``None`` unless the status is ok.
        samples: The raw per-repetition samples, in the order they were measured.
        n_items: Canonical items this cell covers.
        block_size: ``B``, items per block in the block-shuffled read.
        block_bytes: The byte budget one block was sized to.
        tier_a_bytes: Tier A artifact bytes for this dataset and representation.
        bytes_per_s: Bytes moved per second, recorded beside every rate.
        cache_state: Page-cache state before the timed region.
        derived_cache_state: State of the lane's derived cache, declared per lane.
        repeats_declared: Repetitions the campaign asked for. ``len(samples)`` may be lower.
        checksum: Digest of every value touched inside the timed region.
        status: ``ok``, ``failed`` or ``excluded``.
        notes: Free text. Carries the reason when the status is not ok.
        allocation: What the read cost in memory. A failed cell keeps it, because a read that
            completes at no budget is a result and its baseline and partial scaling still hold.
        provenance: What the machine and the checkout looked like.
        measured_at_utc: When the run finished.
    """

    run_id: str
    dataset: str
    representation: str
    consumer: str
    lane: str
    loader_variant: str
    metric: str
    value: float | None
    samples: tuple[float, ...]
    n_items: int
    block_size: int
    block_bytes: int
    tier_a_bytes: int
    bytes_per_s: float | None
    cache_state: str
    derived_cache_state: str
    repeats_declared: int
    checksum: str | None
    status: str
    provenance: RunProvenance
    allocation: AllocationProfile | None = None
    notes: str = ""
    measured_at_utc: str = ""
    schema_version: int = SCHEMA_VERSION

    def __post_init__(self) -> None:
        self._check_vocabulary()
        self._check_counts()
        self._check_value()
        self._check_allocation()

    def _check_vocabulary(self) -> None:
        """Reject a field whose value is outside its declared vocabulary.

        Raises:
            CellRecordError: If a field is empty or outside its vocabulary.
        """
        for name, value in (("run_id", self.run_id), ("dataset", self.dataset), ("lane", self.lane)):
            if not value.strip():
                raise CellRecordError(f"{name} must not be empty")
        for name, value, allowed in (
            ("representation", self.representation, REPRESENTATIONS),
            ("consumer", self.consumer, CONSUMERS),
            ("metric", self.metric, METRICS),
            ("status", self.status, STATUSES),
            ("cache_state", self.cache_state, CACHE_STATES),
            ("derived_cache_state", self.derived_cache_state, DERIVED_CACHE_STATES),
        ):
            if value not in allowed:
                raise CellRecordError(f"{name} must be one of {list(allowed)}, got {value!r}")
        if not self.loader_variant.strip():
            raise CellRecordError("loader_variant must not be empty")
        storage = self.metric == STORAGE_BYTES
        if storage and self.consumer != NO_CONSUMER:
            raise CellRecordError(f"{STORAGE_BYTES} has no consumer, got {self.consumer!r}")
        if not storage and self.consumer == NO_CONSUMER:
            raise CellRecordError(f"{self.metric} needs a consumer")

    def _check_counts(self) -> None:
        """Reject a non-positive count.

        Raises:
            CellRecordError: If a count is below one, or fewer samples than repetitions were kept.
        """
        for name, value in (
            ("n_items", self.n_items),
            ("block_size", self.block_size),
            ("block_bytes", self.block_bytes),
            ("tier_a_bytes", self.tier_a_bytes),
            ("repeats_declared", self.repeats_declared),
        ):
            if value < 1:
                raise CellRecordError(f"{name} must be >= 1, got {value}")
        if self.block_size > self.n_items:
            raise CellRecordError(f"block_size {self.block_size} exceeds n_items {self.n_items}")
        if len(self.samples) > self.repeats_declared:
            raise CellRecordError(
                f"{len(self.samples)} samples exceed the {self.repeats_declared} repetitions declared"
            )

    def _check_value(self) -> None:
        """Hold the reported value to the samples behind it.

        Raises:
            CellRecordError: If the value and the samples contradict each other.
        """
        if self.status != OK:
            if self.value is not None or self.samples:
                raise CellRecordError(f"a {self.status} cell carries no value and no samples")
            return
        if not self.samples:
            raise CellRecordError("an ok cell needs at least one sample")
        if self.value is None:
            raise CellRecordError("an ok cell needs a value")
        for sample in self.samples:
            if not math.isfinite(sample) or sample <= 0.0:
                raise CellRecordError(f"samples must be finite and positive, got {sample!r}")
        expected = statistics.median(self.samples)
        if not math.isclose(self.value, expected, rel_tol=1e-9, abs_tol=0.0):
            raise CellRecordError(f"value {self.value} is not the median {expected} of its samples")
        if self.bytes_per_s is not None and (not math.isfinite(self.bytes_per_s) or self.bytes_per_s <= 0.0):
            raise CellRecordError(f"bytes_per_s must be finite and positive, got {self.bytes_per_s!r}")

    def _check_allocation(self) -> None:
        """Hold the memory profile to the cell it sits on.

        Raises:
            CellRecordError: If a peak-memory cell carries no profile, if a profile's peak and the
                reported median disagree, or if a cell that produced no number still claims a peak.
        """
        if self.metric == PEAK_MEMORY_BYTES and self.allocation is None:
            raise CellRecordError(f"{PEAK_MEMORY_BYTES} cell {self.key.as_slug()} carries no allocation profile")
        if self.allocation is None:
            return
        peak = self.allocation.peak_bytes
        if self.status != OK:
            if peak is not None:
                raise CellRecordError(f"a {self.status} cell reports a peak of {peak} bytes")
            return
        if self.metric != PEAK_MEMORY_BYTES:
            return
        if peak is None:
            raise CellRecordError(f"an ok {PEAK_MEMORY_BYTES} cell needs a peak")
        if self.value is not None and not math.isclose(self.value, peak, rel_tol=1e-9, abs_tol=0.0):
            raise CellRecordError(f"value {self.value} is not the profile's peak {peak}")

    @property
    def key(self) -> CellKey:
        """Return the identity of this cell.

        Returns:
            The five-part key.
        """
        return CellKey(self.dataset, self.representation, self.consumer, self.metric, self.loader_variant)

    @property
    def repeats_kept(self) -> int:
        """Return how many repetitions survived to the median.

        Returns:
            The number of samples.
        """
        return len(self.samples)

    def as_dict(self) -> dict[str, Any]:
        """Return the JSON form of the record.

        Returns:
            A plain mapping, ready for :func:`json.dumps`.
        """
        allocation = self.allocation.as_dict() if self.allocation is not None else {}
        return {
            **allocation,
            "schema_version": self.schema_version,
            "run_id": self.run_id,
            "measured_at_utc": self.measured_at_utc,
            "dataset": self.dataset,
            "representation": self.representation,
            "consumer": self.consumer,
            "lane": self.lane,
            "loader_variant": self.loader_variant,
            "metric": self.metric,
            "value": self.value,
            "samples": list(self.samples),
            "n_items": self.n_items,
            "block_size": self.block_size,
            "block_bytes": self.block_bytes,
            "tier_a_bytes": self.tier_a_bytes,
            "bytes_per_s": self.bytes_per_s,
            "cache_state": self.cache_state,
            "derived_cache_state": self.derived_cache_state,
            "repeats_declared": self.repeats_declared,
            "checksum": self.checksum,
            "status": self.status,
            "notes": self.notes,
            "provenance": self.provenance.as_dict(),
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> CellRecord:
        """Rebuild a record from its JSON form.

        Returns:
            The parsed record.

        Raises:
            CellRecordError: If a field is missing, has the wrong type, or the schema version is
                not the one this module writes.
        """
        version = payload.get("schema_version")
        if version != SCHEMA_VERSION:
            raise CellRecordError(f"cell record schema version {version!r}, expected {SCHEMA_VERSION}")
        try:
            value = payload["value"]
            bytes_per_s = payload["bytes_per_s"]
            checksum = payload["checksum"]
            return cls(
                run_id=str(payload["run_id"]),
                measured_at_utc=str(payload.get("measured_at_utc", "")),
                dataset=str(payload["dataset"]),
                representation=str(payload["representation"]),
                consumer=str(payload["consumer"]),
                lane=str(payload["lane"]),
                loader_variant=str(payload["loader_variant"]),
                metric=str(payload["metric"]),
                value=None if value is None else float(value),
                samples=tuple(float(sample) for sample in payload["samples"]),
                n_items=int(payload["n_items"]),
                block_size=int(payload["block_size"]),
                block_bytes=int(payload["block_bytes"]),
                tier_a_bytes=int(payload["tier_a_bytes"]),
                bytes_per_s=None if bytes_per_s is None else float(bytes_per_s),
                cache_state=str(payload["cache_state"]),
                derived_cache_state=str(payload["derived_cache_state"]),
                repeats_declared=int(payload["repeats_declared"]),
                checksum=None if checksum is None else str(checksum),
                status=str(payload["status"]),
                notes=str(payload.get("notes", "")),
                allocation=AllocationProfile.from_dict(payload),
                provenance=RunProvenance.from_dict(payload["provenance"]),
            )
        except CellRecordError:
            raise
        except (KeyError, TypeError, ValueError) as error:
            raise CellRecordError(f"cell record is malformed: {error}") from error

    def path_in(self, matrix_dir: Path) -> Path:
        """Return where this record belongs under a matrix directory.

        Returns:
            ``<matrix_dir>/<dataset slug>/<run_id>.json``.
        """
        return matrix_dir / dataset_slug(self.dataset) / f"{self.run_id}.json"

    def write(self, matrix_dir: Path) -> Path:
        """Write the record under a matrix directory.

        Args:
            matrix_dir: The ``results/matrix`` directory.

        Returns:
            The path written.
        """
        path = self.path_in(matrix_dir)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.as_dict(), indent=2, sort_keys=True) + "\n", encoding="utf-8")
        return path


def measured(  # noqa: PLR0913 - one keyword per record field, so a caller cannot mix two up
    *,
    dataset: str,
    representation: str,
    consumer: str,
    lane: str,
    loader_variant: str,
    metric: str,
    samples: Iterable[float],
    n_items: int,
    block_size: int,
    block_bytes: int,
    tier_a_bytes: int,
    cache_state: str,
    derived_cache_state: str,
    repeats_declared: int,
    provenance: RunProvenance,
    bytes_per_s: float | None = None,
    checksum: str | None = None,
    allocation: AllocationProfile | None = None,
    notes: str = "",
    run_id: str | None = None,
    measured_at_utc: str | None = None,
) -> CellRecord:
    """Build an ok record, taking its value as the median of the samples.

    The campaign runner never computes the reported statistic itself. That keeps a record's value
    and its samples from drifting apart.

    Returns:
        The record.
    """
    ordered = tuple(float(sample) for sample in samples)
    key = CellKey(dataset, representation, consumer, metric, loader_variant)
    return CellRecord(
        run_id=run_id or key.as_slug(),
        measured_at_utc=measured_at_utc if measured_at_utc is not None else _now(),
        dataset=dataset,
        representation=representation,
        consumer=consumer,
        lane=lane,
        loader_variant=loader_variant,
        metric=metric,
        value=statistics.median(ordered) if ordered else None,
        samples=ordered,
        n_items=n_items,
        block_size=block_size,
        block_bytes=block_bytes,
        tier_a_bytes=tier_a_bytes,
        bytes_per_s=bytes_per_s,
        cache_state=cache_state,
        derived_cache_state=derived_cache_state,
        repeats_declared=repeats_declared,
        checksum=checksum,
        status=OK,
        notes=notes,
        allocation=allocation,
        provenance=provenance,
    )


def failed(record: CellRecord, reason: str) -> CellRecord:
    """Return the same cell marked failed, with its value and samples dropped.

    The memory profile survives, minus its peak. A read that completes at no budget the machine can
    offer has still reported its import baseline, whichever fractions of the corpus it managed, and
    the search that found no ceiling. Those are the finding, not the absence of one.

    Args:
        record: The cell that was attempted.
        reason: Why it failed. It reaches the caption, so write it as one short sentence.

    Returns:
        The failed record.
    """
    return replace(
        record,
        status=FAILED,
        value=None,
        samples=(),
        checksum=None,
        bytes_per_s=None,
        allocation=_without_peak(record.allocation),
        notes=reason,
    )


def _without_peak(allocation: AllocationProfile | None) -> AllocationProfile | None:
    """Drop the peak of a profile whose cell produced no number.

    Returns:
        The profile with every peak cleared, or ``None``.
    """
    if allocation is None:
        return None
    return replace(allocation, peak_bytes=None, arrow_bytes=None, python_bytes=None, torch_bytes=None)


def read_record(path: Path) -> CellRecord:
    """Read one record file.

    Returns:
        The parsed record.

    Raises:
        CellRecordError: If the file is not JSON, or the record is malformed.
    """
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise CellRecordError(f"{path} is not valid JSON: {error}") from error
    if not isinstance(payload, dict):
        raise CellRecordError(f"{path} does not hold a cell record object")
    return CellRecord.from_dict(payload)


def iter_records(matrix_dir: Path) -> Iterator[CellRecord]:
    """Read every record under a matrix directory, in sorted path order.

    Args:
        matrix_dir: The ``results/matrix`` directory.

    Yields:
        One record per JSON file found.

    Raises:
        CellRecordError: If the directory does not exist.
    """
    if not matrix_dir.is_dir():
        raise CellRecordError(f"matrix directory {matrix_dir} does not exist")
    for path in sorted(matrix_dir.rglob("*.json")):
        yield read_record(path)


def block_size_for(*, tier_a_bytes: int, n_items: int, block_bytes: int) -> int:
    """Return ``B``, the number of items in one block.

    A block is a byte budget, not a run of contiguous bytes: contiguity is not implementable on a
    directory of EDF files, and the TimeF row groups of one record are not contiguous either.

    Args:
        tier_a_bytes: Tier A artifact bytes for this representation.
        n_items: Canonical items in the dataset.
        block_bytes: The byte budget per block.

    Returns:
        At least one item per block.
    """
    bytes_per_item = tier_a_bytes / n_items
    return max(1, min(n_items, round(block_bytes / bytes_per_item)))


def block_bytes_for(tier_a_bytes: int, *, cap: int = 64_000_000, floor_blocks: int = 16) -> int:
    """Return the byte budget of one block for a dataset.

    64 MB, decimal to match the storage rule, with a floor of 16 blocks so a small artifact still
    shuffles.

    Returns:
        The block byte budget.
    """
    return max(1, min(cap, tier_a_bytes // floor_blocks))


def _now() -> str:
    """Return the current UTC timestamp, to the second.

    Returns:
        An ISO 8601 string.
    """
    return datetime.now(UTC).isoformat(timespec="seconds")


def _optional_float(value: Any) -> float | None:
    """Return a float, or ``None`` for a JSON null.

    Returns:
        The parsed value.
    """
    return None if value is None else float(value)


def digest_of(payload: Any) -> str:
    """Return the SHA-256 of a JSON-serializable payload in canonical form.

    Returns:
        The lowercase hex digest.
    """
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()
