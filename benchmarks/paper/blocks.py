"""The block sampler: what "block-shuffled" means when contiguity is not implementable.

The draft called a block "a fixed 64 MB of contiguous bytes on both sides". Neither side can offer
that. On a directory of EDF or pickle files APFS gives no byte ordering at all, and inside a TimeF
build one record's row groups are spread across the file: zero of Sleep-EDF's 197 records have
contiguous row groups.

So a block is a declared logical order plus a byte budget. Take ``bytes_per_item`` as Tier A bytes
over ``n_items``, set ``B = max(1, round(block_bytes / bytes_per_item))``, permute the blocks with a
declared seed, and keep canonical order inside each block. ``block_bytes`` is 64 MB decimal, or a
sixteenth of the artifact when that is smaller, so a small dataset still shuffles into 16 blocks.

Full read is the ``B = n_items`` case of this same plan, so the two access patterns differ only in
the permutation.

What the plan cannot promise is locality, so it measures it: :func:`locality_of` reports the runs
the emitted order actually produced, and :func:`switch_stats` counts group switches once a lane can
say which row group each item landed in.
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from itertools import pairwise
import random
from typing import Any

from benchmarks.paper.errors import ItemRegistryError
from benchmarks.paper.record import block_bytes_for, block_size_for


DEFAULT_SEED = 20260905
"""The declared permutation seed. It is part of the method, so it is a constant, not a default."""

BLOCK_BYTES_CAP = 64_000_000
"""64 MB, decimal to match the storage rule."""

FLOOR_BLOCKS = 16
"""Smallest number of blocks an artifact is cut into, however small it is."""


@dataclass(frozen=True)
class BlockPlan:
    """One pass over a dataset: which items, in which order, in which blocks.

    Attributes:
        n_items: Canonical items in the dataset.
        block_size: ``B``, items per block. The last block may be shorter.
        block_bytes: The byte budget ``B`` was sized to.
        tier_a_bytes: Tier A bytes of this representation.
        seed: The permutation seed.
        blocks: The blocks, in visit order. Each holds canonical ordinals in canonical order.
    """

    n_items: int
    block_size: int
    block_bytes: int
    tier_a_bytes: int
    seed: int
    blocks: tuple[tuple[int, ...], ...]

    @property
    def n_blocks(self) -> int:
        """Return how many blocks the pass visits.

        Returns:
            The block count.
        """
        return len(self.blocks)

    @property
    def bytes_per_item(self) -> float:
        """Return the mean Tier A bytes one item carries.

        Returns:
            Tier A bytes over the item count.
        """
        return self.tier_a_bytes / self.n_items

    def order(self) -> Iterator[int]:
        """Walk the ordinals the pass delivers, in delivery order.

        Yields:
            One canonical ordinal per item.
        """
        for block in self.blocks:
            yield from block

    def as_dict(self) -> dict[str, Any]:
        """Return the JSON form, without the blocks themselves.

        The blocks are derivable from the seed and the size, and a full ARFBench plan is 750 numbers
        while Sleep-EDF's is 458,161. What the appendix prints is the four parameters.

        Returns:
            A plain mapping of the plan's parameters.
        """
        return {
            "n_items": self.n_items,
            "block_size": self.block_size,
            "block_bytes": self.block_bytes,
            "tier_a_bytes": self.tier_a_bytes,
            "bytes_per_item": self.bytes_per_item,
            "n_blocks": self.n_blocks,
            "seed": self.seed,
        }


@dataclass(frozen=True)
class Locality:
    """What locality the emitted order actually achieved.

    A block plan promises the reader that consecutive items usually sit next to each other in the
    canonical order. This says how often that held, so the claim is measured rather than asserted.

    Attributes:
        n_items: Items in the pass.
        runs: Maximal stretches of canonically consecutive ordinals in the emitted order.
        mean_run_length: Items per run.
        adjacent_fraction: Share of consecutive delivered pairs that are canonically adjacent.
    """

    n_items: int
    runs: int
    mean_run_length: float
    adjacent_fraction: float

    def as_dict(self) -> dict[str, Any]:
        """Return the JSON form.

        Returns:
            A plain mapping of every field.
        """
        return {
            "n_items": self.n_items,
            "runs": self.runs,
            "mean_run_length": self.mean_run_length,
            "adjacent_fraction": self.adjacent_fraction,
        }


@dataclass(frozen=True)
class SwitchStats:
    """How often a pass moved from one storage group to another.

    A group is whatever the lane can name: a Parquet row group on the TimeF side, a source file on
    the Original side. Counting switches is the only locality claim that is about bytes rather than
    about ordinals.

    Attributes:
        reads: How many items the pass delivered.
        switches: How often two consecutive items came from different groups.
        distinct_groups: How many groups the pass touched.
    """

    reads: int
    switches: int
    distinct_groups: int

    @property
    def switches_per_item(self) -> float:
        """Return the switch rate.

        Returns:
            Switches over reads, or zero for an empty pass.
        """
        return self.switches / self.reads if self.reads else 0.0

    def as_dict(self) -> dict[str, Any]:
        """Return the JSON form.

        Returns:
            A plain mapping of every field plus the derived rate.
        """
        return {
            "reads": self.reads,
            "switches": self.switches,
            "distinct_groups": self.distinct_groups,
            "switches_per_item": self.switches_per_item,
        }


def block_bytes_of(tier_a_bytes: int) -> int:
    """Return the byte budget one block is sized to.

    Returns:
        64 MB, or a sixteenth of the artifact when that is smaller.
    """
    return block_bytes_for(tier_a_bytes, cap=BLOCK_BYTES_CAP, floor_blocks=FLOOR_BLOCKS)


def plan_blocks(*, n_items: int, tier_a_bytes: int, seed: int = DEFAULT_SEED) -> BlockPlan:
    """Cut a dataset into blocks and permute them.

    A non-positive item or byte count raises :class:`ItemRegistryError`.

    Args:
        n_items: Canonical items in the dataset.
        tier_a_bytes: Tier A bytes of this representation.
        seed: The permutation seed.

    Returns:
        The plan, with blocks in visit order.
    """
    _check(n_items, tier_a_bytes)
    block_bytes = block_bytes_of(tier_a_bytes)
    block_size = block_size_for(tier_a_bytes=tier_a_bytes, n_items=n_items, block_bytes=block_bytes)
    blocks = _cut(n_items, block_size)
    random.Random(seed).shuffle(blocks)  # noqa: S311 - a declared, reproducible permutation, not a secret
    return BlockPlan(
        n_items=n_items,
        block_size=block_size,
        block_bytes=block_bytes,
        tier_a_bytes=tier_a_bytes,
        seed=seed,
        blocks=tuple(tuple(block) for block in blocks),
    )


def plan_full_read(*, n_items: int, tier_a_bytes: int) -> BlockPlan:
    """Return the full-read plan: one block holding every item, in canonical order.

    Full read is the ``B = n_items`` case of the block-shuffled plan, so a lane runs it through the
    same code path with the same per-item delivery contract. Without that the two access patterns
    are differently-constructed measurements, and "sequential bounds block-shuffled from above" is
    an artifact of the harness rather than a claim about the format.

    A non-positive item or byte count raises :class:`ItemRegistryError`.

    Args:
        n_items: Canonical items in the dataset.
        tier_a_bytes: Tier A bytes of this representation.

    Returns:
        The plan.
    """
    _check(n_items, tier_a_bytes)
    return BlockPlan(
        n_items=n_items,
        block_size=n_items,
        block_bytes=tier_a_bytes,
        tier_a_bytes=tier_a_bytes,
        seed=DEFAULT_SEED,
        blocks=(tuple(range(n_items)),),
    )


def locality_of(order: Sequence[int]) -> Locality:
    """Measure the locality of an emitted order.

    Args:
        order: The canonical ordinals, in delivery order.

    Returns:
        The runs the order produced and the share of adjacent pairs.
    """
    n_items = len(order)
    if n_items == 0:
        return Locality(n_items=0, runs=0, mean_run_length=0.0, adjacent_fraction=0.0)
    adjacent = sum(1 for previous, current in pairwise(order) if current == previous + 1)
    runs = n_items - adjacent
    pairs = n_items - 1
    return Locality(
        n_items=n_items,
        runs=runs,
        mean_run_length=n_items / runs,
        adjacent_fraction=adjacent / pairs if pairs else 1.0,
    )


def switch_stats(group_ids: Sequence[object]) -> SwitchStats:
    """Count how often a pass moved between storage groups.

    Args:
        group_ids: The group each delivered item came from, in delivery order.

    Returns:
        The reads, the switches and how many groups were touched.
    """
    reads = len(group_ids)
    switches = sum(1 for previous, current in pairwise(group_ids) if current != previous)
    return SwitchStats(reads=reads, switches=switches, distinct_groups=len(set(group_ids)))


def decode_bound_fraction(bytes_per_s: float, cold_bytes_per_s: float) -> float:
    """Return how much of the volume's cold bandwidth a pass used.

    Near one the pass is I/O bound, and its items/s reduces to bandwidth times items over bytes,
    which is a restatement of the storage column. Well under one it is decode bound, and the rate
    columns say something the storage column does not. The paper has to tell the reader which
    regime each row is in.

    Args:
        bytes_per_s: Bytes the pass moved per second of wall clock.
        cold_bytes_per_s: The volume's cold throughput, from the canary band.

    Returns:
        The ratio.

    Raises:
        ItemRegistryError: If the cold bandwidth is not positive.
    """
    if cold_bytes_per_s <= 0.0:
        raise ItemRegistryError(f"cold bandwidth must be positive, got {cold_bytes_per_s}")
    return bytes_per_s / cold_bytes_per_s


def _cut(n_items: int, block_size: int) -> list[tuple[int, ...]]:
    """Cut the canonical order into blocks of ``block_size``.

    Returns:
        The blocks, still in canonical order. The last one may be shorter.
    """
    return [tuple(range(start, min(start + block_size, n_items))) for start in range(0, n_items, block_size)]


def _check(n_items: int, tier_a_bytes: int) -> None:
    """Refuse a plan that cannot be sized.

    Raises:
        ItemRegistryError: If either count is below one.
    """
    if n_items < 1:
        raise ItemRegistryError(f"a block plan needs at least one item, got {n_items}")
    if tier_a_bytes < 1:
        raise ItemRegistryError(f"a block plan needs positive Tier A bytes, got {tier_a_bytes}")
