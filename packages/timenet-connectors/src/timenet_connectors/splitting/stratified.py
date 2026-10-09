"""Deterministic, approximate stratification with indivisible sample groups."""

from collections import Counter
from collections.abc import Callable, Hashable, Mapping, Sequence
from dataclasses import dataclass, field
import math
from random import Random

from timenet.errors import TimeFValidationError
from timenet.types.splits import Split
from timenet_connectors.splitting.base import DatasetSplits, Splitter, T


@dataclass
class _Group:
    indices: list[int] = field(default_factory=list)
    categories: Counter[Hashable] = field(default_factory=Counter)


@dataclass
class _Partition:
    split: Split
    fraction: float
    size: int = 0
    categories: Counter[Hashable] = field(default_factory=Counter)

    def cost(self, group: _Group, totals: Counter[Hashable], sample_count: int) -> float:
        """Measure the change in normalized squared category and size errors.

        Returns:
            The allocation's error change; smaller values are better.
        """
        cost = _error_change(len(group.indices), self.size, sample_count, self.fraction)
        for category, count in group.categories.items():
            cost += _error_change(count, self.categories[category], totals[category], self.fraction)
        return cost

    def add(self, group: _Group) -> None:
        self.size += len(group.indices)
        self.categories.update(group.categories)


def _error_change(added: int, current: int, total: int, fraction: float) -> float:
    target = total * fraction
    return (2 * added * (current - target) + added**2) / total


def _split_fractions(train: float, validation: float | None, test: float) -> dict[Split, float]:
    fractions = {Split.TRAIN: train, Split.TEST: test}
    if validation is not None:
        fractions[Split.VALIDATION] = validation
    for split, fraction in fractions.items():
        if not math.isfinite(fraction) or not 0 < fraction <= 1:
            raise TimeFValidationError(f"{split} fraction must be finite and in (0, 1]")
    if not math.isclose(sum(fractions.values()), 1.0, rel_tol=0, abs_tol=1e-9):
        raise TimeFValidationError("split fractions must sum to one")
    return fractions


class StratifiedSplitter(Splitter[T]):
    """Balance category counts while assigning whole groups to partitions.

    ``stratify_by`` returns one category per sample; ``group_by`` identifies
    samples that must stay together (for example, questions about one signal).
    Without ``group_by``, each input occurrence is an independent group.

    Fractions target sample counts, not group counts. A greedy allocator balances
    size and category errors, processing larger groups first and shuffling ties
    with a local seed. The same ordered input and seed give the same result.
    Exact ratios and category coverage are not guaranteed: small datasets or
    indivisible groups can leave requested partitions empty. Group isolation
    always takes priority. This is not a temporal split strategy.
    """

    def __init__(  # noqa: PLR0913 -- keep partition fractions and callbacks explicit.
        self,
        *,
        train: float,
        test: float,
        stratify_by: Callable[[T], Hashable],
        validation: float | None = None,
        group_by: Callable[[T], Hashable] | None = None,
        seed: int = 42,
    ) -> None:
        self._fractions = _split_fractions(train, validation, test)
        self._stratify_by = stratify_by
        self._group_by = group_by
        self._seed = seed

    def split(self, samples: Sequence[T]) -> DatasetSplits[T]:
        """Assign every sample once, preserving input order within partitions.

        Returns:
            Requested partitions and their original samples, including empty partitions.
        """
        groups = self._group_samples(samples)
        assignments = self._assign_groups(groups)
        return self._collect_splits(samples, assignments)

    def _group_samples(self, samples: Sequence[T]) -> list[_Group]:
        groups: dict[Hashable, _Group] = {}
        for index, sample in enumerate(samples):
            key = index if self._group_by is None else self._group_by(sample)
            group = groups.setdefault(key, _Group())
            group.indices.append(index)
            group.categories[self._stratify_by(sample)] += 1
        ordered = list(groups.values())
        Random(self._seed).shuffle(ordered)  # noqa: S311 -- reproducible sampling, not cryptography.
        ordered.sort(key=lambda group: len(group.indices), reverse=True)
        return ordered

    def _assign_groups(self, groups: Sequence[_Group]) -> dict[int, Split]:
        totals: Counter[Hashable] = Counter()
        for group in groups:
            totals.update(group.categories)
        sample_count = totals.total()
        partitions = [_Partition(split, fraction) for split, fraction in self._fractions.items()]
        assignments: dict[int, Split] = {}
        for group in groups:
            partition = min(partitions, key=lambda candidate: candidate.cost(group, totals, sample_count))
            partition.add(group)
            for index in group.indices:
                assignments[index] = partition.split
        return assignments

    def _collect_splits(self, samples: Sequence[T], assignments: Mapping[int, Split]) -> DatasetSplits[T]:
        partitions: dict[Split, list[T]] = {split: [] for split in self._fractions}
        for index, sample in enumerate(samples):
            partitions[assignments[index]].append(sample)
        return DatasetSplits(partitions)
