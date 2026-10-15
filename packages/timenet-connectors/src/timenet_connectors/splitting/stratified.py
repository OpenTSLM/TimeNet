"""Named dataset partitions backed by scikit-learn's stratified group folds."""

from collections.abc import Callable, Hashable, Iterable, Iterator, Sequence
from fractions import Fraction
import math

import numpy as np
from numpy.typing import NDArray
from sklearn.model_selection import StratifiedGroupKFold

from timenet.errors import TimeFValidationError
from timenet.types.splits import Split
from timenet_connectors.splitting.base import DatasetSplits, Splitter, T


def _split_fold_counts(train: float, validation: float | None, test: float) -> dict[Split, int]:
    fractions = {Split.TRAIN: train, Split.TEST: test}
    if validation is not None:
        fractions[Split.VALIDATION] = validation
    ratios: dict[Split, Fraction] = {}
    for split, fraction in fractions.items():
        if not math.isfinite(fraction) or not 0 < fraction <= 1:
            raise TimeFValidationError(f"{split} fraction must be finite and in (0, 1]")
        # Recover simple ratios such as 1/3 without treating float rounding as
        # a request for trillions of folds. Fraction uses a 1,000,000 denominator limit.
        ratio = Fraction(fraction).limit_denominator()
        if ratio == 0 or not math.isclose(float(ratio), fraction, rel_tol=0, abs_tol=1e-9):
            raise TimeFValidationError(f"{split} fraction cannot be represented as a fold ratio")
        ratios[split] = ratio
    if sum(ratios.values()) != 1:
        raise TimeFValidationError("split fractions must sum to one")
    fold_count = math.lcm(*(ratio.denominator for ratio in ratios.values()))
    return {split: int(ratio * fold_count) for split, ratio in ratios.items()}


def _encode_labels(samples: Sequence[T], key: Callable[[T], Hashable]) -> list[int]:
    labels: dict[Hashable, int] = {}
    encoded = []
    for sample in samples:
        encoded.append(labels.setdefault(key(sample), len(labels)))
    return encoded


class StratifiedSplitter(Splitter[T]):
    """Balance category counts while assigning whole groups to partitions.

    ``stratify_by`` returns one category per sample; ``group_by`` identifies
    samples that must stay together (for example, questions about one signal).
    Without ``group_by``, each input occurrence is an independent group.

    Fractions select proportions of stratified group folds: 80/20 uses four of
    five folds for training; 80/10/10 uses eight of ten. Each group stays in one
    fold. Nonempty input must contain at least as many groups as folds and meet
    scikit-learn's category-count requirements. Invalid inputs raise
    ``TimeFValidationError``.

    Sample counts and category proportions are approximate for unequal groups.
    The same ordered input, seed, and scikit-learn version give the same result.
    This is not a temporal split strategy.
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
        self._fold_counts = _split_fold_counts(train, validation, test)
        self._stratify_by = stratify_by
        self._group_by = group_by
        self._seed = seed

    def split(self, samples: Sequence[T]) -> DatasetSplits[T]:
        """Assign every sample once, preserving input order within partitions.

        Returns:
            Requested partitions and their original samples, including empty partitions.
        """
        partitions: dict[Split, list[T]] = {split: [] for split in self._fold_counts}
        if not samples:
            return DatasetSplits(partitions)
        assignments = self._assign_folds(samples)
        for index, sample in enumerate(samples):
            partitions[assignments[index]].append(sample)
        return DatasetSplits(partitions)

    def _assign_folds(self, samples: Sequence[T]) -> dict[int, Split]:
        categories = _encode_labels(samples, self._stratify_by)
        groups = self._get_group_labels(samples)
        folds = self._split_into_folds(categories, groups)
        return self._assign_folds_to_splits(folds)

    def _get_group_labels(self, samples: Sequence[T]) -> list[int]:
        if self._group_by is None:
            return list(range(len(samples)))
        return _encode_labels(samples, self._group_by)

    def _split_into_folds(self, categories: list[int], groups: list[int]) -> Iterator[NDArray[np.intp]]:
        fold_count = sum(self._fold_counts.values())
        group_count = len(set(groups))
        if group_count < fold_count:
            raise TimeFValidationError(f"{fold_count} folds require at least {fold_count} groups; found {group_count}")
        splitter = StratifiedGroupKFold(n_splits=fold_count, shuffle=True, random_state=self._seed)
        try:
            folds = splitter.split(np.zeros((len(categories), 1)), y=categories, groups=groups)
            for _, indices in folds:
                yield indices
        except ValueError as error:
            raise TimeFValidationError(f"cannot stratify requested folds: {error}") from error

    def _assign_folds_to_splits(self, folds: Iterable[NDArray[np.intp]]) -> dict[int, Split]:
        fold_splits = [split for split, count in self._fold_counts.items() for _ in range(count)]
        assignments: dict[int, Split] = {}
        for split, indices in zip(fold_splits, folds, strict=True):
            for index in indices:
                assignments[int(index)] = split
        return assignments
