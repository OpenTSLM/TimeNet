"""Named dataset partitions backed by scikit-learn's stratified folds."""

from collections.abc import Callable, Hashable, Iterable, Iterator, Sequence

import numpy as np
from numpy.typing import NDArray
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold

from timenet.errors import TimeFValidationError
from timenet.types.splits import Split
from timenet_connectors.splitting.base import DatasetSplits, Splitter, T


def _split_fold_counts(train_folds: int, validation_folds: int | None, test_folds: int) -> dict[Split, int]:
    counts = {Split.TRAIN: train_folds, Split.TEST: test_folds}
    if validation_folds is not None:
        counts[Split.VALIDATION] = validation_folds
    for split, count in counts.items():
        if isinstance(count, bool) or not isinstance(count, int) or count <= 0:
            raise TimeFValidationError(f"{split} fold count must be a positive integer")
    return counts


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
    Without ``group_by``, StratifiedKFold assigns independent input occurrences;
    otherwise StratifiedGroupKFold keeps related samples together.

    ``train_folds``, ``test_folds``, and optional ``validation_folds`` are positive integer fold
    counts. Their sum is the total number of folds: ``train_folds=4, test_folds=1`` uses
    five folds; ``train_folds=8, validation_folds=1, test_folds=1`` uses ten. Each group stays in one
    fold. Nonempty input must contain at least as many groups as folds and meet
    scikit-learn's category-count requirements. Invalid inputs raise
    ``TimeFValidationError``.

    Sample counts and category proportions are approximate for unequal groups.
    The same ordered input, seed, and scikit-learn version give the same result.
    This is not a temporal split strategy.
    """

    def __init__(  # noqa: PLR0913 -- keep partition fold counts and callbacks explicit.
        self,
        *,
        train_folds: int,
        test_folds: int,
        stratify_by: Callable[[T], Hashable],
        validation_folds: int | None = None,
        group_by: Callable[[T], Hashable] | None = None,
        seed: int = 42,
    ) -> None:
        self._fold_counts = _split_fold_counts(train_folds, validation_folds, test_folds)
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
        folds = list(self._split_into_folds(categories, groups))
        self._validate(folds, groups)
        return self._assign_folds_to_splits(folds)

    def _validate(self, folds: Sequence[NDArray[np.intp]], groups: Sequence[int]) -> None:
        # Validate before building the assignment dict, which would hide duplicate indices.
        if len(folds) != sum(self._fold_counts.values()):
            raise TimeFValidationError("unexpected number of stratified folds")

        seen: set[int] = set()
        group_folds: dict[int, int] = {}
        for fold, indices in enumerate(folds):
            for raw_index in indices:
                index = int(raw_index)
                if not 0 <= index < len(groups):
                    raise TimeFValidationError(f"sample index {index} is out of range")
                if index in seen:
                    raise TimeFValidationError(f"sample index {index} occurs in multiple fold assignments")
                seen.add(index)
                group = groups[index]
                if group_folds.setdefault(group, fold) != fold:
                    raise TimeFValidationError(f"group {group} occurs in multiple folds")

        if len(seen) != len(groups):
            raise TimeFValidationError("stratified folds do not cover every input sample")

    def _get_group_labels(self, samples: Sequence[T]) -> list[int]:
        if self._group_by is None:
            return list(range(len(samples)))
        return _encode_labels(samples, self._group_by)

    def _split_into_folds(self, categories: list[int], groups: list[int]) -> Iterator[NDArray[np.intp]]:
        fold_count = sum(self._fold_counts.values())
        group_count = len(set(groups))
        if group_count < fold_count:
            raise TimeFValidationError(f"{fold_count} folds require at least {fold_count} groups; found {group_count}")
        try:
            inputs = np.zeros((len(categories), 1))
            if self._group_by is None:
                # Avoid StratifiedGroupKFold's dense group-by-category matrix for independent samples.
                splitter = StratifiedKFold(n_splits=fold_count, shuffle=True, random_state=self._seed)
                folds = splitter.split(inputs, y=categories)
            else:
                group_splitter = StratifiedGroupKFold(n_splits=fold_count, shuffle=True, random_state=self._seed)
                folds = group_splitter.split(inputs, y=categories, groups=groups)
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
