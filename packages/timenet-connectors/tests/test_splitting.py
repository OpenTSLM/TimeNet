from collections import Counter
from dataclasses import dataclass
import math
import random

from hypothesis import given, strategies as st
import numpy as np
import pytest

from timenet.errors import TimeFValidationError
from timenet.types.splits import Split
from timenet_connectors.splitting import DatasetSplits, Splitter, StratifiedSplitter
from timenet_connectors.splitting.stratified import _split_fold_counts


@dataclass
class Sample:
    signal_id: int
    category: str


def get_category(sample: Sample) -> str:
    return sample.category


def get_signal_id(sample: Sample) -> int:
    return sample.signal_id


@pytest.mark.parametrize(
    ("train", "validation", "test", "expected"),
    [
        (0.8, None, 0.2, {Split.TRAIN: 4, Split.TEST: 1}),
        (0.8, 0.1, 0.1, {Split.TRAIN: 8, Split.TEST: 1, Split.VALIDATION: 1}),
        (1 / 3, 1 / 3, 1 / 3, {Split.TRAIN: 1, Split.TEST: 1, Split.VALIDATION: 1}),
        (0.1 + 0.2, None, 0.7, {Split.TRAIN: 3, Split.TEST: 7}),
        (0.799, None, 0.201, {Split.TRAIN: 799, Split.TEST: 201}),
    ],
)
def test_fold_counts_preserve_proportions_and_partition_order(train, validation, test, expected):
    counts = _split_fold_counts(train, validation, test)

    assert counts == expected
    assert list(counts) == list(expected)


@given(st.lists(st.integers(min_value=1, max_value=100), min_size=2, max_size=3))
def test_fold_counts_recover_minimal_integer_weights(weights):
    total = sum(weights)
    splits = [Split.TRAIN, Split.TEST, Split.VALIDATION][: len(weights)]
    validation = weights[2] / total if len(weights) == 3 else None

    counts = _split_fold_counts(train=weights[0] / total, validation=validation, test=weights[1] / total)

    common_factor = math.gcd(*weights)
    assert counts == {split: weight // common_factor for split, weight in zip(splits, weights, strict=True)}


def test_train_test_preserves_category_proportions():
    samples = [Sample(index, "trend" if index < 90 else "season") for index in range(100)]
    splitter = StratifiedSplitter(train=0.8, test=0.2, stratify_by=get_category)

    splits = splitter.split(samples)

    assert splits.counts == {Split.TRAIN: 80, Split.TEST: 20}
    assert Counter(sample.category for sample in splits[Split.TRAIN]) == {"trend": 72, "season": 8}
    assert Counter(sample.category for sample in splits[Split.TEST]) == {"trend": 18, "season": 2}
    assert Split.VALIDATION not in splits
    with pytest.raises(KeyError):
        _ = splits[Split.VALIDATION]


def test_train_validation_test_and_source_order():
    samples = [Sample(index, str(index % 4)) for index in range(200)]
    splitter = StratifiedSplitter(train=0.8, validation=0.1, test=0.1, stratify_by=get_category)

    splits = splitter.split(samples)

    assert splits.counts == {Split.TRAIN: 160, Split.VALIDATION: 20, Split.TEST: 20}
    for split, expected in [(Split.TRAIN, 40), (Split.VALIDATION, 5), (Split.TEST, 5)]:
        assert Counter(sample.category for sample in splits[split]) == dict.fromkeys(map(str, range(4)), expected)
        indices = [sample.signal_id for sample in splits[split]]
        assert indices == sorted(indices)


def test_mixed_category_groups_stay_together():
    samples = [Sample(signal, category) for signal in range(20) for category in ("trend", "season")]
    splitter = StratifiedSplitter(train=0.8, test=0.2, stratify_by=get_category, group_by=get_signal_id)

    splits = splitter.split(samples)

    assert splits.counts == {Split.TRAIN: 32, Split.TEST: 8}
    train_signals = {sample.signal_id for sample in splits[Split.TRAIN]}
    test_signals = {sample.signal_id for sample in splits[Split.TEST]}
    assert train_signals.isdisjoint(test_signals)
    assert Counter(sample.category for sample in splits[Split.TEST]) == {"trend": 4, "season": 4}


@given(st.lists(st.tuples(st.integers(0, 10), st.sampled_from(["trend", "season"])), max_size=100))
def test_group_isolation_and_complete_assignment(rows):
    # Ten folds require ten groups; provide enough examples of both categories.
    samples = [Sample(signal, category) for signal in range(10) for category in ("trend", "season")]
    samples.extend(Sample(signal, category) for signal, category in rows)
    splitter = StratifiedSplitter(train=0.7, validation=0.1, test=0.2, stratify_by=get_category, group_by=get_signal_id)

    splits = splitter.split(samples)

    seen_signals = set()
    assigned = []
    for partition in splits.values():
        signals = {sample.signal_id for sample in partition}
        assert seen_signals.isdisjoint(signals)
        seen_signals.update(signals)
        assigned.extend(partition)
    assert Counter(map(id, assigned)) == Counter(map(id, samples))


def test_seed_is_repeatable_and_does_not_change_global_random_state():
    samples = [Sample(index, "trend") for index in range(100)]
    splitter = StratifiedSplitter(train=0.8, test=0.2, stratify_by=get_category, seed=42)
    other_seed = StratifiedSplitter(train=0.8, test=0.2, stratify_by=get_category, seed=43)
    random_state = random.getstate()
    numpy_state = np.random.get_state()

    first = splitter.split(samples)

    assert first == splitter.split(samples)
    assert first != other_seed.split(samples)
    assert random.getstate() == random_state
    np.testing.assert_equal(np.random.get_state(), numpy_state)


def test_too_few_groups_are_rejected():
    samples = [Sample(1, "trend"), Sample(1, "season")]
    splitter = StratifiedSplitter(train=0.8, test=0.2, stratify_by=get_category, group_by=get_signal_id)

    with pytest.raises(TimeFValidationError, match=r"5 folds.*found 1"):
        splitter.split(samples)


@pytest.mark.parametrize(
    ("folds", "message"),
    [
        ([[0, 1], [2]], "do not cover every input sample"),
        ([[0, 1], [1, 2, 3]], "multiple fold assignments"),
        ([[0, 0, 1], [2, 3]], "multiple fold assignments"),
        ([[-1, 0, 1], [2, 3]], "out of range"),
        ([[0, 1], [2, 3, 4]], "out of range"),
        ([[0, 2], [1, 3]], "group 0 occurs in multiple folds"),
        ([[0, 1, 2, 3]], "unexpected number"),
    ],
)
def test_invalid_fold_output_is_rejected(monkeypatch, folds, message):
    samples = [Sample(index // 2, "trend") for index in range(4)]
    splitter = StratifiedSplitter(train=0.5, test=0.5, stratify_by=get_category, group_by=get_signal_id)
    monkeypatch.setattr(
        splitter, "_split_into_folds", lambda categories, groups: iter(np.array(fold, dtype=np.intp) for fold in folds)
    )

    with pytest.raises(TimeFValidationError, match=message):
        splitter.split(samples)


def test_category_counts_that_cannot_support_folds_are_rejected():
    samples = [Sample(index, str(index)) for index in range(10)]
    splitter = StratifiedSplitter(train=0.8, test=0.2, stratify_by=get_category)

    with pytest.raises(TimeFValidationError, match="cannot stratify requested folds"):
        splitter.split(samples)


def test_rare_category_warns_without_dropping_samples():
    samples = [Sample(index, "rare" if index == 0 else "common") for index in range(20)]
    splitter = StratifiedSplitter(train=0.8, test=0.2, stratify_by=get_category)

    with pytest.warns(UserWarning, match="least populated class"):
        splits = splitter.split(samples)

    assert Counter(id(sample) for partition in splits.values() for sample in partition) == Counter(map(id, samples))


@pytest.mark.parametrize(
    ("train", "validation", "test", "counts"),
    [
        (0.7, None, 0.3, {Split.TRAIN: 42, Split.TEST: 18}),
        (1 / 3, 1 / 3, 1 / 3, {Split.TRAIN: 20, Split.VALIDATION: 20, Split.TEST: 20}),
    ],
)
def test_ratios_combine_equal_folds(train, validation, test, counts):
    samples = [Sample(index, "trend") for index in range(60)]
    splitter = StratifiedSplitter(train=train, validation=validation, test=test, stratify_by=get_category)

    assert splitter.split(samples).counts == counts


def test_tuple_category_and_group_keys_are_encoded_as_scalar_labels():
    samples = [Sample(signal, category) for signal in range(20) for category in ("trend", "season")]
    splitter = StratifiedSplitter(
        train=0.8,
        test=0.2,
        stratify_by=lambda sample: (sample.category, "task"),
        group_by=lambda sample: ((float(sample.signal_id),),),
    )

    splits = splitter.split(samples)

    assert splits.counts == {Split.TRAIN: 32, Split.TEST: 8}
    assert {sample.signal_id for sample in splits[Split.TRAIN]}.isdisjoint(
        sample.signal_id for sample in splits[Split.TEST]
    )


def test_empty_input_preserves_configured_partitions():
    splitter = StratifiedSplitter(train=0.8, test=0.2, stratify_by=get_category)
    assert splitter.split([]).counts == {Split.TRAIN: 0, Split.TEST: 0}


def test_repeated_sample_occurrences_are_not_deduplicated():
    sample = Sample(1, "trend")
    splitter = StratifiedSplitter(train=0.5, test=0.5, stratify_by=get_category)
    splits = splitter.split([sample, sample])
    assert splits[Split.TRAIN][0] is sample
    assert splits[Split.TEST][0] is sample


@pytest.mark.parametrize(
    ("train", "validation", "test"),
    [
        (0.8, None, 0.3),
        (0.8, None, 0.1),
        (0, None, 1),
        (1, None, 0),
        (-0.1, None, 1.1),
        (float("nan"), None, 0.2),
        (0.8, None, float("inf")),
        (0.8, 0, 0.2),
        (0.8, -0.1, 0.3),
        (0.8, float("nan"), 0.2),
        (1 - 1e-12, None, 1e-12),
    ],
)
def test_invalid_fractions_are_rejected(train, validation, test):
    with pytest.raises(TimeFValidationError):
        StratifiedSplitter(train=train, validation=validation, test=test, stratify_by=get_category)


def test_custom_splitter_uses_same_result_contract():
    class AllTraining(Splitter[int]):
        def split(self, samples):
            return DatasetSplits({Split.TRAIN: samples})

    assert AllTraining().split([1, 2])[Split.TRAIN] == (1, 2)
