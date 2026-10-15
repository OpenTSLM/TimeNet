from collections import Counter
from dataclasses import dataclass
import random

from hypothesis import given, strategies as st
import numpy as np
import pytest

from timenet.errors import TimeFValidationError
from timenet.types.splits import Split
from timenet_connectors.splitting import StratifiedSplitter, stratified


@dataclass
class Sample:
    signal_id: int
    category: str


def get_category(sample: Sample) -> str:
    return sample.category


def get_signal_id(sample: Sample) -> int:
    return sample.signal_id


def test_train_test_preserves_category_proportions():
    samples = [Sample(index, "trend" if index < 90 else "season") for index in range(100)]
    splitter = StratifiedSplitter(train_folds=4, test_folds=1, stratify_by=get_category)

    splits = splitter.split(samples)

    assert splits.counts == {Split.TRAIN: 80, Split.TEST: 20}
    assert Counter(sample.category for sample in splits[Split.TRAIN]) == {"trend": 72, "season": 8}
    assert Counter(sample.category for sample in splits[Split.TEST]) == {"trend": 18, "season": 2}
    assert Split.VALIDATION not in splits
    with pytest.raises(KeyError):
        _ = splits[Split.VALIDATION]


def test_many_independent_categories_do_not_use_grouped_splitter(monkeypatch):
    def reject_grouped_splitter(*args, **kwargs):
        pytest.fail("independent samples must not use the dense grouped splitter")

    monkeypatch.setattr(stratified, "StratifiedGroupKFold", reject_grouped_splitter)
    samples = [Sample(index, str(index % 5000)) for index in range(50000)]

    splits = StratifiedSplitter(train_folds=4, test_folds=1, stratify_by=get_category).split(samples)

    assert splits.counts == {Split.TRAIN: 40000, Split.TEST: 10000}
    assert Counter(sample.category for sample in splits[Split.TRAIN]) == dict.fromkeys(map(str, range(5000)), 8)
    assert Counter(sample.category for sample in splits[Split.TEST]) == dict.fromkeys(map(str, range(5000)), 2)
    assert Counter(id(sample) for partition in splits.values() for sample in partition) == Counter(map(id, samples))


def test_train_validation_test_and_source_order():
    samples = [Sample(index, str(index % 4)) for index in range(200)]
    splitter = StratifiedSplitter(train_folds=8, validation_folds=1, test_folds=1, stratify_by=get_category)

    splits = splitter.split(samples)

    assert splits.counts == {Split.TRAIN: 160, Split.VALIDATION: 20, Split.TEST: 20}
    for split, expected in [(Split.TRAIN, 40), (Split.VALIDATION, 5), (Split.TEST, 5)]:
        assert Counter(sample.category for sample in splits[split]) == dict.fromkeys(map(str, range(4)), expected)
        indices = [sample.signal_id for sample in splits[split]]
        assert indices == sorted(indices)


def test_mixed_category_groups_stay_together(monkeypatch):
    def reject_ungrouped_splitter(*args, **kwargs):
        pytest.fail("grouped samples must use the grouped splitter")

    monkeypatch.setattr(stratified, "StratifiedKFold", reject_ungrouped_splitter)
    samples = [Sample(signal, category) for signal in range(20) for category in ("trend", "season")]
    splitter = StratifiedSplitter(train_folds=4, test_folds=1, stratify_by=get_category, group_by=get_signal_id)

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
    splitter = StratifiedSplitter(
        train_folds=7, validation_folds=1, test_folds=2, stratify_by=get_category, group_by=get_signal_id
    )

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
    splitter = StratifiedSplitter(train_folds=4, test_folds=1, stratify_by=get_category, seed=42)
    other_seed = StratifiedSplitter(train_folds=4, test_folds=1, stratify_by=get_category, seed=43)
    random_state = random.getstate()
    numpy_state = np.random.get_state()

    first = splitter.split(samples)

    assert first == splitter.split(samples)
    assert first != other_seed.split(samples)
    assert random.getstate() == random_state
    np.testing.assert_equal(np.random.get_state(), numpy_state)


def test_too_few_groups_are_rejected():
    samples = [Sample(1, "trend"), Sample(1, "season")]
    splitter = StratifiedSplitter(train_folds=4, test_folds=1, stratify_by=get_category, group_by=get_signal_id)

    with pytest.raises(TimeFValidationError, match=r"5 folds.*found 1"):
        splitter.split(samples)


def test_fold_counts_are_not_reduced_to_a_ratio():
    samples = [Sample(index, "trend") for index in range(5)]
    assert StratifiedSplitter(train_folds=4, test_folds=1, stratify_by=get_category).split(samples).counts == {
        Split.TRAIN: 4,
        Split.TEST: 1,
    }
    with pytest.raises(TimeFValidationError, match=r"10 folds.*found 5"):
        StratifiedSplitter(train_folds=8, test_folds=2, stratify_by=get_category).split(samples)


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
    splitter = StratifiedSplitter(train_folds=1, test_folds=1, stratify_by=get_category, group_by=get_signal_id)
    monkeypatch.setattr(
        splitter, "_split_into_folds", lambda categories, groups: iter(np.array(fold, dtype=np.intp) for fold in folds)
    )

    with pytest.raises(TimeFValidationError, match=message):
        splitter.split(samples)


def test_category_counts_that_cannot_support_folds_are_rejected():
    samples = [Sample(index, str(index)) for index in range(10)]
    splitter = StratifiedSplitter(train_folds=4, test_folds=1, stratify_by=get_category)

    with pytest.raises(TimeFValidationError, match="cannot stratify requested folds"):
        splitter.split(samples)


def test_rare_category_warns_without_dropping_samples():
    samples = [Sample(index, "rare" if index == 0 else "common") for index in range(20)]
    splitter = StratifiedSplitter(train_folds=4, test_folds=1, stratify_by=get_category)

    with pytest.warns(UserWarning, match="least populated class"):
        splits = splitter.split(samples)

    assert Counter(id(sample) for partition in splits.values() for sample in partition) == Counter(map(id, samples))


@pytest.mark.parametrize(
    ("train_folds", "validation_folds", "test_folds", "counts"),
    [
        (7, None, 3, {Split.TRAIN: 42, Split.TEST: 18}),
        (1, 1, 1, {Split.TRAIN: 20, Split.VALIDATION: 20, Split.TEST: 20}),
    ],
)
def test_fold_counts_combine_equal_folds(train_folds, validation_folds, test_folds, counts):
    samples = [Sample(index, "trend") for index in range(60)]
    splitter = StratifiedSplitter(
        train_folds=train_folds, validation_folds=validation_folds, test_folds=test_folds, stratify_by=get_category
    )

    assert splitter.split(samples).counts == counts


def test_tuple_category_and_group_keys_are_encoded_as_scalar_labels():
    samples = [Sample(signal, category) for signal in range(20) for category in ("trend", "season")]
    splitter = StratifiedSplitter(
        train_folds=4,
        test_folds=1,
        stratify_by=lambda sample: (sample.category, "task"),
        group_by=lambda sample: ((float(sample.signal_id),),),
    )

    splits = splitter.split(samples)

    assert splits.counts == {Split.TRAIN: 32, Split.TEST: 8}
    assert {sample.signal_id for sample in splits[Split.TRAIN]}.isdisjoint(
        sample.signal_id for sample in splits[Split.TEST]
    )


def test_empty_input_preserves_configured_partitions():
    splitter = StratifiedSplitter(train_folds=4, test_folds=1, stratify_by=get_category)
    assert splitter.split([]).counts == {Split.TRAIN: 0, Split.TEST: 0}


def test_repeated_sample_occurrences_are_not_deduplicated():
    sample = Sample(1, "trend")
    splitter = StratifiedSplitter(train_folds=1, test_folds=1, stratify_by=get_category)
    splits = splitter.split([sample, sample])
    assert splits[Split.TRAIN][0] is sample
    assert splits[Split.TEST][0] is sample


@pytest.mark.parametrize(
    ("train_folds", "validation_folds", "test_folds"),
    [
        (0, None, 1),
        (1, None, 0),
        (-1, None, 1),
        (1, 0, 1),
        (1, -1, 1),
        (0.8, None, 0.2),
        (4.0, None, 1),
        (4, 1.5, 1),
        (4, None, 1.0),
        (True, None, 1),
        (4, False, 1),
        (4, None, True),
        (float("nan"), None, 1),
        (4, None, float("inf")),
    ],
)
def test_invalid_fold_counts_are_rejected(train_folds, validation_folds, test_folds):
    with pytest.raises(TimeFValidationError, match="fold count must be a positive integer"):
        StratifiedSplitter(
            train_folds=train_folds, validation_folds=validation_folds, test_folds=test_folds, stratify_by=get_category
        )
