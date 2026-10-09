from collections import Counter
from dataclasses import dataclass
import random

from hypothesis import given, strategies as st
import pytest

from timenet.errors import TimeFValidationError
from timenet.types.splits import Split
from timenet_connectors.splitting import DatasetSplits, Splitter, StratifiedSplitter


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
    samples = [Sample(signal, category) for signal, category in rows]
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

    first = splitter.split(samples)

    assert first == splitter.split(samples)
    assert first != other_seed.split(samples)
    assert random.getstate() == random_state


def test_indivisible_group_may_leave_partition_empty():
    samples = [Sample(1, "trend"), Sample(1, "season")]
    splitter = StratifiedSplitter(train=0.8, test=0.2, stratify_by=get_category, group_by=get_signal_id)

    splits = splitter.split(samples)

    assert splits[Split.TRAIN] == tuple(samples)
    assert splits[Split.TEST] == ()


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
