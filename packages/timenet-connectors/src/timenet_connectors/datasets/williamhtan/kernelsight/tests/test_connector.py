import json
from pathlib import Path

import numpy as np
import pytest

from timenet.dataset import TimeFDataset
from timenet.dataset.axis import RegularAxis
from timenet.errors import TimeNetConnectorError
from timenet.reader import TimeFReader
from timenet.registry import open_local_version
from timenet.testing import assert_datasets_equal
from timenet.types import ClassificationTask, Split, TimeInterval
from timenet.types.text import read_value
from timenet_connectors.bases.huggingface import HubFileRef
from timenet_connectors.datasets.williamhtan.kernelsight.connector import (
    CHANNEL_GROUPS,
    COUNTER_NAMES,
    MULTIHOT_TRACK,
    N_BINS,
    VOCAB_L1,
    VOCAB_L2,
    WORKLOAD_L1,
    KernelSightConnector,
    Snapshot,
    assign_splits,
    read_snapshots,
)


_BIN_NS = 1_000_000
"""One millisecond per bin, so bin ``i`` starts at ``i * 1000`` microseconds."""
_MATMUL, _MEMORY = VOCAB_L1.index("matmul"), VOCAB_L1.index("memory_movement")
_GEMM = VOCAB_L2.index("matmul_gemm")


def _trace(path: str, motif: str, dominant: str) -> dict:
    return {"path": path, "motif": motif, "dominant_l1": dominant}


def _write_snapshot(
    root: Path,
    path: str,
    *,
    seed: int,
    segments: list[tuple[int, int, int, int]],
    multihot: np.ndarray | None = None,
) -> np.ndarray:
    # Synthetic files in the shape of the release, including its pickled object arrays, which the
    # connector must never read.
    data = np.random.default_rng(seed).random((len(COUNTER_NAMES), N_BINS), dtype=np.float32)
    edges = 5_000_000 + np.arange(N_BINS + 1, dtype=np.int64) * _BIN_NS
    input_path = root / path
    input_path.parent.mkdir(parents=True)
    np.savez(input_path, data=data, time_edges_ns=edges, counter_names=np.array(COUNTER_NAMES, dtype=object))
    starts, ends, l1, l2 = (
        np.array(column, dtype=dtype)
        for column, dtype in zip(zip(*segments, strict=True), (np.int64, np.int64, np.int32, np.int32), strict=True)
    )
    if multihot is None:
        multihot = np.zeros((N_BINS, len(VOCAB_L1)), dtype=np.uint8)
    labels_path = root / path.replace("/input/tensor_input.npz", "/labels/labels.npz")
    labels_path.parent.mkdir(parents=True)
    np.savez(
        labels_path,
        segment_starts=starts,
        segment_ends=ends,
        segment_label_l1=l1,
        segment_label_l2=l2,
        workload_l1_multihot=multihot,
        vocab_l1=np.array(VOCAB_L1, dtype=object),
        vocab_l2=np.array(VOCAB_L2, dtype=object),
    )
    return data


def _write_splits(root: Path, official: dict[str, list[dict]], extra: dict[str, list[dict]] | None = None) -> None:
    splits = root / "splits"
    splits.mkdir()
    files = {"train": [], "val": [], "test": [], "iid": [], "param_ood": [], "composed": [], "length_ood": []}
    files |= official | (extra or {})
    for name, traces in files.items():
        (splits / f"{name}.json").write_text(json.dumps({"split": name, "n": len(traces), "traces": traces}))


def _refs(root: Path) -> list[HubFileRef]:
    return [
        HubFileRef(root / "kernels", "inputs"),
        HubFileRef(root / "kernels", "labels"),
        HubFileRef(root / "splits", "splits"),
    ]


def _gemm_path(index: int) -> str:
    return f"kernels/cutlass_gemm/_out/m{index}/input/tensor_input.npz"


def _overlap_multihot() -> np.ndarray:
    # A GEMM in bins [100, 200) with TMA loads in [120, 130) and [150, 160) that no segment holds.
    multihot = np.zeros((N_BINS, len(VOCAB_L1)), dtype=np.uint8)
    multihot[100:200, _MATMUL] = 1
    multihot[120:130, _MEMORY] = 1
    multihot[150:160, _MEMORY] = 1
    return multihot


def _convert(root: Path) -> tuple[TimeFDataset, dict[str, np.ndarray]]:
    # Five problems, the fewest that the 3/1/1 split accepts.
    values = {}
    traces = []
    for index in range(5):
        path = _gemm_path(index)
        multihot = _overlap_multihot() if index == 0 else None
        values[path] = _write_snapshot(root, path, seed=index, segments=[(100, 200, _MATMUL, _GEMM)], multihot=multihot)
        traces.append(_trace(path, "cutlass_gemm", "matmul"))
    _write_splits(root, {"train": traces[:3], "val": traces[3:4], "test": traces[4:]}, {"param_ood": traces[:1]})
    return KernelSightConnector().convert(_refs(root)), values


def test_each_snapshot_becomes_a_record_and_a_classification_task(tmp_path):
    dataset, values = _convert(tmp_path)
    tasks = dataset.tasks_of(ClassificationTask)
    assert len(tasks) == len(values)
    assert {task.split for task in tasks} == {Split.TRAIN, Split.VALIDATION, Split.TEST}
    for task in tasks:
        (record,) = task.inputs
        assert task.label_set == WORKLOAD_L1
        assert task.targets is not None
        assert read_value(task.targets[0]) == "matmul"
        assert [source.name for source in record.sources] == [group for group, _ in CHANNEL_GROUPS]
        signals = [signal for source in record.sources for signal in source.signals]
        assert [signal.metadata["counter"] for signal in signals] == list(COUNTER_NAMES)
        for row, signal in enumerate(signals):
            np.testing.assert_array_equal(signal.to_numpy(), values[str(record.metadata["path"])][row])
        axis = signals[0].time_axis
        assert isinstance(axis, RegularAxis)
        assert axis.period_us == 1000


def test_record_metadata_keeps_the_released_split_memberships(tmp_path):
    dataset, _ = _convert(tmp_path)
    by_path = {
        task.inputs[0].metadata["path"]: task.inputs[0].metadata for task in dataset.tasks_of(ClassificationTask)
    }
    assert by_path[_gemm_path(0)] == {
        "path": _gemm_path(0),
        "motif": "cutlass_gemm",
        "problem": "cutlass_gemm/m0",
        "variant": "m0",
        "official_split": "train",
        "kernelsight_splits": ["train", "param_ood"],
    }


def test_segments_and_multihot_only_runs_become_interval_annotations(tmp_path):
    dataset, _ = _convert(tmp_path)
    by_path = {task.inputs[0].metadata["path"]: task.inputs[0] for task in dataset.tasks_of(ClassificationTask)}
    annotations = by_path[_gemm_path(0)].annotations
    found = {(a.key, read_value(a.value), a.span, a.occurrence_metadata.get("track")) for a in annotations}
    segment = TimeInterval.micros(100_000, 200_000)
    assert found == {
        ("workload_l1", "matmul", segment, None),
        ("workload_l2", "matmul_gemm", segment, None),
        ("workload_l1", "memory_movement", TimeInterval.micros(120_000, 130_000), MULTIHOT_TRACK),
        ("workload_l1", "memory_movement", TimeInterval.micros(150_000, 160_000), MULTIHOT_TRACK),
    }
    # A snapshot without an overlap track has only its segment.
    assert len(by_path[_gemm_path(1)].annotations) == 2


def test_variants_of_one_kernelbench_problem_share_a_partition():
    snapshots = [
        Snapshot(
            path=f"kernels/kernelbench/_out/L1_P{problem}/dt-{dtype}_s42/input/tensor_input.npz",
            motif="kernelbench",
            dominant_l1="conv" if problem % 2 else "matmul",
            official_split="train",
            memberships=("train",),
        )
        for problem in range(20)
        for dtype in ("bf16", "fp16", "fp32")
    ]
    assert snapshots[0].problem == "kernelbench/L1_P0"
    partition_of = {snapshot.path: split for split, members in assign_splits(snapshots).items() for snapshot in members}
    for problem in range(20):
        assert len({split for path, split in partition_of.items() if f"/L1_P{problem}/" in path}) == 1


def test_a_snapshot_directly_in_the_output_directory_is_the_baseline_variant():
    snapshot = Snapshot(
        path="kernels/cutlass_ws_overlap/_out/input/tensor_input.npz",
        motif="cutlass_ws_overlap",
        dominant_l1="matmul",
        official_split="train",
        memberships=("train",),
    )
    assert snapshot.problem == "cutlass_ws_overlap/baseline"


def test_a_trace_in_two_official_partitions_raises(tmp_path):
    trace = _trace(_gemm_path(0), "cutlass_gemm", "matmul")
    _write_splits(tmp_path, {"train": [trace], "test": [trace]})
    with pytest.raises(TimeNetConnectorError, match="more than one official partition"):
        read_snapshots(tmp_path / "splits")


def test_an_l2_class_outside_its_l1_class_raises(tmp_path):
    for index in range(5):
        l2 = VOCAB_L2.index("conv_conv1d_standard") if index == 0 else _GEMM
        _write_snapshot(tmp_path, _gemm_path(index), seed=index, segments=[(0, 10, _MATMUL, l2)])
    traces = [_trace(_gemm_path(index), "cutlass_gemm", "matmul") for index in range(5)]
    _write_splits(tmp_path, {"train": traces})
    with pytest.raises(TimeNetConnectorError, match="conv_conv1d_standard does not refine L1 class matmul"):
        KernelSightConnector().convert(_refs(tmp_path))


def test_the_dataset_round_trips_through_storage(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    dataset, _ = _convert(source)
    version_dir = dataset.write(path=tmp_path / "registry")
    with TimeFReader(open_local_version(version_dir)) as reader:
        reader.verify()
        assert_datasets_equal(dataset, reader.read())
