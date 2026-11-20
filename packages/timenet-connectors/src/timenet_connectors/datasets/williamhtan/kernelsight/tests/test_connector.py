import json
from pathlib import Path
import shutil
import struct
from typing import Any
import zipfile

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
from timenet_connectors.datasets.williamhtan.kernelsight.connector import (
    CHANNEL_GROUPS,
    COUNTER_NAMES,
    INPUT_SUFFIX,
    LABELS_SUFFIX,
    MULTIHOT_TRACK,
    N_BINS,
    RELEASE_SPLITS,
    VOCAB_L1,
    VOCAB_L2,
    WORKLOAD_L1,
    KernelSightConnector,
    Snapshot,
    assign_splits,
    read_snapshots,
)


FIXTURES = Path(__file__).parent / "fixtures"
"""One real Snapshot of the pinned release: a warp-specialized GEMM with a multi-hot-only TMA track."""
_REAL_PATH = f"kernels/cutlass_ws_overlap/_out/m1024_n1024_k1024_rH_s1_b0{INPUT_SUFFIX}"
_BIN_NS = 1_000_000
"""One millisecond per bin, so bin ``i`` starts at ``i * 1000`` microseconds."""
_MATMUL, _MEMORY = VOCAB_L1.index("matmul"), VOCAB_L1.index("memory_movement")
_GEMM, _COPY = VOCAB_L2.index("matmul_gemm"), VOCAB_L2.index("memory_movement_copy")


def _trace(path: str, motif: str, dominant: str) -> dict:
    return {"path": path, "motif": motif, "dominant_l1": dominant}


def _snapshot(path: str, motif: str, dominant: str = "matmul") -> Snapshot:
    return Snapshot(path=path, motif=motif, dominant_l1=dominant, official_split="train", release_splits=("train",))


def _write_snapshot(
    root: Path,
    path: str,
    *,
    seed: int,
    segments: list[tuple[int, int, int, int]],
    overlap: bool = False,
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
    l1_multihot = np.zeros((N_BINS, len(VOCAB_L1)), dtype=np.uint8)
    l2_multihot = np.zeros((N_BINS, len(VOCAB_L2)), dtype=np.uint8)
    if overlap:
        # A GEMM in bins [100, 200) with TMA loads in [120, 130) and [150, 160) that no Segment holds.
        l1_multihot[100:200, _MATMUL] = l2_multihot[100:200, _GEMM] = 1
        for start, end in ((120, 130), (150, 160)):
            l1_multihot[start:end, _MEMORY] = l2_multihot[start:end, _COPY] = 1
    labels_path = root / (path.removesuffix(INPUT_SUFFIX) + LABELS_SUFFIX)
    labels_path.parent.mkdir(parents=True)
    np.savez(
        labels_path,
        segment_starts=starts,
        segment_ends=ends,
        segment_label_l1=l1,
        segment_label_l2=l2,
        workload_l1_multihot=l1_multihot,
        workload_l2_multihot=l2_multihot,
        vocab_l1=np.array(VOCAB_L1, dtype=object),
        vocab_l2=np.array(VOCAB_L2, dtype=object),
    )
    return data


def _write_splits(root: Path, official: dict[str, list[dict]], extra: dict[str, list[dict]] | None = None) -> None:
    splits = root / "splits"
    splits.mkdir()
    files = {name: [] for name in RELEASE_SPLITS} | official | (extra or {})
    for name, traces in files.items():
        (splits / f"{name}.json").write_text(json.dumps({"split": name, "n": len(traces), "traces": traces}))


def _files(root: Path) -> dict[str, Path]:
    return {"inputs": root / "kernels", "labels": root / "kernels", "splits": root / "splits"}


def _gemm_path(index: int) -> str:
    return f"kernels/cutlass_gemm/_out/m{index}{INPUT_SUFFIX}"


_DOMINANT = {_gemm_path(index): "matmul" if index < 5 else "conv" for index in range(10)}
"""Five Problems of each class, the fewest that the 3/1/1 split accepts."""
_TRACE = _trace(_gemm_path(0), "cutlass_gemm", "matmul")


def _convert(root: Path) -> tuple[TimeFDataset, dict[str, np.ndarray]]:
    values = {}
    for index, path in enumerate(_DOMINANT):
        values[path] = _write_snapshot(
            root, path, seed=index, segments=[(100, 200, _MATMUL, _GEMM)], overlap=index == 0
        )
    traces = [_trace(path, "cutlass_gemm", dominant) for path, dominant in _DOMINANT.items()]
    _write_splits(root, {"train": traces[:6], "val": traces[6:8], "test": traces[8:]}, {"param_ood": traces[:1]})
    return KernelSightConnector().convert(_files(root)), values


def _replace_arrays(path: Path, **arrays: np.ndarray) -> None:
    # The file is one that the test wrote, so its object arrays are safe to unpickle.
    with np.load(path, allow_pickle=True) as archive:
        kept: dict[str, Any] = {name: archive[name] for name in archive.files}
    kept.update(arrays)
    np.savez(path, **kept)


def _found(annotations) -> set:
    return {(a.key, read_value(a.value), a.span, a.occurrence_metadata.get("track")) for a in annotations}


def test_each_snapshot_becomes_a_record_and_a_classification_task(tmp_path):
    dataset, values = _convert(tmp_path)
    tasks = dataset.tasks_of(ClassificationTask)
    assert len(tasks) == len(values)
    assert {task.split for task in tasks} == {Split.TRAIN, Split.VALIDATION, Split.TEST}
    for task in tasks:
        (record,) = task.inputs
        assert task.label_set == WORKLOAD_L1
        assert task.targets is not None
        assert read_value(task.targets[0]) == _DOMINANT[str(record.metadata["path"])]
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
        "release_splits": ["train", "param_ood"],
    }


def test_segments_and_multihot_only_runs_become_interval_annotations(tmp_path):
    dataset, _ = _convert(tmp_path)
    by_path = {task.inputs[0].metadata["path"]: task.inputs[0] for task in dataset.tasks_of(ClassificationTask)}
    segment = TimeInterval.micros(100_000, 200_000)
    loads = (TimeInterval.micros(120_000, 130_000), TimeInterval.micros(150_000, 160_000))
    assert _found(by_path[_gemm_path(0)].annotations) == {
        ("workload_l1", "matmul", segment, None),
        ("workload_l2", "matmul_gemm", segment, None),
        *(("workload_l1", "memory_movement", span, MULTIHOT_TRACK) for span in loads),
        *(("workload_l2", "memory_movement_copy", span, MULTIHOT_TRACK) for span in loads),
    }
    # A Snapshot without an overlap track has only its Segment.
    assert len(by_path[_gemm_path(1)].annotations) == 2


def test_a_real_snapshot_of_the_release_becomes_a_record():
    record = _snapshot(_REAL_PATH, "cutlass_ws_overlap").create_record(FIXTURES)
    signals = [signal for source in record.sources for signal in source.signals]
    assert len(signals) == len(COUNTER_NAMES)
    with np.load(FIXTURES / _REAL_PATH, allow_pickle=False) as archive:
        np.testing.assert_array_equal(signals[5].to_numpy(), archive["data"][5])
    axis = signals[0].time_axis
    assert isinstance(axis, RegularAxis)
    assert axis.period_us == pytest.approx(203.73275)
    segment = TimeInterval.micros(49_914, 54_396)
    loads = (TimeInterval.micros(49_914, 53_785), TimeInterval.micros(53_989, 54_396))
    assert _found(record.annotations) == {
        ("workload_l1", "matmul", segment, None),
        ("workload_l2", "matmul_gemm", segment, None),
        *(("workload_l1", "memory_movement", span, MULTIHOT_TRACK) for span in loads),
        *(("workload_l2", "memory_movement_copy", span, MULTIHOT_TRACK) for span in loads),
    }


def test_variants_of_one_kernelbench_problem_share_a_partition():
    snapshots = [
        _snapshot(
            f"kernels/kernelbench/_out/L1_P{problem}/dt-{dtype}_s42{INPUT_SUFFIX}",
            "kernelbench",
            "conv" if problem % 2 else "matmul",
        )
        for problem in range(20)
        for dtype in ("bf16", "fp16", "fp32")
    ]
    assert snapshots[0].problem == "kernelbench/L1_P0"
    partition_of = {snapshot.path: split for split, members in assign_splits(snapshots).items() for snapshot in members}
    for problem in range(20):
        assert len({split for path, split in partition_of.items() if f"/L1_P{problem}/" in path}) == 1


@pytest.mark.parametrize(
    ("motif", "variant", "problem"),
    [
        ("cutlass_gemm", "m1024_n1024_k8192_rH_s8_b1", "cutlass_gemm/m1024_n1024_k8192"),
        ("cutlass_gemm", "m4096_rM", "cutlass_gemm/m4096"),
        ("cutlass_gemm", "", "cutlass_gemm/m4096"),
        ("cutlass_gemm", "m2048_rN_b1", "cutlass_gemm/m2048_n2048_k2048"),
        ("cutlass_gemm", "kheavy", "cutlass_gemm/m1024_n1024_k8192"),
        ("cutlass_gemm", "m8192_k1024", "cutlass_gemm/m8192_k1024"),
        # cutlass_ws_overlap runs the kernel of cutlass_gemm, but a short name has the defaults of its own motif.
        ("cutlass_ws_overlap", "m512_n512_k8192_rM_s1_b0", "cutlass_gemm/m512_n512_k8192"),
        ("cutlass_ws_overlap", "m4096_rN", "cutlass_gemm/m4096_n4096_k2048"),
        ("cutlass_ws_overlap", "m2048_sw4", "cutlass_gemm/m2048_n2048_k2048"),
        ("cutlass_ws_overlap", "m1024_k2048", "cutlass_gemm/m1024_n1024_k2048"),
        ("cutlass_ws_overlap", "", "cutlass_gemm/k2048"),
        ("vector_add", "N16M_T1024", "vector_add/N16M"),
        ("reduction", "N16M_T256_I4", "reduction/N16M_I4"),
        ("scatter", "N1M_B16_T512", "scatter/N1M_B16"),
        ("cutlass_fmha", "b2h16_q1024_d64_causal", "cutlass_fmha/b2h16_q1024_d64_causal"),
    ],
)
def test_the_problem_of_a_variant_is_its_shape_without_launch_options(motif, variant, problem):
    directory = f"{variant}/" if variant else ""
    assert _snapshot(f"kernels/{motif}/_out/{directory}{INPUT_SUFFIX.lstrip('/')}", motif).problem == problem


@pytest.mark.parametrize(
    ("official", "extra", "message"),
    [
        ({"train": [_TRACE], "test": [_TRACE]}, {}, "more than one official partition"),
        ({}, {"iid": [_TRACE]}, "in no official partition"),
        ({"train": [{"path": _gemm_path(0)}]}, {}, r"train\.json is not a KernelSight split file"),
        ({"train": [_trace(_gemm_path(0), "cutlass_fmha", "matmul")]}, {}, "is not under kernels/cutlass_fmha/_out/"),
        ({"train": [_trace(_gemm_path(0), "cutlass_gemm", "gemm")]}, {}, "'gemm' is not an L1 class"),
    ],
)
def test_a_malformed_split_index_raises(tmp_path, official, extra, message):
    _write_splits(tmp_path, official, extra)
    with pytest.raises(TimeNetConnectorError, match=message):
        read_snapshots(tmp_path / "splits")


@pytest.mark.parametrize(
    ("suffix", "arrays", "message"),
    [
        (LABELS_SUFFIX, {"segment_ends": np.array([10, 20])}, "different lengths"),
        (LABELS_SUFFIX, {"segment_ends": np.array([N_BINS + 1])}, "outside the 512 bins"),
        (LABELS_SUFFIX, {"segment_label_l1": np.array([-1]), "segment_label_l2": np.array([-1])}, "out of range"),
        (LABELS_SUFFIX, {"workload_l2_multihot": np.zeros((N_BINS, 3), np.uint8)}, "workload_l2_multihot has shape"),
        (INPUT_SUFFIX, {"time_edges_ns": np.arange(N_BINS)}, "time_edges_ns has shape"),
        (INPUT_SUFFIX, {"data": np.zeros((len(COUNTER_NAMES), N_BINS - 1), np.float32)}, "data has shape"),
    ],
)
def test_a_malformed_array_raises(tmp_path, suffix, arrays, message):
    _write_snapshot(tmp_path, _gemm_path(0), seed=0, segments=[(0, 10, _MATMUL, _GEMM)])
    _replace_arrays(tmp_path / (_gemm_path(0).removesuffix(INPUT_SUFFIX) + suffix), **arrays)
    with pytest.raises(TimeNetConnectorError, match=message):
        record = _snapshot(_gemm_path(0), "cutlass_gemm").create_record(tmp_path)
        # The data shape is checked only when a value is read.
        record.sources[0].signals[0].to_numpy()


def test_a_corrupt_labels_file_raises(tmp_path):
    _write_snapshot(tmp_path, _gemm_path(0), seed=0, segments=[(0, 10, _MATMUL, _GEMM)])
    labels = tmp_path / _snapshot(_gemm_path(0), "cutlass_gemm").labels_path
    labels.write_bytes(b"not a zip archive")
    with pytest.raises(TimeNetConnectorError, match="cannot read segment_starts"):
        _snapshot(_gemm_path(0), "cutlass_gemm").create_record(tmp_path)


def test_a_damaged_compressed_member_raises(tmp_path):
    shutil.copytree(FIXTURES, tmp_path, dirs_exist_ok=True)
    snapshot = _snapshot(_REAL_PATH, "cutlass_ws_overlap")
    labels = tmp_path / snapshot.labels_path
    with zipfile.ZipFile(labels) as archive:
        offset = archive.getinfo("segment_starts.npy").header_offset
    raw = bytearray(labels.read_bytes())
    name_length, extra_length = struct.unpack_from("<HH", raw, offset + 26)
    # The member is deflate-compressed. A first byte of 0xFF starts a block of the reserved type, so zlib fails.
    raw[offset + 30 + name_length + extra_length] = 0xFF
    labels.write_bytes(raw)
    with pytest.raises(TimeNetConnectorError, match="cannot read segment_starts"):
        snapshot.create_record(tmp_path)


def test_an_l2_class_outside_its_l1_class_raises(tmp_path):
    for index in range(5):
        l2 = VOCAB_L2.index("conv_conv1d_standard") if index == 0 else _GEMM
        _write_snapshot(tmp_path, _gemm_path(index), seed=index, segments=[(0, 10, _MATMUL, l2)])
    traces = [_trace(_gemm_path(index), "cutlass_gemm", "matmul") for index in range(5)]
    _write_splits(tmp_path, {"train": traces})
    with pytest.raises(TimeNetConnectorError, match="conv_conv1d_standard does not refine L1 class matmul"):
        KernelSightConnector().convert(_files(tmp_path))


def test_the_dataset_round_trips_through_storage(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    dataset, _ = _convert(source)
    version_dir = dataset.write(path=tmp_path / "registry")
    with TimeFReader(open_local_version(version_dir)) as reader:
        reader.verify()
        assert_datasets_equal(dataset, reader.read())
