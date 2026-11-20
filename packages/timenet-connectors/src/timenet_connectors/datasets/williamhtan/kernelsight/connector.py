"""The KernelSight connector: H100 GPU counter traces with per-segment workload labels.

Each snapshot of the release is a ``[24, 512]`` counter image: 24 hardware-counter channels over 512
equal-width bins. Each snapshot becomes one Record with one Signal per channel, grouped into the
five channel groups of the dataset card. Each labeled segment becomes a time-interval annotation
with its L1 and L2 workload class. Each snapshot gets one ClassificationTask whose target is the
authors' dominant L1 class.

The release's own train/val/test split puts dtype and seed variants of one KernelBench problem in
different partitions. The connector therefore assigns a split grouped by problem and keeps the
authors' split memberships in the record metadata.

The release stores its vocabularies and counter names as pickled object arrays. Unpickling a
downloaded file can run code, so the connector keeps those names as constants and reads only the
numeric arrays, with ``allow_pickle=False``.
"""

from collections import defaultdict
from collections.abc import Callable, Iterator, Mapping
from dataclasses import dataclass
from fractions import Fraction
import json
from operator import attrgetter
from pathlib import Path
import re
import zipfile
import zlib

import numpy as np
import pyarrow as pa

from timenet.dataset import Record, Signal, Source, TimeFDataset
from timenet.dataset.axis import RegularAxis
from timenet.errors import TimeNetConnectorError
from timenet.types import Annotation, ClassificationTask, LabelSet, TimeInterval, TimeSeriesSpec, ureg
from timenet_connectors.bases.huggingface import BaseHuggingFaceConnector
from timenet_connectors.splitting import DatasetSplits, StratifiedSplitter


N_BINS = 512

# The five channel groups of the dataset card, in row order of the ``data`` array.
CHANNEL_GROUPS: tuple[tuple[str, tuple[str, ...]], ...] = (
    (
        "Pipe signature",
        (
            "pipe: tensor_op_hmma",
            "pipe: xu",
            "pipe: fma",
            "pipe: alu",
            "pipe: lsu",
            "pipe: cbu",
            "pipe: tma",
        ),
    ),
    ("Memory access", ("hit: l2", "atom: lts_atomic_input_pct")),
    (
        "Discriminators",
        ("stall: short_scoreboard", "stall: barrier", "pred_on_per_inst_ratio", "gmem_coalesce_ratio"),
    ),
    (
        "System bandwidth",
        (
            "SMs Active [Throughput %]",
            "DRAM Read Bandwidth [Throughput %]",
            "DRAM Write Bandwidth [Throughput %]",
            "Tensor Active [Throughput %]",
        ),
    ),
    (
        "SASS modality",
        (
            "sass_compute_fma",
            "sass_compute_tensor",
            "sass_memory_global",
            "sass_memory_shared",
            "sass_memory_tma",
            "sass_control",
            "sass_misc",
        ),
    ),
)
COUNTER_NAMES: tuple[str, ...] = tuple(name for _, names in CHANNEL_GROUPS for name in names)

# The ``vocab_l1`` and ``vocab_l2`` arrays of every labels file, in index order.
VOCAB_L1: tuple[str, ...] = (
    "matmul",
    "conv",
    "activation",
    "normalization",
    "softmax",
    "pooling",
    "reduction",
    "attention",
    "loss",
    "elementwise",
    "memory_movement",
    "other",
)
VOCAB_L2: tuple[str, ...] = (
    "matmul_bmm",
    "matmul_gemm",
    "matmul_matvec",
    "conv_conv1d_standard",
    "conv_conv2d_depthwise",
    "conv_conv2d_pointwise",
    "conv_conv2d_standard",
    "conv_conv3d_standard",
    "conv_convtranspose1d",
    "conv_convtranspose2d",
    "conv_convtranspose3d",
    "activation_elu",
    "activation_gelu",
    "activation_hardsigmoid",
    "activation_hardswish",
    "activation_hardtanh",
    "activation_leaky_relu",
    "activation_mish",
    "activation_other",
    "activation_relu",
    "activation_selu",
    "activation_sigmoid",
    "activation_softplus",
    "activation_softsign",
    "activation_swish",
    "activation_tanh",
    "normalization_batchnorm",
    "normalization_frobeniusnorm",
    "normalization_groupnorm",
    "normalization_instancenorm",
    "normalization_l1norm",
    "normalization_l2norm",
    "normalization_layernorm",
    "normalization_rmsnorm",
    "softmax_log_softmax",
    "softmax_logsumexp",
    "softmax_softmax",
    "pooling_avg_pool",
    "pooling_global_avg_pool",
    "pooling_max_pool",
    "reduction_argmax",
    "reduction_argmin",
    "reduction_cumprod",
    "reduction_cumsum",
    "reduction_max",
    "reduction_mean",
    "reduction_min",
    "reduction_prod",
    "reduction_sum",
    "attention_scaled_dot_product",
    "loss_cross_entropy",
    "loss_hinge",
    "loss_huber",
    "loss_kldiv",
    "loss_mse",
    "loss_triplet_margin",
    "elementwise_add",
    "elementwise_bias_add",
    "elementwise_cast",
    "elementwise_clamp",
    "elementwise_div",
    "elementwise_mul",
    "elementwise_residual_add",
    "elementwise_scalar_multiplication",
    "elementwise_scaling",
    "elementwise_sub",
    "memory_movement_copy",
    "memory_movement_embedding",
    "memory_movement_gather",
    "memory_movement_scatter",
    "memory_movement_transpose",
    "other_dropout",
    "other_misc",
)

WORKLOAD_L1 = LabelSet(VOCAB_L1, annotations=[Annotation("name", value="workload_l1")])

OFFICIAL_SPLITS = ("train", "val", "test")
"""The released partition files. Each Snapshot is in exactly one of them."""
RELEASE_SPLITS = (*OFFICIAL_SPLITS, "iid", "param_ood", "composed", "length_ood")
"""Every released split file. A Snapshot can be in several; the record metadata keeps them all."""

MULTIHOT_TRACK = "multihot"
"""The ``track`` of an interval that only the multi-hot overlap track of the release holds."""

INPUT_SUFFIX = "/input/tensor_input.npz"
LABELS_SUFFIX = "/labels/labels.npz"

# The launch options of a Variant: the CUTLASS raster, swizzle and ``b`` options, alone or together
# (``m1024_n1024_k8192_rH_s8_b1``, ``m4096_rM``, ``m2048_sw4``, ``m2048_rN_b1``), and the thread-block size
# of a microbenchmark (``N1M_T256``).
_LAUNCH_OPTIONS = re.compile(r"_(?:r[A-Z]|s\d+|sw\d+|b\d+)(?=_|$)|_T\d+")

# NOTE: cutlass_ws_overlap runs the CUTLASS ex48 GEMM of cutlass_gemm with device timer markers added. One
# shape of the two motifs gives near-identical traces, so it is one Problem.
_SAME_KERNEL = {"cutlass_ws_overlap": "cutlass_gemm"}


def _slug(name: str) -> str:
    """Turn a counter name such as ``"SMs Active [Throughput %]"`` into ``"sms_active_throughput"``.

    Returns:
        The lowercase name, with each run of other characters replaced by one underscore.
    """
    return re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")


# One spec per channel, because a spec type has one contract and each counter has its own name. The
# values are dimensionless because the authors divided each channel by a fixed physical-scale divisor.
_SPECS: Mapping[str, TimeSeriesSpec] = {
    name: TimeSeriesSpec(spec_type=f"gpu_counter_{_slug(name)}", name=name, unit_value=ureg.dimensionless)
    for name in COUNTER_NAMES
}


def _read_arrays(path: Path, *names: str) -> list[np.ndarray]:
    """Read numeric arrays from an ``.npz`` file without unpickling.

    Returns:
        The arrays, in the order of ``names``.

    Raises:
        TimeNetConnectorError: If the file is missing or corrupt, lacks one of the arrays, or one of the
            arrays holds Python objects.
    """
    try:
        with np.load(path, allow_pickle=False) as archive:
            return [archive[name] for name in names]
    except (OSError, EOFError, ValueError, KeyError, zipfile.BadZipFile, zlib.error) as error:
        raise TimeNetConnectorError(f"{path}: cannot read {', '.join(names)}: {error}") from error


def _channel_loader(path: Path, row: int) -> Callable[[], pa.Array]:
    """Build a lazy loader for one channel of a Snapshot.

    Returns:
        A loader that reads row ``row`` of the ``data`` array as a float32 Arrow array.
    """

    def load() -> pa.Array:
        # NOTE: The shape is checked here, not in ``convert``, so that ``convert`` never reads the values.
        (data,) = _read_arrays(path, "data")
        if data.shape != (len(COUNTER_NAMES), N_BINS):
            raise TimeNetConnectorError(f"{path}: data has shape {data.shape}")
        return pa.array(data[row])

    return load


def _true_runs(active: np.ndarray) -> Iterator[tuple[int, int]]:
    """Yield the ``[start, end)`` bin ranges of each run of consecutive true values."""
    padded = np.concatenate(([False], active.astype(bool), [False]))
    edges = np.flatnonzero(np.diff(padded.astype(np.int8)))
    yield from zip(edges[::2].tolist(), edges[1::2].tolist(), strict=True)


@dataclass(frozen=True)
class Snapshot:
    """One Snapshot of the release, as its split files describe it."""

    path: str
    """The input tensor, relative to the dataset root, such as
    ``kernels/kernelbench/_out/L1_P100/dt-bf16_s42/input/tensor_input.npz``."""
    motif: str
    dominant_l1: str
    official_split: str
    release_splits: tuple[str, ...]

    def __post_init__(self) -> None:
        """Check that the path has the layout of the release and the class is in the vocabulary.

        Raises:
            TimeNetConnectorError: If the path or the dominant class is not as expected.
        """
        if not (self.path.startswith(self._variants_dir) and self.path.endswith(INPUT_SUFFIX)):
            raise TimeNetConnectorError(f"snapshot path {self.path!r} is not under {self._variants_dir}")
        if self.dominant_l1 not in VOCAB_L1:
            raise TimeNetConnectorError(f"{self.path}: dominant_l1 {self.dominant_l1!r} is not an L1 class")

    @property
    def _variants_dir(self) -> str:
        return f"kernels/{self.motif}/_out/"

    @property
    def variant(self) -> str:
        """The Variant directory below ``_out/``, or ``"baseline"`` for a Snapshot directly in it."""
        return self.path[len(self._variants_dir) : -len(INPUT_SUFFIX)] or "baseline"

    @property
    def problem(self) -> str:
        """The leakage group: the motif of its kernel, and the Variant without dtype, seed and launch options."""
        if self.motif == "kernelbench":
            return f"{self.motif}/{self.variant.split('/')[0]}"
        return f"{_SAME_KERNEL.get(self.motif, self.motif)}/{_LAUNCH_OPTIONS.sub('', self.variant)}"

    @property
    def labels_path(self) -> str:
        """The labels file of the Snapshot, relative to the dataset root."""
        return self.path.removesuffix(INPUT_SUFFIX) + LABELS_SUFFIX

    def create_record(self, root: Path) -> Record:
        """Create the Record: one Signal per channel, grouped by channel group, with Segment annotations.

        Args:
            root: The dataset root that the paths of the Snapshot are relative to.

        Returns:
            The Record, not yet added to a dataset.

        Raises:
            TimeNetConnectorError: If a file of the Snapshot is missing or corrupt, or does not have the
                layout of the release.
        """
        input_path = root / self.path
        (edges,) = _read_arrays(input_path, "time_edges_ns")
        if edges.shape != (N_BINS + 1,):
            raise TimeNetConnectorError(f"{self.path}: time_edges_ns has shape {edges.shape}")
        # The bins are equal-width up to one nanosecond of rounding, so the exact mean width is the period.
        axis = RegularAxis(period_us=Fraction(int(edges[-1] - edges[0]), N_BINS * 1000))
        signals = iter(
            Signal.from_loader(
                spec=_SPECS[name],
                time_axis=axis,
                name=_slug(name),
                loader=_channel_loader(input_path, row),
                n_values=N_BINS,
                metadata={"counter": name},
            )
            for row, name in enumerate(COUNTER_NAMES)
        )
        sources = tuple(Source(group, signals=tuple(next(signals) for _ in names)) for group, names in CHANNEL_GROUPS)
        return Record(
            sources=sources,
            metadata={
                "path": self.path,
                "motif": self.motif,
                "problem": self.problem,
                "variant": self.variant,
                "official_split": self.official_split,
                "release_splits": list(self.release_splits),
            },
            annotations=self._segment_annotations(root, axis),
        )

    def _segment_annotations(self, root: Path, axis: RegularAxis) -> list[Annotation]:
        """Annotate each labeled Segment, and each multi-hot run of a class that no Segment has.

        Returns:
            The interval annotations of the Record.

        Raises:
            TimeNetConnectorError: If the labels file is missing or corrupt, an array has the wrong shape,
                a label index or a Segment bound is out of range, or an L2 class does not refine the L1
                class of its Segment.
        """
        path = self.labels_path

        def interval(start: int, end: int) -> TimeInterval:
            if not 0 <= start < end <= N_BINS:
                raise TimeNetConnectorError(f"{path}: segment [{start}, {end}) is outside the {N_BINS} bins")
            return TimeInterval.micros(int(start * axis.period_us), int(end * axis.period_us))

        starts, ends, l1, l2, l1_multihot, l2_multihot = _read_arrays(
            root / path,
            "segment_starts",
            "segment_ends",
            "segment_label_l1",
            "segment_label_l2",
            "workload_l1_multihot",
            "workload_l2_multihot",
        )
        if not len(starts) == len(ends) == len(l1) == len(l2):
            raise TimeNetConnectorError(f"{path}: the segment arrays have different lengths")
        tracks = (("workload_l1", l1, l1_multihot, VOCAB_L1), ("workload_l2", l2, l2_multihot, VOCAB_L2))
        for key, _, multihot, vocab in tracks:
            if multihot.shape != (N_BINS, len(vocab)):
                raise TimeNetConnectorError(f"{path}: {key}_multihot has shape {multihot.shape}")
        annotations = []
        for start, end, l1_index, l2_index in zip(starts, ends, l1, l2, strict=True):
            if not (0 <= l1_index < len(VOCAB_L1) and 0 <= l2_index < len(VOCAB_L2)):
                raise TimeNetConnectorError(f"{path}: label index ({l1_index}, {l2_index}) is out of range")
            l1_name, l2_name = VOCAB_L1[l1_index], VOCAB_L2[l2_index]
            if not l2_name.startswith(f"{l1_name}_"):
                raise TimeNetConnectorError(f"{path}: L2 class {l2_name} does not refine L1 class {l1_name}")
            span = interval(int(start), int(end))
            annotations += [
                Annotation("workload_l1", value=l1_name, span=span),
                Annotation("workload_l2", value=l2_name, span=span),
            ]
        # The overlap track of the release can hold a class that no Segment has, such as the TMA loads
        # (memory_movement, memory_movement_copy) that run during a warp-specialized GEMM. Keep each of
        # its runs as an interval.
        for key, segment_classes, multihot, vocab in tracks:
            only_multihot = set(np.flatnonzero(multihot.any(axis=0)).tolist()) - set(segment_classes.tolist())
            annotations += [
                Annotation(
                    key,
                    value=vocab[class_index],
                    span=interval(start, end),
                    occurrence_metadata={"track": MULTIHOT_TRACK},
                )
                for class_index in sorted(only_multihot)
                for start, end in _true_runs(multihot[:, class_index])
            ]
        return annotations


def _read_split_file(path: Path) -> list[tuple[str, str, str]]:
    """Read the ``(path, motif, dominant_l1)`` of each Snapshot that a released split file lists.

    Returns:
        One tuple per Snapshot, in the order of the file.

    Raises:
        TimeNetConnectorError: If the file is missing or does not have the layout of the release.
    """
    try:
        index = json.loads(path.read_text(encoding="utf-8"))
        return [(entry["path"], entry["motif"], entry["dominant_l1"]) for entry in index["traces"]]
    except (OSError, ValueError, KeyError, TypeError) as error:
        raise TimeNetConnectorError(f"{path} is not a KernelSight split file: {error!r}") from error


def read_snapshots(splits_dir: Path) -> list[Snapshot]:
    """Read every Snapshot from the released split files, sorted by path.

    The sort makes the split assignment independent of the order of the files.

    Args:
        splits_dir: The ``splits`` directory of the release.

    Returns:
        One Snapshot per entry of the train, val and test files.

    Raises:
        TimeNetConnectorError: If a split file is missing or malformed, a Snapshot is in more than one
            official partition or in none, or its path or dominant class is not as in the release.
    """
    release_splits: dict[str, list[str]] = defaultdict(list)
    official: dict[str, tuple[str, str, str]] = {}
    for name in RELEASE_SPLITS:
        for path, motif, dominant_l1 in _read_split_file(splits_dir / f"{name}.json"):
            release_splits[path].append(name)
            if name in OFFICIAL_SPLITS:
                if path in official:
                    raise TimeNetConnectorError(f"{path} is in more than one official partition")
                official[path] = (name, motif, dominant_l1)
    unknown = sorted(release_splits.keys() - official.keys())
    if unknown:
        raise TimeNetConnectorError(f"{len(unknown)} snapshots are in no official partition, such as {unknown[0]}")
    return [
        Snapshot(
            path=path,
            motif=motif,
            dominant_l1=dominant_l1,
            official_split=split,
            release_splits=tuple(release_splits[path]),
        )
        for path, (split, motif, dominant_l1) in sorted(official.items())
    ]


def assign_splits(snapshots: list[Snapshot]) -> DatasetSplits[Snapshot]:
    """Partition the Snapshots 3/1/1, grouped by Problem and stratified by the dominant L1 class.

    Returns:
        The train, validation and test Snapshots.
    """
    # NOTE: 3/1/1, not the 8/1/1 of docs/connectors.md. Several classes have only 6 to 8 Problems, and with
    # 8/1/1 loss gets no validation or test Snapshot.
    splitter = StratifiedSplitter(
        train_folds=3,
        validation_folds=1,
        test_folds=1,
        stratify_by=attrgetter("dominant_l1"),
        group_by=attrgetter("problem"),
        seed=42,
    )
    return splitter.split(snapshots)


# A glob's part path is its directory, so both kernel parts resolve to ``kernels/``. The ``*`` matches
# across directories, which selects the input and labels files and skips the fingerprints.
class KernelSightConnector(
    BaseHuggingFaceConnector,
    hf_repo="williamhtan/kernelsight",
    source_revision="d8cd8e804e5b619aadbb082a1fb3a8791b60a680",
    source_files={
        "inputs": "kernels/*tensor_input.npz",
        "labels": "kernels/*labels.npz",
        "splits": "splits/*.json",
    },
):
    """Connector for the KernelSight release on the Hub (``williamhtan/kernelsight``)."""

    def convert(self, files: Mapping[str, Path]) -> TimeFDataset:
        """Create one Record and one ClassificationTask for each Snapshot.

        Args:
            files: The kernel and split directories from :meth:`download`.

        Returns:
            The populated dataset.
        """
        root = files["inputs"].parent
        dataset = TimeFDataset(metadata=self.metadata())
        splits = assign_splits(read_snapshots(files["splits"]))
        for split, snapshots in splits.items():
            for snapshot in snapshots:
                record = dataset.add_record(snapshot.create_record(root))
                dataset.add_task(
                    ClassificationTask(
                        inputs=(record,),
                        targets=snapshot.dominant_l1,
                        label_set=WORKLOAD_L1,
                        split=split,
                    )
                )
        return dataset


CONNECTOR = KernelSightConnector
