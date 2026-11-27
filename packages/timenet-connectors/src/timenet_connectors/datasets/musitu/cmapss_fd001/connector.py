"""NASA C-MAPSS FD001 TimeNet connector (NASA PCoE simulated engine data).

The original archive is obtained from NASA's Zenodo-hosted record at build
time and verified against its published MD5. It is not committed to TimeNet.

Source: https://zenodo.org/records/15346912
Reference: A. Saxena, K. Goebel, D. Simon, N. Eklund (PHM 2008),
"Damage Propagation Modeling for Aircraft Engine Run-to-Failure Simulation".

The dataset record does not specify a license. We label it "other" rather than
assert rights we cannot verify; review NASA's general data policy and source
terms before distributing any derived data artifact.
"""
from __future__ import annotations

import hashlib
import os
from pathlib import Path
from urllib.request import Request, urlopen
from zipfile import ZipFile

import numpy as np
import pandas as pd

from timenet.connectors import BaseConnector
from timenet.dataset import OrdinalAxis, Record, Signal, Source, TimeFDataset
from timenet.types import Annotation, ScalarPredictionTask, Split, TimeSeriesSpec, ureg

ARCHIVE_URL = "https://zenodo.org/records/15346912/files/CMAPSSData.zip?download=1"
ARCHIVE_MD5 = "79a22f36e80606c69d0e9e4da5bb2b7a"
COLS = ["unit", "cycle"] + [f"setting_{i}" for i in range(1, 4)] + [
    f"sensor_{i}" for i in range(1, 22)
]
SPEC = TimeSeriesSpec(
    spec_type="engine_telemetry", name="Engine sensor observation",
    unit_value=ureg.dimensionless,
)
WINDOW = 30
STRIDE = 15


def _dataset_files(root: Path) -> list[Path]:
    return [root / n for n in ("train_FD001.txt", "test_FD001.txt", "RUL_FD001.txt")]


def _read(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path, sep=r"\s+", header=None, engine="python")
    frame = frame.dropna(axis=1, how="all")
    if frame.shape[1] != len(COLS):
        raise ValueError(f"Expected 26 columns at {path}; got {frame.shape[1]}")
    frame.columns = COLS
    numerical = frame.to_numpy(dtype=float)
    if not np.isfinite(numerical).all():
        raise ValueError(f"Non-finite value in {path}")
    ids_and_cycles = frame[["unit", "cycle"]].to_numpy(dtype=float)
    if np.any(ids_and_cycles < 1) or not np.array_equal(ids_and_cycles, np.floor(ids_and_cycles)):
        raise ValueError(f"Engine and cycle IDs must be positive integers in {path}")
    frame[["unit", "cycle"]] = frame[["unit", "cycle"]].astype(int)
    if frame.duplicated(["unit", "cycle"]).any():
        raise ValueError(f"Duplicate engine-cycle rows in {path}")
    for unit, group in frame.groupby("unit", sort=True):
        expected = np.arange(1, len(group) + 1)
        if not np.array_equal(group["cycle"].to_numpy(), expected):
            raise ValueError(f"Noncontiguous or out-of-order cycles for engine {unit} in {path}")
    return frame


class CmapssFd001Connector(BaseConnector[Path]):
    """Build a leak-aware RUL prediction dataset from NASA FD001 trajectories."""

    def download(self, cache_dir: Path) -> Path:
        """Use local source files when provided; else fetch the checksummed NASA archive."""
        root = Path(os.environ.get("CMAPSS_DATA_DIR", str(cache_dir))).resolve()
        root.mkdir(parents=True, exist_ok=True)
        if all(path.is_file() for path in _dataset_files(root)):
            return root

        archive = root / "CMAPSSData.zip"
        if not archive.exists() or hashlib.md5(archive.read_bytes()).hexdigest() != ARCHIVE_MD5:
            request = Request(ARCHIVE_URL, headers={"User-Agent": "TimeNet-cmapss-fd001/0.1"})
            parts, received = [], 0
            with urlopen(request, timeout=90) as stream:
                while part := stream.read(1024 * 1024):
                    received += len(part)
                    if received > 100 * 1024 * 1024:
                        raise ValueError("Unexpectedly large NASA dataset download")
                    parts.append(part)
            payload = b"".join(parts)
            if hashlib.md5(payload).hexdigest() != ARCHIVE_MD5:
                raise ValueError("NASA data archive MD5 mismatch")
            archive.write_bytes(payload)
        with ZipFile(archive) as z:
            required = [path.name for path in _dataset_files(root)]
            members = {name: [] for name in required}
            for item in z.namelist():
                name = Path(item).name
                if name in members:
                    members[name].append(item)
            for path in _dataset_files(root):
                if len(members[path.name]) != 1:
                    raise ValueError(f"NASA archive must contain exactly one {path.name}")
                path.write_bytes(z.read(members[path.name][0]))
        return root

    def convert(self, root: Path) -> TimeFDataset:
        """Create one 30-cycle multivariate record per sampled training window."""
        train, test = _read(root / "train_FD001.txt"), _read(root / "test_FD001.txt")
        offset_df = pd.read_csv(root / "RUL_FD001.txt", sep=r"\s+", header=None)
        offset_df = offset_df.dropna(axis=1, how="all")
        test_ids = sorted(test.unit.unique())
        if len(test_ids) != len(offset_df) or offset_df.shape[1] != 1:
            raise ValueError("NASA official test RUL labels do not match engine count")
        offsets = pd.to_numeric(offset_df.iloc[:, 0], errors="raise").to_numpy(dtype=float)
        if not np.isfinite(offsets).all() or (offsets < 0).any() or not np.array_equal(offsets, np.floor(offsets)):
            raise ValueError("NASA official test RUL labels must be nonnegative integers")
        test_offsets = dict(zip(test_ids, offsets.astype(int).tolist(), strict=True))
        dataset = TimeFDataset(metadata=self.metadata())
        train_ids = sorted(train.unit.unique())
        # Deterministic by ENGINE, not by window; no shared engine in train and validation.
        val_ids = set(train_ids[::5])

        for cohort, source in (("development", train), ("official_test", test)):
            for unit, group in source.groupby("unit", sort=True):
                group = group.sort_values("cycle").reset_index(drop=True)
                if len(group) < WINDOW:
                    continue
                if cohort == "official_test":
                    split = Split.TEST
                    ends = [len(group)-1]
                else:
                    split = Split.VALIDATION if unit in val_ids else Split.TRAIN
                    ends = range(WINDOW-1, len(group), STRIDE)
                for end in ends:
                    observed = group.iloc[end-WINDOW+1:end+1]
                    current_cycle = int(group.cycle.iloc[end])
                    target = (
                        int(group.cycle.iloc[-1]) - current_cycle
                        if cohort == "development" else test_offsets[unit]
                    )
                    signals = [
                        Signal.from_values(
                            observed[col].to_numpy(dtype=np.float32),
                            spec=SPEC, name=col, time_axis=OrdinalAxis(),
                        ) for col in COLS[2:]
                    ]
                    record = Record(
                        sources=[Source("NASA simulated engine telemetry", signals=signals)],
                        annotations=[
                            Annotation("engine_id", value=int(unit)),
                            Annotation("cohort", value=cohort),
                            Annotation("observation_cycle", value=current_cycle),
                        ],
                    )
                    dataset.add_record(record)
                    dataset.add_task(
                        ScalarPredictionTask(
                            inputs=[record],
                            targets=float(target),
                            unit="cycle",
                            target_name="remaining_useful_life_cycles",
                            split=split,
                        )
                    )
        return dataset


CONNECTOR = CmapssFd001Connector
