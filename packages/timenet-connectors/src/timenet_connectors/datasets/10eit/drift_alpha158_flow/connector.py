"""Build 20-session Alpha158 records from an authorized OHLCV and DRIFT task release."""

import asyncio
from collections import defaultdict
import hashlib
from importlib import import_module
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

from timenet.connectors import BaseConnector
from timenet.dataset import OrdinalAxis, Record, Signal, Source, TimeFDataset
from timenet.types import Annotation, AnswerTask, Split, TimeSeriesSpec, ureg
from timenet_connectors.download import Artifact, download_files


alpha158 = import_module(".alpha158", __package__)
ORDER = alpha158.ORDER
compute_alpha158_ohlcv = alpha158.compute_alpha158_ohlcv


CORE = ("KMID", "KLEN", "ROC5", "STD5", "WVMA5")
SPLITS = {"train": Split.TRAIN, "valid": Split.VALIDATION, "test": Split.TEST}
FACTOR = TimeSeriesSpec(
    spec_type="alpha158_factor",
    name="Alpha158 factor",
    dtype="float32",
    unit_value=ureg.dimensionless,
)
RET = TimeSeriesSpec(
    spec_type="daily_return",
    name="Daily close-to-close return",
    dtype="float32",
    unit_value=ureg.dimensionless,
)


class DriftAlpha158Flow(BaseConnector[Path]):
    """Create point-in-time stock records from the original research provider."""

    @staticmethod
    def download(cache_dir):
        """Locate or download the pinned OHLCV and task bundle."""
        location = os.environ.get("DRIFT_ALPHA158_BUNDLE")
        if location:
            path = Path(location)
        else:
            url = os.environ.get("DRIFT_ALPHA158_BUNDLE_URL", "").rstrip("/")
            if not url.startswith("https://"):
                raise FileNotFoundError(
                    "Set DRIFT_ALPHA158_BUNDLE or an authorized HTTPS DRIFT_ALPHA158_BUNDLE_URL. For an anonymous build use 10eit/drift-alpha158-flow-public."
                )
            expected = json.loads(Path(__file__).with_name("source_hashes.json").read_text())
            path = Path(cache_dir) / "pinned-bundle"
            path.mkdir(parents=True, exist_ok=True)
            asyncio.run(
                download_files(
                    [Artifact(url=f"{url}/{name}", dest=path / name, sha256=pin) for name, pin in expected.items()]
                )
            )
        if not all((path / k).is_file() for k in ("observations.json", "ohlcv.npz")):
            raise FileNotFoundError(path)
        return path

    def convert(self, source):
        """Build the dataset after validating both source files."""
        expected = json.loads(Path(__file__).with_name("source_hashes.json").read_text())
        for name, want in expected.items():
            with (source / name).open("rb") as f:
                if hashlib.file_digest(f, "sha256").hexdigest() != want:
                    raise ValueError(f"Unexpected source revision: {name}")
        records = json.loads((source / "observations.json").read_text())
        with np.load(source / "ohlcv.npz", allow_pickle=False) as f:
            dates = f["dates"].tolist()
            tickers = f["tickers"].tolist()
            cols = f["columns"].tolist()
            bars = f["values"]
        return self.build_records(
            records,
            dates,
            lambda name: pd.DataFrame(bars[tickers.index(name)], columns=cols, index=dates),
        )

    def build_records(self, records, dates, read_frame, source_annotations=None):  # ruff: ignore[too-many-locals] -- construct all signals before registering related tasks
        """Assemble dated signals and annotations for both price providers."""
        jobs = defaultdict(list)
        for ix, r in enumerate(records):
            for name in dict.fromkeys([r["target"], *r["senders"], r["other"]]):
                jobs[name].append((ix, name in {r["target"], r["other"]}))
        ts = [{} for _ in records]
        for name, jobs_for_stock in jobs.items():
            ends = [dates.index(records[ix]["date"]) + 1 for ix, _ in jobs_for_stock]
            start = max(0, min(ends) - 81)
            end = max(ends)
            frame = read_frame(name).reindex(dates[start:end])
            factors = compute_alpha158_ohlcv(frame)
            close = frame.close.to_numpy(dtype="float64")
            with np.errstate(divide="ignore", invalid="ignore"):
                returns = np.r_[np.nan, close[1:] / close[:-1] - 1].astype("float32")
            for (ix, full), stop in zip(jobs_for_stock, ends, strict=True):
                sl = slice(stop - start - 20, stop - start)
                names = ORDER if full else CORE
                if list(frame.index[sl]) != records[ix]["dates"]:
                    raise ValueError("Misaligned factor window")
                sig = [
                    Signal.from_values(
                        returns[sl],
                        name="daily_return",
                        spec=RET,
                        time_axis=OrdinalAxis(),
                    )
                ]
                for k in names:
                    sig.append(
                        Signal.from_values(
                            factors[k].to_numpy()[sl],
                            name=k,
                            spec=FACTOR,
                            time_axis=OrdinalAxis(),
                        )
                    )
                ts[ix][name] = Source(name, signals=sig)
        dataset = TimeFDataset(metadata=self.metadata())
        for r, stocks in zip(records, ts, strict=True):
            ann = dict(
                sample_id=r["date"] + "|" + r["target"],
                date=r["date"],
                target_ticker=r["target"],
                sector=r["sector"],
                sender_tickers=json.dumps(r["senders"]),
                pair_ticker=r["other"],
                trading_dates=json.dumps(r["dates"]),
                factor_order=json.dumps(ORDER),
                backbone_id="drift-sp500-alpha158-seed0",
                **(source_annotations or {}),
            )
            sources = [stocks[k] for k in dict.fromkeys([r["target"], *r["senders"], r["other"]])]
            record = Record(
                sources=sources,
                annotations=[Annotation(k, value=v) for k, v in ann.items()],
            )
            dataset.add_record(record)
            for task in r["tasks"]:
                dataset.add_task(
                    AnswerTask(
                        inputs=(record,),
                        prompt=task["prompt"],
                        targets=task["target"],
                        split=SPLITS[r["split"]],
                    )
                )
        return dataset


CONNECTOR = DriftAlpha158Flow
