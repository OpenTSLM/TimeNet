"""Immutable public OHLCV signals with date-aligned historical DRIFT annotations."""

import asyncio
import gzip
import hashlib
from importlib import import_module
from io import BytesIO
import json
import os
from pathlib import Path
import zipfile

import numpy as np
import pandas as pd

from timenet_connectors.download import Artifact, download_files


DriftAlpha158Flow = import_module("timenet_connectors.datasets.10eit.drift_alpha158_flow.connector").DriftAlpha158Flow

URL = "https://www.kaggle.com/api/v1/datasets/download/joebeachcapital/s-and-p500-index-stocks-daily-updated?dataset_version_number=613"
SHA = "1400038a83f5ba2702adf059829d4e1d2be0f977170fea5455e10bbd22b2c9a2"
NAME = "sp500-kaggle-v613.zip"


def public_frame(archive, ticker):
    """Read split- and dividend-adjusted bars in the provider's date order."""
    with archive.open(f"sp500_cache/{ticker}.csv") as stream:
        raw = pd.read_csv(BytesIO(stream.read()))
    raw = raw.drop_duplicates("Date", keep="last").set_index("Date").sort_index()
    ratio = raw["Adj Close"] / raw["Close"]
    out = raw[["Open", "High", "Low", "Close"]].mul(ratio, axis=0)
    out.columns = ["open", "high", "low", "close"]
    out["volume"] = raw["Volume"] / ratio
    return out.replace([np.inf, -np.inf], np.nan).astype("float32")


class PublicAlpha158Flow(DriftAlpha158Flow):
    """Reconstruct the price-matched public subset from its pinned archive."""

    @staticmethod
    def download(cache_dir):
        """Acquire and verify the same price snapshot in online and offline modes."""
        local = os.environ.get("DRIFT_ALPHA158_BUNDLE")
        path = Path(local) / NAME if local else Path(cache_dir) / NAME
        if not local:
            path.parent.mkdir(parents=True, exist_ok=True)
            asyncio.run(download_files([Artifact(url=URL, dest=path, sha256=SHA)]))
        with path.open("rb") as stream:
            if hashlib.file_digest(stream, "sha256").hexdigest() != SHA:
                raise ValueError("Unexpected public OHLCV revision")
        return path

    def convert(self, source):
        """Compute factors and attach the released DRIFT task annotations."""
        with gzip.open(Path(__file__).with_name("observation_index.json.gz"), "rt") as f:
            records = json.load(f)
        with zipfile.ZipFile(source) as archive:
            calendar = public_frame(archive, "AAPL").index.tolist()
            return self.build_records(
                records,
                calendar,
                lambda name: public_frame(archive, name),
                {"price_source": "Kaggle joebeachcapital v613", "label_source": "frozen DRIFT history-v3"},
            )


CONNECTOR = PublicAlpha158Flow
