"""Check the public column contract and prefix invariance on permitted source data."""

from importlib import import_module
import os
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


module = import_module("timenet_connectors.datasets.10eit.drift_alpha158_flow.alpha158")


def test_column_contract():
    assert len(module.ORDER) == len(set(module.ORDER)) == 157
    assert module.ORDER[:3] == ["KMID", "KLEN", "KMID2"]
    assert "VWAP0" not in module.ORDER


def test_actual_ohlcv_prefix():
    path = os.environ.get("DRIFT_ALPHA158_BUNDLE")
    if not path:
        pytest.skip("requires authorized OHLCV source")
    with np.load(Path(path) / "ohlcv.npz", allow_pickle=False) as f:
        calendar = f["dates"].tolist()
        i = calendar.index("2023-01-03")
        frame = pd.DataFrame(
            f["values"][f["tickers"].tolist().index("AAPL"), i - 90 : i + 10],
            columns=f["columns"].tolist(),
            index=calendar[i - 90 : i + 10],
        )
    original = module.compute_alpha158_ohlcv(frame)
    cut = module.compute_alpha158_ohlcv(frame.iloc[:-9])
    pd.testing.assert_frame_equal(cut, original.iloc[:-9])
