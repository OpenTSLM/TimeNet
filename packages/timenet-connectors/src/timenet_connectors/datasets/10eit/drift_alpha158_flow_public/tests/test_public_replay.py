"""Check released observation keys and public-provider point-in-time factor construction."""

from collections import Counter
import gzip
from importlib import import_module
import json
import os
from pathlib import Path
import zipfile

import pandas as pd
import pytest


public = import_module("timenet_connectors.datasets.10eit.drift_alpha158_flow_public.connector")
factors = import_module("timenet_connectors.datasets.10eit.drift_alpha158_flow.alpha158")


def test_release_index():
    assert public.__file__ is not None
    p = Path(public.__file__).with_name("observation_index.json.gz")
    with gzip.open(p, "rt") as f:
        records = json.load(f)
    assert len({(r["date"], r["target"]) for r in records}) == 98
    assert len({r["target"] for r in records}) == 30
    assert Counter(r["split"] for r in records) == {
        "train": 39,
        "valid": 17,
        "test": 42,
    }
    for r in records:
        assert r["date"] == r["dates"][-1] and len(r["dates"]) == 20
        assert len(r["senders"]) == 5 and len(r["tasks"]) == 4


def test_factor_prefix_on_actual_public_bars():
    bundle = os.environ.get("DRIFT_ALPHA158_BUNDLE")
    if not bundle:
        pytest.skip("Requires the pinned public OHLCV ZIP")
    with zipfile.ZipFile(Path(bundle) / public.NAME) as archive:
        frame = public.public_frame(archive, "AAPL").loc["2023-01-01":"2023-06-30"]
    end = "2023-06-01"
    expected = factors.compute_alpha158_ohlcv(frame).loc[:end]
    actual = factors.compute_alpha158_ohlcv(frame.loc[:end])
    pd.testing.assert_frame_equal(expected, actual)
