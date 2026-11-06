from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any, cast

import numpy as np
import pytest

from timenet.errors import TimeNetConnectorError
from timenet_connectors.bases.physionet import BasePhysioNetConnector, WfdbHeader, _read_dat16


def header(*, direct=True, length=3):
    return WfdbHeader(100, length, ["I", "II"], ("record.dat",) * 2, ("16",) * 2, (3.0, 7.0), (1, -2), direct)


@dataclass
class FallbackReader:
    matrix: np.ndarray

    def _wfdb(self):
        return SimpleNamespace(rdsamp=lambda *args, **kwargs: (self.matrix, {}))


def test_direct_calibration_keeps_float32_arithmetic(tmp_path):
    raw = np.array([[1, -2], [200, 900], [-32768, 32767]], dtype="<i2")
    raw.tofile(tmp_path / "record.dat")
    contract = header()
    for index in range(2):
        load = BasePhysioNetConnector._lead_loaders(cast(Any, FallbackReader(raw)))(
            tmp_path / "record", contract, index
        )
        expected = (raw[:, index].astype("float32") - contract.baselines[index]) / contract.gains[index]
        np.testing.assert_array_equal(load().to_numpy(), expected)


def test_fallback_preserves_rounding_and_validates_before_cast(tmp_path):
    matrix = np.array([[0.1, 0.1], [0.2, 0.2], [0.3, 0.3]], dtype=np.float64)
    load = BasePhysioNetConnector._lead_loaders(cast(Any, FallbackReader(matrix)))(
        tmp_path / "record", header(direct=False), 1
    )
    np.testing.assert_array_equal(load().to_numpy(), matrix[:, 0].astype("float32"))
    matrix[1, 1] = np.inf
    with pytest.raises(TimeNetConnectorError, match=r"record: II\[1\]: expected a finite number"):
        BasePhysioNetConnector._lead_loaders(cast(Any, FallbackReader(matrix)))(
            tmp_path / "record", header(direct=False), 1
        )()


@pytest.mark.parametrize("matrix", [np.zeros((2, 2)), np.zeros((3, 1)), np.zeros(3)])
def test_fallback_checks_declared_layout(tmp_path, matrix):
    load = BasePhysioNetConnector._lead_loaders(cast(Any, FallbackReader(matrix)))(
        tmp_path / "record", header(direct=False), 0
    )
    with pytest.raises(TimeNetConnectorError, match="record: waveforms:"):
        load()


def test_direct_rejects_incomplete_frame_and_length(tmp_path):
    path = tmp_path / "record.dat"
    np.zeros(3, dtype="<i2").tofile(path)
    with pytest.raises(TimeNetConnectorError, match=r"record\.dat: 3 samples"):
        _read_dat16(str(path), 2)
    np.zeros(4, dtype="<i2").tofile(path)
    load = BasePhysioNetConnector._lead_loaders(cast(Any, FallbackReader(np.zeros(0))))(
        tmp_path / "record", header(), 0
    )
    with pytest.raises(TimeNetConnectorError, match="expected 3 samples, got 2"):
        load()
