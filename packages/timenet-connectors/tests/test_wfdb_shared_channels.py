from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pyarrow as pa
import pytest
import wfdb

from timenet.errors import TimeNetConnectorError
from timenet_connectors.bases import physionet
from timenet_connectors.datasets.physionet.ptb_xl.connector import PtbXlConnector


def _record(tmp_path: Path, *, fmt: str = "16") -> tuple[Path, np.ndarray]:
    values = np.array([[0.5, -1.0], [1.25, 2.0], [-0.5, 0.0]], dtype=np.float64)
    wfdb.wrsamp(
        "record",
        fs=500,
        units=["mV", "mV"],
        sig_name=["I", "II"],
        p_signal=values,
        fmt=[fmt, fmt],
        adc_gain=[100, 200],
        baseline=[10, -20],
        write_dir=str(tmp_path),
    )
    return tmp_path / "record", values


@pytest.mark.parametrize("fmt", ["16", "32"])
def test_all_leads_share_one_decode_and_match_wfdb(tmp_path, monkeypatch, fmt):
    base, _ = _record(tmp_path, fmt=fmt)
    expected, _ = wfdb.rdsamp(str(base))
    connector = PtbXlConnector()
    header = connector._read_header(base)
    calls = []
    if header.direct_read:
        read = physionet._read_dat16

        def counted(*args):
            calls.append(args)
            return read(*args)

        monkeypatch.setattr(physionet, "_read_dat16", counted)
    else:
        read = wfdb.rdsamp

        def counted(*args, **kwargs):
            calls.append((args, kwargs))
            return read(*args, **kwargs)

        # Other tests reload optional modules. Patch the import boundary rather than a module
        # object that may no longer be the one imported by the connector.
        monkeypatch.setattr(connector, "_wfdb", lambda: SimpleNamespace(rdsamp=counted))

    factory = connector._lead_loaders()
    first, second = (factory(base, header, index) for index in range(2))
    assert calls == []
    for index, load in ((1, second), (0, first), (1, second)):
        actual = load()
        assert actual.type == pa.float32()
        np.testing.assert_array_equal(actual.to_numpy(), expected[:, index].astype("float32"))
    assert len(calls) == 1


def test_a_new_conversion_does_not_reuse_the_previous_conversions_samples(tmp_path):
    base, _ = _record(tmp_path)
    connector = PtbXlConnector()
    header = connector._read_header(base)
    old_factory = connector._lead_loaders()
    before = old_factory(base, header, 0)().to_numpy().copy()
    dat = tmp_path / "record.dat"
    samples = np.fromfile(dat, dtype="<i2").reshape(-1, 2)
    samples[:, 0] += 100
    samples.tofile(dat)
    after = connector._lead_loaders()(base, header, 0)().to_numpy()
    np.testing.assert_array_equal(after, before + 1)


@pytest.mark.parametrize("fmt", ["16", "32"])
def test_rejects_a_decoded_shape_that_disagrees_with_the_header(tmp_path, monkeypatch, fmt):
    base, _ = _record(tmp_path, fmt=fmt)
    connector = PtbXlConnector()
    header = connector._read_header(base)
    if header.direct_read:
        # A whole frame is still missing; the direct reader's divisibility check alone accepts it.
        dat = tmp_path / "record.dat"
        dat.write_bytes(dat.read_bytes()[:-4])
    else:
        monkeypatch.setattr(connector, "_wfdb", lambda: SimpleNamespace(rdsamp=lambda path: (np.zeros((2, 2)), {})))
    with pytest.raises(TimeNetConnectorError, match=r"record: expected samples with shape \(3, 2\)"):
        connector._lead_loaders()(base, header, 0)()


def test_reports_the_record_when_fallback_reading_fails(tmp_path, monkeypatch):
    base, _ = _record(tmp_path, fmt="32")
    connector = PtbXlConnector()
    header = connector._read_header(base)

    def broken(path):
        raise ValueError("invalid source samples")

    monkeypatch.setattr(connector, "_wfdb", lambda: SimpleNamespace(rdsamp=broken))
    with pytest.raises(TimeNetConnectorError, match="record: cannot read WFDB samples: invalid source samples"):
        connector._lead_loaders()(base, header, 0)()
