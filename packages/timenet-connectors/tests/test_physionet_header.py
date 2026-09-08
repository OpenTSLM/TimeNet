from pathlib import Path

import pytest

from timenet.errors import TimeFFormatError
from timenet_connectors.bases.physionet import BasePhysioNetConnector


def _write_hea(tmp_path: Path, body: str) -> Path:
    (tmp_path / "rec.hea").write_text(body, encoding="utf-8")
    return tmp_path / "rec"


def test_read_header_parses_a_valid_signal_line(tmp_path):
    rec = _write_hea(tmp_path, "rec 1 250 100\nrec.dat 16 200(0)/mV 16 0 0 0 0 II\n")
    header = BasePhysioNetConnector._read_header(rec)
    assert header.sig_name == ["II"]
    assert header.gains == (200.0,)
    assert header.baselines == (0,)


def test_read_header_defaults_an_unspecified_gain_to_the_wfdb_200(tmp_path):
    rec = _write_hea(tmp_path, "rec 1 250 100\nrec.dat 16 0 16 0 0 0 0 II\n")
    assert BasePhysioNetConnector._read_header(rec).gains == (200.0,)


def test_read_header_rejects_fewer_signal_lines_than_declared(tmp_path):
    rec = _write_hea(tmp_path, "rec 12 250 100\nrec.dat 16 200(0)/mV 16 0 0 0 0 II\n")
    with pytest.raises(TimeFFormatError, match="declares 12 signals"):
        BasePhysioNetConnector._read_header(rec)


def test_read_header_rejects_a_signal_line_without_a_signal_name(tmp_path):
    rec = _write_hea(tmp_path, "rec 1 250 100\nrec.dat 16 200(0)/mV 16 0 0 0\n")
    with pytest.raises(TimeFFormatError, match="names no signal"):
        BasePhysioNetConnector._read_header(rec)


def test_read_header_rejects_an_unparseable_gain(tmp_path):
    rec = _write_hea(tmp_path, "rec 1 250 100\nrec.dat 16 bad/mV 16 0 0 0 0 II\n")
    with pytest.raises(TimeFFormatError, match="cannot parse the gain"):
        BasePhysioNetConnector._read_header(rec)
