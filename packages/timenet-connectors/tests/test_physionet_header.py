from pathlib import Path

import pytest

from timenet.errors import TimeNetConnectorError
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
    with pytest.raises(TimeNetConnectorError, match="declares 12 signals"):
        BasePhysioNetConnector._read_header(rec)


def test_read_header_rejects_a_signal_line_without_a_signal_name(tmp_path):
    rec = _write_hea(tmp_path, "rec 1 250 100\nrec.dat 16 200(0)/mV 16 0 0 0\n")
    with pytest.raises(TimeNetConnectorError, match="names no signal"):
        BasePhysioNetConnector._read_header(rec)


def test_read_header_rejects_an_unparseable_gain(tmp_path):
    rec = _write_hea(tmp_path, "rec 1 250 100\nrec.dat 16 bad/mV 16 0 0 0 0 II\n")
    with pytest.raises(TimeNetConnectorError, match="cannot parse the gain"):
        BasePhysioNetConnector._read_header(rec)


def test_read_header_uses_the_direct_path_for_plain_single_dat_16(tmp_path):
    (tmp_path / "plain.hea").write_text(
        "plain 2 500 100\nplain.dat 16 1000(0)/mV 16 0 0 0 0 I\nplain.dat 16 1000(0)/mV 16 0 0 0 0 II\n"
    )
    assert BasePhysioNetConnector._read_header(tmp_path / "plain").direct_read is True


def test_read_header_falls_back_when_the_record_is_not_plain_16(tmp_path):
    # 16x2 (two WFDB samples per frame) breaks the (-1, n_sig) reshape, so the record must route to wfdb; a
    # multi-word signal description is kept whole rather than truncated to its last token.
    (tmp_path / "framed.hea").write_text("framed 1 500 100\nframed.dat 16x2 1000(0)/mV 16 0 0 0 0 my lead II\n")
    header = BasePhysioNetConnector._read_header(tmp_path / "framed")
    assert header.direct_read is False
    assert header.sig_name == ["my lead II"]


def test_read_header_falls_back_when_signals_span_multiple_dat_files(tmp_path):
    (tmp_path / "multi.hea").write_text(
        "multi 2 500 100\na.dat 16 1000(0)/mV 16 0 0 0 0 I\nb.dat 16 1000(0)/mV 16 0 0 0 0 II\n"
    )
    assert BasePhysioNetConnector._read_header(tmp_path / "multi").direct_read is False


@pytest.mark.parametrize(
    "body",
    ["", "r x 500 10\n", "r 1 500 10\nr.dat 16\n"],
    ids=["empty", "record-count-not-a-number", "short-signal-line"],
)
def test_read_header_raises_a_connector_error_for_a_truncated_header(tmp_path, body):
    with pytest.raises(TimeNetConnectorError, match=r"rec\.hea: malformed header"):
        BasePhysioNetConnector._read_header(_write_hea(tmp_path, body))


def test_read_header_rejects_text_that_is_not_utf8(tmp_path):
    (tmp_path / "rec.hea").write_bytes(b"rec 1 250 100\nrec.dat 16 200(0)/mV 16 0 0 0 0 \xff\n")
    with pytest.raises(TimeNetConnectorError, match=r"rec\.hea: malformed header"):
        BasePhysioNetConnector._read_header(tmp_path / "rec")
