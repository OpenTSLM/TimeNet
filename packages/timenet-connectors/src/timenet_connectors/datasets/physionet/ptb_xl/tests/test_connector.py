import csv
from fractions import Fraction
from pathlib import Path

import numpy as np
import pytest
import wfdb

from timenet.dataset import RegularAxis
from timenet.engine import store_dataset
from timenet.reader import TimeFReader
from timenet.registry import DatasetVersion
from timenet_connectors.bases import physionet
from timenet_connectors.bases.physionet import BasePhysioNetConnector
from timenet_connectors.datasets.physionet.ptb_xl.connector import PtbXlConnector, PtbXlSource


_LEADS = ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"]


@pytest.fixture
def source(tmp_path: Path) -> PtbXlSource:
    records = tmp_path / "records500" / "00000"
    records.mkdir(parents=True)
    rng = np.random.default_rng(42)
    rows = []
    for ecg_id, fold in ((1, 9), (2, 10)):
        name = f"{ecg_id:05d}_hr"
        wfdb.wrsamp(
            name,
            fs=500,
            units=["mV"] * len(_LEADS),
            sig_name=list(_LEADS),
            p_signal=rng.standard_normal((50, len(_LEADS))),
            fmt=["16"] * len(_LEADS),
            write_dir=str(records),
        )
        rows.append(
            {
                "ecg_id": ecg_id,
                "patient_id": 100 + ecg_id,
                "age": "nan" if ecg_id == 2 else "41",
                "sex": ecg_id % 2,
                "height": "",
                "weight": 70,
                "report": f"synthetic report {ecg_id}",
                "heart_axis": "MID",
                "device": "synthetic device",
                "scp_codes": "{'NORM': 100.0, 'SR': 0.0}",
                "strat_fold": fold,
                "filename_hr": f"records500/00000/{name}",
            }
        )
    database = tmp_path / "ptbxl_database.csv"
    with database.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return PtbXlSource(root=tmp_path, database_csv=database)


def test_conversion_is_lazy_and_preserves_clinical_metadata(source, monkeypatch):
    def forbidden(*args):
        pytest.fail("conversion decoded a WFDB signal")

    with monkeypatch.context() as guard:
        guard.setattr(physionet, "_read_dat16", forbidden)
        dataset = PtbXlConnector().convert([source])
    assert dataset.metadata.dataset_id == "physionet/ptb-xl"
    assert dataset.metadata.source_revision == "1.0.3"
    assert dataset.tasks == ()
    assert len(dataset.records) == 2
    assert [signal.name for signal in dataset.records[0].signals] == sorted(_LEADS)
    assert all(signal.n_values == 50 for record in dataset.records for signal in record.signals)
    axis = dataset.records[0].signals[0].time_axis
    assert isinstance(axis, RegularAxis)
    assert axis.period_us == Fraction(2000)
    annotations = {record.id: {item.key: item for item in record.annotations} for record in dataset.records}
    assert annotations["ptbxl-1"]["split"].value() == "validation"
    assert annotations["ptbxl-2"]["split"].value() == "test"
    assert annotations["ptbxl-1"]["scp_codes"].value() == ["NORM", "SR"]
    assert annotations["ptbxl-1"]["scp_codes"].metadata["likelihoods"] == {"NORM": 100.0, "SR": 0.0}
    assert annotations["ptbxl-1"]["subject_ids"].value() == ["101"]
    assert annotations["ptbxl-1"]["age"].unit == "year"
    assert "age" not in annotations["ptbxl-2"]
    assert "height" not in annotations["ptbxl-1"]


def test_round_trip_retains_all_leads_and_their_physical_values(source, tmp_path):
    dataset = PtbXlConnector().convert([source])
    version_dir = store_dataset(dataset, tmp_path / "registry")
    with TimeFReader(DatasetVersion.open_local(version_dir)) as reader:
        restored = reader.read()
    assert [record.id for record in restored.records] == ["ptbxl-1", "ptbxl-2"]
    assert restored.tasks == ()
    for record in restored.records:
        base = source.root / "records500" / "00000" / f"{int(record.id.split('-')[1]):05d}_hr"
        expected, _ = wfdb.rdsamp(str(base))
        assert [signal.name for signal in record.signals] == sorted(_LEADS)
        for signal in record.signals:
            index = _LEADS.index(signal.name)
            assert signal.spec.unit_value == "millivolt"
            np.testing.assert_allclose(signal.to_numpy(), expected[:, index].astype("float32"), rtol=1e-5, atol=1e-7)


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
