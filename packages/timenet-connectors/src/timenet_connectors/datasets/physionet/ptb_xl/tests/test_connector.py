import csv
from fractions import Fraction
from pathlib import Path

import numpy as np
import wfdb

from timenet.connectors import BaseConnector
from timenet.dataset import RegularAxis
from timenet.engine import store_dataset
from timenet.reader import TimeFReader
from timenet.registry import DatasetVersion
from timenet_connectors.bases.physionet import BasePhysioNetConnector
from timenet_connectors.datasets.physionet.ptb_xl.connector import (
    PtbXlConnector,
    PtbXlSource,
)


_LEADS = ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"]


def _source(tmp_path: Path) -> PtbXlSource:
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
                "age": 40 + ecg_id,
                "sex": ecg_id % 2,
                "height": 170,
                "weight": 70,
                "report": f"synthetic report {ecg_id}",
                "heart_axis": "MID",
                "device": "synthetic device",
                "scp_codes": "{'NORM': 100.0}",
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


def test_is_taskless_base_connector(tmp_path):
    connector = PtbXlConnector()
    dataset = connector.convert([_source(tmp_path)])

    assert isinstance(connector, BaseConnector)
    assert isinstance(connector, BasePhysioNetConnector)
    assert dataset.metadata.dataset_id == "physionet/ptb-xl"
    assert len(dataset.records) == 2
    assert dataset.tasks == ()


def test_records_keep_all_twelve_high_rate_leads(tmp_path):
    dataset = PtbXlConnector().convert([_source(tmp_path)])

    assert all(len(record.signals) == 12 for record in dataset.records)
    assert all(record.signals[0].n_values == 50 for record in dataset.records)
    assert all(isinstance(record.signals[0].time_axis, RegularAxis) for record in dataset.records)
    assert all(
        record.signals[0].time_axis.period_us == Fraction(2_000)
        for record in dataset.records
        if isinstance(record.signals[0].time_axis, RegularAxis)
    )


def test_records_include_clinical_metadata_and_standard_split(tmp_path):
    dataset = PtbXlConnector().convert([_source(tmp_path)])
    annotations = {
        record.id: {annotation.key: annotation.value for annotation in record.annotations} for record in dataset.records
    }

    assert annotations["ptbxl-1"]["split"] == "validation"
    assert annotations["ptbxl-2"]["split"] == "test"
    assert annotations["ptbxl-1"]["scp_codes"] == ["NORM"]
    assert dataset.records[0].subject_ids == ("101",)


def test_round_trip_preserves_records_without_tasks(tmp_path):
    dataset = PtbXlConnector().convert([_source(tmp_path / "source")])
    dataset.derive_schema()
    version_dir = store_dataset(dataset, tmp_path / "registry")

    with TimeFReader(DatasetVersion.open_local(version_dir)) as reader:
        restored = reader.read()

    assert {record.id for record in restored.records} == {"ptbxl-1", "ptbxl-2"}
    assert restored.tasks == ()
    assert len(restored.records[0].signals[0].to_numpy()) == 50
