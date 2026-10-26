import csv

import numpy as np
import wfdb

from timenet.reader import LoadPlan, TimeFReader
from timenet.registry import open_local_version
from timenet.writer import TimeFWriter
from timenet_connectors.datasets.physionet.ptb_xl.connector import PtbXlConnector, PtbXlSource


def test_taskless_clinical_records_round_trip(tmp_path):
    leads = ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"]
    values = np.random.default_rng(4).normal(size=(20, 12))
    wfdb.wrsamp(
        "record", fs=500, units=["mV"] * 12, sig_name=leads, p_signal=values, fmt=["16"] * 12, write_dir=str(tmp_path)
    )
    row = {
        "ecg_id": "1",
        "filename_hr": "record",
        "age": "56",
        "weight": "63",
        "patient_id": "15709",
        "sex": "1",
        "scp_codes": "{'NORM': 100.0}",
        "strat_fold": "3",
        "report": "Synthetic report",
    }
    database = tmp_path / "ptbxl_database.csv"
    with database.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(row))
        writer.writeheader()
        writer.writerow(row)
    dataset = PtbXlConnector().convert([PtbXlSource(tmp_path, database)])
    assert dataset.tasks == ()
    assert len(dataset.records[0].signals) == 12
    assert {ann.key: ann.value for ann in dataset.records[0].annotations}["age"] == 56
    with TimeFWriter(tmp_path / "out", dataset) as writer:
        writer.write()
    with TimeFReader(open_local_version(tmp_path / "out/physionet/ptb-xl/1.0.0")) as reader:
        restored = reader.read(load=LoadPlan.ALL_EAGER)
    reference, _ = wfdb.rdsamp(str(tmp_path / "record"), channels=[0])
    np.testing.assert_allclose(restored.records[0].signals[0].to_numpy(), reference[:, 0], rtol=1e-5, atol=1e-6)
