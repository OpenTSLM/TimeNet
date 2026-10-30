import csv
from pathlib import Path
import re
import sys

import numpy as np
import pytest
import wfdb

from timenet.errors import TimeFFormatError
from timenet.reader import LoadPlan, TimeFReader
from timenet.types import Annotation
from timenet.types.text import read_value
from timenet.writer import TimeFWriter
from timenet_connectors.datasets.physionet.ptb_xl.connector import PtbXlConnector, PtbXlSource


_LEADS = ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"]
# A synthetic ptbxl_database.csv row. The static-noise and electrode cells hold only whitespace, so they
# state no finding.
_ROW = {
    "ecg_id": "1",
    "filename_hr": "record",
    "age": "56",
    "weight": "63",
    "patient_id": "15709",
    "sex": "1",
    "scp_codes": "{'NORM': 100.0}",
    "strat_fold": "3",
    "report": "Synthetic report",
    "recording_date": "1984-11-09 09:17:34",
    "baseline_drift": " , II,III,AVF",
    "static_noise": "  ",
    "burst_noise": "alles",
    "electrodes_problems": "",
    "extra_beats": "1ES",
    "pacemaker": "ja, pacemaker",
}
# The descriptions of ECG-QA-CoT 1.1.0, which stores sex and age in the same form.
_SEX_DESCRIPTION = "The patient's sex, as PTB-XL states it."
_AGE_DESCRIPTION = (
    "The patient's age in whole years, as PTB-XL states it. PTB-XL states 300 for every patient older than 89."
)


def _write_source(tmp_path: Path, **cells: str) -> PtbXlSource:
    """Write one format-16 recording and its ptbxl_database.csv row, with ``cells`` in place of the row's cells."""
    values = np.random.default_rng(4).normal(size=(20, 12))
    wfdb.wrsamp(
        "record", fs=500, units=["mV"] * 12, sig_name=_LEADS, p_signal=values, fmt=["16"] * 12, write_dir=str(tmp_path)
    )
    database = tmp_path / "ptbxl_database.csv"
    with database.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(_ROW))
        writer.writeheader()
        writer.writerow(_ROW | cells)
    return PtbXlSource(tmp_path, database)


def _annotations(tmp_path: Path, **cells: str) -> dict[str, Annotation]:
    """Convert the fixture row with ``cells`` in place of its cells, and return its annotations by key."""
    dataset = PtbXlConnector().convert([_write_source(tmp_path, **cells)])
    return {ann.key: ann for ann in dataset.records[0].annotations}


def test_taskless_clinical_records_round_trip(tmp_path):
    dataset = PtbXlConnector().convert([_write_source(tmp_path)])
    assert dataset.tasks == ()
    assert len(dataset.records[0].signals) == 12
    # Children find a recording by its upstream ecg_id, which the record's metadata carries.
    assert dataset.records[0].metadata == {"ecg_id": 1}
    assert {ann.key: ann.value for ann in dataset.records[0].annotations}["age"] == 56
    with TimeFWriter(tmp_path / "out", dataset) as writer:
        writer.write()
    with TimeFReader.open(tmp_path / "out/physionet/ptb-xl/1.0.0") as reader:
        restored = reader.read(load=LoadPlan.ALL_EAGER)
    assert restored.records[0].metadata == {"ecg_id": 1}
    # Sex and age keep their form and their descriptions in the stored release, which children import.
    by_key = {ann.key: ann for ann in restored.records[0].annotations}
    assert (read_value(by_key["sex"].value), by_key["sex"].description) == ("female", _SEX_DESCRIPTION)
    assert type(by_key["age"].value) is int
    assert (by_key["age"].value, by_key["age"].unit, by_key["age"].description) == (56, "year", _AGE_DESCRIPTION)
    reference, _ = wfdb.rdsamp(str(tmp_path / "record"), channels=[0])
    np.testing.assert_allclose(restored.records[0].signals[0].to_numpy(), reference[:, 0], rtol=1e-5, atol=1e-6)


@pytest.mark.parametrize(("code", "sex"), [("0", "male"), ("1", "female")])
def test_sex_is_male_or_female(tmp_path, code, sex):
    annotation = _annotations(tmp_path, sex=code)["sex"]
    assert (read_value(annotation.value), annotation.description) == (sex, _SEX_DESCRIPTION)


def test_a_sex_code_other_than_0_or_1_raises(tmp_path):
    with pytest.raises(TimeFFormatError, match=re.escape("ecg_id 1: sex '2' is not 0 or 1")):
        _annotations(tmp_path, sex="2")


@pytest.mark.parametrize(("cell", "years"), [("56", 56), ("56.0", 56), ("300.0", 300)])
def test_age_is_an_int_in_years(tmp_path, cell, years):
    # PTB-XL states 300 for every patient older than 89. The description says so for every age.
    annotation = _annotations(tmp_path, age=cell)["age"]
    assert type(annotation.value) is int
    assert (annotation.value, annotation.unit, annotation.description) == (years, "year", _AGE_DESCRIPTION)


@pytest.mark.parametrize("cell", ["56.5", "unknown"])
def test_an_age_that_is_not_a_whole_number_raises(tmp_path, cell):
    with pytest.raises(TimeFFormatError, match=re.escape(f"ecg_id 1: age {cell!r} is not a whole number")):
        _annotations(tmp_path, age=cell)


@pytest.mark.parametrize("column", ["sex", "age"])
def test_an_empty_sex_or_age_cell_adds_no_annotation(tmp_path, column):
    annotations = _annotations(tmp_path, **{column: ""})
    assert column not in annotations
    # The other of the two stays.
    assert {"sex", "age"} - {column} <= annotations.keys()


def test_recording_date_and_findings_are_annotated_when_their_cell_has_a_value(tmp_path):
    dataset = PtbXlConnector().convert([_write_source(tmp_path)])
    by_key = {ann.key: ann.value for ann in dataset.records[0].annotations}
    # Each value loses its surrounding whitespace. Cells with only whitespace add no annotation.
    expected = {
        "recording_date": "1984-11-09 09:17:34",
        "baseline_drift": ", II,III,AVF",
        "static_noise": None,
        "burst_noise": "alles",
        "electrodes_problems": None,
        "extra_beats": "1ES",
        "pacemaker": "ja, pacemaker",
    }
    assert {key: by_key.get(key) for key in expected} == expected
    # The other clinical annotations stay as they are.
    assert {key: by_key[key] for key in ("weight", "report", "strat_fold", "split", "subject_ids")} == {
        "weight": 63.0,
        "report": "Synthetic report",
        "strat_fold": 3,
        "split": "train",
        "subject_ids": ["15709"],
    }


def test_convert_needs_no_wfdb_for_format16(tmp_path, monkeypatch):
    # Format-16 records are read straight from the .hea/.dat, so convert and lead loading work with no
    # wfdb installed. wfdb is only a fallback for other signal formats.
    source = _write_source(tmp_path)
    monkeypatch.setitem(sys.modules, "wfdb", None)  # `import wfdb` -> ImportError if anything reaches for it
    dataset = PtbXlConnector().convert([source])
    assert dataset.records
    assert len(dataset.records[0].signals[0].to_numpy())  # runs the direct loader, no wfdb
