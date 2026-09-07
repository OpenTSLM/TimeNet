"""What the card states about each folder, and the specs those folders map onto."""

from timenet.types import ureg
from timenet_connectors.datasets.leochen085.slip_eval import folders, specs


def test_every_folder_has_a_spec() -> None:
    assert {f.name for f in folders.FOLDERS} == set(specs.BY_FOLDER)


def test_the_ppg_folders_are_folders() -> None:
    assert set(folders.PPG_FOLDERS) <= {f.name for f in folders.FOLDERS}


def test_the_signal_names_match_the_counts_the_card_states() -> None:
    stated = {
        "AsphaltObstacles": 1,
        "Beijing_AQI": 7,
        "PPG_CVA": 1,
        "PPG_DM": 1,
        "PPG_HTN": 1,
        "ptbxl": 12,
        "sleepEDF": 2,
        "studentlife": 10,
        "uci_har": 6,
        "wesad": 13,
        "wisdm": 3,
    }
    assert {f.name: len(f.signals) for f in folders.FOLDERS} == stated


def test_every_rate_is_exact() -> None:
    # A float rate of 1/3600 gives a period whose numerator does not fit int64, and the axis
    # refuses it. Every rate is a Fraction so the period stays exact.
    for folder in folders.FOLDERS:
        assert folder.rate_hz.denominator >= 1
        assert (1_000_000 / folder.rate_hz).numerator < 2**63


def test_the_four_folders_that_kept_a_unit_are_not_dimensionless() -> None:
    for name in ("sleepEDF", "wisdm", "uci_har", "AsphaltObstacles"):
        assert specs.BY_FOLDER[name].unit_value != ureg.dimensionless


def test_the_rescaled_folders_are_dimensionless() -> None:
    for name in ("ptbxl", "wesad", "Beijing_AQI", "studentlife", "PPG_CVA", "PPG_DM", "PPG_HTN"):
        assert specs.BY_FOLDER[name].unit_value == ureg.dimensionless


def test_a_gyroscope_signal_is_an_angular_rate_and_not_an_acceleration() -> None:
    assert specs.spec_for("uci_har", "acc0") is specs.ACCELERATION_G
    assert specs.spec_for("uci_har", "gyro0") is specs.ANGULAR_RATE
