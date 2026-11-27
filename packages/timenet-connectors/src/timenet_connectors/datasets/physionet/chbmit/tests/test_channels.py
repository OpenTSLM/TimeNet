import pytest

from timenet.errors import TimeFFormatError
from timenet_connectors.datasets.physionet.chbmit.channels import select_channels, spec_for_label


def test_placeholders_become_no_channel_and_a_repeated_label_is_numbered():
    labels = ("FP1-F7", "-", "T8-P8", ".", "T8-P8")
    channels = select_channels(labels, ("uV",) * len(labels), "chb90_01.edf")
    assert [(channel.index, channel.name) for channel in channels] == [(0, "FP1-F7"), (2, "T8-P8"), (4, "T8-P8 (2)")]


@pytest.mark.parametrize(
    ("label", "spec_type"),
    [
        ("FT9-FT10", "eeg"),
        ("FC1-Ref", "eeg"),
        ("F7-CS2", "eeg"),
        ("CZ", "eeg"),
        ("01", "eeg"),
        ("ECG", "ecg"),
        ("LOC-ROC", "eog"),
        ("VNS", "vns"),
        ("LUE-RAE", "biopotential"),
    ],
)
def test_each_label_form_of_the_release_maps_to_a_spec(label, spec_type):
    assert spec_for_label(label, "chb90_01.edf").spec_type == spec_type


@pytest.mark.parametrize("label", ["EEG FP1-F7", "FP1-XX", "Resp"])
def test_an_unknown_label_is_refused(label):
    with pytest.raises(TimeFFormatError, match="names no channel this release records"):
        spec_for_label(label, "chb90_01.edf")


def test_a_channel_in_another_unit_is_refused():
    with pytest.raises(TimeFFormatError, match="states the unit 'mV'"):
        select_channels(("FP1-F7",), ("mV",), "chb90_01.edf")
