from pathlib import Path

import pytest

from timenet.errors import TimeFFormatError
from timenet_connectors.bases.seizures import SeizureInterval
from timenet_connectors.datasets.physionet.chbmit.summary import CaseSummary


_FIXTURES = Path(__file__).parent / "fixtures"


def _summary(tmp_path: Path, text: str) -> Path:
    path = tmp_path / "chb90-summary.txt"
    path.write_text(text, encoding="utf-8")
    return path


def test_a_channel_change_holds_for_the_entries_after_it():
    summary = CaseSummary.from_file(_FIXTURES / "chb21-summary.txt")
    assert [entry.channels for entry in summary.entries] == [
        ("FP1-F7", "T8-P8", "T8-P8"),
        ("FP1-F7", ".", "T8-P8", "ECG"),
    ]


def test_numbered_and_padded_seizure_lines_parse_to_whole_second_intervals():
    entry = CaseSummary.from_file(_FIXTURES / "chb21-summary.txt").entry_for("chb21_02.edf")
    assert entry.seizures == (SeizureInterval.from_whole_seconds(2, 4), SeizureInterval.from_whole_seconds(8, 10))


def test_a_seizure_count_that_disagrees_with_the_listed_seizures_is_refused(tmp_path):
    path = _summary(
        tmp_path,
        "Channels in EDF Files:\nChannel 1: FP1-F7\n\n"
        "File Name: chb90_01.edf\nNumber of Seizures in File: 2\n"
        "Seizure Start Time: 3 seconds\nSeizure End Time: 6 seconds\n",
    )
    with pytest.raises(TimeFFormatError, match="states 2 seizures and lists 1 onsets and 1 ends"):
        CaseSummary.from_file(path)


def test_an_unknown_line_is_refused(tmp_path):
    path = _summary(
        tmp_path,
        "Channels in EDF Files:\nChannel 1: FP1-F7\n\nFile Name: chb90_01.edf\nNumber of Seizures in File: 0\n"
        "Seizure Onset: 3 s\n",
    )
    with pytest.raises(TimeFFormatError, match="a line this parser does not know"):
        CaseSummary.from_file(path)


def test_a_file_listed_before_any_channel_list_is_refused(tmp_path):
    path = _summary(tmp_path, "File Name: chb90_01.edf\nNumber of Seizures in File: 0\n")
    with pytest.raises(TimeFFormatError, match="before any channel list"):
        CaseSummary.from_file(path)
