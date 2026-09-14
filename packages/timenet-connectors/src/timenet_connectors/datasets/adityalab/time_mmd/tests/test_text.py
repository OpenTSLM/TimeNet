from datetime import date
from pathlib import Path

import pytest

from timenet.errors import TimeFFormatError
from timenet.types import TimeInterval
from timenet_connectors.datasets.adityalab.time_mmd import text
from timenet_connectors.datasets.adityalab.time_mmd.keys import AnnotationKey, TextKind
from timenet_connectors.datasets.adityalab.time_mmd.timeline import US_PER_DAY, Timeline


_RECORD_ID = "time-mmd-energy"


@pytest.mark.parametrize(
    "field",
    ["", "   ", "NA", "N/A", "NA (no relevant information found)", "NA\n\nNote: nothing relevant.", "  NA;NA"],
)
def test_a_field_that_states_nothing(field):
    assert text.states_nothing(field)


@pytest.mark.parametrize(
    "field",
    ["NASA launched a satellite.", "NAFTA changed.", "The price rose.", "na is a syllable.", "Not available."],
)
def test_a_field_that_states_something(field):
    assert not text.states_nothing(field)


def test_predictions_state_nothing_only_when_every_part_does():
    assert text.predictions_state_nothing("NA;NA")
    assert text.predictions_state_nothing("NA (nothing for the long term); NA (nothing for the short term)")
    assert text.predictions_state_nothing("")
    assert not text.predictions_state_nothing("NA;Short term: prices fall.")
    assert not text.predictions_state_nothing("Long term: prices rise.;NA")
    assert not text.predictions_state_nothing("Long term only, with no separator.")


def test_the_annotation_id_names_the_record_the_file_the_field_and_the_row():
    assert text.annotation_id(_RECORD_ID, TextKind.SEARCH, "prediction", 7) == "time-mmd-energy-search-prediction-7"


def _rows(*rows):
    return tuple(
        text.TextRow(
            row=index, start=date.fromisoformat(start), end=date.fromisoformat(end), fact=fact, predictions=preds
        )
        for index, (start, end, fact, preds) in enumerate(rows)
    )


def test_a_row_gives_a_fact_and_a_prediction_over_its_days():
    timeline = Timeline(start=date(2020, 1, 6))
    built = text.build_annotations(
        _RECORD_ID, TextKind.REPORT, _rows(("2020-01-13", "2020-01-17", "A fact.", "Long.;Short.")), timeline
    )
    assert [one.key for one in built] == [AnnotationKey.REPORT_FACT, AnnotationKey.REPORT_PREDICTION]
    assert [one.value for one in built] == ["A fact.", "Long.;Short."]
    assert {one.span for one in built} == {TimeInterval.micros(7 * US_PER_DAY, 12 * US_PER_DAY)}
    assert [one.id for one in built] == ["time-mmd-energy-report-fact-0", "time-mmd-energy-report-prediction-0"]


def test_a_search_row_takes_the_search_keys():
    timeline = Timeline(start=date(2020, 1, 6))
    built = text.build_annotations(
        _RECORD_ID, TextKind.SEARCH, _rows(("2020-01-06", "2020-01-12", "A fact.", "NA;NA")), timeline
    )
    assert [one.key for one in built] == [AnnotationKey.SEARCH_FACT]


def test_every_annotation_of_one_key_carries_one_description():
    timeline = Timeline(start=date(2020, 1, 6))
    built = text.build_annotations(
        _RECORD_ID,
        TextKind.REPORT,
        _rows(("2020-01-06", "2020-01-10", "One.", "NA;NA"), ("2020-01-13", "2020-01-17", "Two.", "NA;NA")),
        timeline,
    )
    assert len({one.description for one in built}) == 1


def test_a_text_is_kept_verbatim():
    timeline = Timeline(start=date(2020, 1, 6))
    raw = "  A fact with spaces around it, and a\nline break.  "
    built = text.build_annotations(_RECORD_ID, TextKind.REPORT, _rows(("2020-01-06", "2020-01-10", raw, "")), timeline)
    assert built[0].value == raw


def _write(path: Path, lines: list[str]) -> Path:
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def test_read_rows_counts_rows_from_zero_and_keeps_the_texts(tmp_path: Path):
    path = _write(
        tmp_path / "Energy_report.csv",
        [
            ",start_date,end_date,fact,preds",
            '0,2020-01-06,2020-01-10,"A fact, with a comma.",Long.;Short.',
            "1,2020-01-13,2020-01-17,NA,NA;NA",
        ],
    )
    rows = text.read_rows(path)
    assert [one.row for one in rows] == [0, 1]
    assert rows[0].fact == "A fact, with a comma."
    assert rows[1].start == date(2020, 1, 13)


def test_read_rows_reads_an_empty_file(tmp_path: Path):
    assert text.read_rows(_write(tmp_path / "Energy_report.csv", [",start_date,end_date,fact,preds"])) == ()


def test_a_row_that_ends_before_it_starts_raises(tmp_path: Path):
    path = _write(
        tmp_path / "Energy_report.csv", [",start_date,end_date,fact,preds", "0,2020-01-13,2020-01-06,A fact.,NA;NA"]
    )
    with pytest.raises(TimeFFormatError, match="line 2: ends on 2020-01-06, before it starts"):
        text.read_rows(path)


def test_a_file_that_lacks_a_column_raises(tmp_path: Path):
    path = _write(tmp_path / "Energy_report.csv", [",start_date,end_date,preds", "0,2020-01-06,2020-01-10,NA;NA"])
    with pytest.raises(TimeFFormatError, match=r"lacks the columns \['fact'\]"):
        text.read_rows(path)


def test_the_readme_spelling_of_the_predictions_column_is_read_too(tmp_path: Path):
    path = _write(
        tmp_path / "Energy_report.csv",
        [",start_date,end_date,fact,pred", "0,2020-01-06,2020-01-10,A fact.,Long.;Short."],
    )
    assert text.read_rows(path)[0].predictions == "Long.;Short."


def test_a_file_with_both_spellings_of_the_predictions_column_raises(tmp_path: Path):
    path = _write(
        tmp_path / "Energy_report.csv",
        [",start_date,end_date,fact,preds,pred", "0,2020-01-06,2020-01-10,A fact.,NA;NA,NA;NA"],
    )
    with pytest.raises(TimeFFormatError, match=r"holds the prediction columns \['preds', 'pred'\]"):
        text.read_rows(path)


def test_a_file_with_no_predictions_column_raises(tmp_path: Path):
    path = _write(tmp_path / "Energy_report.csv", [",start_date,end_date,fact", "0,2020-01-06,2020-01-10,A fact."])
    with pytest.raises(TimeFFormatError, match=r"holds the prediction columns \[\]"):
        text.read_rows(path)


def test_a_short_row_raises(tmp_path: Path):
    path = _write(tmp_path / "Energy_report.csv", [",start_date,end_date,fact,preds", "0,2020-01-06,2020-01-10"])
    with pytest.raises(TimeFFormatError, match="line 2: holds no"):
        text.read_rows(path)
