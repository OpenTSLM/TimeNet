import asyncio
import calendar
import csv
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import cast

import numpy as np
import pytest

from timenet.client import TimeNet
from timenet.dataset import Signal
from timenet.dataset.axis import IrregularAxis, RegularAxis
from timenet.errors import TimeFFormatError
from timenet.types import AnswerTask, ForecastingTask, Task, TimeInterval
from timenet.types.clock import unix_us
from timenet_connectors.datasets.adityalab.time_mmd import connector
from timenet_connectors.datasets.adityalab.time_mmd.connector import SOURCE_FILES, TimeMMDConnector
from timenet_connectors.discovery import resolve


_FIXTURES = Path(__file__).with_name("fixtures")


def _refs(root=_FIXTURES):
    return [root / relative for relative in SOURCE_FILES]


def _times(signal: Signal):
    if isinstance(signal.time_axis, RegularAxis):
        return np.array([signal.time_axis.time_offset_us(i) for i in range(signal.n_values)])
    return signal.time_offsets_us()


def _long_refs(tmp_path, difference="0.002", future="0.25"):
    refs = _copy(tmp_path)
    for domain in ("Economy", "Energy"):
        path = next(p for p in refs if p.name == f"{domain}.csv")
        with path.open(newline="") as stream:
            reader = csv.DictReader(stream)
            headers = reader.fieldnames
            template = next(reader)
        assert headers is not None
        rows = []
        for index in range(65):
            row = template.copy()
            if domain == "Energy":
                start = date(1993, 4, 5) + timedelta(days=7 * index)
                end = start + timedelta(days=6)
            else:
                year, month = divmod(1987 * 12 + index, 12)
                start = date(year, month + 1, 1)
                end = start.replace(day=calendar.monthrange(year, month + 1)[1])
            row.update(date=start.isoformat(), start_date=start.isoformat(), end_date=end.isoformat())
            delta = str(float(difference) * 1000) if domain == "Economy" else difference
            row["OT"] = future if index == 64 else delta if index >= 56 else "0"
            rows.append(row)
        with path.open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=headers)
            writer.writeheader()
            writer.writerows(rows)
    return refs


def _input_values(task: Task):
    assert isinstance(task.scope, TimeInterval)
    assert task.scope.time_series_ids is not None
    signal = next(s for s in task.inputs[0].signals if s.id in task.scope.time_series_ids)
    times = _times(signal)
    return signal.to_numpy()[(times >= task.scope.start_us) & (times < task.scope.end_us)]


def _copy(tmp_path):
    for path in _refs():
        target = tmp_path / path.relative_to(_FIXTURES)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(path.read_bytes())
    return _refs(tmp_path)


def _edit_numeric(tmp_path, domain, edit):
    refs = _copy(tmp_path)
    path = next(p for p in refs if p.name == f"{domain}.csv")
    with path.open(newline="") as stream:
        reader = csv.DictReader(stream)
        headers, rows = reader.fieldnames, list(reader)
    assert headers is not None
    edit(rows)
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]) if rows else headers)
        writer.writeheader()
        writer.writerows(rows)
    return refs


def test_discovery_and_card():
    assert resolve("adityalab/time-mmd") is TimeMMDConnector
    metadata = TimeMMDConnector().metadata()
    assert metadata.dataset_id == "adityalab/time-mmd"
    assert str(metadata.license) == "ODC-By-1.0"


def test_shared_domain_records_and_calendar():
    economy, energy = TimeMMDConnector().convert(_refs()).records
    assert len(economy.signals) == 3
    assert len(energy.signals) == 9
    assert economy.metadata["numerical_source_rows"] == [1, 2, 0]
    ot = next(s for s in economy.signals if s.name == "OT")
    assert isinstance(ot.time_axis, IrregularAxis)
    assert ot.spec.dtype == "float64"
    np.testing.assert_array_equal(ot.to_numpy(), [-13452.6, -10956.4, -10628.8])
    np.testing.assert_array_equal(np.diff(_times(ot)), [31, 28] * np.array(86_400_000_000))
    energy_ot = next(s for s in energy.signals if s.name == "OT")
    assert isinstance(energy_ot.time_axis, RegularAxis)
    np.testing.assert_array_equal(np.diff(_times(energy_ot)), [7, 7] * np.array(86_400_000_000))
    assert energy_ot.metadata["unit"] == "USD/US gallon"
    date_columns = cast("list[dict[str, str]]", economy.metadata["numerical_date_columns"])
    assert date_columns[0]["Month"] == "January 1987"


def test_lazy_values_and_offsets(monkeypatch):
    original = connector._values
    calls = []

    def capture(rows, column):
        calls.append(column)
        return original(rows, column)

    monkeypatch.setattr(connector, "_values", capture)
    dataset = TimeMMDConnector().convert(_refs())
    assert calls == []
    dataset.records[0].signals[0].to_numpy()
    assert len(calls) == 1


def test_additional_numerical_channels_are_preserved(tmp_path):
    def add_channel(rows):
        for row in rows:
            row["additional_measurement"] = "2.5"

    refs = _edit_numeric(tmp_path, "Economy", add_channel)
    economy = TimeMMDConnector().convert(refs).records[0]
    signal = next(s for s in economy.signals if s.name == "additional_measurement")
    assert len(economy.signals) == 4
    np.testing.assert_array_equal(signal.to_numpy(), [2.5, 2.5, 2.5])


def test_missing_ot_channel_is_rejected(tmp_path):
    def remove_ot(rows):
        for row in rows:
            del row["OT"]

    refs = _edit_numeric(tmp_path, "Economy", remove_ot)
    with pytest.raises(TimeFFormatError, match="OT channel"):
        TimeMMDConnector().convert(refs)


def test_text_preserves_missing_markers_duplicate_rows_and_unknown_availability():
    economy = TimeMMDConnector().convert(_refs()).records[0]
    annotations = [a for a in economy.annotations if a.key == "search_preds"]
    assert [a.value for a in annotations] == ["NA;NA", "NA;NA", ""]
    assert len({a.id for a in annotations}) == 3
    assert len({a.occurrence_id for a in annotations}) == 3
    assert isinstance(annotations[0].span, TimeInterval)
    assert isinstance(economy.start_time.timestamp, int)
    assert economy.start_time.timestamp + annotations[0].span.start_us == unix_us(datetime(1979, 12, 31, tzinfo=UTC))
    assert all(a.metadata["available_at"] is None for a in annotations)
    facts = [a.value for a in economy.annotations if a.key == "report_fact"]
    assert facts == ['Reported "goods and services", not goods-only.\nOriginal second line.']


def test_roundtrip_preserves_hierarchy_values_dates_and_annotations(tmp_path):
    dataset = TimeMMDConnector().convert(_refs())
    dataset.write(path=tmp_path)
    restored = TimeNet(registry=tmp_path).load("adityalab/time-mmd", auto_build=False)
    assert len(restored.records) == 2
    for before, after in zip(dataset.records, restored.records, strict=True):
        assert before.metadata == after.metadata
        assert [a.value for a in before.annotations] == [a.value for a in after.annotations]
        assert [a.metadata for a in before.annotations] == [a.metadata for a in after.annotations]
        assert [a.span for a in before.annotations] == [a.span for a in after.annotations]
        for original, reread in zip(before.signals, after.signals, strict=True):
            np.testing.assert_array_equal(original.to_numpy(), reread.to_numpy())
            np.testing.assert_array_equal(_times(original), _times(reread))


@pytest.mark.parametrize("value", ["", "NA", "NaN", "Infinity", "1e1000"])
def test_invalid_numerical_values_are_rejected(tmp_path, value):
    refs = _edit_numeric(tmp_path, "Energy", lambda rows: rows[0].update(OT=value))
    with pytest.raises(TimeFFormatError):
        TimeMMDConnector().convert(refs)


def test_duplicate_numerical_dates_are_rejected(tmp_path):
    refs = _edit_numeric(tmp_path, "Energy", lambda rows: rows.append(rows[0].copy()))
    with pytest.raises(TimeFFormatError, match="duplicate"):
        TimeMMDConnector().convert(refs)


def test_calendar_gap_is_rejected(tmp_path):
    refs = _edit_numeric(tmp_path, "Energy", lambda rows: rows.pop(1))
    with pytest.raises(TimeFFormatError, match="calendar"):
        TimeMMDConnector().convert(refs)


def test_missing_or_repeated_file_is_rejected():
    refs = _refs()
    for invalid in (refs[:-1], [*refs, refs[0]]):
        with pytest.raises(TimeFFormatError, match="six"):
            TimeMMDConnector().convert(invalid)


def test_download_uses_existing_helpers_and_pinned_checksums(monkeypatch, tmp_path):
    seen = []

    async def download(artifacts):
        seen.extend(artifacts)
        return [a.dest for a in artifacts]

    monkeypatch.setattr(connector, "download_files", download)
    refs = asyncio.run(TimeMMDConnector().download_async(tmp_path))
    assert len(refs) == 6
    assert all(connector.SOURCE_REVISION in a.url for a in seen)
    assert [a.sha256 for a in seen] == list(SOURCE_FILES.values())


def test_corrupted_cache_is_rejected_before_network(tmp_path):
    relative = next(iter(SOURCE_FILES))
    path = tmp_path / connector.SOURCE_REVISION / relative
    path.parent.mkdir(parents=True)
    path.write_text("truncated")
    with pytest.raises(TimeFFormatError, match="checksum"):
        TimeMMDConnector().download(tmp_path)


@pytest.mark.parametrize(
    "difference,expected",
    [("0.002", "higher"), ("-0.002", "lower"), ("0.001", "within"), ("-0.001", "within"), ("0", "within")],
)
def test_qa_truth_and_inclusive_tolerance(tmp_path, difference, expected):
    dataset = TimeMMDConnector().convert(_long_refs(tmp_path, difference))
    answers = dataset.tasks_of(AnswerTask)
    assert len(answers) == 2
    assert all(task.targets == (expected,) for task in answers)
    assert all(len(_input_values(task)) == 64 for task in answers)
    assert all(task.input_annotations == () for task in dataset.tasks)
    assert {task.metadata["tolerance"] for task in answers} == {"1", "0.001"}


def test_future_target_cannot_change_qa_or_scoped_inputs(tmp_path):
    first = TimeMMDConnector().convert(_long_refs(tmp_path / "first", future="0.25"))
    changed = TimeMMDConnector().convert(_long_refs(tmp_path / "changed", future="99999"))
    for before, after in zip(first.tasks, changed.tasks, strict=True):
        np.testing.assert_array_equal(_input_values(before), _input_values(after))
        assert before.scope == after.scope
        if isinstance(before, AnswerTask):
            assert before.targets == after.targets
        else:
            assert before.targets == (0.25,)
            assert after.targets == (99999.0,)


def test_tasks_roundtrip_and_evaluator_metadata_are_not_inputs(tmp_path):
    dataset = TimeMMDConnector().convert(_long_refs(tmp_path / "raw"))
    dataset.write(path=tmp_path / "registry")
    loaded = TimeNet(registry=tmp_path / "registry").load("adityalab/time-mmd", auto_build=False)
    original = {task.id: task for task in dataset.tasks}
    assert len(loaded.tasks) == 4
    for task in loaded.tasks:
        before = original[task.id]
        assert task.targets == before.targets
        assert task.metadata == before.metadata
        assert task.scope == before.scope
        values = _input_values(task).copy()
        task.targets = ("sentinel",)
        task.rationale = "sentinel"
        task.metadata = {"sentinel": "not model input"}
        np.testing.assert_array_equal(values, _input_values(task))
    forecasts = dataset.tasks_of(ForecastingTask)
    assert all(t.metadata["horizon"] == 1 for t in forecasts)
