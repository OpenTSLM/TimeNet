import csv
from dataclasses import replace
from datetime import UTC, datetime, timedelta
import gzip
from hashlib import file_digest
import json

import pytest

from timenet import TimeNet
from timenet.dataset import TimeFDataset
from timenet.errors import TimeFFormatError, TimeFValidationError, TimeNetConnectorError
from timenet.testing import assert_datasets_equal
from timenet.types import Access, License, Split, read_value, ureg
from timenet.types.tasks import TextHandle
from timenet_connectors import discovery
from timenet_connectors.datasets.wienkers.swot_river_slope import connector
from timenet_connectors.datasets.wienkers.swot_river_slope.coastal import (
    DAY,
    FIELDS,
    Pass,
    astronomical_phase,
    coastal_issues,
    daily_visits,
    read_pass,
)
from timenet_connectors.datasets.wienkers.swot_river_slope.model_inputs import model_inputs


REACH = 23303400021
ISSUE = datetime(2024, 1, 22, tzinfo=UTC)
EXAMPLE = {
    "reach_id": str(REACH),
    "time_str": "2024-03-12T18:34:21Z",
    "wse": "1.7345",
    "wse_u": "0.09457",
    "slope": "-8.55239e-5",
    "slope_u": "1.985184e-5",
    "reach_q": "1",
}


def _answer(task):
    assert task.targets is not None and isinstance(task.targets[0], TextHandle)
    return task.targets[0].read()


def _passes() -> list[Pass]:
    falls = [0.05, 0.10, 0.15, 0.20, -0.8, -0.4]
    rows = [Pass(REACH, ISSUE - (len(falls) - i) * DAY, 1.0, 0.1, f / 1e4, 1e-5, 1) for i, f in enumerate(falls)]
    rows.append(Pass(REACH, ISSUE + 3 * DAY, 1.1, 0.1, -2e-5, 1e-5, 1))
    return rows


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("slope", "-9.999e15"),
        ("wse", "-999999999999.0"),
        ("wse", "8001"),
        ("slope", ".051"),
        ("slope_u", "0"),
        ("slope_u", "-9.999e15"),
        ("slope_u", "5.1e-5"),
        ("wse_u", "nan"),
        ("reach_q", "3"),
        ("reach_q", "0.5"),
        ("reach_q", "-1"),
    ],
)
def test_fill_and_qc(field, value):
    pass_ = read_pass(EXAMPLE, REACH)
    assert pass_ is not None
    assert pass_.fall == pytest.approx(-0.855239)
    assert pass_.fall_u == pytest.approx(0.1985184)
    assert pass_.z == pytest.approx(-4.308109475)
    assert read_pass({**EXAMPLE, field: value}, REACH) is None


def test_sealed_year_before_decoding_values():
    row = {"reach_id": str(REACH), "time_str": "2025-01-01T00:00:00Z"}
    with pytest.raises(TimeFValidationError, match="outside"):
        read_pass(row, REACH)
    with pytest.raises(TimeFValidationError, match="sealed"):
        read_pass({"reach_id": "33303400021"}, 33303400021)
    assert (2025, {3, 4, 5}, Split.TEST) == connector.SEALED_TEST_SPLIT


def test_t2_and_reversal_scope_and_datum(tmp_path):  # ruff: ignore[too-many-locals]
    reach = connector.Reach(REACH, 10464.0, -0.73744, 53.58339, 2024, "coastal")
    rows = _passes()
    case = next(i for i in coastal_issues(rows, 2024) if i.at == ISSUE)
    assert case.baseline == pytest.approx(0.125)
    assert case.reversal and case.reversal_sig and not case.steepening
    assert case.verified.time == ISSUE + 3 * DAY
    assert all(p.time < ISSUE for p in case.history)
    record, tasks = connector.record_and_tasks(reach, case)
    forecast, reversal, steepening, caption = tasks
    assert forecast.targets == (-0.2, 1, 1, 0)
    assert _answer(reversal) == "Yes" and _answer(steepening) == "No"
    assert reversal.rationale is not None and "M2/S2/K1/O1" in reversal.rationale.read()
    assert caption.metadata["task"] == "caption_pre_issue_evidence"
    assert {s.name for s in record.sources} == {"central_reach"}
    for task in tasks:
        assert not task.from_tasks
        assert task.input_annotations == record.annotations
        assert "verification_utc" not in {annotation.key for annotation in task.input_annotations}
    input_context = {annotation.key: read_value(annotation.value) for annotation in forecast.input_annotations}
    assert input_context["issue_utc"] == "2024-01-22T00:00:00Z"
    assert input_context["baseline_nonneg_m_per_10km"] == pytest.approx(0.125)
    for signal in record.sources[0].signals:
        if signal.name in {"fall_m_per_10km", "fall_u_m_per_10km"}:
            unit = signal.spec.unit_value
            assert unit is not None
            assert (1 * unit).to(ureg.dimensionless).magnitude == pytest.approx(1e-4)
    predicted_input = model_inputs(reversal)
    other = replace(case, verified=replace(case.verified, slope=1e-3, time=case.verified.time + DAY))
    _, (_, other_reversal, other_steepening, _) = connector.record_and_tasks(reach, other)
    assert _answer(other_reversal) == "No" and _answer(other_steepening) == "Yes"
    assert json.dumps(predicted_input) == json.dumps(model_inputs(other_reversal))
    assert "verification_utc" not in json.dumps(predicted_input)
    assert str(case.verified.time.date()) not in json.dumps(predicted_input)
    shifted = replace(case, history=tuple(replace(p, wse=p.wse + 50) for p in case.history))
    _, (_, shifted_reversal, _, _) = connector.record_and_tasks(reach, shifted)
    assert _answer(shifted_reversal) == _answer(reversal)
    assert shifted_reversal.rationale is not None and shifted_reversal.rationale.read() == reversal.rationale.read()

    producer = connector.SwotRiverSlopeConnector()
    dataset = TimeFDataset(metadata=producer.metadata())
    dataset.add_record(record=record)
    dataset.add_tasks(tasks=tasks)
    dataset.write(path=tmp_path)
    with TimeNet(sources=[tmp_path]).open("wienkers/swot-river-slope") as reader:
        reader.verify()
        assert_datasets_equal(dataset, reader.read())
        _, answer, _, stored_caption = reader.iter_tasks(split=Split.VALIDATION)
        assert _answer(answer) == "Yes"
        assert model_inputs(answer) == predicted_input
        assert not stored_caption.from_tasks
        assert not list(reader.iter_tasks(split=Split.TEST))


def test_daily_tie_break_and_regime_boundary():
    first = _passes()[0]
    second = replace(first, time=first.time + DAY / 8, wse_u=first.wse_u / 2)
    assert daily_visits([second, first]) == [first]
    assert daily_visits([first, replace(second, slope_u=first.slope_u / 2)])[0].time == second.time

    case = next(i for i in coastal_issues(_passes(), 2024) if i.at == ISSUE)
    latest = replace(case.history[-1], slope=2e-5)
    reach = connector.Reach(REACH, 10464.0, -0.73744, 53.58339, 2024, "coastal")
    _, (_, reversal, _, _) = connector.record_and_tasks(reach, replace(case, history=(*case.history[:-1], latest)))
    assert reversal.rationale is not None and "low gradient" in reversal.rationale.read()


@pytest.mark.parametrize(
    ("lead", "eligible"),
    [
        (2 * DAY - timedelta(microseconds=1), False),
        (2 * DAY, True),
        (21 * DAY, True),
        (21 * DAY + timedelta(microseconds=1), False),
    ],
)
def test_strict_history_and_inclusive_lead(lead, eligible):
    rows = _passes()
    at_issue = replace(rows[-1], time=ISSUE, slope=0.05)
    target = replace(rows[-1], time=ISSUE + lead)
    case = next((c for c in coastal_issues([*rows[:-1], at_issue, target], 2024) if c.at == ISSUE), None)
    if eligible:
        assert case is not None
        assert case.verified == target
        assert all(visit.time < ISSUE for visit in case.history)
        assert case.baseline == pytest.approx(0.125)
    else:
        assert case is None


def test_prompts_match_the_selected_daily_representative():
    rows = _passes()
    earlier = rows[-1]
    later = replace(earlier, time=earlier.time + DAY / 8, slope=2e-5, slope_u=5e-6)
    case = next(c for c in coastal_issues([*rows, later], 2024) if c.at == ISSUE)
    assert case.verified == later
    reach = connector.Reach(REACH, 10000.0, 0.0, 50.0, 2024, "coastal")
    _, (forecast, reversal, steepening, _) = connector.record_and_tasks(reach, case)
    for task in (forecast, reversal, steepening):
        assert task.prompt is not None
        assert "first daily representative whose timestamp is 2 to 21 days after issue" in task.prompt.read()
        assert "smallest slope uncertainty" in task.prompt.read()
    assert forecast.metadata["verification_utc"] == "2024-01-25T03:00:00Z"
    assert _answer(reversal) == "No"


def test_astronomical_phase_reference():
    phase = astronomical_phase(datetime(2024, 3, 12, 18, 34, 21, tzinfo=UTC), -0.73744)
    assert all(0 <= phase[key] < 360 for key in ("M2", "S2", "K1", "O1"))
    # UTide mean equilibrium arguments without nodal corrections.
    assert [phase[key] for key in ("M2", "S2", "K1", "O1")] == pytest.approx(
        [134.023, 195.700, 358.752, 135.271], abs=0.02
    )
    assert phase["spring_index"] == pytest.approx(0.73722, abs=0.00001)
    assert astronomical_phase(datetime(2024, 3, 12, 18, 34, 21, tzinfo=UTC), 359.26256) == pytest.approx(phase)


@pytest.fixture
def pinned_fixture(tmp_path, monkeypatch):
    snapshot = tmp_path / connector.SNAPSHOT
    lookup = tmp_path / connector.REACH_INDEX
    with gzip.open(snapshot, "wt", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(FIELDS)
        for visit in _passes():
            writer.writerow(
                (REACH, visit.time.isoformat(), visit.wse, visit.wse_u, visit.slope, visit.slope_u, visit.reach_q)
            )
    lookup.write_text(
        json.dumps(
            [{"reach_id": REACH, "length_m": 10000.0, "lon": 0.0, "lat": 50.0, "phase": 1, "stratum": "coastal"}]
        ),
        encoding="utf-8",
    )
    pins = {}
    for file in (snapshot, lookup):
        with file.open("rb") as handle:
            pins[file.name] = file_digest(handle, "sha256").hexdigest()
    monkeypatch.setattr(connector, "_pins", lambda: pins)
    return [snapshot, lookup]


def test_discovery_download_convert_and_digest(pinned_fixture, tmp_path, monkeypatch):
    producer = discovery.resolve("wienkers/swot-river-slope")()
    assert producer.metadata().license is License.OTHER
    assert producer.metadata().access is Access.OPEN

    async def fetch(artifacts):
        items = list(artifacts)
        assert [artifact.dest for artifact in items] == pinned_fixture
        for artifact in items:
            assert artifact.url == f"{connector.RELEASE_BASE}/{artifact.dest.name}"
            assert artifact.sha256 == connector._pins()[artifact.dest.name]
        return pinned_fixture

    monkeypatch.setattr(connector, "download_files", fetch)
    raw_refs = producer.download(tmp_path)
    dataset = producer.convert(raw_refs)
    assert len(dataset.records) == 1
    assert len(dataset.tasks) == 4
    assert {task.split for task in dataset.tasks} == {Split.VALIDATION}
    snapshot, _ = raw_refs
    snapshot.write_bytes(snapshot.read_bytes() + b"bad")
    with pytest.raises(TimeFFormatError, match="SHA-256"):
        producer.convert(raw_refs)


def test_malformed_pinned_index(pinned_fixture, monkeypatch):
    producer = connector.SwotRiverSlopeConnector()
    _, lookup = pinned_fixture
    pins = connector._pins()
    lookup.write_text('[{"phase": 1}]', encoding="utf-8")
    with lookup.open("rb") as handle:
        pins[lookup.name] = file_digest(handle, "sha256").hexdigest()
    monkeypatch.setattr(connector, "_pins", lambda: pins)
    with pytest.raises(TimeNetConnectorError, match="Invalid reach entry 1"):
        producer.convert(pinned_fixture)
