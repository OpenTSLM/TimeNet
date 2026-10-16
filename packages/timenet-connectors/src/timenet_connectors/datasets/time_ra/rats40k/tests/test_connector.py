import json
from pathlib import Path

import numpy as np
import pytest

from timenet.dataset.axis import OrdinalAxis
from timenet.errors import TimeFValidationError
from timenet.types import ClassificationTask, Split
from timenet_connectors.datasets.time_ra.rats40k import RATS40KConnector, _json, connector as module


_FIXTURES = Path(__file__).parent / "fixtures"
_EXPECTED_REVISION = "dc968d6a88f396215be50023c93c3928e06096df"


@pytest.fixture(autouse=True)
def _fixture_inventory(monkeypatch):
    """Pin the inventory to the fixtures: one row per split, and the uni test null entry 106."""
    rows = {"train": 1, "test": 1}
    monkeypatch.setattr(module, "SOURCE_ROWS", {"uni": dict(rows), "multi": dict(rows)})
    monkeypatch.setattr(module, "NULL_KEYS", frozenset({"106"}))


def _refs() -> list[module._FileRef]:
    return [{"path": str(_FIXTURES / f"{variant}.json"), "variant": variant} for variant in ("uni", "multi")]


def _annotations(owner):
    return {annotation.key: annotation.value for annotation in owner.annotations}


def test_conversion_preserves_signals_and_published_labels(monkeypatch, tmp_path):
    # The connector's lazy loaders seek by byte offset, including across CRLF and UTF-8 text.
    refs = []
    for ref in _refs():
        path = tmp_path / Path(ref["path"]).name
        path.write_bytes(Path(ref["path"]).read_bytes().replace(b"\n", b"\r\n"))
        refs.append({**ref, "path": str(path)})
    monkeypatch.setattr(_json, "_CHUNK_SIZE", 17)
    dataset = RATS40KConnector().convert(refs)
    # A partial preview must not consume the connector's task stream.
    next(dataset.iter_tasks())
    tasks = list(dataset.iter_tasks())
    annotations = {annotation.id: annotation for annotation in dataset.annotations}
    assert len(dataset.records) == 4
    assert len(tasks) == 4
    expected_channels = (
        {"series": [0.123456789, -0.2, 657372000]},
        {"series": [1, 8]},
        {"pH": [8.33, 8.527], "Redox": [753.0, 729.281]},
        {"ECG1": [0.39, 0.46, 0.45], "ECG2": [0.22, 0.22, 0.2]},
    )
    source_rows = [
        row
        for variant in ("uni", "multi")
        for rows in json.loads((_FIXTURES / f"{variant}.json").read_text()).values()
        for row in rows.values()
        if row is not None
    ]
    records_by_id = {record.id: record for record in dataset.records}
    tasks_by_id = {task.id: task for task in tasks}
    for index, (channels, row) in enumerate(zip(expected_channels, source_rows, strict=True)):
        variant = "uni" if index < 2 else "multi"
        split = Split.TRAIN if index % 2 == 0 else Split.TEST
        record = records_by_id[f"rats40k-{variant}-{split}-0"]
        assert len(record.sources) == 1
        assert record.sources[0].name == row["Source"]
        assert record.sources[0].metadata["channel_order"] == list(channels)
        assert {signal.name for signal in record.signals} == set(channels)
        for signal in record.signals:
            assert signal.spec.unit_value is None
            values = signal.to_numpy()
            assert values.dtype == np.dtype("float64")
            assert isinstance(signal.time_axis, OrdinalAxis)
            np.testing.assert_array_equal(values, np.asarray(channels[signal.name], dtype=np.float64))
        provenance = _annotations(record)
        assert provenance["source_key"] == "0"
        assert provenance["figure_path"] == row["FigurePath"]
        assert not {"action_id", "ActionID", "Action", "Label", "Thought"} & provenance.keys()
        task = tasks_by_id[f"{record.id}-category"]
        assert type(task) is ClassificationTask
        assert task.targets == (row["Action"],)
        assert task.rationale == row["Thought"]
        assert task.target_schema == annotations[f"rats40k-{variant}-category-vocabulary"].id
        assert task.metadata == {
            "ActionID": row["ActionID"],
            "Label": row["Label"],
            **({"Rank": row["Rank"]} if "Rank" in row else {}),
        }
        assert task.inputs == (record,)
        assert task.prompt == module._PROMPTS[variant]
        assert {a.key: a.value for a in task.input_annotations} == {"source_domain": row["Source"]}
        assert task.split is split


def test_download_requests_pinned_source_files(monkeypatch, tmp_path):
    calls = []

    def snapshot(repo_id, revision, cache_dir, filenames):
        calls.append((repo_id, revision, cache_dir, filenames))
        return tmp_path

    monkeypatch.setattr(module, "hub_snapshot", snapshot)
    refs = RATS40KConnector().download(tmp_path)
    assert calls == [
        (
            "Time-RA/RATs40K",
            _EXPECTED_REVISION,
            tmp_path,
            ("RATs-Uni-TSImage_Reason.json", "RATs-Multi-TSImage_Reason.json"),
        )
    ]
    assert refs == [
        {"path": str(tmp_path / filename), "variant": variant}
        for variant, filename in (
            ("uni", "RATs-Uni-TSImage_Reason.json"),
            ("multi", "RATs-Multi-TSImage_Reason.json"),
        )
    ]


@pytest.mark.parametrize(
    ("text", "message"),
    [
        # json.load would silently keep only the last duplicate row.
        ('{"TSAD_train": {"0": null, "0": null}, "TSAD_test": {}}', "distinct string key"),
        ('{"TSAD_train": {}, "TSAD_val": {}}', "unknown split 'TSAD_val'"),
        ('{"TSAD_train": {}}', "expected TSAD_train and TSAD_test"),
    ],
)
def test_malformed_split_structure_is_rejected(tmp_path, text, message):
    path = tmp_path / "broken.json"
    path.write_text(text)
    with pytest.raises(TimeFValidationError, match=message):
        list(_json.iter_entries(path))


def test_pinned_inventory_mismatch_fails(monkeypatch):
    monkeypatch.setattr(module, "NULL_KEYS", frozenset())
    with pytest.raises(TimeFValidationError, match="pinned row counts or null entry keys differ"):
        RATS40KConnector().convert(_refs())


@pytest.mark.parametrize(
    ("variant", "field", "value", "message"),
    [
        ("multi", "Observation", '"a": [1]; "a": [2]', "duplicate channel"),
        ("multi", "Observation", '"a": [1]; "b": [2,3]', "lengths differ"),
        # Python-written JSON can contain NaN, which json accepts by default.
        ("uni", "Observation", [1.0, float("nan")], "finite float64"),
        ("uni", "ActionID", True, "ActionID must be a nonnegative integer"),
        ("uni", "Label", "anomaly", "Label must be Normal or Anomaly"),
        # Only explicit nulls are skipped; any other non-object row fails.
        ("uni", None, "x", "expected a row object"),
    ],
)
def test_malformed_rows_have_source_context(tmp_path, variant, field, value, message):
    data = json.loads((_FIXTURES / f"{variant}.json").read_text())
    if field is None:
        data["TSAD_train"]["0"] = value
    else:
        data["TSAD_train"]["0"][field] = value
    path = tmp_path / "broken.json"
    path.write_text(json.dumps(data))
    with pytest.raises(TimeFValidationError, match=message) as error:
        RATS40KConnector().convert([{"path": str(path), "variant": variant}])
    assert str(path) in str(error.value) and "train, key 0" in str(error.value)


def test_published_names_are_targets_whatever_their_codes(monkeypatch, tmp_path):
    rows = {"train": 3, "test": 1}
    monkeypatch.setattr(module, "SOURCE_ROWS", {"uni": dict(rows), "multi": dict(rows)})
    refs = []
    for variant in ("uni", "multi"):
        data = json.loads((_FIXTURES / f"{variant}.json").read_text())
        for rows in data.values():
            for row in rows.values():
                if row is not None:
                    row["ActionID"] = 2
        # Two names under one code, and one name under two codes, all stay published targets.
        data["TSAD_train"]["1"] = {**data["TSAD_train"]["0"], "Action": "Another published name"}
        data["TSAD_train"]["2"] = {**data["TSAD_train"]["0"], "ActionID": 10}
        path = tmp_path / f"{variant}.json"
        path.write_text(json.dumps(data))
        refs.append({"path": str(path), "variant": variant})
    dataset = RATS40KConnector().convert(refs)
    vocabularies = {a.key: a for a in dataset.annotations if a.key.endswith("category_vocabulary")}
    for variant, original_name in (("uni", "Normal Sequence"), ("multi", "Joint Space Anomaly")):
        vocabulary = vocabularies[f"{variant}_category_vocabulary"]
        # The vocabulary is the prompt's, whatever names and codes the rows publish.
        assert vocabulary.value == list(module._CATEGORY_NAMES[variant])
        categories = [
            task
            for task in dataset.iter_tasks()
            if isinstance(task, ClassificationTask) and task.id.startswith(f"rats40k-{variant}-")
        ]
        assert len(categories) == 4
        # A name outside the prompt's categories stays the published target.
        test_name = {"uni": "Sudden Spike Anomaly", "multi": "Normal Sequence"}[variant]
        assert {task.targets for task in categories} == {
            ("Another published name",),
            (original_name,),
            (test_name,),
        }
        assert {task.metadata["ActionID"] for task in categories} == {2, 10}
        assert all(task.target_schema == vocabulary.id for task in categories)
