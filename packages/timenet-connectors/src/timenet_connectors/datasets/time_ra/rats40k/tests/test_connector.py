import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from timenet.dataset.axis import OrdinalAxis
from timenet.errors import TimeFValidationError
from timenet.types import ClassificationTask, Split
from timenet_connectors.bases.huggingface import HubFileRef
from timenet_connectors.datasets.time_ra.rats40k import RATS40KConnector, _json, connector as module


_FIXTURES = Path(__file__).parent / "fixtures"
_EXPECTED_REVISION = "dc968d6a88f396215be50023c93c3928e06096df"
_EXPECTED_CHANNELS = {
    ("uni", Split.TRAIN): {"series": [0.123456789, -0.2, 657372000]},
    ("uni", Split.TEST): {"series": [1, 8]},
    ("multi", Split.TRAIN): {"pH": [8.33, 8.527], "Redox": [753.0, 729.281]},
    ("multi", Split.TEST): {"ECG1": [0.39, 0.46, 0.45], "ECG2": [0.22, 0.22, 0.2]},
}
# Each prompt's category count, and names at the authors' codes for them.
_EXPECTED_VOCABULARY = {
    "uni": (15, {0: "Normal Sequence", 12: "Sudden Spike Anomaly"}),
    "multi": (8, {0: "Normal Sequence", 4: "Local Structural Jump Anomaly"}),
}


@pytest.fixture(autouse=True)
def _fixture_inventory(monkeypatch):
    """Pin the inventory to the fixtures: one row per split, and the uni test null entry 106."""
    rows = {"train": 1, "test": 1}
    monkeypatch.setattr(module, "SPLIT_ROWS", {"uni": dict(rows), "multi": dict(rows)})
    monkeypatch.setattr(module, "NULL_KEYS", frozenset({"106"}))


def _refs() -> list[HubFileRef]:
    return [
        HubFileRef(_FIXTURES / f"{variant}.json", variant, RATS40KConnector.SOURCE_FILES[variant])
        for variant in ("uni", "multi")
    ]


def test_conversion_preserves_signals_and_published_labels(monkeypatch, tmp_path):
    # The connector's lazy loaders seek by byte offset, including across CRLF and UTF-8 text.
    refs = []
    for ref in _refs():
        path = tmp_path / ref.path.name
        path.write_bytes(ref.path.read_bytes().replace(b"\n", b"\r\n"))
        refs.append(HubFileRef(path, ref.part, ref.filename))
    monkeypatch.setattr(_json, "_CHUNK_SIZE", 17)
    dataset = RATS40KConnector().convert(refs)
    # A partial preview must not consume the connector's task stream.
    next(dataset.iter_tasks())
    records = {record.id: record for record in dataset.records}
    tasks = {task.id: task for task in dataset.iter_tasks()}
    vocabularies = {annotation.id: annotation.value for annotation in dataset.annotations}
    assert len(records) == len(tasks) == 4
    for variant in ("uni", "multi"):
        schema = f"rats40k-{variant}-category-vocabulary"
        size, names = _EXPECTED_VOCABULARY[variant]
        assert len(vocabularies[schema]) == size
        assert {code: vocabularies[schema][code] for code in names} == names
        for source_split, rows in json.loads((_FIXTURES / f"{variant}.json").read_text()).items():
            row, split = rows["0"], Split(source_split.removeprefix("TSAD_"))
            channels = _EXPECTED_CHANNELS[variant, split]
            record = records[f"rats40k-{variant}-{split}-0"]
            assert len(record.sources) == 1
            assert record.sources[0].name == row["Source"]
            assert record.sources[0].metadata["channel_order"] == list(channels)
            assert {signal.name for signal in record.signals} == set(channels)
            for signal in record.signals:
                assert signal.spec.unit_value is None
                assert signal.to_numpy().dtype == np.dtype("float64")
                assert isinstance(signal.time_axis, OrdinalAxis)
                np.testing.assert_array_equal(signal.to_numpy(), np.asarray(channels[signal.name], dtype=np.float64))
            provenance = {annotation.key: annotation.value for annotation in record.annotations}
            assert provenance["source_key"] == "0"
            assert provenance["figure_path"] == row["FigurePath"]
            assert not {"action_id", "ActionID", "Action", "Label", "Thought"} & provenance.keys()
            task = tasks[f"{record.id}-category"]
            assert type(task) is ClassificationTask
            # The published name is the target whatever its ActionID, even outside the vocabulary.
            assert task.targets == (row["Action"],)
            assert task.rationale == row["Thought"]
            assert task.target_schema == schema
            assert task.metadata == {
                "ActionID": row["ActionID"],
                "Label": row["Label"],
                **({"Rank": row["Rank"]} if "Rank" in row else {}),
            }
            assert task.inputs == (record,)
            assert task.prompt == module._PROMPTS[variant]
            assert {a.key: a.value for a in task.input_annotations} == {"source_domain": row["Source"]}
            assert task.split is split
    # The uni test fixture carries a multivariate name, as most real uni test rows do.
    assert tasks["rats40k-uni-test-0-category"].targets == ("Temporal Dependency Anomaly",)
    assert "Temporal Dependency Anomaly" not in vocabularies["rats40k-uni-category-vocabulary"]


def test_download_requests_pinned_source_files(monkeypatch, tmp_path):
    calls = []

    def snapshot_download(repo_id, *, revision, cache_dir, allow_patterns, **kwargs):
        calls.append((repo_id, revision, cache_dir, allow_patterns))
        return str(tmp_path)

    monkeypatch.setitem(sys.modules, "huggingface_hub", SimpleNamespace(snapshot_download=snapshot_download))
    refs = RATS40KConnector().download(tmp_path / "cache")
    filenames = ["RATs-Uni-TSImage_Reason.json", "RATs-Multi-TSImage_Reason.json"]
    assert calls == [("Time-RA/RATs40K", _EXPECTED_REVISION, str(tmp_path / "cache"), filenames)]
    assert refs == [
        HubFileRef(tmp_path / name, variant, name) for name, variant in zip(filenames, ("uni", "multi"), strict=True)
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
        RATS40KConnector().convert([HubFileRef(path, variant, RATS40KConnector.SOURCE_FILES[variant])])
    assert str(path) in str(error.value) and "train, key 0" in str(error.value)
