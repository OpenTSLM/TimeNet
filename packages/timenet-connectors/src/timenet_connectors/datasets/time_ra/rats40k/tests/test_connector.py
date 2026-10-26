import json
from pathlib import Path

import numpy as np
import pytest

from timenet.dataset.axis import OrdinalAxis
from timenet.errors import TimeFValidationError
from timenet.types import ClassificationTask, Split
from timenet_connectors.bases.huggingface import HubFileRef
from timenet_connectors.datasets.time_ra.rats40k import RATS40KConnector, _source, connector as module, request
from timenet_connectors.sources import json_stream


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
    return [HubFileRef(_FIXTURES / f"{variant}.json", variant) for variant in ("uni", "multi")]


def test_conversion_preserves_signals_and_published_labels(monkeypatch, tmp_path):
    # The connector's lazy loaders seek by byte offset, including across CRLF and UTF-8 text.
    refs = []
    for ref in _refs():
        path = tmp_path / ref.path.name
        path.write_bytes(ref.path.read_bytes().replace(b"\n", b"\r\n"))
        refs.append(HubFileRef(path, ref.part))
    monkeypatch.setattr(json_stream, "_CHUNK_SIZE", 17)
    dataset = RATS40KConnector().convert(refs)
    # Iterating tasks more than once gives every task each time.
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
            assert record.metadata == {"source_key": "0", "source_variant": variant, "figure_path": row["FigurePath"]}
            assert {annotation.key: annotation.value for annotation in record.annotations} == {
                "source_domain": row["Source"]
            }
            task = tasks[f"{record.id}-category"]
            assert type(task) is ClassificationTask
            # The published name is the target whatever its ActionID, even outside the vocabulary.
            assert task.targets == (row["Action"],)
            assert task.rationale == row["Thought"]
            assert task.target_schema == schema
            assert task.metadata == {
                "ActionID": row["ActionID"],
                "Label": row["Label"],
                "target_in_vocabulary": row["Action"] in vocabularies[schema],
                **({"Rank": row["Rank"]} if "Rank" in row else {}),
            }
            assert task.inputs == (record,)
            assert task.prompt == request.TASK_PROMPTS[variant]
            assert {a.key: a.value for a in task.input_annotations} == {"source_domain": row["Source"]}
            assert task.split is split
    # The uni test fixture carries a multivariate name, as most real uni test rows do.
    assert tasks["rats40k-uni-test-0-category"].targets == ("Temporal Dependency Anomaly",)
    assert "Temporal Dependency Anomaly" not in vocabularies["rats40k-uni-category-vocabulary"]
    assert tasks["rats40k-uni-test-0-category"].metadata["target_in_vocabulary"] is False
    assert tasks["rats40k-uni-train-0-category"].metadata["target_in_vocabulary"] is True


def test_request_rebuilds_the_authors_text():
    dataset = RATS40KConnector().convert(_refs())
    tasks = {task.id: task for task in dataset.iter_tasks()}
    for variant in ("uni", "multi"):
        template = request.TEMPLATES[variant]
        # The stored template is the authors' str.format text, placeholders included.
        assert "{our_source}" in template and "{our_observation}" in template
        for source_split, rows in json.loads((_FIXTURES / f"{variant}.json").read_text()).items():
            row = rows["0"]
            observation = row["Observation"]
            # The authors format the published value: a list printed by str(), or the multivariate
            # string as is. The real release spells every value as a float; the fixture has ints.
            if isinstance(observation, list):
                observation = [float(value) for value in observation]
            expected = template.format(our_source=row["Source"], our_observation=observation)
            task = tasks[f"rats40k-{variant}-{source_split.removeprefix('TSAD_')}-0-category"]
            assert request.authors_request(task, dataset.annotations) == expected


def test_template_without_placeholders_is_rejected():
    with pytest.raises(TimeFValidationError, match="one domain clause and one Observation line"):
        request.task_prompt("Here is a time series.\nThought: \\boxed1{{}}\n")


def test_declares_pinned_source_files():
    # The base's download() turns these declarations into refs; pin the declarations themselves.
    assert RATS40KConnector.HF_REPO == "Time-RA/RATs40K"
    assert RATS40KConnector.SOURCE_REVISION == _EXPECTED_REVISION
    assert RATS40KConnector.SOURCE_FILES == {
        "uni": "RATs-Uni-TSImage_Reason.json",
        "multi": "RATs-Multi-TSImage_Reason.json",
    }


@pytest.mark.parametrize(
    ("variants", "message"),
    [(("uni", "uni"), "distinct part references"), (("bi",), "has no source part bi")],
)
def test_conversion_rejects_repeated_or_unknown_variants(variants, message):
    refs = [HubFileRef(_FIXTURES / "uni.json", variant) for variant in variants]
    with pytest.raises(TimeFValidationError, match=message):
        RATS40KConnector().convert(refs)


@pytest.mark.parametrize(
    ("text", "message"),
    [
        # json.load would silently keep only the last duplicate row.
        ('{"TSAD_train": {"0": null, "0": null}, "TSAD_test": {}}', "duplicate key '0'"),
        ('{"TSAD_train": {}, "TSAD_val": {}}', "unknown split 'TSAD_val'"),
        ('{"TSAD_train": {}}', "expected TSAD_train and TSAD_test"),
    ],
)
def test_malformed_split_structure_is_rejected(tmp_path, text, message):
    path = tmp_path / "broken.json"
    path.write_text(text)
    with pytest.raises(TimeFValidationError, match=message):
        list(_source.iter_entries(path))


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
        RATS40KConnector().convert([HubFileRef(path, variant)])
    assert str(path) in str(error.value) and "train, key 0" in str(error.value)
