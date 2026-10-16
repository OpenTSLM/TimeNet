import json
from pathlib import Path

import numpy as np
import pytest

from timenet.config import settings
from timenet.dataset.axis import OrdinalAxis
from timenet.engine import run_pipeline
from timenet.errors import TimeFValidationError
from timenet.format import control_writer
from timenet.reader import TimeFReader
from timenet.registry import DatasetVersion
from timenet.types import Access, AnswerTask, ClassificationTask, Split
from timenet_connectors.datasets.time_ra.rats40k import RATS40KConnector, _json, connector as module
from timenet_connectors.discovery import resolve


_FIXTURES = Path(__file__).parent / "fixtures"
_EXPECTED_REVISION = "dc968d6a88f396215be50023c93c3928e06096df"


def _refs() -> list[module._FileRef]:
    return [{"path": str(_FIXTURES / f"{variant}.json"), "variant": variant} for variant in ("uni", "multi")]


def _annotations(owner):
    return {annotation.key: annotation.value for annotation in owner.annotations}


def _assert_mapping(dataset):
    records, tasks = dataset.records, dataset.tasks
    vocabularies = {annotation.key: annotation for annotation in dataset.annotations}
    assert len(records) == 4
    assert len(tasks) == 12
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
    records_by_id = {record.id: record for record in records}
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
            assert signal.spec.dtype == "float64"
            values = signal.to_numpy()
            assert values.dtype == np.dtype("float64")
            assert isinstance(signal.time_axis, OrdinalAxis)
            np.testing.assert_array_equal(values, np.asarray(channels[signal.name], dtype=np.float64))
        provenance = _annotations(record)
        assert provenance["source_key"] == "0"
        assert provenance["figure_path"] == row["FigurePath"]
        assert not {"action_id", "ActionID", "Action", "Label", "Thought"} & provenance.keys()
        assert provenance["source_split"] == f"TSAD_{split}"
        detection, category, reasoning = (
            tasks_by_id[f"{record.id}-{suffix}"] for suffix in ("detection", "type", "reasoning")
        )
        assert type(detection) is type(category) is ClassificationTask
        assert detection.targets == (row["Label"],)
        assert category.targets == (str(row["ActionID"]),)
        assert type(reasoning) is AnswerTask
        assert reasoning.targets == (row["Thought"],)
        assert category.rationale is None
        assert detection.target_schema == vocabularies["label_vocabulary"].id
        assert category.target_schema == vocabularies[f"{variant}_action_vocabulary"].id
        assert category.targets[0] in vocabularies[f"{variant}_action_vocabulary"].value
        assert category.metadata == {"Action": row["Action"], **({"Rank": row["Rank"]} if "Rank" in row else {})}
        for task in (detection, category, reasoning):
            assert task.inputs == (record,) and task.inputs[0] is record
            assert task.prompt is None
            assert {a.key: a.value for a in task.input_annotations} == {"source_domain": row["Source"]}
            assert task.split is split


# Download and discovery


def test_discovery_and_pinned_authenticated_download(monkeypatch, tmp_path):
    import huggingface_hub  # noqa: PLC0415

    calls = []

    def snapshot(repo_id, **kwargs):
        calls.append((repo_id, kwargs))
        return str(tmp_path)

    monkeypatch.setattr(huggingface_hub, "snapshot_download", snapshot)
    assert resolve("time-ra/rats40k") is RATS40KConnector
    assert RATS40KConnector().metadata().access is Access.CREDENTIALED
    refs = RATS40KConnector().download(tmp_path)
    assert calls == [
        (
            "Time-RA/RATs40K",
            {
                "revision": _EXPECTED_REVISION,
                "repo_type": "dataset",
                "cache_dir": str(tmp_path),
                "allow_patterns": ["RATs-Uni-TSImage_Reason.json", "RATs-Multi-TSImage_Reason.json"],
            },
        )
    ]
    assert refs == [
        {"path": str(tmp_path / filename), "variant": variant, "check_inventory": True}
        for variant, filename in (
            ("uni", "RATs-Uni-TSImage_Reason.json"),
            ("multi", "RATs-Multi-TSImage_Reason.json"),
        )
    ]


# Source structure and row validation


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


def test_pinned_inventory_mismatch_fails():
    refs: list[module._FileRef] = [{**ref, "check_inventory": True} for ref in _refs()]
    with pytest.raises(TimeFValidationError, match="pinned row counts or null entry keys differ"):
        RATS40KConnector().convert(refs)


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


# Conversion and category mapping


def test_category_names_preserve_conflicts_and_variant_scope(tmp_path):
    refs = []
    for variant in ("uni", "multi"):
        data = json.loads((_FIXTURES / f"{variant}.json").read_text())
        for rows in data.values():
            for row in rows.values():
                if row is not None:
                    row["ActionID"] = 2
        # Two names for the same label within one split must not overwrite each other.
        data["TSAD_train"]["1"] = {**data["TSAD_train"]["0"], "Action": "Another published name"}
        data["TSAD_train"]["2"] = {**data["TSAD_train"]["0"], "ActionID": 10}
        path = tmp_path / f"{variant}.json"
        path.write_text(json.dumps(data))
        refs.append({"path": str(path), "variant": variant})
    dataset = RATS40KConnector().convert(refs)
    vocabularies = {a.key: a for a in dataset.annotations if a.key.endswith("action_vocabulary")}
    for variant, original_name in (("uni", "Normal Sequence"), ("multi", "Joint Space Anomaly")):
        vocabulary = vocabularies[f"{variant}_action_vocabulary"]
        # Sort class codes numerically before converting them to strings.
        assert vocabulary.value == ["2", "10"]
        names = vocabulary.metadata["observed_names"]
        assert isinstance(names, list)
        assert names[1] == {
            "split": "train",
            "action_id": 2,
            "actions": ["Another published name", original_name],
        }
        categories = [
            task
            for task in dataset.iter_tasks()
            if isinstance(task, ClassificationTask)
            and task.id.startswith(f"rats40k-{variant}-")
            and task.id.endswith("-type")
        ]
        assert len(categories) == 4
        assert {task.targets for task in categories} == {("2",), ("10",)}
        assert all(task.target_schema == vocabulary.id for task in categories)


# End-to-end build and reload


def test_real_pipeline_roundtrip_and_cache_cleanup(monkeypatch, tmp_path):
    monkeypatch.setenv("TIMENET_CACHE", str(tmp_path / "cache"))
    monkeypatch.setattr(control_writer, "_TASK_BATCH_SIZE", 5)
    original_convert = RATS40KConnector.convert

    def convert(self, refs):
        dataset = original_convert(self, refs)
        # A preview must not consume the task stream used by the writer.
        next(dataset.iter_tasks())
        return dataset

    def download(self, cache_dir):
        assert cache_dir == settings().cache_dir / "time-ra/rats40k"
        refs = []
        for ref in _refs():
            path = cache_dir / Path(ref["path"]).name
            # CRLF and non-ASCII data exercise byte offsets across chunk boundaries.
            path.write_bytes(Path(ref["path"]).read_bytes().replace(b"\n", b"\r\n"))
            refs.append({**ref, "path": str(path)})
        return refs

    monkeypatch.setattr(RATS40KConnector, "download", download)
    monkeypatch.setattr(RATS40KConnector, "convert", convert)
    monkeypatch.setattr(_json, "_CHUNK_SIZE", 17)
    version_dir = run_pipeline(RATS40KConnector(), tmp_path / "registry")
    assert not (settings().cache_dir / "time-ra/rats40k").exists()
    with TimeFReader(DatasetVersion.open_local(version_dir)) as reader:
        restored = reader.read()
        _assert_mapping(restored)
        provenance = _annotations(restored)
        assert provenance["source_revision"] == _EXPECTED_REVISION
        assert provenance["skipped_null_entries"] == 1
        assert provenance["label_action_disagreements"] == 1
        assert provenance["label_vocabulary"] == ["Normal", "Anomaly"]
        vocabularies = {a.key: a for a in restored.annotations if a.key.endswith("action_vocabulary")}
        assert vocabularies["uni_action_vocabulary"].value == ["0", "12"]
        assert vocabularies["multi_action_vocabulary"].value == ["0", "4"]
        assert vocabularies["uni_action_vocabulary"].metadata["observed_names"] == [
            {"split": "test", "action_id": 12, "actions": ["Sudden Spike Anomaly"]},
            {"split": "train", "action_id": 0, "actions": ["Normal Sequence"]},
        ]
        assert set(restored.derive_schema().tasks) == {ClassificationTask, AnswerTask}
        reader.verify()
