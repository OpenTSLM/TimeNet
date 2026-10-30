from collections.abc import Mapping
import hashlib
import json
from pathlib import Path
import re
from typing import Any, ClassVar

import pytest

from timenet.dataset.axis import OrdinalAxis
from timenet.errors import TimeFFormatError, TimeFValidationError
from timenet.reader import TimeFReader
from timenet.registry import LocalRegistry
from timenet.types import ClassificationTask, Split, read_value
from timenet_connectors.bases.huggingface import HubFileRef
from timenet_connectors.datasets.time_ra.rats40k import RATS40KConnector, request, rows
from timenet_connectors.datasets.time_ra.rats40k.records import record_metadata
from timenet_connectors.datasets.time_ra.rats40k.release import PINNED_INVENTORY, PinnedInventory, Variant
from timenet_connectors.sources import json_stream


_FIXTURES = Path(__file__).parent / "fixtures"
_EXPECTED_REVISION = "dc968d6a88f396215be50023c93c3928e06096df"
# Marks a field the test removes from a row.
_MISSING = object()
# The fixtures' inventory: one row per split, and the uni test null entry 106.
_FIXTURE_INVENTORY = {
    variant: PinnedInventory(
        rows={Split.TRAIN: 1, Split.TEST: 1},
        null_keys={Split.TRAIN: frozenset(), Split.TEST: frozenset({"106"} if variant is Variant.UNI else ())},
    )
    for variant in Variant
}


class _FixtureConnector(RATS40KConnector):
    """The connector, pinned to the fixtures' inventory instead of the release's."""

    CARD = Path(__file__).parents[1] / "dataset.yaml"
    SOURCE_INVENTORY: ClassVar[Mapping[Variant, PinnedInventory]] = _FIXTURE_INVENTORY


def _refs() -> list[HubFileRef]:
    return [HubFileRef(_FIXTURES / f"{variant}.json", variant) for variant in ("uni", "multi")]


def _row_of(record) -> tuple[str, Split, str]:
    """The source row a Record holds: its variant, split and key."""
    metadata = record_metadata(record)
    return metadata.variant, metadata.split, metadata.key


def _tasks_by_row(dataset) -> dict[tuple[str, Split, str], ClassificationTask]:
    return {_row_of(task.inputs[0]): task for task in dataset.iter_tasks()}


# The whole converted fixture, written out by hand from the fixture files and the authors' prompts,
# not from connector constants. Long texts are pinned by sha256.
_PROMPT_SHA256 = {
    "uni": "e7d360ac08d475f487dd25bf3dcb1be195d2c3c074444f7dc16b2a70b4ebb295",
    "multi": "a6be9753c9d63bc269213cd9dac6c23d410d097a2d9651f2715deba4969d1a71",
}
# The vocabularies are the authors' ANOMALY_DICT keys in code order (prompts/*.py at Time-RA 200b244).
_EXPECTED_DATASET_ANNOTATIONS: dict[str, Any] = {
    "source_repository": "Time-RA/RATs40K",
    "source_revision": _EXPECTED_REVISION,
    "uni_category_vocabulary": [
        "Normal Sequence",
        "Point Anomaly",
        "Periodic Change Anomaly",
        "Trend Change Anomaly",
        "Change Point Anomaly",
        "Distributional Change Anomaly",
        "Amplitude Anomaly",
        "Pattern Change Anomaly",
        "Sparse Anomaly",
        "Repeated Value Anomaly",
        "Sudden Flatline Anomaly",
        "Drift Anomaly",
        "Sudden Spike Anomaly",
        "Continuous Segment Anomaly",
        "Nonlinear Pattern Anomaly",
    ],
    "multi_category_vocabulary": [
        "Normal Sequence",
        "Covariance Structure Anomaly",
        "Temporal Dependency Anomaly",
        "Trend Divergence Anomaly",
        "Local Structural Jump Anomaly",
        "Joint Space Anomaly",
        "Principal Component Space Anomaly",
        "Collinearity Shift Anomaly",
    ],
    "uni_request_template": "sha256:070c3721a9c83c9b35c9727f1787bad3ddf2699938f2a3fdc1cbbe357819cb9d",
    "multi_request_template": "sha256:63b87a7924d44b71347304bb00ebf5839a81f0171e03879cbc705b48e5f2ab5f",
}


_LABEL_DESCRIPTION = "The release's Label, Normal or Anomaly, as published; it disagrees with the target in 1,184 rows."
_ACTION_ID_DESCRIPTION = "The release's ActionID, as published; it doesn't always follow the prompt's numbering."


# What each fixture row publishes, by variant and split. Every fixture row has key "0".
_FIXTURE_ROWS: dict[tuple[str, str], dict[str, Any]] = {
    ("uni", "train"): {
        "domain": "Healthcare-ECG",
        "channels": {"series": [0.123456789, -0.2, 657372000.0]},
        "task_metadata": {"target_in_vocabulary": True},
        "label": "Normal",
        "action_id": 0,
        "target": "Normal Sequence",
        "rationale": "A normal ECG waveform.\nNo abnormal changes — café.",
    },
    ("uni", "test"): {
        "domain": "Server",
        "channels": {"series": [1.0, 8.0]},
        "task_metadata": {"target_in_vocabulary": False, "Rank": ["model-b", "model-a"]},
        "label": "Normal",
        "action_id": 2,
        "target": "Temporal Dependency Anomaly",
        "rationale": "A sudden spike at the second sample.",
    },
    ("multi", "train"): {
        "domain": "IoT-Drinking Water Quality",
        "channels": {"pH": [8.33, 8.527], "Redox": [753.0, 729.281]},
        "task_metadata": {"target_in_vocabulary": True},
        "label": "Anomaly",
        "action_id": 4,
        "target": "Joint Space Anomaly",
        "rationale": "The joint behavior changes.\nKeep this complete explanation.",
    },
    ("multi", "test"): {
        "domain": "Medical-ECG",
        "channels": {"ECG1": [0.39, 0.46, 0.45], "ECG2": [0.22, 0.22, 0.2]},
        "task_metadata": {"target_in_vocabulary": True},
        "label": "Normal",
        "action_id": 0,
        "target": "Normal Sequence",
        "rationale": "The ECG channels behave normally.",
    },
}


def _expected_output(variant: str, split: str) -> tuple[dict, dict]:
    """The Record and the task one fixture row becomes, as ``_output`` writes them."""
    row = _FIXTURE_ROWS[variant, split]
    figures = "figures" if variant == "uni" else "figures_multi"
    record = {
        "metadata": {"variant": variant, "split": split, "key": "0"},
        "annotations": {"source_domain": row["domain"]},
        "source": row["domain"],
        "source_metadata": {"channel_order": list(row["channels"])},
        "signals": {
            name: (values, "float64", OrdinalAxis, "rats40k_series", None) for name, values in row["channels"].items()
        },
    }
    task = {
        "type": ClassificationTask,
        "inputs": 1,
        "prompt": f"sha256:{_PROMPT_SHA256[variant]}",
        "input_annotations": {"source_domain": row["domain"]},
        "targets": (row["target"],),
        "target_schema": f"{variant}_category_vocabulary",
        "rationale": row["rationale"],
        "metadata": row["task_metadata"] | {"figure_path": f"{figures}/{split}/0.jpg"},
        "annotations": {
            "label": (str, row["label"], _LABEL_DESCRIPTION),
            "action_id": (int, row["action_id"], _ACTION_ID_DESCRIPTION),
        },
        "split": split,
    }
    return record, task


def _plain(value):
    """Read a text value, pin a long text by its sha256, and keep anything else as is."""
    value = read_value(value)
    if isinstance(value, str) and len(value) > 1000:
        return f"sha256:{hashlib.sha256(value.encode()).hexdigest()}"
    return value


def _row_key(record) -> tuple[str, str, str]:
    variant, split, key = _row_of(record)
    return str(variant), split.value, key


def _output(dataset) -> dict:
    """Everything the dataset holds, as plain values, with records and tasks keyed by source row."""
    records, tasks = {}, {}
    for record in dataset.records:
        (source,) = record.sources
        records[_row_key(record)] = {
            "metadata": dict(record.metadata),
            "annotations": {a.key: a.value for a in record.annotations},
            "source": source.name,
            "source_metadata": dict(source.metadata),
            "signals": {
                s.name: (
                    s.to_numpy().tolist(),
                    str(s.to_numpy().dtype),
                    type(s.time_axis),
                    s.spec.spec_type,
                    s.spec.unit_value,
                )
                for s in record.signals
            },
        }
    for task in dataset.iter_tasks():
        tasks[_row_key(task.inputs[0])] = {
            "type": type(task),
            "inputs": len(task.inputs),
            "prompt": _plain(task.prompt),
            "input_annotations": {a.key: a.value for a in task.input_annotations},
            "targets": task.targets,
            "target_schema": task.target_schema,
            "rationale": task.rationale,
            "metadata": dict(task.metadata),
            # The type too, since 0 == 0.0: a code must come back as an int.
            "annotations": {a.key: (type(_plain(a.value)), _plain(a.value), a.description) for a in task.annotations},
            "split": task.split.value,
        }
    annotations = {a.key: _plain(a.value) for a in dataset.annotations}
    return {"annotations": annotations, "records": records, "tasks": tasks}


def test_storage_round_trip_keeps_the_whole_conversion(monkeypatch, tmp_path):
    # The lazy loaders seek by byte offset, including across CRLF line ends, UTF-8 text and small reads.
    refs = []
    for ref in _refs():
        path = tmp_path / ref.path.name
        path.write_bytes(ref.path.read_bytes().replace(b"\n", b"\r\n"))
        refs.append(HubFileRef(path, ref.part))
    monkeypatch.setattr(json_stream, "_CHUNK_SIZE", 17)
    original = _FixtureConnector().convert(refs)
    registry = LocalRegistry(tmp_path / "registry")
    registry.store(original)
    with TimeFReader(registry.open_version("time-ra/rats40k", "1.0.0")) as reader:
        restored = reader.read()
    expected_rows = {(variant, split, "0"): _expected_output(variant, split) for variant, split in _FIXTURE_ROWS}
    expected = {
        "annotations": _EXPECTED_DATASET_ANNOTATIONS,
        "records": {row: record for row, (record, _) in expected_rows.items()},
        "tasks": {row: task for row, (_, task) in expected_rows.items()},
    }
    # The writer drew every lazy signal and the tasks already, so this pass also checks they load again.
    assert _output(original) == expected
    assert _output(restored) == expected


def test_request_rebuilds_the_authors_text():
    dataset = _FixtureConnector().convert(_refs())
    tasks = _tasks_by_row(dataset)
    for variant in Variant:
        template = request.TEMPLATES[variant]
        # The stored template is the authors' str.format text, placeholders included.
        assert "{our_source}" in template and "{our_observation}" in template
        for source_split, entries in json.loads((_FIXTURES / f"{variant}.json").read_text()).items():
            row = entries["0"]
            # The authors format the published value: a list printed by str(), or the multivariate
            # string as is.
            expected = template.format(our_source=row["Source"], our_observation=row["Observation"])
            task = tasks[variant, Split(source_split.removeprefix("TSAD_")), "0"]
            assert request.authors_request(task, dataset.annotations) == expected


def _authors_code(response: str, vocabulary: list[str]) -> int | None:
    """Parse an answer's category code as the authors' evaluation does, or None where it skips the row.

    Ported line for line from eval_anomaly_multiclf_reason.py, lines 161-172, at Time-RA 200b244; the
    multivariate script has the same lines. ANOMALY_DICT maps each name to its code, its prompt number.
    """
    anomaly_dict = {name: (code,) for code, name in enumerate(vocabulary)}
    action_match = re.search(r"Action:\s*(.*)", response)
    action = action_match.group(1).strip() if action_match else ""
    action = re.sub(r"(?:\\boxed\d+{)?\d*\.?\s*(.*?)}?$", r"\1", action).strip()
    action = re.sub(r"\\boxed\{(.+)$", r"\1", action)
    if action in anomaly_dict:
        return anomaly_dict[action][0]
    # The script skips the row, so its prediction file has no entry and the row is never scored.
    return None


def test_targets_fit_the_authors_answer_format():
    for variant in Variant:
        vocabulary = _EXPECTED_DATASET_ANNOTATIONS[f"{variant}_category_vocabulary"]
        for code, name in enumerate(vocabulary):
            # A name in \boxed2{}, with or without the prompt's number, parses to its code.
            for answer in (f"Action: \\boxed2{{{name}}}", f"Action: \\boxed2{{{code}. {name}}}"):
                assert _authors_code(f"Thought: \\boxed1{{x}}\n{answer}", vocabulary) == code
    dataset = _FixtureConnector().convert(_refs())
    for (variant, _, _), task in _tasks_by_row(dataset).items():
        vocabulary = _EXPECTED_DATASET_ANNOTATIONS[f"{variant}_category_vocabulary"]
        answer = request.authors_answer(task)
        assert task.targets is not None
        (target,) = task.targets
        rationale, name = read_value(task.rationale), read_value(target)
        # The authors' SFT target: train/train_sft_parser.py, line 76, at Time-RA 200b244.
        assert answer == f"Thought: \\boxed1{{{rationale}}}\nAction: \\boxed2{{{name}}}"
        # The uni test fixture's multivariate name is skipped, as the authors' parser skips the 3,790 real ones.
        expected = vocabulary.index(name) if task.metadata["target_in_vocabulary"] else None
        assert _authors_code(answer, vocabulary) == expected


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
    # Literal pins, so moving to another revision is a deliberate test edit.
    assert RATS40KConnector.SOURCE_INVENTORY is PINNED_INVENTORY
    assert {variant: dict(pinned.rows) for variant, pinned in PINNED_INVENTORY.items()} == {
        Variant.UNI: {Split.TRAIN: 30266, Split.TEST: 6034},
        Variant.MULTI: {Split.TRAIN: 2728, Split.TEST: 546},
    }
    assert PINNED_INVENTORY[Variant.UNI].null_keys[Split.TEST] == {
        "106", "386", "560", "875", "1632", "2081", "2581", "2644", "2721", "3584", "5558"
    }  # fmt: skip


@pytest.mark.parametrize(
    ("text", "message"),
    [
        ('{"TSAD_train": {}, "TSAD_val": {}}', "unknown split 'TSAD_val'"),
        ('{"TSAD_train": {}}', "expected TSAD_train and TSAD_test"),
    ],
)
def test_malformed_split_structure_is_rejected(tmp_path, text, message):
    path = tmp_path / "broken.json"
    path.write_text(text)
    with pytest.raises(TimeFValidationError, match=message):
        list(rows.iter_entries(path, Variant.UNI))


def test_a_row_is_reread_once_for_all_its_channels(monkeypatch):
    reads = []
    read_entry = rows.read_entry
    monkeypatch.setattr(rows, "read_entry", lambda *args: reads.append(args) or read_entry(*args))
    dataset = _FixtureConnector().convert(_refs())
    signals = [signal for record in dataset.records for signal in record.signals]
    for signal in signals:
        signal.to_numpy()
    # Six signals over four rows: each multivariate row's two channels share one reread.
    assert (len(signals), len(reads)) == (6, 4)


def test_a_reread_that_no_longer_matches_its_row_fails(tmp_path):
    path = tmp_path / "uni.json"
    path.write_bytes((_FIXTURES / "uni.json").read_bytes())
    dataset = _FixtureConnector().convert([HubFileRef(path, "uni")])
    # Same byte length, so the range still holds one JSON value, but now with four values, not three.
    text = path.read_text()
    path.write_text(text.replace("657372000.0]", "6573, 72000]"))
    record = next(record for record in dataset.records if _row_of(record) == ("uni", Split.TRAIN, "0"))
    with pytest.raises(TimeFFormatError, match=r"train, key 0.*channels differ from the converted row"):
        record.signals[0].to_numpy()


def test_pinned_inventory_mismatch_fails():
    # The release's inventory, against the fixtures.
    with pytest.raises(TimeFValidationError, match="pinned row counts or null entry keys differ"):
        RATS40KConnector().convert(_refs())


@pytest.mark.parametrize(
    ("variant", "field", "value", "message"),
    [
        ("multi", "Observation", '"a": [1]; "a": [2]', "duplicate channel"),
        ("multi", "Observation", '"a": [1]; "b": [2,3]', "lengths differ"),
        # Python-written JSON can contain NaN, which json accepts by default.
        ("uni", "Observation", [1.0, float("nan")], r"(?s)Observation\.series\.1.*finite number"),
        # A JSON true is an int to Python, so the code must reject it explicitly.
        ("uni", "ActionID", True, r"(?s)ActionID.*valid integer"),
        # Observation reaches the row through its alias and the channel validator.
        ("uni", "Observation", _MISSING, r"(?s)Observation.*Field required"),
        # A field the release doesn't have is rejected, not dropped.
        ("uni", "Index", 0, r"(?s)Index.*Extra inputs are not permitted"),
        # The row's reference comes from where it was read, never from the JSON.
        ("uni", "ref", "x", "a published row has no 'ref' key"),
        # Only explicit nulls are skipped; any other non-object row fails.
        ("uni", None, "x", "expected a row object"),
    ],
)
def test_malformed_rows_have_source_context(tmp_path, variant, field, value, message):
    data = json.loads((_FIXTURES / f"{variant}.json").read_text())
    if field is None:
        data["TSAD_train"]["0"] = value
    elif value is _MISSING:
        del data["TSAD_train"]["0"][field]
    else:
        data["TSAD_train"]["0"][field] = value
    path = tmp_path / "broken.json"
    path.write_text(json.dumps(data))
    with pytest.raises(TimeFValidationError, match=message) as error:
        _FixtureConnector().convert([HubFileRef(path, variant)])
    assert str(path) in str(error.value) and "train, key 0" in str(error.value)
