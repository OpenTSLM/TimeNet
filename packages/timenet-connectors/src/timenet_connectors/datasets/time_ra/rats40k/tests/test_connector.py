from collections.abc import Iterator, Mapping
import hashlib
import json
from pathlib import Path
import re
from typing import Any, ClassVar, cast

import pytest

from timenet.dataset import Record, TimeFDataset
from timenet.dataset.axis import OrdinalAxis
from timenet.errors import TimeFFormatError, TimeFValidationError
from timenet.reader import TimeFReader
from timenet.registry import LocalRegistry
from timenet.types import ClassificationTask, Label, LabelSet, Split, read_value
from timenet_connectors.bases.huggingface import HubFileRef
from timenet_connectors.datasets.time_ra.rats40k import RATS40KConnector, authors_answer, connector
from timenet_connectors.datasets.time_ra.rats40k.connector import SourceInventory
from timenet_connectors.sources import json_stream


_FIXTURES = Path(__file__).parent / "fixtures"
_REVISION = "dc968d6a88f396215be50023c93c3928e06096df"
# Marks a field the test removes from a row.
_MISSING = object()
_ROW_KEY = tuple[str, str, str]
"""A row's variant, split and key."""


def _inventory(train: int, test: int, null_test_keys: frozenset[str] = frozenset()) -> SourceInventory:
    return SourceInventory(
        rows={Split.TRAIN: train, Split.TEST: test},
        null_keys={Split.TRAIN: frozenset(), Split.TEST: null_test_keys},
    )


# The fixtures: one row per split, and the null uni test entry 106.
_FIXTURE_INVENTORY = {"uni": _inventory(1, 1, frozenset({"106"})), "multi": _inventory(1, 1)}


class _FixtureConnector(RATS40KConnector):
    """The connector with the card of this dataset, which the base reads, and the fixtures' inventory."""

    CARD = Path(__file__).parents[1] / "dataset.yaml"
    SOURCE_INVENTORY: ClassVar[Mapping[str, SourceInventory]] = _FIXTURE_INVENTORY


def _refs(directory: Path = _FIXTURES) -> list[HubFileRef]:
    return [HubFileRef(directory / f"{variant}.json", variant) for variant in ("uni", "multi")]


def _row_of(record: Record) -> _ROW_KEY:
    metadata = record.metadata
    return str(metadata["variant"]), str(metadata["split"]), str(metadata["key"])


def _plain(value: Any) -> Any:
    """Read a text value, and pin a long text by its sha256."""
    value = read_value(value)
    if isinstance(value, str) and len(value) > 1000:
        return f"sha256:{hashlib.sha256(value.encode()).hexdigest()}"
    return value


# The whole converted fixture, written out by hand from the fixture files and the authors' prompts,
# not from connector constants. Long texts are pinned by sha256.
_PROMPT_SHA256 = {
    "uni": "e7d360ac08d475f487dd25bf3dcb1be195d2c3c074444f7dc16b2a70b4ebb295",
    "multi": "a6be9753c9d63bc269213cd9dac6c23d410d097a2d9651f2715deba4969d1a71",
}
_EXPECTED_DATASET_ANNOTATIONS: dict[str, Any] = {
    "source_repository": "Time-RA/RATs40K",
    "source_revision": _REVISION,
    "uni_request_template": "sha256:070c3721a9c83c9b35c9727f1787bad3ddf2699938f2a3fdc1cbbe357819cb9d",
    "multi_request_template": "sha256:63b87a7924d44b71347304bb00ebf5839a81f0171e03879cbc705b48e5f2ab5f",
}
# The prompt categories are the authors' ANOMALY_DICT keys in code order (prompts/*.py at Time-RA 200b244).
_PROMPT_CATEGORIES: dict[str, list[str]] = {
    "uni": [
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
    "multi": [
        "Normal Sequence",
        "Covariance Structure Anomaly",
        "Temporal Dependency Anomaly",
        "Trend Divergence Anomaly",
        "Local Structural Jump Anomaly",
        "Joint Space Anomaly",
        "Principal Component Space Anomaly",
        "Collinearity Shift Anomaly",
    ],
}
_OUTSIDE_PROMPT = "A published target name that is not one of this variant's prompt categories."
# Each label set: the prompt categories, then the other target names of the fixture rows, with a
# description. Only the uni test row's target is outside its prompt.
_EXPECTED_LABEL_SETS: dict[str, tuple[str, tuple[tuple[str, str | None], ...]]] = {
    variant: (
        f"{variant}_category",
        tuple((name, None) for name in categories)
        + ((("Temporal Dependency Anomaly", _OUTSIDE_PROMPT),) if variant == "uni" else ()),
    )
    for variant, categories in _PROMPT_CATEGORIES.items()
}
_LABEL_DESCRIPTION = "The release's Label, Normal or Anomaly, as published. It can disagree with the target."
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
        "task_metadata": {"target_in_vocabulary": False, "rank": ["model-b", "model-a"]},
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
        "label_set": _EXPECTED_LABEL_SETS[variant],
        "rationale": row["rationale"],
        "metadata": row["task_metadata"] | {"figure_path": f"{figures}/{split}/0.jpg"},
        "annotations": {
            "label": (str, row["label"], _LABEL_DESCRIPTION),
            "action_id": (int, row["action_id"], _ACTION_ID_DESCRIPTION),
        },
        "split": split,
    }
    return record, task


def _output(dataset: TimeFDataset) -> dict:
    """Everything the dataset holds, as plain values, with records and tasks keyed by source row."""
    records, tasks = {}, {}
    for record in dataset.records:
        (source,) = record.sources
        records[_row_of(record)] = {
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
    for task in cast(Iterator[Any], dataset.iter_tasks()):
        tasks[_row_of(task.inputs[0])] = {
            "type": type(task),
            "inputs": len(task.inputs),
            "prompt": _plain(task.prompt),
            "input_annotations": {a.key: a.value for a in task.input_annotations},
            "targets": task.targets,
            "label_set": (
                task.label_set.name,
                tuple((label.name, label.description) for label in task.label_set.labels),
            ),
            "rationale": task.rationale,
            "metadata": dict(task.metadata),
            # The type too, since 0 == 0.0: a code must come back as an int.
            "annotations": {a.key: (type(_plain(a.value)), _plain(a.value), a.description) for a in task.annotations},
            "split": str(task.split),
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


def test_declares_the_hub_files():
    assert RATS40KConnector.SOURCE_FILES == {
        "uni": "RATs-Uni-TSImage_Reason.json",
        "multi": "RATs-Multi-TSImage_Reason.json",
    }


def test_the_dataset_has_each_annotation_and_task_once():
    dataset = _FixtureConnector().convert(_refs())
    assert [a.key for a in dataset.annotations] == [
        "source_repository",
        "source_revision",
        "uni_request_template",
        "multi_request_template",
    ]
    assert sum(1 for _ in dataset.iter_tasks()) == 4


def test_the_published_rows_can_be_rebuilt_from_the_dataset():
    # Rebuild each row's observation from its stored signals, and its full prompt from the stored
    # template and domain, and compare them with the published text.
    dataset = _FixtureConnector().convert(_refs())
    annotations: dict[str, Any] = {a.key: read_value(a.value) for a in dataset.annotations}
    rebuilt = {}
    for task in dataset.iter_tasks():
        (record,) = task.inputs
        variant, split, key = _row_of(record)
        values: dict[str, Any] = {signal.name: signal.to_numpy().tolist() for signal in record.signals}
        order = cast(list[str], record.sources[0].metadata["channel_order"])
        if variant == "uni":
            observation = str(values["series"])
        else:
            observation = "; ".join(f"{json.dumps(name)}: {values[name]}" for name in order)
        (domain,) = (read_value(a.value) for a in task.input_annotations)
        rebuilt[variant, split, key] = annotations[f"{variant}_request_template"].format(
            our_source=domain, our_observation=observation
        )
    for variant in ("uni", "multi"):
        template = (Path(connector.__file__).parent / "prompts" / f"{variant}.txt").read_text(encoding="utf-8")
        published = json.loads((_FIXTURES / f"{variant}.json").read_text(encoding="utf-8"))
        for source_split, entries in published.items():
            row = entries["0"]
            # The authors fill the template with the published value: a list printed by str(), or the
            # multivariate string as is.
            expected = template.format(our_source=row["Source"], our_observation=row["Observation"])
            assert rebuilt[variant, source_split.removeprefix("TSAD_"), "0"] == expected


def test_task_prompt_is_the_template_without_what_the_inputs_supply():
    for variant in ("uni", "multi"):
        template = (Path(connector.__file__).parent / "prompts" / f"{variant}.txt").read_text(encoding="utf-8")
        task_prompt = connector._prompts(connector._Variant(variant)).task_prompt
        assert "{our_source}" in template
        assert "{our_observation}" in template
        # The domain clause and the observation line are gone, and no placeholder is left.
        assert "our_source" not in task_prompt
        assert "our_observation" not in task_prompt
        assert "oberservation" not in task_prompt
        assert not task_prompt.endswith("\n")


def test_a_prompt_that_defines_no_category_is_rejected(monkeypatch, tmp_path):
    (tmp_path / "uni.txt").write_text("Name the anomaly.\nObservation: {our_observation}\n", encoding="utf-8")
    monkeypatch.setattr(connector, "_PROMPTS", tmp_path)
    connector._prompts.cache_clear()
    try:
        with pytest.raises(TimeFValidationError, match="defines no category"):
            connector._prompts(connector._Variant.UNI)
    finally:
        connector._prompts.cache_clear()


def test_a_prompt_that_does_not_ask_for_the_answer_fields_is_rejected(monkeypatch, tmp_path):
    prompt = "1. Normal Sequence: no anomaly.\nObservation: {our_observation}\nAnswer: \\boxed{{}}\n"
    (tmp_path / "uni.txt").write_text(prompt, encoding="utf-8")
    monkeypatch.setattr(connector, "_PROMPTS", tmp_path)
    connector._prompts.cache_clear()
    try:
        with pytest.raises(TimeFValidationError, match="does not end with the answer fields"):
            connector._prompts(connector._Variant.UNI)
    finally:
        connector._prompts.cache_clear()


def _authors_code(response: str, categories: list[str]) -> int | None:
    """Parse an answer's category code as the authors' evaluation does, or None where it skips the row.

    Ported line for line from eval_anomaly_multiclf_reason.py, lines 161-172, at Time-RA 200b244. The
    multivariate script has the same lines. ANOMALY_DICT maps each name to its code, its prompt number.
    """
    anomaly_dict = {name: (code,) for code, name in enumerate(categories)}
    action_match = re.search(r"Action:\s*(.*)", response)
    action = action_match.group(1).strip() if action_match else ""
    action = re.sub(r"(?:\\boxed\d+{)?\d*\.?\s*(.*?)}?$", r"\1", action).strip()
    action = re.sub(r"\\boxed\{(.+)$", r"\1", action)
    if action in anomaly_dict:
        return anomaly_dict[action][0]
    # The script skips the row, so its prediction file has no entry and the row is never scored.
    return None


def test_the_ported_parser_reads_every_category_name():
    for categories in _PROMPT_CATEGORIES.values():
        for code, name in enumerate(categories):
            # A name in \boxed2{}, with or without the prompt's number, parses to its code.
            for answer in (f"Action: \\boxed2{{{name}}}", f"Action: \\boxed2{{{code}. {name}}}"):
                assert _authors_code(f"Thought: \\boxed1{{x}}\n{answer}", categories) == code


# Each fixture row's answer, written out by hand. It is also the authors' SFT target
# (train/train_sft_parser.py, line 76, at Time-RA 200b244). Next to it, the code that the authors'
# parser reads from it.
_EXPECTED_ANSWERS: dict[_ROW_KEY, tuple[str, int | None]] = {
    ("uni", "train", "0"): (
        "Thought: \\boxed1{A normal ECG waveform.\nNo abnormal changes — café.}\nAction: \\boxed2{Normal Sequence}",
        0,
    ),
    # A multivariate name in the univariate test split: the authors' parser skips it.
    ("uni", "test", "0"): (
        "Thought: \\boxed1{A sudden spike at the second sample.}\nAction: \\boxed2{Temporal Dependency Anomaly}",
        None,
    ),
    ("multi", "train", "0"): (
        "Thought: \\boxed1{The joint behavior changes.\nKeep this complete explanation.}\n"
        "Action: \\boxed2{Joint Space Anomaly}",
        5,
    ),
    ("multi", "test", "0"): (
        "Thought: \\boxed1{The ECG channels behave normally.}\nAction: \\boxed2{Normal Sequence}",
        0,
    ),
}


def test_each_task_gives_the_answer_that_its_prompt_asks_for(tmp_path):
    registry = LocalRegistry(tmp_path / "registry")
    registry.store(_FixtureConnector().convert(_refs()))
    with TimeFReader(registry.open_version("time-ra/rats40k", "1.0.0")) as reader:
        tasks = {_row_of(task.inputs[0]): task for task in reader.read().iter_tasks()}
    assert tasks.keys() == _EXPECTED_ANSWERS.keys()
    for (variant, split, key), task in tasks.items():
        answer, code = _EXPECTED_ANSWERS[variant, split, key]
        # The prompt ends with the two fields empty, and the answer fills them.
        prompt = read_value(task.prompt)
        assert isinstance(prompt, str)
        assert prompt.splitlines()[-2:] == ["Thought: \\boxed1{}", "Action: \\boxed2{}"]
        assert authors_answer(task) == answer
        assert _authors_code(answer, _PROMPT_CATEGORIES[variant]) == code


def test_a_task_without_a_rationale_has_no_answer():
    record = next(iter(_FixtureConnector().convert(_refs()).records))
    labels = LabelSet(name="uni_category", labels=(Label(name="Normal Sequence"),))
    task = ClassificationTask(inputs=(record,), targets=("Normal Sequence",), label_set=labels)
    with pytest.raises(TimeFValidationError, match="needs a rationale and exactly one target"):
        authors_answer(task)


def test_every_value_of_unusual_rows_reaches_its_signal(monkeypatch, tmp_path):
    # Names with ';', ':' and quotes, integers, exponents, extra whitespace, a null Rank and a null entry.
    def row(observation, rank=_MISSING):
        published = {
            "Observation": observation,
            "Source": "IoT;Mixed",
            "FigurePath": "figures_multi/test/5.jpg",
            "Thought": "Because.",
            "Action": "Normal Sequence",
            "ActionID": 0,
            "Label": "Normal",
        }
        if rank is not _MISSING:
            published["Rank"] = rank
        return published

    uni = {"TSAD_train": {}, "TSAD_test": {"1": row([1, -2, 3.5, 1e-07, 123456789012.5], rank=None), "2": None}}
    multi = {
        "TSAD_train": {"5": row('"a;b: c": [1, 2, 3]; "é\\"x": [3.5, 4, 5];"z": [0, 0, 1e-3]')},
        "TSAD_test": {"0": row('  "only": [1.0, 2.0]  ', rank=["x"])},
    }
    for name, data in (("uni", uni), ("multi", multi)):
        (tmp_path / f"{name}.json").write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
    inventory = {"uni": _inventory(0, 1, frozenset({"2"})), "multi": _inventory(1, 1)}
    monkeypatch.setattr(_FixtureConnector, "SOURCE_INVENTORY", inventory)
    dataset = _FixtureConnector().convert(_refs(tmp_path))
    signals = {_row_of(r): {s.name: s.to_numpy().tolist() for s in r.signals} for r in dataset.records}
    assert signals == {
        ("uni", "test", "1"): {"series": [1.0, -2.0, 3.5, 1e-07, 123456789012.5]},
        ("multi", "train", "5"): {"a;b: c": [1.0, 2.0, 3.0], 'é"x': [3.5, 4.0, 5.0], "z": [0.0, 0.0, 0.001]},
        ("multi", "test", "0"): {"only": [1.0, 2.0]},
    }
    records = {_row_of(r): r for r in dataset.records}
    assert records["multi", "train", "5"].sources[0].metadata["channel_order"] == ["a;b: c", 'é"x', "z"]
    tasks = {_row_of(t.inputs[0]): t for t in dataset.iter_tasks()}
    # A null Rank is no rank, and the null entry makes no row.
    assert "rank" not in tasks["uni", "test", "1"].metadata
    assert tasks["multi", "test", "0"].metadata["rank"] == ["x"]
    assert sum(1 for _ in dataset.iter_tasks()) == 3


def test_the_pinned_inventory_is_the_releases():
    # The counts of the dataset card's description: 32,994 train and 6,580 test rows.
    inventory = RATS40KConnector.SOURCE_INVENTORY
    assert sum(pinned.rows[Split.TRAIN] for pinned in inventory.values()) == 32_994
    assert sum(pinned.rows[Split.TEST] for pinned in inventory.values()) == 6_580
    assert {variant: len(pinned.null_keys[Split.TEST]) for variant, pinned in inventory.items()} == {
        "uni": 11,
        "multi": 0,
    }


@pytest.mark.parametrize(
    ("inventory", "message"),
    [
        # Another row count.
        (_inventory(2, 1, frozenset({"106"})), r"expected test: 1 rows, null keys \['106'\], train: 2 rows"),
        # The same counts, but another null key.
        (_inventory(1, 1, frozenset({"999"})), r"found test: 1 rows, null keys \['106'\]"),
    ],
)
def test_files_that_differ_from_the_pinned_inventory_fail(monkeypatch, inventory, message):
    monkeypatch.setattr(_FixtureConnector, "SOURCE_INVENTORY", {**_FIXTURE_INVENTORY, "uni": inventory})
    with pytest.raises(
        TimeFValidationError, match=rf"uni\.json: the rows differ from the pinned inventory: .*{message}"
    ):
        _FixtureConnector().convert(_refs())


def test_a_row_is_reread_once_for_all_its_channels(monkeypatch):
    reads = []
    read_entry = connector.read_entry
    monkeypatch.setattr(connector, "read_entry", lambda *args: reads.append(args) or read_entry(*args))
    dataset = _FixtureConnector().convert(_refs())
    signals = [signal for record in dataset.records for signal in record.signals]
    for signal in signals:
        signal.to_numpy()
    # Six signals over four rows: each multivariate row's two channels share one reread.
    assert (len(signals), len(reads)) == (6, 4)


@pytest.mark.parametrize(
    ("old", "new", "message"),
    [
        # Same byte length and same shape, but another value: only the values digest sees it.
        ("657372000.0]", "657372001.0]", "channels differ from the converted row"),
        # Same byte length, but four values, not three.
        ("657372000.0]", "6573, 72000]", "channels differ from the converted row"),
    ],
)
def test_a_reread_that_no_longer_matches_its_row_fails(tmp_path, old, new, message):
    path = tmp_path / "uni.json"
    path.write_bytes((_FIXTURES / "uni.json").read_bytes())
    dataset = _FixtureConnector().convert([HubFileRef(path, "uni")])
    path.write_text(path.read_text().replace(old, new))
    record = next(record for record in dataset.records if _row_of(record) == ("uni", "train", "0"))
    with pytest.raises(TimeFFormatError, match=rf"train, key 0.*{message}"):
        record.signals[0].to_numpy()


def test_a_reread_that_is_no_longer_json_fails(tmp_path):
    path = tmp_path / "uni.json"
    path.write_bytes((_FIXTURES / "uni.json").read_bytes())
    dataset = _FixtureConnector().convert([HubFileRef(path, "uni")])
    path.write_text(path.read_text().replace("657372000.0]", "657372000.0}"))
    record = next(record for record in dataset.records if _row_of(record) == ("uni", "train", "0"))
    with pytest.raises(TimeFFormatError, match=r"uni\.json \(bytes \d+-\d+\)"):
        record.signals[0].to_numpy()


def test_a_reread_with_other_channel_names_fails(tmp_path):
    path = tmp_path / "multi.json"
    path.write_bytes((_FIXTURES / "multi.json").read_bytes())
    dataset = _FixtureConnector().convert([HubFileRef(path, "multi")])
    path.write_text(path.read_text().replace("pH", "ph"))
    record = next(record for record in dataset.records if _row_of(record) == ("multi", "train", "0"))
    with pytest.raises(TimeFFormatError, match="channels differ from the converted row"):
        record.signals[0].to_numpy()


@pytest.mark.parametrize(
    ("text", "message"),
    [
        ('{"TSAD_train": {}, "TSAD_val": {}}', "unknown split 'TSAD_val'"),
        ('{"TSAD_train": {}}', "expected TSAD_train and TSAD_test"),
        ('{"TSAD_train": {"a1": null}, "TSAD_test": {}}', "expected a decimal row key, got 'a1'"),
        ('{"TSAD_train": {}, "TSAD_test": {}} []', "unexpected content after the JSON document"),
    ],
)
def test_malformed_split_structure_is_rejected(tmp_path, text, message):
    path = tmp_path / "broken.json"
    path.write_text(text)
    with pytest.raises(TimeFValidationError, match=message) as error:
        _FixtureConnector().convert([HubFileRef(path, "uni")])
    assert str(path) in str(error.value)


@pytest.mark.parametrize(
    ("variant", "field", "value", "message"),
    [
        ("multi", "Observation", '"a": [1]; "a": [2]', "duplicate channel"),
        ("multi", "Observation", '"a": [1]; "b": [2,3]', "channels differ in length"),
        ("multi", "Observation", '"a" [1]', "followed by ':'"),
        ("multi", "Observation", '"a": [1];', "after the last ';'"),
        ("multi", "Observation", '"a": [1] "b": [2]', "between channels"),
        ("multi", "Observation", '" ": [1]', "nonblank channel name"),
        ("multi", "Observation", "", "no channels"),
        ("multi", "Observation", [1.0], "string of named channels"),
        ("multi", "Observation", '"a": []', "nonempty list of finite numbers"),
        ("uni", "Observation", [], "nonempty list of finite numbers"),
        ("uni", "Observation", "1, 2", "nonempty list of finite numbers"),
        ("uni", "Observation", [1.0, "2"], "nonempty list of finite numbers"),
        ("uni", "Observation", [True], "nonempty list of finite numbers"),
        # Python-written JSON can contain NaN, which json accepts by default.
        ("uni", "Observation", [1.0, float("nan")], "nonempty list of finite numbers"),
        ("uni", "Observation", [10**400], "int too large"),
        ("uni", "Observation", _MISSING, "missing fields"),
        # A field the release doesn't have is rejected, not dropped.
        ("uni", "Index", 0, "unknown fields"),
        # A JSON true is an int to Python, so the code must reject it explicitly.
        ("uni", "ActionID", True, "ActionID must be a nonnegative integer"),
        ("uni", "ActionID", -1, "ActionID must be a nonnegative integer"),
        ("uni", "ActionID", 1.0, "ActionID must be a nonnegative integer"),
        ("uni", "Label", "normal", "Label must be one of"),
        ("uni", "Label", ["Normal"], "Label must be one of"),
        ("uni", "Source", "  ", "Source must be nonblank text"),
        ("uni", "Thought", None, "Thought must be nonblank text"),
        ("uni", "Action", 3, "Action must be nonblank text"),
        ("uni", "FigurePath", "", "FigurePath must be nonblank text"),
        ("uni", "Rank", "model-a", "Rank must be a list"),
        ("uni", "Rank", ["model-a", ""], "Rank must be nonblank text"),
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
