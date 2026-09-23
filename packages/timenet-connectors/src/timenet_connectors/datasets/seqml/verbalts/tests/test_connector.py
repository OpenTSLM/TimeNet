from collections.abc import Iterable
from fractions import Fraction
import hashlib
import json
import logging
from pathlib import Path

import numpy as np
import pytest

from timenet.connectors import BaseConnector
from timenet.dataset import TimeFDataset
from timenet.dataset.axis import OrdinalAxis, RegularAxis
from timenet.dataset.record import Record
from timenet.engine import store_dataset
from timenet.errors import TimeFFormatError, TimeFValidationError
from timenet.reader import TimeFReader
from timenet.registry import DatasetVersion
from timenet.types import Annotation, Task, TSGenerationTask
from timenet_connectors.datasets.seqml.verbalts import (
    VerbalTsComponent,
    VerbalTsConnector,
    connector as connector_module,
)
from timenet_connectors.datasets.seqml.verbalts.files import FILES, check_drive_body


# The arrays and caption strings below are hand-written for these tests. Nothing here is derived
# from real VerbalTS bytes, so the repo ships no dataset data. A reader can tell: the real release
# stores 120 to 600 steps per window and these hold 3 to 6, and every caption names its own
# component, split and row. Five of the six components are present, which covers both caption
# arities, both axis kinds and every way a signal gets its name. Weather has 21 signals, three
# captions per window and a regular axis. synthetic_u has one signal and an ordinal axis, and
# synthetic_m has two. BlindWays names its 72 signals by joint and axis. ETTm1 names its single
# signal from the window's own var_id code, and carries a `season` attribute that deliberately
# collides with Weather's `season`.
_SPLITS = ("train", "valid", "test")
_ROWS_PER_SPLIT = 2
_WEATHER_STEPS = 4
_WEATHER_SIGNALS = 21
_SYNTHETIC_STEPS = 5
_BLINDWAYS_STEPS = 3
_BLINDWAYS_SIGNALS = 72
_ETTM1_STEPS = 6
_ETTM1_COLUMNS = ("HUFL", "HULL", "MUFL", "MULL", "LUFL", "LULL", "OT")

_SHAPE = {
    "Weather": (_WEATHER_STEPS, _WEATHER_SIGNALS),
    "synthetic_u": (_SYNTHETIC_STEPS, 1),
    "synthetic_m": (_SYNTHETIC_STEPS, 2),
    "BlindWays": (_BLINDWAYS_STEPS, _BLINDWAYS_SIGNALS),
    "ETTm1": (_ETTM1_STEPS, 1),
}

_FIXTURE_META = {
    "Weather": {"attr_list": ["season", "time"], "attr_n_ops": [4, 4]},
    "synthetic_u": {"attr_list": ["trend_types"], "attr_n_ops": [4]},
    "synthetic_m": {"attr_list": ["trend_types"], "attr_n_ops": [4]},
    "BlindWays": {"attr_list": ["scene"], "attr_n_ops": [4]},
    "ETTm1": {"attr_list": ["var_id", "season"], "attr_n_ops": [7, 10]},
}

# Release order (files.COMPONENTS), not the fixture's dict order.
_FIXTURE_COMPONENTS = ("synthetic_u", "synthetic_m", "Weather", "BlindWays", "ETTm1")


def _values(component: str, split: str) -> np.ndarray:
    """Deterministic float64 windows, distinct per component, split and cell."""
    steps, signals = _SHAPE[component]
    base = float(len(component) * 1000 + _SPLITS.index(split) * 100)
    flat = base + np.arange(_ROWS_PER_SPLIT * steps * signals, dtype=np.float64) / 8.0
    return flat.reshape(_ROWS_PER_SPLIT, steps, signals)


def _attributes(component: str, split: str) -> np.ndarray:
    n_attrs = len(_FIXTURE_META[component]["attr_list"])
    offset = _SPLITS.index(split)
    return np.array(
        [[(row + offset + index) % 4 for index in range(n_attrs)] for row in range(_ROWS_PER_SPLIT)],
        dtype=np.int64,
    )


def _captions(component: str, split: str) -> np.ndarray:
    n_captions = 3 if component == "Weather" else 1
    return np.array(
        [
            [f"{component} {split} window {row} caption {index}." for index in range(n_captions)]
            for row in range(_ROWS_PER_SPLIT)
        ],
        dtype=np.str_,
    )


def _write_corpus(root: Path) -> Path:
    for component, meta in _FIXTURE_META.items():
        folder = root / component
        folder.mkdir(parents=True)
        (folder / "meta.json").write_text(json.dumps(meta), encoding="utf-8")
        for split in _SPLITS:
            np.save(folder / f"{split}_ts.npy", _values(component, split))
            np.save(folder / f"{split}_attrs_idx.npy", _attributes(component, split))
            np.save(folder / f"{split}_text_caps.npy", _captions(component, split))
    return root


@pytest.fixture(autouse=True)
def _fresh_maps():
    # _open_npy is module-global. A test that rewrites a fixture file would otherwise read the
    # memory map another test left behind.
    connector_module._open_npy.cache_clear()
    yield
    connector_module._open_npy.cache_clear()


@pytest.fixture
def corpus(tmp_path):
    return _write_corpus(tmp_path / "cache")


def _components(root: Path) -> list[VerbalTsComponent]:
    return [VerbalTsComponent(name=name, folder=root / name) for name in _FIXTURE_COMPONENTS]


def _convert(root: Path) -> TimeFDataset:
    return VerbalTsConnector().convert(_components(root))


def _by_id(dataset: TimeFDataset) -> dict[str, Record]:
    return {record.record_id: record for record in dataset.records}


def _generation_tasks_for(tasks: Iterable[Task], record_id: str) -> list[TSGenerationTask]:
    return [t for t in tasks if isinstance(t, TSGenerationTask) and t.record_ids == (record_id,)]


def test_is_a_connector():
    assert isinstance(VerbalTsConnector(), BaseConnector)


def test_metadata_declares_the_id_and_a_licence_url():
    metadata = VerbalTsConnector().metadata()
    assert metadata.dataset_id == "seqml/verbalts"
    assert str(metadata.license) == "other"
    assert metadata.license_url


def test_the_pinned_file_table_covers_every_component_and_split():
    assert len(FILES) == 60
    assert len({file_id for _c, _n, file_id, _b, _d in FILES}) == 60
    assert sum(n_bytes for _c, _n, _f, n_bytes, _d in FILES) == 799_997_660
    per_component = {}
    for component, _name, _file_id, _n_bytes, _digest in FILES:
        per_component[component] = per_component.get(component, 0) + 1
    assert set(per_component.values()) == {10}


def test_every_pinned_file_carries_a_distinct_digest():
    digests = [digest for _c, _n, _f, _b, digest in FILES]
    assert all(len(digest) == 64 and set(digest) <= set("0123456789abcdef") for digest in digests)
    assert len(set(digests)) == 60


def test_download_pins_every_artifact():
    # The real table is 800 MB, so the download test below runs on a stand-in. This one checks the
    # pins the connector actually ships: every file has a size, a digest and a confirm=t URL.
    assert all(n_bytes and digest for _c, _n, _f, n_bytes, digest in FILES)
    urls = [connector_module.DRIVE_URL.format(file_id=file_id) for _c, _n, file_id, _b, _d in FILES]
    assert all("confirm=t" in url for url in urls)


# Bodies small enough to hash in a test, one per file the stand-in table below pins.
_FAKE_NPY = b"\x93NUMPY" + b"stand-in body"
_FAKE_META = b'{"attr_list": [], "attr_n_ops": []}'


def _stand_in_table():
    """Pin two small files per component, with the digests of the bodies the fake writes."""
    table = []
    for component in connector_module.COMPONENTS:
        for name, body in (("meta.json", _FAKE_META), ("train_ts.npy", _FAKE_NPY)):
            table.append((component, name, f"drive-{component}-{name}", len(body), hashlib.sha256(body).hexdigest()))
    return tuple(table)


def test_download_checks_every_body_against_its_pinned_digest(tmp_path, monkeypatch):
    seen = {}

    async def fake_download_files(artifacts, **kwargs):
        seen["artifacts"] = list(artifacts)
        seen["kwargs"] = kwargs
        for artifact in seen["artifacts"]:
            artifact.dest.parent.mkdir(parents=True, exist_ok=True)
            artifact.dest.write_bytes(_FAKE_NPY if artifact.dest.suffix == ".npy" else _FAKE_META)
        return [artifact.dest for artifact in seen["artifacts"]]

    monkeypatch.setattr(connector_module, "download_files", fake_download_files)
    monkeypatch.setattr(connector_module, "FILES", _stand_in_table())
    components = VerbalTsConnector().download(tmp_path)
    assert [component.name for component in components] == list(connector_module.COMPONENTS)
    assert len(seen["artifacts"]) == 12
    assert seen["kwargs"]["max_concurrency"] == 4
    assert all(artifact.sha256 for artifact in seen["artifacts"])


def test_download_rejects_a_body_that_does_not_match_its_digest(tmp_path, monkeypatch):
    async def fake_download_files(artifacts, **kwargs):
        for artifact in artifacts:
            artifact.dest.parent.mkdir(parents=True, exist_ok=True)
            # Right length and right magic, replaced contents: only the digest can see this.
            artifact.dest.write_bytes((_FAKE_NPY if artifact.dest.suffix == ".npy" else _FAKE_META)[:-1] + b"X")
        return []

    monkeypatch.setattr(connector_module, "download_files", fake_download_files)
    monkeypatch.setattr(connector_module, "FILES", _stand_in_table())
    with pytest.raises(TimeFFormatError, match="was replaced since the table was pinned"):
        VerbalTsConnector().download(tmp_path)


def test_download_rejects_a_short_body(tmp_path, monkeypatch):
    async def fake_download_files(artifacts, **kwargs):
        for artifact in artifacts:
            artifact.dest.parent.mkdir(parents=True, exist_ok=True)
            artifact.dest.write_bytes(b"Google Drive can't scan this file for viruses.")
        return []

    monkeypatch.setattr(connector_module, "download_files", fake_download_files)
    with pytest.raises(TimeFFormatError, match="virus-scan interstitial"):
        VerbalTsConnector().download(tmp_path)


def test_download_rejects_a_file_the_fetch_never_wrote(tmp_path, monkeypatch):
    async def fake_download_files(artifacts, **kwargs):
        return []

    monkeypatch.setattr(connector_module, "download_files", fake_download_files)
    monkeypatch.setattr(connector_module, "FILES", _stand_in_table())
    with pytest.raises(TimeFFormatError, match="is missing after the download"):
        VerbalTsConnector().download(tmp_path)


def test_one_record_per_window_with_padded_ids(corpus):
    dataset = _convert(corpus)
    assert len(dataset.records) == len(_FIXTURE_COMPONENTS) * len(_SPLITS) * _ROWS_PER_SPLIT
    assert "verbalts-Weather-train-00000" in _by_id(dataset)
    assert "verbalts-synthetic_u-test-00001" in _by_id(dataset)


def test_series_shape_and_ids_follow_the_arrays(corpus):
    record = _by_id(_convert(corpus))["verbalts-Weather-valid-00001"]
    assert len(record.time_series) == _WEATHER_SIGNALS
    assert {series.n_values for series in record.time_series} == {_WEATHER_STEPS}
    assert {series.source_id for series in record.time_series} == {"verbalts-Weather-valid-00001"}
    assert record.time_series[0].signal == "p (mbar)"
    assert record.time_series[0].time_series_id == "verbalts-Weather-valid-00001-p (mbar)"


def test_blindways_names_every_joint_and_axis(corpus):
    record = _by_id(_convert(corpus))["verbalts-BlindWays-train-00000"]
    names = tuple(series.signal for series in record.time_series)
    assert names[0] == "j00_x"
    assert names[-1] == "j23_z"
    assert len(names) == _BLINDWAYS_SIGNALS


def test_a_two_signal_synthetic_window_is_named_by_position(corpus):
    record = _by_id(_convert(corpus))["verbalts-synthetic_m-train-00000"]
    assert tuple(series.signal for series in record.time_series) == ("x0", "x1")


def test_an_ettm1_signal_is_named_by_its_own_var_id_code(corpus):
    records = _by_id(_convert(corpus))
    codes = _attributes("ETTm1", "train")
    for row in range(_ROWS_PER_SPLIT):
        record = records[f"verbalts-ETTm1-train-{row:05d}"]
        assert len(record.time_series) == 1
        assert record.time_series[0].signal == _ETTM1_COLUMNS[int(codes[row, 0])]


def test_every_time_series_id_is_unique(corpus):
    dataset = _convert(corpus)
    ids = [series.time_series_id for record in dataset.records for series in record.time_series]
    assert len(set(ids)) == len(ids)


def test_axes_come_from_the_component(corpus):
    records = _by_id(_convert(corpus))
    weather_axis = records["verbalts-Weather-train-00000"].time_series[0].time_axis
    assert isinstance(weather_axis, RegularAxis)
    assert weather_axis.period_us == Fraction(600_000_000)
    assert isinstance(records["verbalts-synthetic_u-train-00000"].time_series[0].time_axis, OrdinalAxis)


def test_values_are_bit_identical_and_the_loader_is_re_callable(corpus):
    record = _by_id(_convert(corpus))["verbalts-Weather-test-00001"]
    source = _values("Weather", "test")
    for signal, series in enumerate(record.time_series):
        assert np.array_equal(series.to_numpy(), source[1, :, signal])
    # The writer calls a loader more than once, so re-slicing must give the same values.
    assert np.array_equal(record.time_series[0].to_numpy(), source[1, :, 0])


def test_convert_maps_the_arrays_and_materialises_no_values(corpus, monkeypatch):
    modes = []
    real_load = np.load
    real_pa_array = connector_module.pa.array

    def load_spy(path, *args, **kwargs):
        modes.append(kwargs.get("mmap_mode"))
        return real_load(path, *args, **kwargs)

    def array_spy(*args, **kwargs):
        pytest.fail("convert() built an Arrow array; the values must stay behind the lazy loader")

    monkeypatch.setattr(connector_module.np, "load", load_spy)
    monkeypatch.setattr(connector_module.pa, "array", array_spy)
    dataset = _convert(corpus)
    assert modes
    assert set(modes) == {"r"}
    # Every array is mapped once, not once per window: values and attributes per split. The caption
    # plane belongs to the task stream, so convert never opens it.
    assert modes.count("r") == len(_FIXTURE_COMPONENTS) * len(_SPLITS) * 2
    monkeypatch.setattr(connector_module.pa, "array", real_pa_array)
    assert len(dataset.records[0].time_series[0].to_numpy()) == _SYNTHETIC_STEPS


def test_annotations_carry_the_context_and_the_codes_but_not_the_captions(corpus):
    record = _by_id(_convert(corpus))["verbalts-Weather-train-00000"]
    by_key = {}
    for annotation in record.annotations:
        by_key.setdefault(annotation.key, []).append(annotation)
    assert by_key["component"][0].value == "Weather"
    assert by_key["split"][0].value == "train"
    codes = _attributes("Weather", "train")[0]
    assert by_key["weather_season"][0].value == int(codes[0])
    assert isinstance(by_key["weather_season"][0].value, int)
    assert by_key["weather_time"][0].value == int(codes[1])
    assert "season" not in by_key
    # The caption is the task prompt, so it is not also stored as an annotation.
    assert "caption" not in by_key


def test_a_one_caption_component_carries_one_task(corpus):
    dataset = _convert(corpus)
    tasks = _generation_tasks_for(dataset.iter_tasks(), "verbalts-synthetic_u-train-00001")
    assert [t.prompt for t in tasks] == list(_captions("synthetic_u", "train")[1])
    assert len(tasks) == 1


def test_codebooks_are_registered_once_and_carry_their_option_count(corpus):
    dataset = _convert(corpus)
    registered = {annotation.id: annotation for annotation in dataset.registered_annotations}
    assert set(registered) == {
        "verbalts-codebook-weather-season",
        "verbalts-codebook-weather-time",
        "verbalts-codebook-synthetic-u-trend-types",
        "verbalts-codebook-synthetic-m-trend-types",
        "verbalts-codebook-blindways-scene",
        "verbalts-codebook-ettm1-var-id",
        "verbalts-codebook-ettm1-season",
    }
    assert registered["verbalts-codebook-weather-season"].key == "weather_season_options"
    assert registered["verbalts-codebook-weather-season"].value == 4


def test_one_generation_task_per_caption_prompted_by_that_caption(corpus):
    dataset = _convert(corpus)
    n_captions = sum(3 if r.record_id.startswith("verbalts-Weather-") else 1 for r in dataset.records)
    assert sum(1 for _ in dataset.iter_tasks()) == n_captions
    tasks = _generation_tasks_for(dataset.iter_tasks(), "verbalts-Weather-train-00000")
    task = next(t for t in tasks if t.prompt == _captions("Weather", "train")[0][2])
    assert task.target is None
    assert task.target_annotation_ids == ()
    assert task.target_record_id == "verbalts-Weather-train-00000"
    assert task.input_annotation_ids == ("verbalts-codebook-weather-season", "verbalts-codebook-weather-time")


def test_a_weather_window_gets_three_specifications_of_one_target(corpus):
    dataset = _convert(corpus)
    tasks = _generation_tasks_for(dataset.iter_tasks(), "verbalts-Weather-train-00000")
    assert len(tasks) == 3
    assert {t.target_record_id for t in tasks} == {"verbalts-Weather-train-00000"}
    assert [t.prompt for t in tasks] == list(_captions("Weather", "train")[0])


def test_the_tasks_stream_and_no_record_lists_its_tasks(corpus):
    # A streamed task carries its own record_ids, and the link runs one way: nothing walks from a
    # record to its tasks any more.
    dataset = _convert(corpus)
    assert dataset.has_task_stream
    assert dataset.tasks == ()
    assert all(record.task_ids == () for record in dataset.records)
    assert all(task.record_ids for task in dataset.iter_tasks())


def test_the_task_stream_gives_the_same_tasks_when_read_again(corpus):
    # set_task_stream needs a source it can read again, so a generator that empties itself is a bug.
    dataset = _convert(corpus)
    once = [(t.record_ids, t.prompt, t.input_annotation_ids) for t in dataset.iter_tasks()]
    again = [(t.record_ids, t.prompt, t.input_annotation_ids) for t in dataset.iter_tasks()]
    assert once == again
    assert len(once) == sum(3 if r.record_id.startswith("verbalts-Weather-") else 1 for r in dataset.records)


def test_every_streamed_task_id_is_distinct(corpus):
    # A stream skips the dataset's cross-task duplicate-id check, so the generated ids are pinned
    # here instead.
    ids = [task.id for task in _convert(corpus).iter_tasks()]
    assert len(set(ids)) == len(ids)


def test_schema_derives_a_prefixed_key_per_attribute_and_one_task_type(corpus):
    schema = _convert(corpus).derive_schema()
    assert {descriptor.key for descriptor in schema.annotations} == {
        "component",
        "split",
        "weather_season",
        "weather_time",
        "weather_season_options",
        "weather_time_options",
        "synth_u_trend_types",
        "synth_u_trend_types_options",
        "synth_m_trend_types",
        "synth_m_trend_types_options",
        "blindways_scene",
        "blindways_scene_options",
        "ettm1_var_id",
        "ettm1_var_id_options",
        "ettm1_season",
        "ettm1_season_options",
    }
    assert schema.tasks == (TSGenerationTask,)
    assert {spec.spec_type for spec in schema.time_series_specs} == {"weather", "synthetic", "pose", "power"}


def test_convert_round_trips_through_the_writer(corpus, tmp_path):
    dataset = _convert(corpus)
    dataset.derive_schema()
    version_dir = store_dataset(dataset, tmp_path / "out")
    with TimeFReader(DatasetVersion.open_local(version_dir)) as reader:
        restored = reader.read()
    record = next(r for r in restored.records if r.record_id == "verbalts-Weather-train-00000")
    tasks = _generation_tasks_for(restored.tasks, "verbalts-Weather-train-00000")
    assert {t.prompt for t in tasks} == set(_captions("Weather", "train")[0])
    assert {t.target_record_id for t in tasks} == {"verbalts-Weather-train-00000"}
    # The build wrote no record task ids; read() rebuilds the reverse link from the task rows.
    assert len(restored.tasks_for(record, TSGenerationTask)) == 3
    assert len(record.time_series) == _WEATHER_SIGNALS
    assert np.array_equal(record.time_series[0].to_numpy(), _values("Weather", "train")[0, :, 0])


def test_unprefixed_attribute_keys_would_collide(corpus):
    # The guard this connector's key prefix exists for: two descriptors under one key abort the build.
    dataset = _convert(corpus)
    dataset.records[0].add_annotation(
        Annotation(key="weather_season", value="fall", id="collide"), warn_when_outside=False
    )
    with pytest.raises(TimeFValidationError, match="conflicting descriptors"):
        dataset.derive_schema()


def test_a_non_finite_value_is_stored_by_the_writer(corpus, tmp_path):
    # A NaN is a measured payload, not an absent measurement, so the writer keeps it.
    broken = _values("synthetic_u", "train")
    broken[0, 2, 0] = np.nan
    np.save(corpus / "synthetic_u" / "train_ts.npy", broken)
    dataset = _convert(corpus)
    dataset.derive_schema()
    version_dir = store_dataset(dataset, tmp_path / "out")
    with TimeFReader(DatasetVersion.open_local(version_dir)) as reader:
        restored = reader.read()
    series = {ts.time_series_id: ts for record in restored.records for ts in record.time_series}
    assert np.isnan(series["verbalts-synthetic_u-train-00000-x0"].to_numpy()[2])


def test_a_truncated_values_file_fails_the_declared_length(corpus, tmp_path):
    short = _values("synthetic_u", "valid")[:, :-1, :]
    dataset = _convert(corpus)  # n_values read from the full-length header
    dataset.derive_schema()
    connector_module._open_npy.cache_clear()
    np.save(corpus / "synthetic_u" / "valid_ts.npy", short)
    with pytest.raises(TimeFValidationError, match="returned 4 values but it declares n_values=5"):
        store_dataset(dataset, tmp_path / "out")


def test_an_unknown_signal_count_is_rejected(corpus):
    np.save(corpus / "Weather" / "train_ts.npy", _values("Weather", "train")[:, :, :3])
    with pytest.raises(TimeFFormatError, match="3 signals, expected 21"):
        _convert(corpus)


def test_a_values_file_that_is_not_three_dimensional_is_rejected(corpus):
    np.save(corpus / "synthetic_u" / "train_ts.npy", _values("synthetic_u", "train")[:, :, 0])
    with pytest.raises(TimeFFormatError, match="holds a 2-dimensional array"):
        _convert(corpus)


def test_a_caption_plane_that_is_not_a_rectangle_is_rejected(corpus):
    # The captions are the stream's input, so the shape is checked when the stream reads them.
    np.save(corpus / "synthetic_u" / "train_text_caps.npy", _captions("synthetic_u", "train")[:, 0])
    with pytest.raises(TimeFFormatError, match=r"train_text_caps.npy holds an array of shape \(2,\)"):
        list(_convert(corpus).iter_tasks())


def test_an_attribute_plane_that_disagrees_with_meta_json_is_rejected(corpus):
    wider = np.repeat(_attributes("Weather", "train"), 2, axis=1)
    np.save(corpus / "Weather" / "train_attrs_idx.npy", wider)
    with pytest.raises(TimeFFormatError, match=r"holds 4 columns, but meta\.json names 2 attributes"):
        _convert(corpus)


def test_a_meta_json_with_lists_of_different_lengths_is_rejected(corpus):
    meta = {"attr_list": ["season", "time"], "attr_n_ops": [4]}
    (corpus / "Weather" / "meta.json").write_text(json.dumps(meta), encoding="utf-8")
    with pytest.raises(TimeFFormatError, match="lists 2 attributes but 1 option counts"):
        _convert(corpus)


def test_a_meta_json_without_the_attribute_lists_is_rejected(corpus):
    (corpus / "Weather" / "meta.json").write_text(json.dumps({"final_split": 3}), encoding="utf-8")
    with pytest.raises(TimeFFormatError, match="states no attr_list and no attr_n_ops"):
        _convert(corpus)


def test_a_meta_json_that_does_not_parse_is_rejected(corpus):
    # json.loads raises JSONDecodeError, which is not one of this repo's error types.
    (corpus / "Weather" / "meta.json").write_text('{"attr_list": ["season"', encoding="utf-8")
    with pytest.raises(TimeFFormatError, match=r"meta\.json is not valid JSON"):
        _convert(corpus)


def test_a_meta_json_that_is_not_an_object_is_rejected(corpus):
    (corpus / "Weather" / "meta.json").write_text("[4, 4]", encoding="utf-8")
    with pytest.raises(TimeFFormatError, match="holds a JSON list, not an object"):
        _convert(corpus)


@pytest.mark.parametrize("count", ["four", 4.5])
def test_a_meta_json_option_count_that_is_not_a_whole_number_is_rejected(corpus, count):
    # 4.5 is the silent case: int(4.5) would have truncated the count to 4 and converted.
    meta = {"attr_list": ["season", "time"], "attr_n_ops": [4, count]}
    (corpus / "Weather" / "meta.json").write_text(json.dumps(meta), encoding="utf-8")
    with pytest.raises(TimeFFormatError, match="attr_n_ops this connector cannot read"):
        _convert(corpus)


def test_a_reordered_attribute_list_is_rejected_where_the_name_needs_var_id(corpus):
    # The one naming rule that reads a data value reads attribute 0. A release that moved var_id
    # would mislabel every window whose code stays inside the column range, and raise for the rest.
    meta = {"attr_list": ["season", "var_id"], "attr_n_ops": [10, 7]}
    (corpus / "ETTm1" / "meta.json").write_text(json.dumps(meta), encoding="utf-8")
    with pytest.raises(TimeFFormatError, match="names 'season' as its first attribute"):
        _convert(corpus)


def test_a_code_outside_the_declared_options_warns_once_and_is_still_converted(corpus, caplog):
    # An out-of-range code is an inconsistency the release ships, not a corrupt file, so the build
    # warns one time for that attribute and writes the code as stated.
    codes = _attributes("Weather", "train")
    codes[1, 0] = 9  # the fixture meta.json declares 4 season options
    np.save(corpus / "Weather" / "train_attrs_idx.npy", codes)
    with caplog.at_level(logging.WARNING):
        dataset = _convert(corpus)
    messages = [record.getMessage() for record in caplog.records]
    assert sum("season codes fall outside the 4 options" in message for message in messages) == 1
    record = _by_id(dataset)["verbalts-Weather-train-00001"]
    assert {a.key: a.value for a in record.annotations}["weather_season"] == 9


def test_a_values_file_that_is_not_float64_is_rejected(corpus):
    # The byte census and the spec both state float64. An int64 file would widen silently.
    np.save(corpus / "synthetic_u" / "train_ts.npy", _values("synthetic_u", "train").astype(np.int64))
    with pytest.raises(TimeFFormatError, match="holds int64 values; the release stores float64"):
        _convert(corpus)


def test_the_drive_interstitial_is_named_in_the_error(tmp_path):
    page = tmp_path / "train_ts.npy"
    page.write_bytes(b"<!DOCTYPE html><html><body>Google Drive can't scan this file.</body></html>")
    with pytest.raises(TimeFFormatError, match="virus-scan interstitial"):
        check_drive_body(page, 284_428_928, "0" * 64)


def test_a_right_sized_body_that_is_not_an_array_is_rejected(tmp_path):
    page = tmp_path / "train_ts.npy"
    page.write_bytes(b"<html>" + b"x" * 26)
    with pytest.raises(TimeFFormatError, match="not an NPY file"):
        check_drive_body(page, 32, "0" * 64)


def test_a_matching_body_passes_every_guard(tmp_path):
    body = b"\x93NUMPY" + b"body"
    path = tmp_path / "train_ts.npy"
    path.write_bytes(body)
    check_drive_body(path, len(body), hashlib.sha256(body).hexdigest())
