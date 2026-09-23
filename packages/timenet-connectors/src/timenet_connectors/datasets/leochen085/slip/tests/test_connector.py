"""The corpus connector, against a synthetic shard written into ``tmp_path``."""

import logging
from pathlib import Path
import re
import sys
from typing import Any

import huggingface_hub
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from timenet.connectors import BaseConnector
from timenet.dataset.axis import OrdinalAxis, RegularAxis
from timenet.errors import TimeFFormatError, TimeNetDownloadError
from timenet.reader import TimeFReader
from timenet.registry import DatasetVersion
from timenet.writer import TimeFWriter
from timenet_connectors.datasets.leochen085.slip import connector as slip
from timenet_connectors.datasets.leochen085.slip.connector import SlipConnector, SlipSource, _iter_tasks


# The release ships float32 values, so a fixture holds them too: the writer checks a series' values
# against the dtype its spec states.
_SCHEMA = pa.schema(
    [
        ("category", pa.string()),
        ("dataset", pa.string()),
        ("caption0", pa.string()),
        ("caption1", pa.string()),
        ("caption2", pa.string()),
        ("caption3", pa.string()),
        ("time_series", pa.list_(pa.list_(pa.float32()))),
    ]
)


@pytest.fixture(autouse=True)
def _clear_the_row_group_cache():
    # The row-group cache is module-level, so one test's row group would still be held in the next.
    slip._row_group.cache_clear()
    yield
    slip._row_group.cache_clear()


def _write_shard(
    directory: Path,
    rows: dict[str, list],
    name: str = "train-00003-of-00017.parquet",
    row_group_size: int | None = None,
) -> Path:
    path = directory / name
    pq.write_table(pa.table(rows, schema=_SCHEMA), path, row_group_size=row_group_size)
    return path


def _shard(directory: Path, name: str = "train-00003-of-00017.parquet") -> Path:
    # Invented, not a slice of the release. A reader can tell: every real caption in the corpus runs
    # to hundreds of characters, and every real series to hundreds of values.
    return _write_shard(
        directory,
        {
            "category": ["Health", "Energy"],
            "dataset": ["BIDMC32HR", "Wind Farms"],
            "caption0": ["a caption.", "another."],
            "caption1": ["a rewrite.", "another rewrite."],
            "caption2": ["a second rewrite.", "and another."],
            "caption3": ["a third rewrite.", "one more."],
            "time_series": [[[1.0, 2.0, 3.0]], [[4.0, 5.0], [6.0, 7.0]]],
        },
        name,
    )


def _meta(directory: Path) -> Path:
    path = directory / "meta.csv"
    path.write_text(
        "Domain,Dataset,Freq,Source\n"
        "Health,BIDMC32HR,-,https://example.test/a\n"
        "Energy,Wind Farms,4 sec,https://example.test/b\n",
        encoding="utf-8",
    )
    return path


def _fake_snapshot(root: Path, seen: dict[str, Any]):
    # snapshot_download is imported inside download(), so patching the module attribute is enough.
    def _download(repo_id: str, **kwargs: Any) -> str:
        seen["repo_id"] = repo_id
        seen.update(kwargs)
        return str(root)

    return _download


def test_is_a_connector() -> None:
    assert isinstance(SlipConnector(), BaseConnector)


def test_metadata() -> None:
    metadata = SlipConnector().metadata()
    assert metadata.dataset_id == "leochen085/slip"
    assert str(metadata.license) == "MIT"


def test_the_revision_is_a_full_commit_sha() -> None:
    # A branch or a tag moves, and the README's promise is that this one cannot.
    assert re.fullmatch(r"[0-9a-f]{40}", SlipConnector.REVISION)


def test_download_pins_the_revision_and_asks_for_the_corpus_only(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # The pin is what makes a build reproducible, and the two patterns are what keep the release's
    # 11 evaluation folders out of the fetch. A refactor that drops either passes every other test.
    root = tmp_path / "snapshot"
    (root / "data").mkdir(parents=True)
    shard = _shard(root / "data")
    meta_csv = _meta(root)
    seen: dict[str, Any] = {}
    monkeypatch.setattr(huggingface_hub, "snapshot_download", _fake_snapshot(root, seen))

    sources = SlipConnector().download(tmp_path / "cache")

    assert seen["repo_id"] == "LeoChen085/SlipDataset"
    assert seen["repo_type"] == "dataset"
    assert seen["revision"] == SlipConnector.REVISION
    assert seen["allow_patterns"] == ["data/*.parquet", "meta.csv"]
    assert seen["cache_dir"] == str(tmp_path / "cache")
    assert sources == [SlipSource(shards=(shard,), meta_csv=meta_csv)]


def test_download_gives_the_shards_in_name_order(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # convert numbers rows per shard, so the order the shards arrive in is part of the record ids.
    root = tmp_path / "snapshot"
    (root / "data").mkdir(parents=True)
    for name in ("train-00002-of-00017.parquet", "train-00000-of-00017.parquet", "train-00001-of-00017.parquet"):
        _shard(root / "data", name)
    _meta(root)
    monkeypatch.setattr(huggingface_hub, "snapshot_download", _fake_snapshot(root, {}))

    shards = SlipConnector().download(tmp_path / "cache")[0].shards

    assert [path.name for path in shards] == [
        "train-00000-of-00017.parquet",
        "train-00001-of-00017.parquet",
        "train-00002-of-00017.parquet",
    ]


def test_download_raises_when_the_fetch_brought_no_shards(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # An empty fetch is a moved revision or a changed layout, not a corpus with no rows.
    root = tmp_path / "snapshot"
    root.mkdir()
    monkeypatch.setattr(huggingface_hub, "snapshot_download", _fake_snapshot(root, {}))
    with pytest.raises(TimeNetDownloadError, match=r"no shards matching"):
        SlipConnector().download(tmp_path / "cache")


def test_download_without_huggingface_hub_names_the_requirements_file(monkeypatch: pytest.MonkeyPatch) -> None:
    # `import huggingface_hub` -> ImportError, the way an environment built without the connector's
    # own requirements behaves.
    monkeypatch.setitem(sys.modules, "huggingface_hub", None)
    with pytest.raises(ImportError, match=r"requirements\.txt"):
        SlipConnector().download(Path("cache"))


def test_one_record_per_row_with_a_positional_id(tmp_path: Path) -> None:
    source = SlipSource(shards=(_shard(tmp_path),), meta_csv=_meta(tmp_path))
    records = SlipConnector().convert([source]).records
    assert [r.record_id for r in records] == ["slip-shard-00003-row-000000", "slip-shard-00003-row-000001"]


def test_a_row_becomes_one_signal_per_inner_list(tmp_path: Path) -> None:
    source = SlipSource(shards=(_shard(tmp_path),), meta_csv=_meta(tmp_path))
    records = SlipConnector().convert([source]).records
    assert [s.signal for s in records[0].time_series] == ["s0"]
    assert [s.signal for s in records[1].time_series] == ["s0", "s1"]
    assert [s.n_values for s in records[1].time_series] == [2, 2]


def test_every_series_states_its_record_as_its_source_id(tmp_path: Path) -> None:
    # The writer groups series by source_id before signal, so this is what makes the write order the
    # walk order. Without it the write goes signal-major and re-reads the row groups of every
    # multivariate row once per signal.
    source = SlipSource(shards=(_shard(tmp_path),), meta_csv=_meta(tmp_path))
    records = SlipConnector().convert([source]).records
    for record in records:
        assert [s.source_id for s in record.time_series] == [record.record_id] * len(record.time_series)


def test_a_shape_is_read_per_row_and_not_per_corpus(tmp_path: Path) -> None:
    # ChatTS rows are 256 values long in one shard and 719 in another, so a per-corpus shape would
    # be wrong for a sixth of the release and nothing would say so.
    shard = _write_shard(
        tmp_path,
        {
            "category": ["Health", "Health"],
            "dataset": ["BIDMC32HR", "BIDMC32HR"],
            "caption0": ["a caption.", "another."],
            "caption1": ["a rewrite.", "another rewrite."],
            "caption2": ["a second rewrite.", "and another."],
            "caption3": ["a third rewrite.", "one more."],
            "time_series": [[[1.0, 2.0]], [[3.0, 4.0, 5.0, 6.0]]],
        },
    )
    records = SlipConnector().convert([SlipSource(shards=(shard,), meta_csv=_meta(tmp_path))]).records
    assert [s.n_values for r in records for s in r.time_series] == [2, 4]


def test_the_axis_comes_from_the_corpus_the_row_names(tmp_path: Path) -> None:
    source = SlipSource(shards=(_shard(tmp_path),), meta_csv=_meta(tmp_path))
    records = SlipConnector().convert([source]).records
    # BIDMC32HR states no rate, so its series records order and claims none.
    assert isinstance(records[0].time_series[0].time_axis, OrdinalAxis)
    axis = records[1].time_series[0].time_axis
    assert isinstance(axis, RegularAxis)
    assert axis.period_us == 4_000_000


def test_the_values_are_read_when_the_loader_is_called(tmp_path: Path) -> None:
    source = SlipSource(shards=(_shard(tmp_path),), meta_csv=_meta(tmp_path))
    records = SlipConnector().convert([source]).records
    assert records[0].time_series[0].to_numpy().tolist() == [1.0, 2.0, 3.0]
    assert records[1].time_series[1].to_numpy().tolist() == [6.0, 7.0]


def test_convert_reads_no_values_through_a_loader(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # A loader called during convert would put the whole values plane in memory before the writer
    # asks for any of it.
    reads: list[object] = []
    original = slip._read_signal

    def _counted(ref):
        reads.append(ref)
        return original(ref)

    monkeypatch.setattr(slip, "_read_signal", _counted)
    source = SlipSource(shards=(_shard(tmp_path),), meta_csv=_meta(tmp_path))
    records = SlipConnector().convert([source]).records
    assert reads == []
    records[0].time_series[0].to_arrow()
    assert len(reads) == 1


def test_a_record_carries_its_corpus_its_domain_and_where_that_corpus_was_published(tmp_path: Path) -> None:
    source = SlipSource(shards=(_shard(tmp_path),), meta_csv=_meta(tmp_path))
    records = SlipConnector().convert([source]).records
    found = {a.key: a.value for a in records[0].annotations if not a.key.startswith("caption")}
    assert found == {
        "source_dataset": "BIDMC32HR",
        "domain": "Health",
        "source_url": "https://example.test/a",
    }


def test_a_record_carries_its_four_captions_under_the_columns_they_came_from(tmp_path: Path) -> None:
    source = SlipSource(shards=(_shard(tmp_path),), meta_csv=_meta(tmp_path))
    records = SlipConnector().convert([source]).records
    captions = [a for a in records[0].annotations if a.key.startswith("caption")]
    assert [(a.key, a.value) for a in captions] == [
        ("caption0", "a caption."),
        ("caption1", "a rewrite."),
        ("caption2", "a second rewrite."),
        ("caption3", "a third rewrite."),
    ]
    # The ids are built, not generated: a streamed task answers by reference to them.
    assert [a.id for a in captions] == [f"slip-shard-00003-row-000000-caption{index}" for index in range(4)]


def test_four_tasks_per_record_each_answering_with_one_caption(tmp_path: Path) -> None:
    source = SlipSource(shards=(_shard(tmp_path),), meta_csv=_meta(tmp_path))
    dataset = SlipConnector().convert([source])
    tasks = list(_iter_tasks(dataset.records))
    assert len(tasks) == 8
    first = tasks[:4]
    assert [t.target_annotation_ids for t in first] == [
        (f"slip-shard-00003-row-000000-caption{index}",) for index in range(4)
    ]
    assert {t.record_ids for t in first} == {("slip-shard-00003-row-000000",)}
    # The answer is the stored annotation, never a second copy of it.
    assert [t.target for t in first] == [None] * 4
    # A streamed task states no id; these are the generated UUIDv7s, and all that matters is that
    # four distinct tasks exist.
    assert len({t.id for t in first}) == 4


def test_the_task_stream_gives_the_same_tasks_when_read_again(tmp_path: Path) -> None:
    # set_task_stream needs a source it can read again, so a generator that empties itself is a bug.
    source = SlipSource(shards=(_shard(tmp_path),), meta_csv=_meta(tmp_path))
    dataset = SlipConnector().convert([source])
    once = [(t.record_ids, t.target_annotation_ids) for t in _iter_tasks(dataset.records)]
    again = [(t.record_ids, t.target_annotation_ids) for t in _iter_tasks(dataset.records)]
    assert once == again
    assert len(once) == 8


def test_a_series_with_no_numbers_is_kept_annotated_and_logged(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    shard = _write_shard(
        tmp_path,
        {
            "category": ["Health", "Health"],
            "dataset": ["BIDMC32HR", "BIDMC32HR"],
            "caption0": ["a caption.", "another."],
            "caption1": ["a rewrite.", "another rewrite."],
            "caption2": ["a second rewrite.", "and another."],
            "caption3": ["a third rewrite.", "one more."],
            "time_series": [
                [[1.0, 2.0], [float("nan"), float("nan")]],
                [[3.0, 4.0], [float("nan"), float("nan")]],
            ],
        },
        "train-00000-of-00017.parquet",
    )
    source = SlipSource(shards=(shard,), meta_csv=_meta(tmp_path))
    with caplog.at_level(logging.WARNING):
        records = SlipConnector().convert([source]).records
    # The series is built, not dropped, and an annotation names it.
    assert [s.signal for s in records[0].time_series] == ["s0", "s1"]
    empty = [a for a in records[0].annotations if a.key == "all_nan_signals"]
    assert [a.value for a in empty] == [["s1"]]
    # Two records, one warning: the count is the fact, not each occurrence.
    lines = [record.getMessage() for record in caplog.records]
    assert len(lines) == 1
    assert lines[0].startswith("2 records hold a series with no numbers at all")


def test_a_paraphraser_stub_is_converted_and_logged(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    shard = _write_shard(
        tmp_path,
        {
            "category": ["Health"],
            "dataset": ["BIDMC32HR"],
            "caption0": ["a caption."],
            "caption1": ["Paraphrase 1:"],
            "caption2": ["The second paraphrase:"],
            "caption3": ["a third rewrite."],
            "time_series": [[[1.0, 2.0]]],
        },
        "train-00000-of-00017.parquet",
    )
    source = SlipSource(shards=(shard,), meta_csv=_meta(tmp_path))
    with caplog.at_level(logging.WARNING):
        records = SlipConnector().convert([source]).records
    assert "2 captions hold a paraphraser stub" in caplog.text
    # The stub is still what the record states: it is what the release ships.
    assert [a.value for a in records[0].annotations if a.key == "caption1"] == ["Paraphrase 1:"]


def test_captions_cut_mid_sentence_are_counted_once(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    shard = _write_shard(
        tmp_path,
        {
            "category": ["Health"],
            "dataset": ["BIDMC32HR"],
            "caption0": ["the readings has no trend,"],
            "caption1": ["and this one stops too,"],
            "caption2": ["a second rewrite."],
            "caption3": ["a third rewrite."],
            "time_series": [[[1.0, 2.0]]],
        },
        "train-00000-of-00017.parquet",
    )
    with caplog.at_level(logging.WARNING):
        SlipConnector().convert([SlipSource(shards=(shard,), meta_csv=_meta(tmp_path))])
    assert "2 captions stop mid-sentence" in caplog.text


def test_the_task_stream_counts_nothing_however_often_it_is_read(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    # The stream is read after convert has reported, so anything counted there is counted twice.
    shard = _write_shard(
        tmp_path,
        {
            "category": ["Health"],
            "dataset": ["BIDMC32HR"],
            "caption0": ["Paraphrase 1:"],
            "caption1": ["stops here,"],
            "caption2": ["a second rewrite."],
            "caption3": ["a third rewrite."],
            "time_series": [[[1.0, 2.0]]],
        },
        "train-00000-of-00017.parquet",
    )
    dataset = SlipConnector().convert([SlipSource(shards=(shard,), meta_csv=_meta(tmp_path))])
    with caplog.at_level(logging.WARNING):
        # convert has already reported this row's stub and its cut caption. The stream adds nothing.
        caplog.clear()
        list(_iter_tasks(dataset.records))
        list(_iter_tasks(dataset.records))
    assert caplog.records == []


def test_a_row_naming_a_corpus_meta_csv_does_not_state_raises(tmp_path: Path) -> None:
    # A corpus the table does not name means the release grew one, and its rate is unknown
    # rather than absent. Falling back to an ordinal axis would hide that.
    shard = _write_shard(
        tmp_path,
        {
            "category": ["Health"],
            "dataset": ["SomethingNew"],
            "caption0": ["a caption."],
            "caption1": ["a rewrite."],
            "caption2": ["a second rewrite."],
            "caption3": ["a third rewrite."],
            "time_series": [[[1.0, 2.0]]],
        },
        "train-00000-of-00017.parquet",
    )
    with pytest.raises(TimeFFormatError, match=r"meta\.csv does not name"):
        SlipConnector().convert([SlipSource(shards=(shard,), meta_csv=_meta(tmp_path))])


def test_a_shard_whose_name_states_no_number_raises(tmp_path: Path) -> None:
    # The record id carries the shard's number. A shard named some other way would renumber every
    # record of the build, and nothing downstream could tell.
    shard = _shard(tmp_path, "train-3-of-17.parquet")
    with pytest.raises(TimeFFormatError, match=r"train-NNNNN-of-NNNNN"):
        SlipConnector().convert([SlipSource(shards=(shard,), meta_csv=_meta(tmp_path))])


def test_a_badly_named_shard_stops_the_build_before_any_row_is_read(tmp_path: Path) -> None:
    # Every shard's name is checked before the first row is decoded, so a release that renames its
    # last shard fails in milliseconds instead of after every shard before it has been converted.
    good = _shard(tmp_path, "train-00000-of-00002.parquet")
    bad = _shard(tmp_path, "train-1-of-00002.parquet")
    with pytest.raises(TimeFFormatError, match=r"train-NNNNN-of-NNNNN"):
        SlipConnector().convert([SlipSource(shards=(good, bad), meta_csv=_meta(tmp_path))])
    assert slip._row_group.cache_info().misses == 0


def test_a_row_that_states_no_caption_raises(tmp_path: Path) -> None:
    # Every row of the pinned revision states four captions. A null is a changed release, and a
    # substituted empty string would report itself as a caption that stops mid-sentence.
    shard = _write_shard(
        tmp_path,
        {
            "category": ["Health"],
            "dataset": ["BIDMC32HR"],
            "caption0": ["a caption."],
            "caption1": [None],
            "caption2": ["a second rewrite."],
            "caption3": ["a third rewrite."],
            "time_series": [[[1.0, 2.0]]],
        },
        "train-00000-of-00017.parquet",
    )
    with pytest.raises(TimeFFormatError, match=r"row 0: caption1 states nothing"):
        SlipConnector().convert([SlipSource(shards=(shard,), meta_csv=_meta(tmp_path))])


def test_a_clean_shard_logs_nothing(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    source = SlipSource(shards=(_shard(tmp_path),), meta_csv=_meta(tmp_path))
    with caplog.at_level(logging.WARNING):
        dataset = SlipConnector().convert([source])
        list(_iter_tasks(dataset.records))
    assert caplog.records == []


def test_a_write_gives_the_values_and_the_captions_back_and_reads_each_row_group_once(tmp_path: Path) -> None:
    shard = _write_shard(
        tmp_path,
        {
            "category": ["Energy"] * 4,
            "dataset": ["Wind Farms"] * 4,
            "caption0": ["a caption.", "another.", "a third.", "a fourth."],
            "caption1": ["a rewrite."] * 4,
            "caption2": ["a second rewrite."] * 4,
            "caption3": ["a third rewrite."] * 4,
            "time_series": [
                [[1.0, 2.0], [3.0, 4.0]],
                [[5.0, 6.0]],
                [[7.0, 8.0], [9.0, 10.0]],
                [[11.0, 12.0]],
            ],
        },
        row_group_size=2,
    )
    dataset = SlipConnector().convert([SlipSource(shards=(shard,), meta_csv=_meta(tmp_path))])
    store = tmp_path / "store"
    with TimeFWriter(store, dataset) as writer:
        writer.write()
    # Two row groups, and the write asked for the values in the order convert built them, so each
    # was decoded once. A signal-major write order would decode the two-signal row groups again.
    assert slip._row_group.cache_info().misses == 2

    version_dir = store / "leochen085" / "slip" / "1.0.0"
    with TimeFReader(DatasetVersion.open_local(version_dir)) as reader:
        read_back = reader.read()
        tasks = reader.tasks
    values = [s.to_numpy().tolist() for r in read_back.records for s in r.time_series]
    assert values == [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0], [9.0, 10.0], [11.0, 12.0]]
    # Every task's answer resolves to a caption the record carries.
    captions = {a.id: a.value for r in read_back.records for a in r.annotations}
    assert len(tasks) == 16
    answers = [captions[a] for t in tasks for a in t.target_annotation_ids]
    assert answers.count("a caption.") == 1
    assert answers.count("a rewrite.") == 4
