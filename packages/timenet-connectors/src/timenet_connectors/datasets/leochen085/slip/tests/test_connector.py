"""The corpus connector, against a synthetic shard written into ``tmp_path``."""

import logging
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from timenet.dataset.axis import OrdinalAxis, RegularAxis
from timenet_connectors.datasets.leochen085.slip.connector import (
    SlipConnector,
    SlipSource,
    _iter_tasks,
)


def _shard(directory: Path, name: str = "train-00003-of-00017.parquet") -> Path:
    table = pa.table(
        {
            "category": ["Health", "Energy"],
            "dataset": ["BIDMC32HR", "Wind Farms"],
            "caption0": ["a caption.", "another."],
            "caption1": ["a rewrite.", "another rewrite."],
            "caption2": ["a second rewrite.", "and another."],
            "caption3": ["a third rewrite.", "one more."],
            "time_series": [[[1.0, 2.0, 3.0]], [[4.0, 5.0], [6.0, 7.0]]],
        }
    )
    path = directory / name
    pq.write_table(table, path)
    return path


def _meta(directory: Path) -> Path:
    path = directory / "meta.csv"
    path.write_text(
        "Domain,Dataset,Freq,Source\n"
        "Health,BIDMC32HR,-,https://example.test/a\n"
        "Energy,Wind Farms,4 sec,https://example.test/b\n",
        encoding="utf-8",
    )
    return path


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


def test_a_record_carries_its_corpus_its_domain_and_where_that_corpus_was_published(tmp_path: Path) -> None:
    source = SlipSource(shards=(_shard(tmp_path),), meta_csv=_meta(tmp_path))
    records = SlipConnector().convert([source]).records
    found = {a.key: a.value for a in records[0].annotations}
    assert found == {
        "source_dataset": "BIDMC32HR",
        "domain": "Health",
        "source_url": "https://example.test/a",
    }


def test_four_tasks_per_row_each_holding_one_caption(tmp_path: Path) -> None:
    shard = _shard(tmp_path)
    tasks = list(_iter_tasks((shard,)))
    assert len(tasks) == 8
    first = tasks[:4]
    assert [t.target for t in first] == ["a caption.", "a rewrite.", "a second rewrite.", "a third rewrite."]
    assert {t.record_ids for t in first} == {("slip-shard-00003-row-000000",)}
    # A streamed task states no id, so nothing here asserts one.
    assert all(t.id for t in first)


def test_the_task_stream_gives_the_same_tasks_when_read_again(tmp_path: Path) -> None:
    # The build reads the stream more than once, so a generator that empties itself is a bug.
    shard = _shard(tmp_path)
    once = [(t.record_ids, t.target) for t in _iter_tasks((shard,))]
    again = [(t.record_ids, t.target) for t in _iter_tasks((shard,))]
    assert once == again
    assert len(once) == 8


def test_a_series_with_no_numbers_is_kept_annotated_and_logged(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    table = pa.table(
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
        }
    )
    shard = tmp_path / "train-00000-of-00017.parquet"
    pq.write_table(table, shard)
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
    table = pa.table(
        {
            "category": ["Health"],
            "dataset": ["BIDMC32HR"],
            "caption0": ["a caption."],
            "caption1": ["Paraphrase 1:"],
            "caption2": ["The second paraphrase:"],
            "caption3": ["a third rewrite."],
            "time_series": [[[1.0, 2.0]]],
        }
    )
    shard = tmp_path / "train-00000-of-00017.parquet"
    pq.write_table(table, shard)
    with caplog.at_level(logging.WARNING):
        tasks = list(_iter_tasks((shard,)))
    # The stub is still the task's target: it is what the release states.
    assert tasks[1].target == "Paraphrase 1:"
    assert "2 captions hold a paraphraser stub" in caplog.text


def test_captions_cut_mid_sentence_are_counted_once(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    table = pa.table(
        {
            "category": ["Health"],
            "dataset": ["BIDMC32HR"],
            "caption0": ["the readings has no trend,"],
            "caption1": ["and this one stops too,"],
            "caption2": ["a second rewrite."],
            "caption3": ["a third rewrite."],
            "time_series": [[[1.0, 2.0]]],
        }
    )
    shard = tmp_path / "train-00000-of-00017.parquet"
    pq.write_table(table, shard)
    with caplog.at_level(logging.WARNING):
        list(_iter_tasks((shard,)))
    assert "2 captions stop mid-sentence" in caplog.text


def test_a_clean_shard_logs_nothing(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    source = SlipSource(shards=(_shard(tmp_path),), meta_csv=_meta(tmp_path))
    with caplog.at_level(logging.WARNING):
        SlipConnector().convert([source])
        list(_iter_tasks(source.shards))
    assert caplog.records == []
