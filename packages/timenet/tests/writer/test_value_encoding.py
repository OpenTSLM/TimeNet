import json
import logging

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from timenet.dataset import TimeFDataset, TimeSeries
from timenet.dataset.axis import RegularAxis
from timenet.errors import TimeFValidationError
from timenet.format.constants import DEFAULT_ROW_GROUP_TARGET_BYTES
from timenet.manifest import Manifest
from timenet.reader import TimeFReader
from timenet.registry import DatasetVersion
from timenet.types import DatasetMetadata, Domain, License, TimeSeriesSpec, Version, ureg
from timenet.writer import TimeFWriter
from timenet.writer.encodings import applied_matches, shard_dictionary, shard_encoding, values_encoding_of
from timenet.writer.value_encoding import (
    DICT_MAX_CARDINALITY,
    SAMPLE_MAX_VALUES,
    ValueEncoding,
    distinct_bit_patterns,
    encoding_for_cardinality,
    sample_values,
    select_value_encoding,
)


_RATE_HZ = 100
_N = 20_000


def _spec(spec_type):
    return TimeSeriesSpec(
        spec_type=spec_type,
        name=spec_type.title(),
        unit_value=ureg.dimensionless,
    )


def quantized(n=_N, step=0.001, seed=0):
    """A signal on a coarse grid: few distinct values, like a wfdb physical conversion."""
    rng = np.random.default_rng(seed)
    return (np.round(rng.normal(scale=0.3, size=n) / step) * step).astype(np.float32)


def continuous(n=_N, seed=1):
    """A signal where almost every value is distinct, like z-normalized TSQA."""
    return np.random.default_rng(seed).normal(size=n).astype(np.float32)


def _dataset(channels, dataset_id="timenet/encoding", **metadata_kwargs):
    """Build a dataset from ``{(spec_type, channel): values}``."""
    dataset = TimeFDataset(
        metadata=DatasetMetadata(
            dataset_id=dataset_id,
            dataset_version=Version(1, 0, 0),
            name="Encoding",
            description="Fixture for values-encoding selection.",
            license=License.CC_BY_4_0,
            domains=(Domain.GENERAL,),
            **metadata_kwargs,
        )
    )
    series = [
        TimeSeries(
            spec=_spec(spec_type),
            channel=channel,
            time_axis=RegularAxis.from_rate_hz(_RATE_HZ),
            loader=(lambda captured=values: pa.array(captured, type=pa.float32())),
            time_series_id=f"ts-{spec_type}-{channel}",
            n_values=len(values),
        )
        for (spec_type, channel), values in channels.items()
    ]
    dataset.add_sample(time_series=tuple(series), sample_id="sample-0")
    dataset.derive_schema()
    return dataset


def _write(tmp_path, dataset, **kwargs):
    with TimeFWriter(tmp_path, dataset, **kwargs) as writer:
        writer.write()
    return tmp_path / dataset.metadata.dataset_id / str(dataset.metadata.dataset_version)


def _manifest(version_dir):
    return Manifest.from_json((version_dir / "manifest.json").read_text())


# ---- the rule ----------------------------------------------------------------------------------


def test_low_cardinality_selects_dictionary():
    assert select_value_encoding([quantized()]) is ValueEncoding.DICTIONARY


def test_high_cardinality_selects_byte_stream_split():
    # 20k values would fit a dictionary; the rule needs enough of them to exceed the threshold.
    assert select_value_encoding([continuous(n=SAMPLE_MAX_VALUES)]) is ValueEncoding.BYTE_STREAM_SPLIT


def test_threshold_is_inclusive_at_the_boundary():
    assert encoding_for_cardinality(DICT_MAX_CARDINALITY) is ValueEncoding.DICTIONARY
    assert encoding_for_cardinality(DICT_MAX_CARDINALITY + 1) is ValueEncoding.BYTE_STREAM_SPLIT


def test_plain_is_never_selected_automatically():
    for distinct in (0, 1, DICT_MAX_CARDINALITY, DICT_MAX_CARDINALITY + 1, 10**9):
        assert encoding_for_cardinality(distinct) is not ValueEncoding.PLAIN


def test_empty_input_selects_dictionary():
    assert select_value_encoding([]) is ValueEncoding.DICTIONARY


def test_signed_zeros_count_as_two_patterns():
    # A Parquet dictionary keys on stored bytes, so -0.0 and 0.0 take two slots despite comparing equal.
    assert distinct_bit_patterns(np.array([0.0, -0.0, 0.0], dtype=np.float32)) == 2


def test_sample_is_bounded_and_strides_the_whole_chunk():
    values = np.arange(SAMPLE_MAX_VALUES * 4, dtype=np.float32)
    sample = sample_values([values])
    assert sample.size == SAMPLE_MAX_VALUES
    assert sample[-1] > values.size / 2, "a prefix would only ever see the start of a chunk"


def test_selection_is_deterministic():
    values = quantized()
    assert len({select_value_encoding([values.copy()]) for _ in range(5)}) == 1


def test_default_row_group_holds_a_full_sample():
    # The rule decides from one buffered row group, so a row-group target below the sample size would
    # silently shrink the sample and invalidate the calibration behind DICT_MAX_CARDINALITY.
    # Lowering DEFAULT_ROW_GROUP_TARGET_BYTES means re-running the value-encoding sweep.
    assert DEFAULT_ROW_GROUP_TARGET_BYTES // 4 >= SAMPLE_MAX_VALUES


# ---- end-to-end selection ----------------------------------------------------------------------


def test_writer_picks_per_modality(tmp_path):
    version_dir = _write(
        tmp_path,
        _dataset(
            {
                ("ecg", "I"): quantized(),
                ("embedding", "e"): continuous(n=SAMPLE_MAX_VALUES),
            }
        ),
    )
    assert _manifest(version_dir).value_encoding == {
        "ecg": ValueEncoding.DICTIONARY.value,
        "embedding": ValueEncoding.BYTE_STREAM_SPLIT.value,
    }


def test_shards_are_single_modality(tmp_path):
    version_dir = _write(
        tmp_path,
        _dataset({("ecg", "I"): quantized(), ("embedding", "e"): continuous(n=SAMPLE_MAX_VALUES)}),
    )
    shards = sorted(version_dir.glob("time_series/shard-*.parquet"))
    assert len(shards) == 2, "one modality per shard, so one encoding per shard is always right"
    for shard in shards:
        spec_types = set(pq.read_table(shard, columns=["spec_type"]).column("spec_type").to_pylist())
        assert len(spec_types) == 1


def test_every_manifest_shard_exists_and_is_non_empty(tmp_path):
    # Modality rotation advances the shard index; a mismatched index would list a file nobody wrote.
    version_dir = _write(
        tmp_path,
        _dataset({("ecg", "I"): quantized(), ("embedding", "e"): continuous(n=SAMPLE_MAX_VALUES)}),
        row_group_target_bytes=4096,
        shard_target_bytes=4096,
        chunk_max_bytes=4096,
    )
    manifest = _manifest(version_dir)
    assert manifest.files.time_series
    for rel in manifest.files.time_series:
        assert pq.ParquetFile(version_dir / rel.path).metadata.num_rows > 0


def test_writing_twice_reaches_the_same_encoding(tmp_path):
    first = _manifest(_write(tmp_path / "a", _dataset({("ecg", "I"): quantized()})))
    second = _manifest(_write(tmp_path / "b", _dataset({("ecg", "I"): quantized()})))
    assert first.value_encoding == second.value_encoding


# ---- overrides ---------------------------------------------------------------------------------


@pytest.mark.parametrize("forced", [e.value for e in ValueEncoding])
def test_writer_argument_forces_an_encoding(tmp_path, forced):
    version_dir = _write(tmp_path, _dataset({("ecg", "I"): quantized()}), value_encoding=forced)
    assert _manifest(version_dir).value_encoding == {"ecg": forced}


def test_unknown_value_encoding_is_rejected_before_staging(tmp_path):
    dataset = _dataset({("ecg", "I"): quantized()})
    with pytest.raises(TimeFValidationError, match="value_encoding"):
        TimeFWriter(tmp_path, dataset, value_encoding="RLE")
    assert not list(tmp_path.rglob("*.parquet"))


# ---- a forced encoding needs a backend that applies it -----------------------------------------


def test_forced_argument_rejects_a_backend_that_cannot_apply_it(tmp_path):
    # Only Parquet applies a values encoding; forcing one onto zarr is a misconfiguration, not a
    # silent no-op.
    dataset = _dataset({("ecg", "I"): quantized()})
    with pytest.raises(TimeFValidationError, match="value_encoding"):
        TimeFWriter(tmp_path, dataset, values_backend="zarr", value_encoding="dictionary")
    assert not list(tmp_path.rglob("*.zarr"))


def test_auto_allows_a_backend_without_an_encoding_choice(tmp_path):
    # auto is every backend's default, so the guard must not trip on it.
    dataset = _dataset({("ecg", "I"): quantized()})
    TimeFWriter(tmp_path, dataset, values_backend="zarr")  # constructs without raising


# ---- decision log ------------------------------------------------------------------------------

_ENCODING_LOGGER = "timenet.values_backends.parquet.writer"


def _encoding_logs(caplog):
    return [record.getMessage() for record in caplog.records if record.name == _ENCODING_LOGGER]


def test_auto_mode_logs_the_decision(tmp_path, caplog):
    with caplog.at_level(logging.INFO, logger=_ENCODING_LOGGER):
        _write(tmp_path, _dataset({("ecg", "I"): quantized()}))
    assert any("ecg" in m and "dictionary" in m and "auto" in m for m in _encoding_logs(caplog))


def test_forced_mode_logs_the_encoding(tmp_path, caplog):
    with caplog.at_level(logging.INFO, logger=_ENCODING_LOGGER):
        _write(tmp_path, _dataset({("ecg", "I"): quantized()}), value_encoding="plain")
    assert any("ecg" in m and "plain" in m and "forced" in m for m in _encoding_logs(caplog))


# ---- self-check and round-trip -------------------------------------------------------------------


@pytest.mark.parametrize("forced", [e.value for e in ValueEncoding])
def test_self_check_accepts_every_encoding(tmp_path, forced):
    version_dir = _write(tmp_path, _dataset({("ecg", "I"): quantized()}), value_encoding=forced)
    shard = next(version_dir.glob("time_series/shard-*.parquet"))
    assert applied_matches(ValueEncoding(forced), values_encoding_of(str(shard)))


@pytest.mark.parametrize("forced", [e.value for e in ValueEncoding])
def test_round_trip_is_bit_exact(tmp_path, forced):
    values = quantized()
    version_dir = _write(tmp_path, _dataset({("ecg", "I"): values}), value_encoding=forced)
    with TimeFReader(DatasetVersion.open_local(version_dir)) as reader:
        restored = reader.read().samples[0].time_series[0].to_arrow().to_numpy(zero_copy_only=False)
    assert np.array_equal(restored.view(np.uint32), values.view(np.uint32))


def test_reader_needs_no_encoding_hint(tmp_path):
    # Parquet is self-describing: drop value_encoding from the manifest and the values still read back,
    # which is why the reader needed no change and the map is provenance rather than contract.
    values = quantized()
    version_dir = _write(tmp_path, _dataset({("ecg", "I"): values}))
    manifest_path = version_dir / "manifest.json"
    document = json.loads(manifest_path.read_text())
    assert document.pop("value_encoding") == {"ecg": ValueEncoding.DICTIONARY.value}
    manifest_path.write_text(json.dumps(document, indent=2))

    assert Manifest.from_json(manifest_path.read_text()).value_encoding == {}
    with TimeFReader(DatasetVersion.open_local(version_dir)) as reader:
        restored = reader.read().samples[0].time_series[0].to_arrow().to_numpy(zero_copy_only=False)
    assert np.array_equal(restored.view(np.uint32), values.view(np.uint32))


def test_self_check_rejects_an_encoding_that_did_not_land():
    # What the check exists for: pyarrow drops a column encoding whose path does not match, and the
    # file then carries pyarrow's default instead of the selection.
    assert not applied_matches(ValueEncoding.BYTE_STREAM_SPLIT, {"PLAIN", "RLE"})
    assert not applied_matches(ValueEncoding.PLAIN, {"BYTE_STREAM_SPLIT", "RLE"})
    assert not applied_matches(ValueEncoding.DICTIONARY, {"BYTE_STREAM_SPLIT", "RLE"})


def test_plain_is_not_satisfied_by_a_dictionary_page():
    # A dictionary chunk carries PLAIN for its dictionary page, so PLAIN alone must not pass as proof
    # that a column asked for plain got it.
    assert not applied_matches(ValueEncoding.PLAIN, {"PLAIN", "RLE", "RLE_DICTIONARY"})


def test_self_check_tolerates_the_dictionary_to_plain_fallback():
    # Parquet abandons a dictionary that outgrows its page limit and finishes the chunk in PLAIN.
    # That is lossless and not something the writer chose, so it must not fail the build.
    assert applied_matches(ValueEncoding.DICTIONARY, {"PLAIN", "RLE"})


def test_a_wrong_column_path_is_caught(tmp_path):
    # Proves the self-check has teeth: ask for BYTE_STREAM_SPLIT on a path that does not exist and
    # pyarrow silently writes its default instead.
    from timenet.format.schemas import default_id_types, shard_schema  # noqa: PLC0415
    from timenet.writer import encodings  # noqa: PLC0415

    schema = shard_schema(default_id_types())
    path = tmp_path / "typo.parquet"
    kwargs = encodings.parquet_kwargs(
        dictionary_columns=shard_dictionary(ValueEncoding.BYTE_STREAM_SPLIT),
        column_encoding={"values.element": "BYTE_STREAM_SPLIT"},  # the wrong path
        compression="zstd",
        compression_level=3,
    )
    values = pa.ListArray.from_arrays(pa.array([0, _N], type=pa.int32()), pa.array(quantized(), type=pa.float32()))
    table = pa.Table.from_pydict(
        {
            "time_series_id": ["ts-0"],
            "spec_type": ["ecg"],
            "channel": ["I"],
            "chunk_idx": [0],
            "n_values": [_N],
            "values": values,
            "time_offsets_us": pa.array([None], type=pa.list_(pa.int64())),
        },
        schema=schema,
    )
    with pq.ParquetWriter(path, schema, **kwargs) as writer:
        writer.write_table(table)
    assert not applied_matches(ValueEncoding.BYTE_STREAM_SPLIT, values_encoding_of(str(path)))


# ---- parquet option mapping ----------------------------------------------------------------------


def test_dictionary_goes_through_use_dictionary_not_column_encoding():
    # pyarrow rejects a column that appears in both lists, so the two must stay disjoint.
    assert "values.list.element" in shard_dictionary(ValueEncoding.DICTIONARY)
    assert "values.list.element" not in shard_encoding(ValueEncoding.DICTIONARY)
    for encoding in (ValueEncoding.PLAIN, ValueEncoding.BYTE_STREAM_SPLIT):
        assert "values.list.element" not in shard_dictionary(encoding)
        assert "values.list.element" in shard_encoding(encoding)
