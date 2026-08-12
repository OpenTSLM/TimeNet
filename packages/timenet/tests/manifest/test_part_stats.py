import pytest

from timenet.errors import InvalidManifestError
from timenet.manifest import Manifest, ManifestFiles, PartStat
from timenet.testing import make_dataset


def _manifest(part_stats, files, id_encoding=None):
    ds = make_dataset()
    return Manifest(
        dataset_id=ds.metadata.dataset_id,
        metadata=ds.metadata,
        files=files,
        part_stats=part_stats,
        id_encoding=id_encoding or {},
    )


def test_part_stats_round_trip_string_ids():
    files = ManifestFiles(
        samples=("samples/part-00000000.parquet", "samples/part-00000001.parquet"),
        annotations=("annotations/part-00000000.parquet",),
        time_series_index=("time_series_index/part-00000000.parquet",),
    )
    part_stats = {
        "samples": (
            PartStat("samples/part-00000000.parquet", 2, ("sample-0",), ("sample-1",)),
            PartStat("samples/part-00000001.parquet", 1, ("sample-2",), ("sample-2",)),
        ),
        "annotations": (PartStat("annotations/part-00000000.parquet", 0, None, None),),
        "time_series_index": (
            PartStat(
                "time_series_index/part-00000000.parquet",
                3,
                ("sample-0", "ts-0"),
                ("sample-2", "ts-9"),
            ),
        ),
    }
    restored = Manifest.from_json(_manifest(part_stats, files).to_json())
    assert restored.part_stats == part_stats


def test_part_stats_uuid16_keys_hex_encode():
    key = b"\x11" * 16
    files = ManifestFiles(
        samples=("samples/part-00000000.parquet",),
        annotations=("annotations/part-00000000.parquet",),
        time_series_index=("time_series_index/part-00000000.parquet",),
    )
    part_stats = {
        "samples": (PartStat("samples/part-00000000.parquet", 1, (key,), (key,)),),
        "annotations": (PartStat("annotations/part-00000000.parquet", 0, None, None),),
        "time_series_index": (PartStat("time_series_index/part-00000000.parquet", 0, None, None),),
    }
    text = _manifest(part_stats, files, id_encoding={"sample_id": "uuid16"}).to_json()
    assert key.hex() in text
    restored = Manifest.from_json(text)
    assert restored.part_stats["samples"][0].first_key == (key,)


def test_part_stats_paths_disagreeing_with_files_raise():
    files = ManifestFiles(
        samples=("samples/part-00000000.parquet",),
        annotations=("annotations/part-00000000.parquet",),
        time_series_index=("time_series_index/part-00000000.parquet",),
    )
    part_stats = {
        "samples": (PartStat("samples/WRONG.parquet", 1, ("a",), ("a",)),),
        "annotations": (PartStat("annotations/part-00000000.parquet", 0, None, None),),
        "time_series_index": (PartStat("time_series_index/part-00000000.parquet", 0, None, None),),
    }
    text = _manifest(part_stats, files).to_json()
    with pytest.raises(InvalidManifestError):
        Manifest.from_json(text)
