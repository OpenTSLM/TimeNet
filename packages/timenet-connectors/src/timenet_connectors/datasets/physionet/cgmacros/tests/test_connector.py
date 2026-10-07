from pathlib import Path
import shutil

from PIL import Image
from pydantic import ValidationError
import pytest

from timenet.dataset import IrregularAxis
from timenet.errors import TimeFFormatError
from timenet.registry import LocalRegistry
from timenet.types import License, ureg
from timenet_connectors.datasets.physionet.cgmacros.connector import CGMacrosConnector, Meal


@pytest.fixture
def source(tmp_path):
    root = tmp_path / "raw"
    participant = root / "CGMacros-001"
    participant.mkdir(parents=True)
    shutil.copyfile(Path(__file__).parent / "fixtures/CGMacros-001.csv", participant / "CGMacros-001.csv")
    for name in ("bio", "microbes", "gut_health_test"):
        (root / f"{name}.csv").write_text("subject,first,Time (t),Time (t)\n1,5.4,09:00,10:00\n")
    return root


def test_original_values_missingness_events_and_supplementary_columns(source):
    dataset = CGMacrosConnector().convert([source])
    assert dataset.metadata.dataset_id == "physionet/cgmacros"
    assert dataset.metadata.license == License.CC_BY_NC_SA_4_0
    assert not dataset.tasks
    (record,) = dataset.records
    signals = {signal.name: signal for signal in record.signals}
    assert signals["Libre GL"].to_arrow().to_pylist() == [98, 100, 102]
    assert signals["Dexcom GL"].to_arrow().to_pylist() == [100, None, 104]
    assert signals["HR"].to_arrow().to_pylist() == [70, 72, None]
    assert signals["Intensity"].to_arrow().to_pylist() == [0, None, 3]
    assert signals["Intensity"].spec.unit_value == ureg.Unit("dimensionless")
    assert isinstance(signals["Libre GL"].time_axis, IrregularAxis)
    assert signals["Libre GL"].time_offsets_us().tolist() == [0, 60_000_000, 180_000_000]
    assert record.subject_ids == ("CGMacros-001",)
    assert record.start_time.timestamp is None
    assert record.metadata["supplementary"] == {
        name: {"columns": ["first", "Time (t)", "Time (t)"], "values": ["5.4", "09:00", "10:00"]}
        for name in ("bio", "microbes", "gut_health_test")
    }
    meals = record.metadata["meals"]
    assert isinstance(meals, list) and isinstance(meals[0], dict)
    assert meals[0]["Carbs"] == 20
    assert record.annotations[0].span is not None
    assert record.annotations[0].span.start_us == 60_000_000


def test_roundtrip_and_photo(source, tmp_path):
    photos = source / "CGMacros-001" / "photos"
    photos.mkdir()
    Image.new("RGB", (2, 3), (10, 20, 30)).save(photos / "meal.jpg")
    connector = CGMacrosConnector()
    dataset = connector.convert([source])
    assert len(dataset.records) == 2
    assert dataset.records[1].signals[0].spec.value_shape == (3, 2, 3)
    registry = LocalRegistry(tmp_path / "registry")
    registry.store(dataset, values_backend=connector.values_backend)
    with registry.open_reader("physionet/cgmacros", "1.0.0") as reader:
        restored = list(reader.iter_records())
        assert {record.id for record in restored} == {record.id for record in dataset.records}
        assert next(s for r in restored for s in r.signals if s.name == "Dexcom GL").to_arrow().to_pylist() == [
            100,
            None,
            104,
        ]


def test_bad_measurement_is_not_loaded_during_convert(source):
    path = source / "CGMacros-001/CGMacros-001.csv"
    path.write_text(path.read_text().replace(",98,", ",broken,"))
    dataset = CGMacrosConnector().convert([source])
    with pytest.raises(ValueError, match="broken"):
        next(s for s in dataset.records[0].signals if s.name == "Libre GL").to_arrow()


def test_meal_validation_rejects_nonfinite_nutrients():
    with pytest.raises(ValidationError):
        Meal.model_validate({"Timestamp": "2024-01-25 09:13:00", "Meal Type": "lunch", "Carbs": float("inf")})


def test_empty_archive_is_rejected(tmp_path):
    with pytest.raises(TimeFFormatError, match="no participant"):
        CGMacrosConnector().convert([tmp_path])
