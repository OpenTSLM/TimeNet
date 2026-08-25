from timenet.types import ClassificationTask
import timenet_connectors


def test_build_then_load_round_trip(monkeypatch, tmp_path):
    # timenet_connectors.builder/load anchor on the same default registry, so a build here loads back.
    monkeypatch.setenv("TIMENET_HOME", str(tmp_path))

    version_dir = timenet_connectors.build("timenet/test-mean")
    assert (version_dir / "manifest.json").exists()

    dataset = timenet_connectors.load("timenet/test-mean")
    assert len(dataset.samples) == 1000

    x, y = dataset.to_features_and_targets(task=ClassificationTask)
    assert len(x) == 1000
    assert set(y.to_pylist()) == {"above_zero", "below_zero"}
