from pathlib import Path
from typing import ClassVar

import numpy as np
import pytest
import wfdb

from timenet.errors import TimeFFormatError, TimeFValidationError
from timenet.reader import TimeFReader
from timenet.registry import LocalRegistry
from timenet.types import (
    ClassificationTask,
    LocalizationMode,
    TemporalLocalizationTask,
    TimePoint,
)
from timenet.types.text import read_value
from timenet_connectors.datasets.physionet.mitdb import MitdbConnector
from timenet_connectors.datasets.physionet.mitdb.annotations import decode_quality
from timenet_connectors.datasets.physionet.mitdb.keys import AnnotationKey
from timenet_connectors.datasets.physionet.mitdb.loader import MitdbDatasetLoader
from timenet_connectors.datasets.physionet.mitdb.records import AXIS, record_metadata
from timenet_connectors.datasets.physionet.mitdb.release import PINNED_COUNTS, PinnedCounts
from timenet_connectors.datasets.physionet.mitdb.vocabulary import SignalQuality


_N_SAMPLES = 3600
"""Ten seconds at 360 Hz."""
_FIXTURE_COUNTS = PinnedCounts(records=2, beats=6, rhythm_labels=4, quality_labels=2, non_beats=4, comments=1)

# Each record's annotations, as the release writes them: (sample, code, subtype, auxiliary text).
_ANNOTATIONS = {
    "201": [
        (0, "+", 0, "(N"),
        (100, "N", 0, ""),
        (150, "~", 1, ""),
        (400, "V", 2, ""),
        (500, '"', 0, "MISSB"),
        (600, "+", 0, "(AFIB"),
        (700, "x", 0, ""),
        (800, "/", 0, ""),
        (900, "~", -1, ""),
        (1000, "|", 3, ""),
        (3599, "N", 0, ""),
    ],
    "202": [
        (5, "[", 0, ""),
        (10, "+", 0, "(VFL"),
        (20, "!", 0, ""),
        (30, "]", 0, ""),
        (40, "+", 0, "(N"),
        (50, "N", 0, ""),
    ],
}
_LEADS = {"201": ("MLII", "V1"), "202": ("V5", "MLII")}
_COMMENTS = {
    "201": ["68 M 1960 2851 x1", "Digoxin, Inderal", "The PVCs are uniform."],
    "202": ["-1 M 1960 2851 x1", "None"],
}


class _FixtureConnector(MitdbConnector):
    """The connector, pinned to the fixture's counts instead of the release's."""

    CARD = Path(__file__).parents[1] / "dataset.yaml"
    SOURCE_COUNTS: ClassVar[PinnedCounts] = _FIXTURE_COUNTS


def _digital(name: str) -> np.ndarray:
    """Each record's ADC samples, one column per signal, distinct per record and signal."""
    offset = 0 if name == "201" else 7
    ramp = (np.arange(_N_SAMPLES) * 3 + offset) % 2000
    return np.stack([ramp + 24, 2047 - ramp], axis=1).astype(np.int64)


def _write_record(root: Path, name: str, annotations: list[tuple[int, str, int, str]]) -> None:
    """Write a record's ``.dat`` and ``.atr`` with wfdb, and its ``.hea`` as the release writes it."""
    digital = _digital(name)
    wfdb.wrsamp(
        name,
        fs=360,
        units=["mV", "mV"],
        sig_name=list(_LEADS[name]),
        d_signal=digital,
        fmt=["212", "212"],
        adc_gain=[200.0, 200.0],
        baseline=[1024, 1024],
        write_dir=str(root),
    )
    # The release's gain field names no baseline and no unit.
    lines = [f"{name} 2 360 {_N_SAMPLES}"]
    for column, lead in enumerate(_LEADS[name]):
        values = digital[:, column]
        checksum = (int(values.sum()) + 32768) % 65536 - 32768
        lines.append(f"{name}.dat 212 200 11 1024 {values[0]} {checksum} 0 {lead}")
    lines += [f"# {comment}" for comment in _COMMENTS[name]]
    (root / f"{name}.hea").write_text("\n".join(lines) + "\n", encoding="utf-8")
    samples, symbols, subtypes, texts = zip(*annotations, strict=True)
    wfdb.wrann(
        name,
        "atr",
        sample=np.array(samples),
        symbol=list(symbols),
        subtype=np.array(subtypes),
        aux_note=list(texts),
        write_dir=str(root),
    )


def _release(tmp_path: Path, annotations: dict | None = None) -> Path:
    root = tmp_path / "mit-bih-arrhythmia-database-1.0.0"
    root.mkdir()
    (root / "RECORDS").write_text("201\n202\n", encoding="utf-8")
    for name, rows in (annotations or _ANNOTATIONS).items():
        _write_record(root, name, rows)
    return root


def _us(sample: int) -> int:
    return AXIS.time_offset_us(sample)


def _span(annotation) -> tuple[int, int]:
    return annotation.span.start_us, annotation.span.exclusive_end


def _output(dataset) -> dict:
    """Everything the dataset holds, as plain values, with records keyed by record name."""
    records = {}
    for record in dataset.records:
        metadata = record_metadata(record)
        records[metadata.record] = {
            "subject_id": metadata.subject_id,
            "leads": list(metadata.leads),
            "comments": list(metadata.comments),
            "facts": {a.key: read_value(a.value) for a in record.annotations if a.span is None},
            "labels": [
                (a.key, read_value(a.value), *_span(a), dict(a.metadata))
                for a in record.annotations
                if a.span is not None
            ],
            "signals": {
                s.name: (
                    s.n_values,
                    str(s.spec.unit_value),
                    [(a.key, read_value(a.value), *_span(a), dict(a.metadata)) for a in s.annotations],
                )
                for s in record.signals
            },
        }
    tasks = []
    for task in dataset.iter_tasks():
        name = record_metadata(task.inputs[0]).record
        if isinstance(task, ClassificationTask):
            tasks.append((name, "beat", task.scope.start_us, task.targets, task.target_schema))
        else:
            assert isinstance(task, TemporalLocalizationTask)
            keys = {a.key for a in task.target_annotations}
            tasks.append((name, *keys, len(task.target_annotations), task.mode))
    annotations = {a.key: read_value(a.value) for a in dataset.annotations}
    return {"annotations": annotations, "records": records, "tasks": tasks}


_END = _us(_N_SAMPLES)

_EXPECTED = {
    "annotations": {
        "source_url": "https://physionet.org/static/published-projects/mitdb/mit-bih-arrhythmia-database-1.0.0.zip",
        "source_sha256": "47b26926927c11bd9174154d367c811afc1b186f650f8a205931ca8c520f0a87",
        "beat_vocabulary": ["N", "L", "R", "A", "a", "J", "S", "V", "F", "!", "e", "j", "E", "/", "f", "Q"],
        # The 19 rhythms of the authors' dbnotes.c, in its order.
        "rhythm_vocabulary": [
            "(N",
            "(SBR",
            "(BII",
            "(B22",
            "(B3",
            "(SAB",
            "(PREX",
            "(AB",
            "(EAR",
            "(SVTA",
            "(AFL",
            "(AFIB",
            "(P",
            "(NOD",
            "(B",
            "(T",
            "(IVR",
            "(VT",
            "(VFL",
        ],
    },
    "records": {
        "201": {
            "subject_id": "201",
            "leads": ["MLII", "V1"],
            "comments": _COMMENTS["201"],
            "facts": {"age": 68, "sex": "M", "medications": "Digoxin, Inderal"},
            "labels": [
                ("beat", "N", _us(100), _us(100) + 1, {}),
                ("beat", "V", _us(400), _us(400) + 1, {"subtype": 2}),
                ("beat", "/", _us(800), _us(800) + 1, {}),
                ("beat", "N", _us(3599), _us(3599) + 1, {}),
                ("rhythm", "(N", 0, _us(600), {}),
                ("rhythm", "(AFIB", _us(600), _END, {}),
                ("event", "x", _us(700), _us(700) + 1, {}),
                ("event", "|", _us(1000), _us(1000) + 1, {"subtype": 3}),
                ("comment", "MISSB", _us(500), _us(500) + 1, {}),
            ],
            "signals": {
                "MLII": (
                    _N_SAMPLES,
                    "millivolt",
                    [
                        ("signal_quality", "noisy", _us(150), _us(900), {"subtype": 1}),
                        ("signal_quality", "unreadable", _us(900), _END, {"subtype": -1}),
                    ],
                ),
                "V1": (
                    _N_SAMPLES,
                    "millivolt",
                    [
                        ("signal_quality", "clean", _us(150), _us(900), {"subtype": 1}),
                        ("signal_quality", "unreadable", _us(900), _END, {"subtype": -1}),
                    ],
                ),
            },
        },
        "202": {
            # Records 201 and 202 come from the same subject.
            "subject_id": "201",
            "leads": ["V5", "MLII"],
            "comments": _COMMENTS["202"],
            # An age of -1 was not recorded, so there is no age.
            "facts": {"sex": "M", "medications": "None"},
            "labels": [
                ("beat", "!", _us(20), _us(20) + 1, {}),
                ("beat", "N", _us(50), _us(50) + 1, {}),
                ("rhythm", "(VFL", _us(10), _us(40), {}),
                ("rhythm", "(N", _us(40), _END, {}),
                ("event", "[", _us(5), _us(5) + 1, {}),
                ("event", "]", _us(30), _us(30) + 1, {}),
            ],
            "signals": {"V5": (_N_SAMPLES, "millivolt", []), "MLII": (_N_SAMPLES, "millivolt", [])},
        },
    },
    "tasks": [
        ("201", "beat", 4, LocalizationMode.EXHAUSTIVE),
        ("201", "rhythm", 2, LocalizationMode.EXHAUSTIVE),
        ("201", "beat", _us(100), ("N",), "beat_vocabulary"),
        ("201", "beat", _us(400), ("V",), "beat_vocabulary"),
        ("201", "beat", _us(800), ("/",), "beat_vocabulary"),
        ("201", "beat", _us(3599), ("N",), "beat_vocabulary"),
        ("202", "beat", 2, LocalizationMode.EXHAUSTIVE),
        ("202", "rhythm", 2, LocalizationMode.EXHAUSTIVE),
        ("202", "beat", _us(20), ("!",), "beat_vocabulary"),
        ("202", "beat", _us(50), ("N",), "beat_vocabulary"),
    ],
}


def test_converts_the_fixture(tmp_path):
    dataset = _FixtureConnector().convert([_release(tmp_path)])
    assert _output(dataset) == _EXPECTED


def test_signals_reread_their_record_in_millivolts(tmp_path):
    dataset = _FixtureConnector().convert([_release(tmp_path)])
    for record in dataset.records:
        name = record_metadata(record).record
        digital = _digital(name)
        # The Source keeps the header's order; Record.signals sorts by name.
        for column, signal in enumerate(record.sources[0].signals):
            values = signal.to_numpy()
            assert values.dtype == np.float32
            np.testing.assert_array_equal(values, ((digital[:, column] - 1024) / 200).astype(np.float32))
            # Every value is a whole ADC step.
            np.testing.assert_array_equal(np.round(values.astype(np.float64) * 200) + 1024, digital[:, column])


def test_storage_round_trip_keeps_the_whole_conversion(tmp_path):
    original = _FixtureConnector().convert([_release(tmp_path)])
    registry = LocalRegistry(tmp_path / "registry")
    registry.store(original)
    with TimeFReader(registry.open_version("physionet/mitdb", "1.0.0")) as reader:
        restored = reader.read()
    assert _output(original) == _EXPECTED
    assert _output(restored) == _EXPECTED


def test_tasks_point_at_the_records_own_labels(tmp_path):
    # The localization targets are the Record's own beat and rhythm annotations, and a classification
    # scope is its beat's point, which maps back to the beat's sample. No task carries anything else.
    loader = MitdbDatasetLoader.parse_from([_release(tmp_path)], counts=_FIXTURE_COUNTS)
    beat_annotations = list(loader.beat_annotation_tasks())
    rhythm_annotations = list(loader.rhythm_annotation_tasks())
    beat_classifications = list(loader.beat_classification_tasks())
    for task in beat_annotations:
        (record,) = task.inputs
        assert task.target_annotations == tuple(a for a in record.annotations if a.key == AnnotationKey.BEAT)
    for task in rhythm_annotations:
        (record,) = task.inputs
        assert task.target_annotations == tuple(a for a in record.annotations if a.key == AnnotationKey.RHYTHM)
    beat_samples = []
    for task in beat_classifications:
        assert isinstance(task.scope, TimePoint)
        beat_samples.append(AXIS.index_at_or_after(task.scope.start_us))
    assert beat_samples == [100, 400, 800, 3599, 20, 50]
    tasks = list(loader.tasks())
    # tasks() holds exactly the tasks of the three methods, record by record.
    by_record = [
        [beat, rhythm, *(task for task in beat_classifications if task.inputs == beat.inputs)]
        for beat, rhythm in zip(beat_annotations, rhythm_annotations, strict=True)
    ]
    assert tasks == [task for record_tasks in by_record for task in record_tasks]
    for task in tasks:
        assert task.split is None
        assert task.prompt is None
        assert task.input_annotations == ()
        assert dict(task.metadata) == {}
        if isinstance(task, ClassificationTask):
            assert isinstance(task.scope, TimePoint)
            assert task.scope.time_series is None


@pytest.mark.parametrize(
    ("subtype", "qualities"),
    [
        (0, ("clean", "clean")),
        (1, ("noisy", "clean")),
        (2, ("clean", "noisy")),
        (3, ("noisy", "noisy")),
        (0x11, ("unreadable", "clean")),
        (0x23, ("noisy", "unreadable")),
        (-1, ("unreadable", "unreadable")),
    ],
)
def test_signal_quality_decodes_as_dbnotes_does(subtype, qualities):
    assert decode_quality(subtype, "test") == tuple(SignalQuality(quality) for quality in qualities)


@pytest.mark.parametrize(("subtype", "message"), [(0x10, "unreadable but not noisy"), (4, "doesn't have")])
def test_signal_quality_rejects_an_undecodable_subtype(subtype, message):
    with pytest.raises(TimeFFormatError, match=message):
        decode_quality(subtype, "test")


def test_pinned_counts_mismatch_fails(tmp_path):
    with pytest.raises(TimeFValidationError, match="the pinned archive holds"):
        MitdbDatasetLoader.parse_from([_release(tmp_path)], counts=PINNED_COUNTS)


def _with(name: str, index: int, row: tuple[int, str, int, str]) -> dict:
    """The fixture's annotations, with one row of one record replaced."""
    rows = list(_ANNOTATIONS[name])
    rows[index] = row
    return {**_ANNOTATIONS, name: rows}


@pytest.mark.parametrize(
    ("annotations", "message"),
    [
        (_with("201", 5, (600, "+", 0, "(XYZ")), "not a rhythm label the authors name"),
        (_with("201", 1, (100, "N", 0, "text")), "with auxiliary text"),
        (_with("201", 4, (500, '"', 1, "MISSB")), "with subtype"),
        (_with("201", 1, (100, "B", 0, "")), "not one the release writes"),
        (_with("201", 8, (900, "~", 0x10, "")), "unreadable but not noisy"),
        (_with("201", 1, (0, "+", 0, "(AFIB")), "two changes of one kind"),
    ],
)
def test_malformed_annotations_fail_with_their_file(tmp_path, annotations, message):
    root = _release(tmp_path, annotations)
    with pytest.raises(TimeFFormatError, match=rf"201\.atr.*{message}"):
        MitdbDatasetLoader.parse_from([root], counts=_FIXTURE_COUNTS)


@pytest.mark.parametrize(
    ("old", "new", "message"),
    [
        ("# 68 M 1960 2851 x1\n", "# 0 M 1960 2851 x1\n", "states an age of 0"),
        ("# 68 M 1960 2851 x1\n", "# 68 X 1960 2851 x1\n", "doesn't start with age and sex"),
        ("# Digoxin, Inderal\n# The PVCs are uniform.\n", "", "a medications line"),
        ("201 2 360", "201 2 250", "expected 2 signals at 360 Hz"),
    ],
)
def test_malformed_headers_fail_with_their_file(tmp_path, old, new, message):
    root = _release(tmp_path)
    header = root / "201.hea"
    text = header.read_text(encoding="utf-8")
    assert old in text
    header.write_text(text.replace(old, new), encoding="utf-8")
    with pytest.raises(TimeFFormatError, match=rf"201\.hea.*{message}"):
        MitdbDatasetLoader.parse_from([root], counts=_FIXTURE_COUNTS)


def test_a_gain_with_a_unit_fails(tmp_path):
    root = _release(tmp_path)
    header = root / "201.hea"
    header.write_text(header.read_text(encoding="utf-8").replace(" 200 11 ", " 200/uV 11 ", 1), encoding="utf-8")
    with pytest.raises(TimeFFormatError, match=r"201\.hea.*default unit"):
        MitdbDatasetLoader.parse_from([root], counts=_FIXTURE_COUNTS)


def test_a_record_without_its_reference_annotations_fails(tmp_path):
    root = _release(tmp_path)
    (root / "202.atr").unlink()
    with pytest.raises(TimeFFormatError, match="record 202 has no atr file"):
        MitdbDatasetLoader.parse_from([root], counts=_FIXTURE_COUNTS)
