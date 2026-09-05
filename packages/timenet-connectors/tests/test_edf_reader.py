from datetime import date, datetime, time
from fractions import Fraction
from pathlib import Path

import edfio
import numpy as np
import pytest

from timenet.errors import TimeFFormatError
from timenet_connectors.bases.edf import reader


_NUM_RECORDS = 10
_FAST_RATE = 100
_SLOW_RATE = 1
# EDF states the subtype in a 44-byte field that starts 192 bytes into the header.
_RESERVED_FIELD = slice(192, 236)
# It states the recording identification in an 80-byte field that starts at byte 88.
_RECORDING_FIELD_START = 88


def _signals() -> list[edfio.EdfSignal]:
    rng = np.random.default_rng(0)
    return [
        edfio.EdfSignal(
            rng.uniform(-100, 100, _FAST_RATE * _NUM_RECORDS),
            _FAST_RATE,
            label="EEG Fpz-Cz",
            physical_dimension="uV",
            physical_range=(-192.0, 192.0),
            digital_range=(-2048, 2047),
        ),
        edfio.EdfSignal(
            rng.uniform(-1, 1, _SLOW_RATE * _NUM_RECORDS),
            _SLOW_RATE,
            label="Resp oro-nasal",
            physical_dimension="V",
        ),
    ]


def _write_psg(tmp_path: Path, name: str = "psg.edf") -> Path:
    edf = edfio.Edf(
        _signals(),
        patient=edfio.Patient(code="X", sex="F", name="Female_33yr"),
        recording=edfio.Recording(startdate=date(1989, 4, 24)),
        starttime=time(16, 13, 0),
        data_record_duration=1,
    )
    path = tmp_path / name
    edf.write(path)
    return path


def _write_tenth_second_psg(tmp_path: Path) -> Path:
    # A record of 0.1 s holds 10 records of a 100 Hz signal and none of a 1 Hz one, thus
    # this file carries the fast signal alone.
    rng = np.random.default_rng(0)
    edf = edfio.Edf(
        [
            edfio.EdfSignal(
                rng.uniform(-100, 100, 10 * _NUM_RECORDS), _FAST_RATE, label="EEG Fpz-Cz", physical_dimension="uV"
            )
        ],
        recording=edfio.Recording(startdate=date(1989, 4, 24)),
        data_record_duration=0.1,
    )
    path = tmp_path / "tenth.edf"
    edf.write(path)
    return path


def _write_scoring(tmp_path: Path, name: str = "scoring.edf") -> Path:
    edf = edfio.Edf(
        [edfio.EdfSignal(np.zeros(120), 1.0, label="marker")],
        recording=edfio.Recording(startdate=date(1989, 4, 24)),
        data_record_duration=30,
        annotations=[
            edfio.EdfAnnotation(0.0, 30.0, "Sleep stage W"),
            edfio.EdfAnnotation(30.0, 60.0, "Sleep stage 1"),
            edfio.EdfAnnotation(90.0, 30.0, "Sleep stage R"),
        ],
    )
    path = tmp_path / name
    edf.write(path)
    return path


def _patch(path: Path, start: int, value: bytes) -> None:
    raw = bytearray(path.read_bytes())
    raw[start : start + len(value)] = value
    path.write_bytes(bytes(raw))


def test_open_edf_reads_every_header_field(tmp_path):
    header = reader.open_edf(_write_psg(tmp_path)).header
    assert header.start_time == datetime(1989, 4, 24, 16, 13, 0)
    assert header.patient_id == "X F X Female_33yr"
    assert header.num_records == _NUM_RECORDS
    assert header.record_duration == Fraction(1)
    assert header.signals == ("EEG Fpz-Cz", "Resp oro-nasal")
    assert header.units == ("uV", "V")
    assert header.samples_per_record == (100, 1)


def test_open_edf_holds_a_record_duration_that_is_not_a_whole_second(tmp_path):
    header = reader.open_edf(_write_tenth_second_psg(tmp_path)).header
    assert header.record_duration == Fraction(1, 10)
    assert header.samples_per_record == (10,)


def test_open_edf_refuses_a_file_that_is_not_edf(tmp_path):
    path = tmp_path / "notes.txt"
    path.write_text("this file holds no EDF header", encoding="utf-8")
    with pytest.raises(TimeFFormatError, match="cannot read this file as EDF"):
        reader.open_edf(path)


def test_open_edf_refuses_a_file_that_lost_whole_records(tmp_path):
    path = _write_psg(tmp_path)
    raw = path.read_bytes()
    record_bytes = (_FAST_RATE + _SLOW_RATE) * 2
    path.write_bytes(raw[: len(raw) - 3 * record_bytes])
    with pytest.raises(TimeFFormatError, match="ends before its own header says"):
        reader.open_edf(path)


def test_open_edf_refuses_a_file_that_lost_part_of_a_record(tmp_path):
    path = _write_psg(tmp_path)
    raw = path.read_bytes()
    path.write_bytes(raw[: len(raw) - 50])
    with pytest.raises(TimeFFormatError, match="ends before its own header says"):
        reader.open_edf(path)


def test_open_edf_keeps_a_file_whose_two_start_dates_disagree(tmp_path):
    # A release states the start date twice, and the two fields can hold different dates.
    # edfio warns about that, and the reader must pass the warning on and read the file.
    path = _write_psg(tmp_path)
    _patch(path, _RECORDING_FIELD_START, b"Startdate 23-APR-1989")
    with pytest.warns(UserWarning, match="Different values in startdate fields"):
        header = reader.open_edf(path).header
    assert header.num_records == _NUM_RECORDS


def test_open_edf_refuses_a_file_whose_start_date_is_anonymized(tmp_path):
    edf = edfio.Edf(_signals(), data_record_duration=1)
    edf.anonymize()
    path = tmp_path / "anonymous.edf"
    edf.write(path)
    with pytest.raises(TimeFFormatError, match="its header is anonymized"):
        reader.open_edf(path)


def test_open_edf_refuses_a_discontinuous_file(tmp_path):
    path = _write_scoring(tmp_path)
    _patch(path, _RESERVED_FIELD.start, b"EDF+D")
    with pytest.raises(TimeFFormatError, match="discontinuous EDF\\+D"):
        reader.open_edf(path)


def test_open_edf_reads_a_continuous_edf_plus(tmp_path):
    path = _write_scoring(tmp_path)
    assert path.read_bytes()[_RESERVED_FIELD].startswith(b"EDF+C")
    assert reader.open_edf(path).header.num_records == 4


def test_open_edf_reads_a_plain_edf_whose_subtype_field_is_empty(tmp_path):
    path = _write_psg(tmp_path)
    assert path.read_bytes()[_RESERVED_FIELD].strip() == b""
    assert reader.open_edf(path).header.signals == ("EEG Fpz-Cz", "Resp oro-nasal")


def test_read_channel_gives_every_sample_of_one_channel(tmp_path):
    file = reader.open_edf(_write_psg(tmp_path))
    assert len(reader.read_signal(file, 0)) == _FAST_RATE * _NUM_RECORDS
    assert len(reader.read_signal(file, 1)) == _SLOW_RATE * _NUM_RECORDS


def test_read_channel_refuses_a_channel_the_file_does_not_hold(tmp_path):
    file = reader.open_edf(_write_psg(tmp_path))
    with pytest.raises(TimeFFormatError, match="signal 2 does not exist"):
        reader.read_signal(file, 2)


def test_read_record_gives_one_array_for_each_channel(tmp_path):
    record = reader.read_record(reader.open_edf(_write_psg(tmp_path)), 3)
    assert [len(one) for one in record] == [_FAST_RATE, _SLOW_RATE]


def test_read_record_gives_the_same_values_as_the_whole_channel(tmp_path):
    file = reader.open_edf(_write_psg(tmp_path))
    # Read the records first. Reading the signal would hold it, and the records would then
    # come from what is held rather than from the file.
    records = [reader.read_record(file, index)[0] for index in (0, 4, _NUM_RECORDS - 1)]
    whole = reader.read_signal(file, 0)
    for index, record in zip((0, 4, _NUM_RECORDS - 1), records, strict=True):
        start = index * _FAST_RATE
        np.testing.assert_array_equal(record, whole[start : start + _FAST_RATE])


def test_read_record_holds_a_record_duration_that_is_not_a_whole_second(tmp_path):
    file = reader.open_edf(_write_tenth_second_psg(tmp_path))
    records = [reader.read_record(file, index)[0] for index in range(_NUM_RECORDS)]
    whole = reader.read_signal(file, 0)
    for index, record in enumerate(records):
        assert len(record) == 10
        np.testing.assert_array_equal(record, whole[index * 10 : index * 10 + 10])


def test_read_record_refuses_a_record_after_the_end(tmp_path):
    file = reader.open_edf(_write_psg(tmp_path))
    with pytest.raises(TimeFFormatError, match="record 10 does not exist"):
        reader.read_record(file, _NUM_RECORDS)


def test_read_record_reads_no_channel_whole(tmp_path, monkeypatch):
    # ``signal.digital`` reads a whole signal and holds it. Reading one record must not do
    # that, so watch what the conversion is handed: one record of values, not the signal.
    file = reader.open_edf(_write_psg(tmp_path))
    handed = []
    convert = reader._convert_signal

    def watch(signal, digital):
        handed.append(len(digital))
        return convert(signal, digital)

    monkeypatch.setattr(reader, "_convert_signal", watch)
    reader.read_record(file, 3)

    assert handed == [_FAST_RATE, _SLOW_RATE]
    assert all(signal._digital is None for signal in file.handle.signals)


def test_build_channel_loader_reads_the_channel_when_it_is_called(tmp_path):
    file = reader.open_edf(_write_psg(tmp_path))
    load = reader.build_signal_loader(file, 1)
    assert len(load()) == _SLOW_RATE * _NUM_RECORDS


def test_read_annotations_gives_each_labelled_entry_in_microseconds(tmp_path):
    entries = reader.read_annotations(reader.open_edf(_write_scoring(tmp_path)))
    assert entries == (
        reader.EdfAnnotation(0, 30_000_000, "Sleep stage W"),
        reader.EdfAnnotation(30_000_000, 60_000_000, "Sleep stage 1"),
        reader.EdfAnnotation(90_000_000, 30_000_000, "Sleep stage R"),
    )


def test_read_annotations_drops_the_lists_that_only_timestamp_a_record(tmp_path):
    edf = edfio.Edf(
        [edfio.EdfSignal(np.zeros(120), 1.0, label="marker")],
        recording=edfio.Recording(startdate=date(1989, 4, 24)),
        data_record_duration=30,
        annotations=[
            edfio.EdfAnnotation(0.0, 30.0, "Sleep stage W"),
            edfio.EdfAnnotation(60.0, None, ""),
            edfio.EdfAnnotation(90.0, 30.0, "Sleep stage R"),
        ],
    )
    path = tmp_path / "timestamped.edf"
    edf.write(path)

    file = reader.open_edf(path)
    assert any(not entry.text for entry in file.handle.annotations)
    assert tuple(entry.label for entry in reader.read_annotations(file)) == ("Sleep stage W", "Sleep stage R")


def test_read_annotations_reads_the_file_the_caller_already_opened(tmp_path, monkeypatch):
    # An EDF+ recording carries its own annotations, thus a caller that already holds the
    # recording reads them from it and opens nothing.
    file = reader.open_edf(_write_scoring(tmp_path))
    monkeypatch.setattr(reader, "open_edf", lambda path: pytest.fail(f"opened {path} a second time"))
    assert len(reader.read_annotations(file)) == 3


def test_compute_signal_end_microseconds_multiplies_the_records_out(tmp_path):
    file = reader.open_edf(_write_scoring(tmp_path))
    assert reader.compute_signal_end_microseconds(file.header) == 120_000_000


def test_compute_signal_end_microseconds_refuses_a_part_of_a_microsecond():
    header = reader.EdfHeader(
        start_time=datetime(1989, 4, 24, 16, 13, 0),
        patient_id="X",
        num_records=1,
        record_duration=Fraction(1, 3),
        signals=("EEG",),
        units=("uV",),
        samples_per_record=(1,),
    )
    with pytest.raises(TimeFFormatError, match="not a whole number of microseconds"):
        reader.compute_signal_end_microseconds(header)


def test_convert_digital_to_physical_uses_the_range_of_its_own_file():
    value = reader.convert_digital_to_physical(
        np.array([220]), digital_min=-2048, digital_max=2047, physical_min=-192.0, physical_max=192.0
    )
    assert value[0] == pytest.approx(20.6769, abs=1e-4)


def test_convert_digital_to_physical_refuses_an_empty_digital_range():
    with pytest.raises(TimeFFormatError, match="no scale"):
        reader.convert_digital_to_physical(
            np.array([0]), digital_min=5, digital_max=5, physical_min=0.0, physical_max=1.0
        )
