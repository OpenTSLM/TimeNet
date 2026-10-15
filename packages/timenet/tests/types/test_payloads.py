import copy
from dataclasses import replace
from fractions import Fraction
import pickle

import pyarrow as pa

from timenet.dataset import Record, RegularAxis, Signal, Source, TimeFDataset
from timenet.testing import make_dataset
from timenet.types import (
    Annotation,
    AnswerTask,
    ArrayPayload,
    InputModality,
    Payload,
    PayloadDescriptor,
    TimeSeriesSpec,
)


def test_schema_and_task_inspection_do_not_read_annotation_or_text_bodies():
    calls = []

    def load_report():
        calls.append("report")
        return "Sinus rhythm."

    report = Payload.from_loader(load_report, descriptor=PayloadDescriptor("str", length=13))
    dataset = TimeFDataset(metadata=make_dataset().metadata)
    signal = Signal.from_values(
        [1.0, 2.0],
        spec=TimeSeriesSpec(spec_type="ecg", name="ECG", unit_value=None),
        name="ECG",
        time_axis=RegularAxis(period_us=Fraction(1000)),
    )
    record = dataset.add_record(record=Record(sources=(Source(name="ECG", signals=(signal,)),)))
    annotation = record.annotate(Annotation(key="report", value=report))
    task = dataset.add_task(task=AnswerTask(inputs=(record,), targets=("normal",), prompt=report))
    task.set_prompt(report)
    task.set_rationale(report)
    task.set_targets((report,))
    assert dataset.derive_schema().annotations[0].value_type == "str"
    assert InputModality.TEXT in task.resolved_input_modalities
    assert "Sinus rhythm" not in repr(task.prompt_handle())
    assert copy.deepcopy(annotation).handle() is report
    assert replace(annotation, description="Updated description").handle() is report
    assert task.prompt_handle() is report
    assert task.rationale_handle() is report
    assert task.targets_handle() == (report,)
    assert replace(task, prompt=report).prompt_handle() is report
    assert calls == []
    assert task.prompt() == "Sinus rhythm."
    assert calls == ["report"]
    assert annotation.value() == "Sinus rhythm."
    assert task.rationale() == "Sinus rhythm."
    assert task.targets() == ("Sinus rhythm.",)
    assert calls == ["report"] * 4


def test_inline_payloads_remain_immutable_through_list_reads_and_pickle():
    values = ["normal", "AF"]
    payload = Payload.from_value(values)
    values.append("changed source list")
    returned = payload.read()
    returned.append("changed returned list")
    assert pickle.loads(pickle.dumps(payload)).read() == ["normal", "AF"]


def test_array_payload_uses_bounded_reads_without_calling_full_loader():
    calls = []
    values = pa.array([1.0, 2.0, 3.0, 4.0])

    def load_all():
        calls.append("all")
        return values

    def load_range(start, stop):
        calls.append((start, stop))
        return values.slice(start, stop - start)

    payload = ArrayPayload.from_loader(
        load_all, descriptor=PayloadDescriptor("array", length=4, dtype="float64"), read_range=load_range
    )
    assert payload.read_steps(1, 3).to_pylist() == [2.0, 3.0]
    assert calls == [(1, 3)]


def test_signal_rejects_reuse_of_a_different_array_contract_without_reading():
    from timenet.errors import TimeFValidationError  # noqa: PLC0415

    calls = []

    def load():
        calls.append("read")
        return pa.array([1, 2], type=pa.int64())

    payload = ArrayPayload.from_loader(load, descriptor=PayloadDescriptor("array", length=2, dtype="int64"))
    import pytest  # noqa: PLC0415

    with pytest.raises(TimeFValidationError, match="different array contract"):
        Signal.from_loader(
            loader=payload,
            n_values=2,
            name="ECG",
            time_axis=RegularAxis.from_rate_hz(100),
            spec=TimeSeriesSpec(spec_type="ecg", name="ECG", unit_value=None),
        )
    assert calls == []
