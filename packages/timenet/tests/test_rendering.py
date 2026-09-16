from fractions import Fraction

import pyarrow as pa

from timenet.dataset import Record, RegularAxis, Signal, Source
from timenet.rendering import TextPromptRenderer
from timenet.testing import CountingLoader
from timenet.types import Annotation, AnswerTask, TimeInterval, TimePoint, TimeSeriesSpec, ureg


def test_text_renderer_walks_the_hierarchy_without_loading_values():
    loader = CountingLoader([1.0, 2.0])
    signal = Signal(
        id="lead-i",
        name="I",
        spec=TimeSeriesSpec(
            spec_type="voltage",
            name="Voltage",
            unit_value=ureg.millivolt,
            dtype="float32",
        ),
        time_axis=RegularAxis(axis_id="ecg-axis", period_us=Fraction(2_000)),
        loader=loader,
        n_values=2,
    )
    signal.annotate(Annotation(key="lead_status", value="attached"))
    ecg = Source(id="ecg", name="ECG", signals=(signal,))
    monitor = Source(id="monitor", name="Bedside monitor", sources=(ecg,))
    record = Record(record_id="record-1", sources=(monitor,))
    patient_sex = record.annotate(Annotation(key="patient_sex", value="male"))
    task = AnswerTask(
        id="task-1",
        prompt="Diagnose this patient.",
        inputs=(record,),
        targets=("Normal", 0.98, TimeInterval.seconds(1, 2)),
        input_annotations=(patient_sex,),
    )

    example = TextPromptRenderer().render(task=task)

    assert example.prompt == (
        "Task: Diagnose this patient.\n"
        "Input annotations:\n"
        '  - patient_sex="male"\n'
        "Inputs:\n"
        "  1. Record record-1\n"
        "    Annotations:\n"
        '      - patient_sex="male"\n'
        "    Source Bedside monitor [monitor]\n"
        "      Source ECG [ecg]\n"
        "        Signal I [lead-i]\n"
        "          Spec voltage: Voltage; unit=millivolt; dtype=float32; values=2\n"
        "          Axis regular [ecg-axis]; period_us=2000; start_index=0\n"
        "          Annotations:\n"
        '            - lead_status="attached"'
    )
    assert example.target == '1. "Normal"\n2. 0.98\n3. time interval [1, 2) s'
    assert loader.calls == 0


def test_text_renderer_preserves_record_and_signal_target_order():
    signal = Signal(
        id="signal-1",
        name="Value",
        spec=TimeSeriesSpec(
            spec_type="value",
            name="Value",
            unit_value=ureg.dimensionless,
        ),
        time_axis=RegularAxis.from_rate_hz(1),
        data=pa.array([1.0], type=pa.float32()),
    )
    record = Record(
        record_id="record-1",
        sources=(Source(id="source-1", name="Sensor", signals=(signal,)),),
    )
    task = AnswerTask(inputs=(), targets=(signal, "then", record, TimePoint.seconds(3)))

    target = TextPromptRenderer().render(task=task).target

    assert target is not None
    assert target.index("1. Signal target") < target.index('2. "then"')
    assert target.index('2. "then"') < target.index("3. Record target")
    assert target.index("3. Record target") < target.index("4. time point at 3 s")
    assert TextPromptRenderer().render(task=AnswerTask(targets=())).target == ""
    assert TextPromptRenderer().render(task=AnswerTask()).target is None
