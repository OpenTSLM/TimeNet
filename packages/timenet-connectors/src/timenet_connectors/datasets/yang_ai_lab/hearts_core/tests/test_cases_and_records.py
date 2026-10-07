import numpy as np
import pandas as pd
import pytest

from timenet.errors import TimeFFormatError
from timenet.types import ClassificationTask, TimeSeriesSpec, ureg
from timenet_connectors.datasets.yang_ai_lab.hearts_core.cases import Case, iter_cases
from timenet_connectors.datasets.yang_ai_lab.hearts_core.records import FrameLayout, frame_record, waveform_record
from timenet_connectors.datasets.yang_ai_lab.hearts_core.release import TaskDef, Waveform


@pytest.mark.parametrize("create_directory", [False, True])
def test_missing_or_empty_expected_task_is_rejected(tmp_path, create_directory):
    directory = "example/task"
    if create_directory:
        (tmp_path / directory).mkdir(parents=True)
    tasks = {directory: TaskDef(ClassificationTask)}
    with pytest.raises(TimeFFormatError, match=r"missing|no cases"):
        tuple(iter_cases(tmp_path, tasks))


def test_frame_record_annotates_corpus_qualified_subject(tmp_path):
    case = Case("example/task", TaskDef(ClassificationTask), 0, tmp_path / "0.pkl", {})
    frame = pd.DataFrame({"time": [0.0, 1.0], "value": [1.0, 2.0]})
    spec = TimeSeriesSpec(spec_type="example", name="Example", unit_value=ureg.dimensionless, dtype="float64")
    built = frame_record(case, None, {("frame",): frame}, FrameLayout({"value": spec}, ("time",)), subject="person")
    assert (
        next(annotation.value for annotation in built.record.annotations if annotation.key == "subject_id")
        == "example:person"
    )


def test_waveform_record_annotates_corpus_qualified_subject(tmp_path):
    case = Case("example/task", TaskDef(ClassificationTask), 0, tmp_path / "0.pkl", {"audio": np.array([0.0, 0.5])})
    record = waveform_record(case, Waveform(("audio",), rate=16000), subject="person")
    assert (
        next(annotation.value for annotation in record.annotations if annotation.key == "subject_id")
        == "example:person"
    )
