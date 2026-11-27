import pyarrow as pa
import pytest

from timenet.dataset import Record, RegularAxis, Signal, Source, TimeFDataset
from timenet.errors import TimeFFormatError
from timenet.format.control_audit import audit_control_database
from timenet.format.control_reader import DuckDBControlReader
from timenet.format.control_writer import DuckDBControlWriter
from timenet.format.duckdb import connect_control
from timenet.format.storage import StorageSources
from timenet.format.stored_values import PackedValueWriter
from timenet.types import (
    AnswerTask,
    ClassificationEnumerator,
    ClassificationTask,
    DatasetMetadata,
    LabelSet,
    License,
    OptionKeyReference,
    OptionKeyStyle,
    PromptOption,
    PromptOptions,
    PromptSpan,
    TimeSeriesSpec,
    Version,
    ureg,
)


SPEC = TimeSeriesSpec(spec_type="voltage", name="Voltage", unit_value=ureg.millivolt)
LETTERS = LabelSet(["(a)", "(b)"])
TRENDS = LabelSet(["up", "down"])

KEYED_PROMPT = "Which trend? (a) up (b) down. Only answer (a) or (b)."
KEYED = PromptOptions(
    options=[
        PromptOption(label="(a)", marker=PromptSpan(start=13, end=16), text=PromptSpan(start=17, end=19)),
        PromptOption(label="(b)", marker=PromptSpan(start=20, end=23), text=PromptSpan(start=24, end=28), pinned=True),
    ],
    key_style=OptionKeyStyle(
        enumerator=ClassificationEnumerator.LATIN_LOWERCASE,
        marker_prefix="(",
        marker_suffix=")",
        reference_prefix="(",
        reference_suffix=")",
    ),
    key_references=[
        OptionKeyReference(option_position=0, span=PromptSpan(start=42, end=45)),
        OptionKeyReference(option_position=1, span=PromptSpan(start=49, end=52)),
    ],
    allowable_enumerators={ClassificationEnumerator.LATIN_LOWERCASE, ClassificationEnumerator.LATIN_UPPERCASE},
)
NAMED_PROMPT = "Is the trend 'up' or 'down'?"
NAMED = PromptOptions(
    options=[
        PromptOption(label="up", text=PromptSpan(start=14, end=16)),
        PromptOption(label="down", text=PromptSpan(start=22, end=26)),
    ]
)
NOT_PRINTED = PromptOptions(options=[PromptOption(label="down"), PromptOption(label="up")])


def _write_release(tmp_path):
    """Write one release with each form of prompt options, a classification task without them, and an answer task.

    Returns:
        The path of the control database.
    """
    dataset = TimeFDataset(
        metadata=DatasetMetadata(
            dataset_id="org/prompt-options",
            dataset_version=Version(1, 0, 0),
            name="Prompt options",
            description="Classification tasks with prompt options",
            license=License.MIT,
        )
    )
    signal = Signal.from_loader(
        name="I",
        spec=SPEC,
        time_axis=RegularAxis.from_rate_hz(1),
        n_values=2,
        loader=lambda: pa.array([1.0, 2.0], type=pa.float32()),
    )
    record = dataset.add_record(record=Record(sources=(Source(name="ECG", signals=(signal,)),)))
    dataset.add_tasks(
        tasks=[
            ClassificationTask(  # 1
                inputs=(record,), prompt=KEYED_PROMPT, targets="(b)", label_set=LETTERS, prompt_options=KEYED
            ),
            ClassificationTask(  # 2
                inputs=(record,), prompt=NAMED_PROMPT, targets="up", label_set=TRENDS, prompt_options=NAMED
            ),
            ClassificationTask(  # 3
                inputs=(record,), prompt="Which trend?", targets="up", label_set=TRENDS, prompt_options=NOT_PRINTED
            ),
            ClassificationTask(inputs=(record,), targets="down", label_set=TRENDS),  # 4
            AnswerTask(inputs=(record,), prompt="Which trend?", targets="It rises."),  # 5
        ]
    )
    path = tmp_path / "control.duckdb"
    DuckDBControlWriter(path, external=PackedValueWriter(path.parent, StorageSources())).write_hierarchy(dataset)
    return path


def _read_prompt_options(path) -> list[PromptOptions | None]:
    with DuckDBControlReader(path) as reader:
        tasks = reader.read_tasks(reader.read_records())
    return [task.prompt_options for task in tasks if isinstance(task, ClassificationTask)]


def test_each_form_of_prompt_options_is_read_back_as_it_was_written(tmp_path):
    path = _write_release(tmp_path)

    with connect_control(path, read_only=True) as connection:
        audit_control_database(connection, require_chunks=False)
    assert _read_prompt_options(path) == [KEYED, NAMED, NOT_PRINTED, None]


def test_a_database_of_the_schema_version_before_prompt_options_is_still_read(tmp_path):
    path = _write_release(tmp_path)
    with connect_control(path) as connection:
        connection.execute("DROP TABLE task_prompt_options")
        connection.execute("UPDATE control_metadata SET value = '3' WHERE key = 'schema_version'")

    with connect_control(path, read_only=True) as connection:
        audit_control_database(connection, require_chunks=False)
    assert _read_prompt_options(path) == [None, None, None, None]


@pytest.mark.parametrize(
    ("change", "problem"),
    [
        ("UPDATE task_prompt_options SET task_id = 5 WHERE task_id = 1", "is not a classification task"),
        (
            "UPDATE task_prompt_options SET options = [{'label': 'sideways', 'marker_start': NULL, "
            "'marker_end': NULL, 'text_start': NULL, 'text_end': NULL, 'pinned': false}] WHERE task_id = 3",
            "a label that is not in its label set",
        ),
        ("UPDATE task_prompt_options SET key_enumerator = 'greek' WHERE task_id = 1", "an invalid key style"),
        ("UPDATE task_prompt_options SET key_marker_suffix = NULL WHERE task_id = 1", "an invalid key style"),
    ],
)
def test_the_audit_rejects_stored_prompt_options_that_break_a_rule(tmp_path, change, problem):
    path = _write_release(tmp_path)
    with connect_control(path) as connection:
        connection.execute(change)

    with connect_control(path, read_only=True) as connection, pytest.raises(TimeFFormatError, match=problem):
        audit_control_database(connection, require_chunks=False)
