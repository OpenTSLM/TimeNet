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
    ClassificationTemplate,
    DatasetMetadata,
    LabelSet,
    License,
    OptionKeyStyle,
    PromptTemplate,
    TemplateOption,
    TimeSeriesSpec,
    Version,
    ureg,
)


SPEC = TimeSeriesSpec(spec_type="voltage", name="Voltage", unit_value=ureg.millivolt)
KEYS = LabelSet(["(a)", "(b)"])
TRENDS = LabelSet(["up", "down"])

KEYED_PROMPT = "Which trend?\n(a) up\n(b) none of the above\nOnly answer (a) or (b)."
KEYED = ClassificationTemplate(
    prompt_template=PromptTemplate(
        source=(
            "{% block prompt %}Which trend?\n{% for option in options %}{{ option.marker }} {{ option.text }}\n"
            '{% endfor %}Only answer {{ keys | join(" or ") }}.{% endblock %}'
            "{% block answer %}{{ option.key }}{% endblock %}"
        )
    ),
    options=[
        TemplateOption(label="(a)", text="up"),
        TemplateOption(label="(b)", text="none of the above", pinned=True),
    ],
    key_style=OptionKeyStyle(
        enumerator=ClassificationEnumerator.LATIN_LOWERCASE,
        marker_prefix="(",
        marker_suffix=")",
        reference_prefix="(",
        reference_suffix=")",
    ),
    allowable_enumerators={ClassificationEnumerator.LATIN_LOWERCASE, ClassificationEnumerator.LATIN_UPPERCASE},
    reproduces_published_prompt=True,
)
# A template without keys, for tasks that have no prompt. Two tasks share it.
LISTED = ClassificationTemplate(
    prompt_template=PromptTemplate(
        source=(
            '{% block prompt %}Which trend? Options: {{ options | map(attribute="text") | join(", ") }}.{% endblock %}'
            "{% block answer %}{{ option.text }}{% endblock %}"
        )
    ),
    options=[TemplateOption(label="up"), TemplateOption(label="down")],
)


def _write_release(tmp_path):
    """Write one release with a keyed template, a template that two tasks share, a task without one, and an answer task.

    Returns:
        The path of the control database.
    """
    dataset = TimeFDataset(
        metadata=DatasetMetadata(
            dataset_id="org/classification-templates",
            dataset_version=Version(1, 0, 0),
            name="Classification templates",
            description="Classification tasks with templates",
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
            ClassificationTask(
                inputs=(record,), prompt=KEYED_PROMPT, targets="(a)", label_set=KEYS, template=KEYED
            ),  # 1
            ClassificationTask(inputs=(record,), targets="up", label_set=TRENDS, template=LISTED),  # 2
            ClassificationTask(inputs=(record,), targets="down", label_set=TRENDS, template=LISTED.model_copy()),  # 3
            ClassificationTask(inputs=(record,), targets="down", label_set=TRENDS),  # 4
            AnswerTask(inputs=(record,), prompt="Which trend?", targets="It rises."),  # 5
        ]
    )
    path = tmp_path / "control.duckdb"
    DuckDBControlWriter(path, external=PackedValueWriter(path.parent, StorageSources())).write_hierarchy(dataset)
    return path


def _read_classification_tasks(path) -> list[ClassificationTask]:
    with DuckDBControlReader(path) as reader:
        tasks = reader.read_tasks(reader.read_records())
    return [task for task in tasks if isinstance(task, ClassificationTask)]


def test_a_template_is_read_back_as_written_and_equal_templates_are_stored_once(tmp_path):
    path = _write_release(tmp_path)

    with connect_control(path, read_only=True) as connection:
        audit_control_database(connection, require_chunks=False)
        stored = connection.execute("SELECT count(*) FROM classification_templates").fetchone()
    keyed, listed, same_listed, without_template = _read_classification_tasks(path)

    assert stored == (2,)
    assert keyed.template == KEYED
    assert listed.template == LISTED
    assert listed.template is same_listed.template
    assert without_template.template is None
    # The task that was read back shows its options as the task that was written.
    assert keyed.get_option_presentation(use_shuffled_label_key=False).prompt == KEYED_PROMPT
    assert keyed.get_option_presentation(seed=4).answers in {("(a)",), ("(A)",)}


def test_a_database_of_the_schema_version_before_templates_is_still_read(tmp_path):
    path = _write_release(tmp_path)
    with connect_control(path) as connection:
        connection.execute("DROP TABLE task_classification_templates")
        connection.execute("DROP TABLE classification_templates")
        connection.execute("UPDATE control_metadata SET value = '3' WHERE key = 'schema_version'")

    with connect_control(path, read_only=True) as connection:
        audit_control_database(connection, require_chunks=False)
    assert [task.template for task in _read_classification_tasks(path)] == [None, None, None, None]


@pytest.mark.parametrize(
    ("change", "problem"),
    [
        ("UPDATE task_classification_templates SET task_id = 5 WHERE task_id = 1", "is not a classification task"),
        (
            "UPDATE task_classification_templates SET classification_template_id = 1 WHERE task_id = 2",
            "an option for a label that is not in its label set",
        ),
        ("UPDATE task_classification_templates SET classification_template_id = 9 WHERE task_id = 2", "missing"),
        ("UPDATE classification_templates SET key_enumerator = 'greek' WHERE key_enumerator IS NOT NULL", "invalid"),
        ("UPDATE classification_templates SET key_marker_suffix = NULL WHERE key_enumerator IS NOT NULL", "invalid"),
    ],
)
def test_the_audit_rejects_stored_templates_that_break_a_rule(tmp_path, change, problem):
    path = _write_release(tmp_path)
    with connect_control(path) as connection:
        connection.execute(change)

    with connect_control(path, read_only=True) as connection, pytest.raises(TimeFFormatError, match=problem):
        audit_control_database(connection, require_chunks=False)


def test_a_stored_template_that_is_not_valid_jinja_is_rejected_when_it_is_read(tmp_path):
    path = _write_release(tmp_path)
    with connect_control(path) as connection:
        connection.execute("UPDATE classification_templates SET source = '{% block prompt %}{{ options.pop() }}'")

    with pytest.raises(TimeFFormatError, match=r"stored classification template .* is not valid"):
        _read_classification_tasks(path)
