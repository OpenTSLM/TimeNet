from timenet.format.schemas import task_schema
from timenet.types import TaskType
from timenet.writer.encodings import task_dictionary


def test_target_columns_are_dictionary_encoded():
    # Guards the target rename: the categorical list and the schema columns must stay in sync, else
    # the target columns silently lose dictionary+RLE encoding with no test or error to catch it.
    dict_cols = task_dictionary(task_schema(TaskType.CLASSIFICATION))
    assert "target" in dict_cols
    assert "target_schema" in dict_cols
