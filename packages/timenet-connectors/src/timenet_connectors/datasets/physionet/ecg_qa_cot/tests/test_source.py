import csv

import pytest

from timenet.errors import TimeNetConnectorError
from timenet_connectors.datasets.physionet.ecg_qa_cot.connector import _load_template_answers as load_template_answers


def _answers_file(tmp_path, classes):
    path = tmp_path / "answers.csv"
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["template_id", "classes", "question_type"])
        writer.writerow([0, "['yes']", "a multiline\nquestion type"])
        writer.writerow([1, classes, "question"])
    return path


@pytest.mark.parametrize("classes", ["'yes'", "('yes', 'no')", "{'yes': 'no'}", "['yes', 1]", "[True]", "[None]"])
def test_answer_options_require_a_list_of_strings(tmp_path, classes):
    path = _answers_file(tmp_path, classes)
    with pytest.raises(TimeNetConnectorError) as error:
        load_template_answers(path)
    assert str(error.value).startswith(f"{path}:4: classes")


@pytest.mark.parametrize(("classes", "cause"), [("[broken", SyntaxError), ("[unknown]", ValueError)])
def test_malformed_answer_literals_keep_source_location_and_cause(tmp_path, classes, cause):
    path = _answers_file(tmp_path, classes)
    with pytest.raises(TimeNetConnectorError) as error:
        load_template_answers(path)
    assert str(error.value) == f"{path}:4: classes: invalid Python literal"
    assert isinstance(error.value.__cause__, cause)


def test_answer_options_preserve_strings_order_and_duplicates(tmp_path):
    classes = [" no ", "yes", "yes", "001", "", "line\nbreak"]
    path = _answers_file(tmp_path, repr(classes))
    assert load_template_answers(path) == {0: ("yes",), 1: tuple(classes)}
