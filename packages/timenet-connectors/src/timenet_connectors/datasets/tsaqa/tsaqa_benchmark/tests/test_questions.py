import pytest

from timenet.errors import TimeFValidationError
from timenet_connectors.datasets.tsaqa.tsaqa_benchmark.questions import parse_question, series_text
from timenet_connectors.datasets.tsaqa.tsaqa_benchmark.vocabulary import QuestionFormat, SourceTask, Transform


_TF = "Respond ONLY with the letter of the correct choice (T or F).\n\nChoices:\nT: True.\nF: False.\n"
_MC = "Respond ONLY with the letter of the correct choice (A, B, C, or D).\n\nChoices:\n"


def test_reads_an_anomaly_question_without_series():
    text = f"Determine whether the given time series contains anomalies.\n{_TF}"
    question, series = parse_question(text, SourceTask.ANOMALY_DETECTION)
    assert question.format is QuestionFormat.TRUE_OR_FALSE
    assert question.letters == ("T", "F")
    assert question.segments == (text,)
    assert series == ()


def test_reads_a_classification_question_with_text_choices():
    text = (
        "Classify the given time series into one of the categories below.\n"
        "Respond ONLY with the letter of the correct choice (A, B, C).\n\n"
        "Choices:\nA: up\nB: down\nC: flat\n"
    )
    question, series = parse_question(text, SourceTask.CLASSIFICATION)
    assert question.format is QuestionFormat.MULTIPLE_CHOICE
    assert question.choices == (("A", "up"), ("B", "down"), ("C", "flat"))
    assert series == ()


def test_reads_a_transform_sequence_and_rebuilds_the_text():
    text = f"Is the following sequence the First Order Difference of the given time series?\n[0, -0.5, 2.0]\n{_TF}"
    question, series = parse_question(text, SourceTask.DATA_TRANSFORMATION)
    assert question.transform is Transform.FIRST_ORDER_DIFFERENCE
    assert question.series_labels == (None,)
    assert series == ([0.0, -0.5, 2.0],)
    assert question.text(["<series 2>"]).startswith("Is the following sequence the First Order Difference")
    assert "\n<series 2>\n" in question.text(["<series 2>"])
    assert question.text([series_text(values) for values in series]) == text


def test_reads_four_series_choices_and_a_puzzle():
    choices = "A: [1.0]\nB: [2.5]\nC: [-0.0]\nD: [0]\n"
    text = f"Which of the following choices is most likely the future continuation of the given time series?\n{_MC}{choices}"
    question, series = parse_question(text, SourceTask.TEMPORAL_RELATIONSHIP)
    assert question.series_labels == ("A", "B", "C", "D")
    assert question.text([series_text(values) for values in series]) == text
    puzzle = (
        "The given time series is the first patch of the sequence.\n"
        "Below are the remaining patches, labeled as A, B, C, and D.\n"
        "Arrange A, B, C, D in the correct order to reconstruct the original sequence.\n\n"
        f"Patches:\n{choices}"
    )
    question, series = parse_question(puzzle, SourceTask.TEMPORAL_RELATIONSHIP)
    assert question.format is QuestionFormat.PUZZLING
    assert [len(values) for values in series] == [1, 1, 1, 1]
    assert question.text(["<a>", "<b>", "<c>", "<d>"]).endswith("Patches:\nA: <a>\nB: <b>\nC: <c>\nD: <d>\n")


def test_series_text_spells_zero_as_the_authors_do():
    assert series_text([0.0, -0.0, 1.0, -0.0588, 12.777]) == "[0, -0.0, 1.0, -0.0588, 12.777]"


@pytest.mark.parametrize(
    ("text", "task", "message"),
    [
        (f"Determine whether it rains.\n{_TF}", SourceTask.ANOMALY_DETECTION, "Appendix B.7"),
        (
            f"Is the following sequence the Hilbert Transform of the given time series?\n[1.0]\n{_TF}",
            SourceTask.DATA_TRANSFORMATION,
            "known transform",
        ),
        (
            f"Is the following sequence the Fourier Transform of the given time series?\n[0.0]\n{_TF}",
            SourceTask.DATA_TRANSFORMATION,
            "not spelled",
        ),
        (
            f"Is the following sequence the Fourier Transform of the given time series?\n[NaN]\n{_TF}",
            SourceTask.DATA_TRANSFORMATION,
            "NaN",
        ),
        (
            f"Is the following sequence the Fourier Transform of the given time series?\n[]\n{_TF}",
            SourceTask.DATA_TRANSFORMATION,
            "nonempty",
        ),
        ("A question about <series 2>?\n" + _TF, SourceTask.CHARACTERIZATION, "<series"),
        (
            "Which is it?\nRespond ONLY with the letter of the correct choice (A, B).\n\nChoices:\nA: x\nB: y\nC: z\n",
            SourceTask.CHARACTERIZATION,
            "the instruction offers",
        ),
        ("Which is it?\n\nChoices:\nA: x\nB: y\n", SourceTask.CHARACTERIZATION, "Respond ONLY"),
        (
            f"Determine whether the given time series contains anomalies.\n{_TF}",
            SourceTask.CLASSIFICATION,
            "multiple-choice",
        ),
    ],
)
def test_malformed_questions_are_rejected(text, task, message):
    with pytest.raises(TimeFValidationError, match=message):
        parse_question(text, task)
