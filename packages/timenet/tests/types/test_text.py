import copy
from dataclasses import dataclass
import pickle

import pytest

from timenet.errors import TimeFValidationError
from timenet.types import (
    Annotation,
    AnswerTask,
    ClassificationTask,
    InputModality,
    TextHandle,
    TextListHandle,
    TimePoint,
    ValueHandle,
    read_value,
    read_values,
    value_type_of,
)
from timenet.types.text import InlineText, InlineTextList


@dataclass(frozen=True)
class _Location:
    """Stand in for a payload location, the content key of a stored handle."""

    handle: int


@dataclass(eq=False)
class _StoredText(TextHandle):
    """A stored string that records every read, like a handle to packed text."""

    text: str
    reads: int = 0

    @property
    def is_loaded(self) -> bool:
        return False

    @property
    def length(self) -> int:
        return len(self.text)

    def read(self) -> str:
        self.reads += 1
        return self.text

    def _content_key(self) -> _Location:
        return _Location(id(self))


@dataclass(eq=False)
class _StoredTextList(TextListHandle):
    """A stored list of strings that records every read."""

    texts: list[str]
    reads: int = 0

    @property
    def is_loaded(self) -> bool:
        return False

    @property
    def length(self) -> int:
        return len(self.texts)

    def read(self) -> list[str]:
        self.reads += 1
        return self.texts

    def _content_key(self) -> _Location:
        return _Location(id(self))


def test_inline_text_holds_one_loaded_string():
    text = InlineText("note")
    assert isinstance(text, TextHandle)
    assert isinstance(text, ValueHandle)
    assert not isinstance(text, TextListHandle)
    assert text.is_loaded
    assert text.length == 4
    assert text.read() == "note"


def test_inline_text_list_holds_one_loaded_list():
    texts = InlineTextList(["a", "bc"])
    assert isinstance(texts, TextListHandle)
    assert isinstance(texts, ValueHandle)
    assert not isinstance(texts, TextHandle)
    assert texts.is_loaded
    assert texts.length == 2
    assert texts.read() == ["a", "bc"]


@pytest.mark.parametrize("value", [["a"], 1, None])
def test_inline_text_takes_only_a_string(value):
    with pytest.raises(TimeFValidationError, match="must be a string"):
        InlineText(value)


@pytest.mark.parametrize("value", ["ab", ("a", "b"), ["a", 1]])
def test_inline_text_list_takes_only_a_list_of_strings(value):
    with pytest.raises(TimeFValidationError, match="list of strings"):
        InlineTextList(value)


@pytest.mark.parametrize("handle", [InlineText(""), InlineTextList([])])
def test_an_empty_handle_is_false(handle):
    assert not handle


@pytest.mark.parametrize("handle", [InlineText("x"), InlineTextList([""])])
def test_a_nonempty_handle_is_true(handle):
    assert handle


def test_length_and_truth_do_not_read_the_value():
    text = _StoredText("stored")
    texts = _StoredTextList(["a", "b"])
    assert text
    assert texts
    assert (text.length, texts.length) == (6, 2)
    assert text.reads == texts.reads == 0


def test_handles_of_one_kind_compare_by_content():
    assert InlineText("note") == InlineText("note")
    assert InlineText("note") != InlineText("nope")
    assert InlineTextList(["a", "b"]) == InlineTextList(["a", "b"])
    assert InlineTextList(["a", "b"]) != InlineTextList(["a", "c"])


def test_handles_compare_with_native_values_of_their_kind_from_either_side():
    assert InlineText("note") == "note"
    assert "note" == InlineText("note")  # noqa: SIM300 - the reflected comparison is the point
    assert InlineTextList(["a", "b"]) == ["a", "b"]
    assert ["a", "b"] == InlineTextList(["a", "b"])  # noqa: SIM300 - the reflected comparison is the point
    assert InlineText("note") != "other"
    assert InlineTextList(["a"]) != ["b"]


def test_a_string_handle_never_equals_a_list_handle_or_a_list():
    assert InlineText("a") != InlineTextList(["a"])
    assert InlineTextList(["a"]) != InlineText("a")
    assert InlineText("a") != ["a"]
    assert InlineTextList(["a"]) != "a"
    assert InlineTextList(["a"]) != ("a",)


def test_handles_of_different_kinds_compare_unequal_without_a_read():
    text = _StoredText("a")
    texts = _StoredTextList(["a"])
    assert text != texts
    assert texts != text
    assert text.__eq__(texts) is False
    assert text.reads == texts.reads == 0


def test_a_handle_never_equals_a_non_text_value():
    assert InlineText("1") != 1
    assert InlineText("note") != None  # noqa: E711
    assert InlineTextList([]) != None  # noqa: E711


def test_stored_and_inline_text_with_the_same_content_are_equal():
    text = _StoredText("note")
    texts = _StoredTextList(["a", "b"])
    assert text == InlineText("note")
    assert texts == InlineTextList(["a", "b"])
    assert text.reads == texts.reads == 1


def test_different_lengths_compare_unequal_without_a_read():
    text = _StoredText("note")
    texts = _StoredTextList(["a"])
    assert text != InlineText("a longer note")
    assert text != "a longer note"
    assert texts != InlineTextList(["a", "b"])
    assert texts != ["a", "b"]
    assert text.reads == texts.reads == 0


def test_a_handle_equals_itself_without_a_read():
    text = _StoredText("note")
    assert text == text
    assert text.reads == 0


@pytest.mark.parametrize("handle", [InlineText("note"), InlineTextList(["a"]), _StoredText("note")])
def test_handles_are_not_hashable(handle):
    with pytest.raises(TypeError, match="unhashable"):
        hash(handle)


def test_inline_text_is_its_own_content_key():
    assert InlineText("note")._content_key() == "note"
    assert InlineTextList(["a", "b"])._content_key() == ("a", "b")
    assert hash(InlineTextList(["a", "b"])._content_key()) == hash(InlineTextList(["a", "b"])._content_key())
    assert InlineText("a")._content_key() != InlineTextList(["a"])._content_key()


def test_a_content_key_reads_no_text():
    text = _StoredText("note")
    texts = _StoredTextList(["a", "b"])
    keys = (
        text._content_key(),
        texts._content_key(),
        Annotation(key="label", value=text)._content_key,
        Annotation(key="labels", value=texts)._content_key,
    )
    # A set hashes every key.
    assert len(set(keys)) == len(keys)
    assert text.reads == texts.reads == 0


def test_a_stored_key_never_equals_an_inline_key():
    # Equal text may get two keys, but two keys never stand for different text.
    text = _StoredText("note")
    assert text == InlineText("note")
    assert text._content_key() != InlineText("note")._content_key()
    assert Annotation(key="label", value=text)._content_key == Annotation(key="label", value=text)._content_key


@pytest.mark.parametrize("handle", [InlineText("note"), InlineTextList(["a", "b"])])
def test_an_inline_handle_survives_pickling_and_copies(handle):
    assert not hasattr(handle, "__dict__")
    for back in (pickle.loads(pickle.dumps(handle)), copy.copy(handle), copy.deepcopy(handle)):
        assert type(back) is type(handle)
        assert back.is_loaded
        assert back == handle


def test_read_value_reads_text_handles():
    assert read_value(InlineText("note")) == "note"
    assert read_value(InlineTextList(["a", "b"])) == ["a", "b"]


@pytest.mark.parametrize("value", [None, 1, 1.5, True, "plain", ["a"]])
def test_read_value_returns_other_values_unchanged(value):
    assert read_value(value) is value


def test_read_values_reads_each_handle_and_keeps_other_items():
    target = InlineText("answer")
    span = TimePoint.seconds(1.0)
    assert read_values([target, 3, span, None]) == ("answer", 3, span, None)
    assert read_values([InlineTextList(["a", "b"]), InlineText("c")]) == (["a", "b"], "c")


def test_read_values_returns_a_tuple():
    read = read_values(item for item in [InlineText("a"), InlineText("b")])
    assert type(read) is tuple
    assert read == ("a", "b")
    assert read_values([]) == ()


def test_read_values_keeps_none_apart_from_no_items():
    assert read_values(None) is None
    assert read_values(()) == ()


def test_read_values_rejects_a_bare_string():
    with pytest.raises(TimeFValidationError, match="read_value for one string"):
        read_values("answer")


def test_text_fields_wrap_strings_and_keep_string_handles():
    rationale = _StoredText("Because.")
    task = AnswerTask(prompt="Question?", rationale=rationale)
    assert isinstance(task.prompt, InlineText)
    assert task.rationale is rationale
    assert rationale.reads == 0
    assert AnswerTask().prompt is None


def test_text_fields_reject_a_list_handle():
    with pytest.raises(TimeFValidationError, match="string or a TextHandle"):
        AnswerTask(prompt=InlineTextList(["a"]))  # ty: ignore[invalid-argument-type]


def test_string_targets_become_string_handles():
    task = ClassificationTask(targets=("sleep", 3))
    assert task.targets is not None
    assert isinstance(task.targets[0], InlineText)
    assert task.targets[1] == 3


def test_a_bare_string_is_not_a_target_sequence():
    # Iterating "normal" would give one target per letter.
    with pytest.raises(TimeFValidationError, match="sequence of items"):
        ClassificationTask(targets="normal")
    with pytest.raises(TimeFValidationError, match="sequence of items"):
        ClassificationTask().targets = "normal"


def test_a_list_handle_is_not_a_target():
    with pytest.raises(TimeFValidationError, match="unsupported type InlineTextList"):
        ClassificationTask(targets=(InlineTextList(["a"]),))  # ty: ignore[invalid-argument-type]


def test_annotation_values_become_handles_of_their_kind():
    assert isinstance(Annotation(key="label", value="sleep").value, InlineText)
    assert isinstance(Annotation(key="labels", value=["a", "b"]).value, InlineTextList)
    assert Annotation(key="age", value=64).value == 64


def test_value_type_and_modality_inference_read_no_text():
    prompt = _StoredText("Question?")
    labels = _StoredTextList(["a", "b"])
    annotation = Annotation(key="labels", value=labels)
    prompted = AnswerTask(prompt=prompt, input_annotations=(annotation,))
    # Without a prompt, only the text annotation makes the task take text input.
    unprompted = AnswerTask(input_annotations=(annotation,))
    assert value_type_of(annotation.value) == "list"
    assert value_type_of(Annotation(key="label", value=_StoredText("sleep")).value) == "str"
    assert prompted.resolved_input_modalities == {InputModality.TEXT, InputModality.NO_INPUT}
    assert InputModality.TEXT in unprompted.resolved_input_modalities
    assert prompt.reads == labels.reads == 0


def test_a_text_annotation_equals_its_copies():
    annotation = Annotation(key="label", value="sleep")
    assert annotation == copy.deepcopy(annotation)
    assert annotation == pickle.loads(pickle.dumps(annotation))


def test_annotations_sharing_a_key_can_be_found_in_a_list():
    sleep = Annotation(key="label", value="sleep")
    wake = Annotation(key="label", value="wake")
    assert wake in [sleep, wake]  # noqa: PLR6201 - annotations are unhashable, so a set cannot hold them
    assert [sleep, wake].index(wake) == 1


def test_a_task_equals_one_whose_prompt_is_stored():
    inline = AnswerTask(prompt="Question?", rationale="Because.")
    stored = AnswerTask(prompt=_StoredText("Question?"), rationale=_StoredText("Because."))
    assert inline == stored
    assert inline == pickle.loads(pickle.dumps(inline))
