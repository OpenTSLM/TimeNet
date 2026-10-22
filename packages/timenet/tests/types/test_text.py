import copy
from dataclasses import dataclass
import pickle
from typing import Literal

import pytest

from timenet.types import Annotation, AnswerTask, InlineText, TextValue


@dataclass(eq=False)
class _CountingText(TextValue[str]):
    """A stored string that records every read, like a handle to packed text."""

    text: str
    reads: int = 0

    @property
    def value_type(self) -> Literal["str", "list"]:
        return "str"

    @property
    def length(self) -> int:
        return len(self.text)

    def read(self) -> str:
        self.reads += 1
        return self.text


@pytest.mark.parametrize("value", ["", []])
def test_an_empty_handle_is_false(value):
    assert not InlineText(value)


@pytest.mark.parametrize("value", ["x", ["x"]])
def test_a_nonempty_handle_is_true(value):
    assert InlineText(value)


def test_truth_does_not_read_the_value():
    stored = _CountingText("stored")
    assert stored
    assert stored.reads == 0


def test_handles_compare_by_content():
    assert InlineText("note") == InlineText("note")
    assert InlineText(["a", "b"]) == InlineText(["a", "b"])
    assert InlineText("note") != InlineText("other")
    assert InlineText("ab") != InlineText(["a", "b"])


def test_handles_compare_with_native_text_from_either_side():
    assert InlineText("note") == "note"
    assert "note" == InlineText("note")  # noqa: SIM300 - the reflected comparison is the point
    assert InlineText(["a", "b"]) == ["a", "b"]
    assert InlineText("note") != "other"


def test_a_handle_never_equals_a_non_text_value():
    assert InlineText("1") != 1
    assert InlineText("note") != None  # noqa: E711


def test_stored_and_inline_text_with_the_same_content_are_equal():
    stored = _CountingText("note")
    assert stored == InlineText("note")
    assert stored.reads == 1


def test_different_lengths_compare_unequal_without_a_read():
    stored = _CountingText("note")
    assert stored != InlineText("a longer note")
    assert stored != "a longer note"
    assert stored.reads == 0


def test_handles_are_not_hashable():
    with pytest.raises(TypeError, match="unhashable"):
        hash(InlineText("note"))


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
    inline = AnswerTask(id="task", prompt="Question?", rationale="Because.")
    stored = AnswerTask(id="task", prompt=_CountingText("Question?"), rationale=_CountingText("Because."))
    assert inline == stored
    assert inline == pickle.loads(pickle.dumps(inline))
