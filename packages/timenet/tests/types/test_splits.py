import pytest

from timenet.errors import TimeFValidationError
from timenet.types import AnswerTask, Split


def test_task_coerces_its_split_and_rejects_an_unknown_one():
    # The field is typed as Split, but a connector that passes the plain value at runtime gets the member.
    assert AnswerTask(targets=("x",), split="test").split is Split.TEST  # ty: ignore[invalid-argument-type]
    with pytest.raises(TimeFValidationError, match="unknown split 'holdout'"):
        AnswerTask(targets=("x",), split="holdout")  # ty: ignore[invalid-argument-type]
