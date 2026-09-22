from typing import Any, cast

import pytest

from timenet.errors import TimeFValidationError
from timenet.types import AnswerTask, TimeInterval
from timenet.types.trust import is_trusted, trusted_construction


def test_trusted_construction_skips_validation_only_inside_the_block():
    assert not is_trusted()
    with trusted_construction():
        assert is_trusted()
        stored = TimeInterval(start_us=5, end_us=1)
        task = AnswerTask(targets=cast("Any", (object(),)), prompt="?")
    assert not is_trusted()
    assert (stored.start_us, stored.end_us) == (5, 1)
    assert len(task.targets or ()) == 1
    with pytest.raises(TimeFValidationError):
        TimeInterval(start_us=5, end_us=1)
    with pytest.raises(TimeFValidationError):
        AnswerTask(targets=cast("Any", (object(),)), prompt="?")


def test_trusted_construction_resets_after_an_exception():
    with pytest.raises(RuntimeError), trusted_construction():
        raise RuntimeError
    assert not is_trusted()
