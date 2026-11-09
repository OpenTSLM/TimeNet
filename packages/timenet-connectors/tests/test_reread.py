import pytest

from timenet.errors import TimeNetConnectorError
from timenet_connectors.sources.reread import checked_reread


def test_returns_matching_payload_without_copying_or_caching():
    payloads = [[1, 2], [1, 2]]
    pending = iter(payloads)
    for expected_payload in payloads:
        assert (
            checked_reread(lambda: next(pending), fingerprint=tuple, expected=(1, 2), context="row 7")
            is expected_payload
        )


def test_rejects_different_content_even_when_length_matches():
    with pytest.raises(TimeNetConnectorError, match="row 7: content differs from the conversion scan"):
        checked_reread(lambda: [1, 3], fingerprint=tuple, expected=(1, 2), context="row 7")


@pytest.mark.parametrize("stage", ["load", "fingerprint"])
@pytest.mark.parametrize("error_type", [TimeNetConnectorError, ValueError, RuntimeError])
def test_callback_errors_and_causes_propagate_unchanged(stage, error_type):
    cause = OSError("source unavailable")
    error = error_type("original diagnostic")

    def fail():
        raise error from cause

    def load():
        if stage == "load":
            fail()
        return b"payload"

    def fingerprint(value):
        if stage == "fingerprint":
            fail()
        pytest.fail("fingerprint must not run after a failed load")

    with pytest.raises(error_type) as raised:
        checked_reread(load, fingerprint=fingerprint, expected=b"digest", context="row 7")
    assert raised.value is error
    assert raised.value.__cause__ is cause
    assert str(raised.value) == "original diagnostic"
