import pytest

from timenet.refs import split_ref


@pytest.mark.parametrize(
    "ref,expected",
    [
        ("timenet/hello-world", ("timenet/hello-world", None)),
        ("org/name", ("org/name", None)),
        ("org/name@latest", ("org/name", None)),
        ("org/name@1.2.3", ("org/name", "1.2.3")),
        ("timenet/hello-world@0.0.1", ("timenet/hello-world", "0.0.1")),
    ],
)
def test_split_ref_ok(ref, expected):
    assert split_ref(ref) == expected


@pytest.mark.parametrize(
    "ref",
    ["@1.0.0", "id@", "id@1.0", "id@1", "id@1.0.0.0", "a@b@c", "id@v1.0.0"],
)
def test_split_ref_rejects(ref):
    with pytest.raises(ValueError):
        split_ref(ref)
