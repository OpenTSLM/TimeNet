import math

import pytest

from timenet.errors import TimeNetConnectorError
from timenet_connectors.demographics import (
    AGE_UNIT,
    SEXES,
    DemographicKey,
    Sex,
    age_annotation,
    sex_annotation,
    subject_ids_annotation,
)


def test_spellings_are_fixed():
    # Published releases store these strings. A change here changes every dataset that uses them.
    assert [key.value for key in DemographicKey] == ["subject_ids", "sex", "age"]
    assert [sex.value for sex in Sex] == ["male", "female"]
    assert SEXES.name == "sex"
    assert SEXES.names == ("male", "female")
    assert AGE_UNIT == "year"


def test_sex_label_set_accepts_every_sex_as_a_target():
    for sex in Sex:
        SEXES.check_targets((sex.value,))


def test_annotations_use_the_shared_keys_and_values():
    subjects = subject_ids_annotation(["p-1", "p-2"])
    assert (subjects.key, subjects.value) == ("subject_ids", ["p-1", "p-2"])

    sex = sex_annotation(Sex.MALE, description="Coded as is_male.")
    assert (sex.key, sex.value, sex.description) == ("sex", "male", "Coded as is_male.")

    age = age_annotation(64)
    assert (age.key, age.value, age.unit) == ("age", 64, "year")


@pytest.mark.parametrize(
    "subject_ids",
    [
        [],
        [""],
        ["  "],
        ["p-1", "p-1"],
        [17],
        "p-1",
        [" a"],
        ["a\n"],
        ["a", " a"],
        {"a", "b"},
        frozenset({"a"}),
        {"a": "b"},
    ],
)
def test_subject_ids_reject_invalid_identifiers(subject_ids):
    with pytest.raises(TimeNetConnectorError):
        subject_ids_annotation(subject_ids)


@pytest.mark.parametrize("age", [-1, 0.5, 64.0, math.nan, math.inf, True, "64"])
def test_age_rejects_invalid_values(age):
    with pytest.raises(TimeNetConnectorError, match="age"):
        age_annotation(age)


@pytest.mark.parametrize("sex", ["F", "male", 1, None])
def test_sex_rejects_undecoded_codes(sex):
    with pytest.raises(TimeNetConnectorError, match="Sex value"):
        sex_annotation(sex)


def test_age_must_fit_the_stored_int64():
    assert age_annotation(2**63 - 1).value == 2**63 - 1
    with pytest.raises(TimeNetConnectorError, match="too large for the 64-bit integer"):
        age_annotation(2**63)
