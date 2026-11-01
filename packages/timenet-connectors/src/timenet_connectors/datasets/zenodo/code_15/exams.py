"""Read CODE-15%'s ``exams.csv``: one row per exam, validated by :class:`ExamRow`.

:func:`read_exams` checks the rules across rows: unique exams, one sex per patient, and mortality stated on
at most one exam per patient.
"""

import csv
from pathlib import Path
from typing import Annotated, Any, Self

from pydantic import (
    BaseModel,
    BeforeValidator,
    ConfigDict,
    Field,
    FiniteFloat,
    NonNegativeInt,
    PositiveInt,
    StringConstraints,
    ValidationError,
    model_validator,
)

from timenet.errors import TimeFFormatError
from timenet_connectors.datasets.zenodo.code_15.vocabulary import Abnormality
from timenet_connectors.demographics import Sex


_HEADER = [
    "exam_id",
    "age",
    "is_male",
    "nn_predicted_age",
    "1dAVb",
    "RBBB",
    "LBBB",
    "SB",
    "ST",
    "AF",
    "patient_id",
    "death",
    "timey",
    "normal_ecg",
    "trace_file",
]


def _published_bool(value: Any) -> Any:
    """Read the release's ``True`` or ``False``, and nothing else, so a typo can't pass as a boolean.

    Returns:
        The boolean, or the value unchanged for the field's own validation to reject.
    """
    if value == "True":
        return True
    if value == "False":
        return False
    return value


def _empty_as_none(value: Any) -> Any:
    """Read an empty ``death`` as absent.

    ``death`` and ``timey`` are the only fields the release leaves empty, and always together; ``timey`` reads
    its own with :func:`_optional_number`, and every other field raises when empty. A patient states them on
    at most one exam: on the pinned release, 233,647 patients on exactly one and 123 on none.

    Returns:
        ``None`` for an empty string, else the value unchanged.
    """
    return None if isinstance(value, str) and not value else value


def _optional_number(value: Any) -> Any:
    """Read an empty field as absent, and any other as a number.

    A before-validator's output is no longer validated as a string, so the number is parsed here.

    Returns:
        ``None`` for an empty string, a float for another string, else the value unchanged.

    Raises:
        ValueError: If the string isn't a number.
    """  # noqa: DOC502 - raised by float
    if not isinstance(value, str):
        return value
    return float(value) if value else None


_Bool = Annotated[bool, BeforeValidator(_published_bool)]
_OptionalBool = Annotated[bool | None, BeforeValidator(_published_bool), BeforeValidator(_empty_as_none)]
_OptionalYears = Annotated[Annotated[float, Field(ge=0, allow_inf_nan=False)] | None, BeforeValidator(_optional_number)]


class ExamRow(BaseModel):
    """One row of ``exams.csv``, as the release writes it.

    The six abnormality fields are read from their columns, such as ``1dAVb``.
    """

    model_config = ConfigDict(strict=True, frozen=True, extra="forbid")

    exam_id: PositiveInt
    age: NonNegativeInt
    """The patient's age in years at the exam."""
    is_male: _Bool
    nn_predicted_age: FiniteFloat
    """The age the authors' network predicts from the trace, in years. Some are negative."""
    first_degree_av_block: _Bool = Field(alias=Abnormality.FIRST_DEGREE_AV_BLOCK.value)
    right_bundle_branch_block: _Bool = Field(alias=Abnormality.RIGHT_BUNDLE_BRANCH_BLOCK.value)
    left_bundle_branch_block: _Bool = Field(alias=Abnormality.LEFT_BUNDLE_BRANCH_BLOCK.value)
    sinus_bradycardia: _Bool = Field(alias=Abnormality.SINUS_BRADYCARDIA.value)
    sinus_tachycardia: _Bool = Field(alias=Abnormality.SINUS_TACHYCARDIA.value)
    atrial_fibrillation: _Bool = Field(alias=Abnormality.ATRIAL_FIBRILLATION.value)
    patient_id: NonNegativeInt
    death: _OptionalBool
    """Whether the patient died in the follow-up. At most one exam per patient states it."""
    timey: _OptionalYears
    """Years to death, or to the end of follow-up for a survivor. Stated with ``death``."""
    normal_ecg: _Bool
    trace_file: Annotated[str, StringConstraints(pattern=r"^exams_part[0-9]+\.hdf5$")]
    """The HDF5 file that holds the exam's trace."""

    @model_validator(mode="after")
    def _mortality_stated_together(self) -> Self:
        if (self.death is None) != (self.timey is None):
            raise ValueError("death and timey must be both stated or both empty")
        return self

    @model_validator(mode="after")
    def _labels_agree(self) -> Self:
        # Both rules hold on every row of the pinned release; a row that breaks one is mislabelled.
        if self.normal_ecg and self.abnormalities:
            raise ValueError(f"normal_ecg is True, but the exam is labelled {[a.value for a in self.abnormalities]}")
        if self.sinus_bradycardia and self.sinus_tachycardia:
            raise ValueError("SB and ST are both True, but a sinus rhythm can't be both slow and fast")
        return self

    @classmethod
    def parse(cls, row: list[str], context: str) -> Self:
        """Validate one CSV row against the release's columns.

        Args:
            row: The row's fields.
            context: The file and line, for the error.

        Returns:
            The row.

        Raises:
            TimeFFormatError: If the row doesn't have one field per column, or a field isn't one the release
                writes.
        """
        if len(row) != len(_HEADER):
            raise TimeFFormatError(f"{context}: expected {len(_HEADER)} fields, got {len(row)}")
        try:
            return cls.model_validate_strings(dict(zip(_HEADER, row, strict=True)))
        except ValidationError as exc:
            raise TimeFFormatError(f"{context}: {exc}") from exc

    @property
    def sex(self) -> Sex:
        """Return the patient's sex, decoded from ``is_male``."""
        return Sex.MALE if self.is_male else Sex.FEMALE

    @property
    def abnormalities(self) -> tuple[Abnormality, ...]:
        """Return the abnormalities the exam is labelled with, in the authors' output order."""
        return tuple(abnormality for abnormality in Abnormality if getattr(self, _FIELD_OF[abnormality]))


# Each abnormality's ExamRow field, read from the fields' aliases, which are the abnormalities' columns.
_FIELD_OF = {Abnormality(field.alias): name for name, field in ExamRow.model_fields.items() if field.alias}


def read_exams(path: Path) -> tuple[ExamRow, ...]:
    """Read ``exams.csv`` and check the rules across its rows.

    Returns:
        Every row, in file order.

    Raises:
        TimeFFormatError: If the header isn't the release's, a row fails :meth:`ExamRow.parse`, an exam is
            listed twice, a patient is listed with two sexes, or a patient's mortality is stated on two exams.
    """
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.reader(handle)
        if next(reader, None) != _HEADER:
            raise TimeFFormatError(f"{path}: expected the header {','.join(_HEADER)}")
        exams = tuple(ExamRow.parse(row, f"{path}:{line}") for line, row in enumerate(reader, 2))
    seen_exams: set[int] = set()
    sex_of: dict[int, Sex] = {}
    mortality_stated: set[int] = set()
    for exam in exams:
        if exam.exam_id in seen_exams:
            raise TimeFFormatError(f"{path}: exam {exam.exam_id} is listed twice")
        seen_exams.add(exam.exam_id)
        if sex_of.setdefault(exam.patient_id, exam.sex) is not exam.sex:
            raise TimeFFormatError(f"{path}: patient {exam.patient_id} is listed as both sexes")
        if exam.death is not None:
            if exam.patient_id in mortality_stated:
                raise TimeFFormatError(f"{path}: patient {exam.patient_id}'s mortality is stated on two exams")
            mortality_stated.add(exam.patient_id)
    return exams
