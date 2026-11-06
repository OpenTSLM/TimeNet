"""Convert CODE-15%'s exams into lazy 12-lead Records, each with an age task and an abnormality task.

Each exam whose trace is not zero everywhere becomes a Record whose leads reread their HDF5 part on demand,
in millivolts and without the authors' zero padding. README.md explains how this follows the authors.

The build has two passes. The first reads ``exams.csv`` and scans every trace once, keeping only small
results: the padding and a digest of each trace. The second builds the Records and streams the tasks. The
samples stay in the HDF5 parts until they are written.

Metadata is provenance and never model input. Task annotations are never model input either.
"""

import asyncio
from collections.abc import Iterator
import csv
from enum import StrEnum, unique
from functools import partial
import hashlib
from pathlib import Path
import re
from types import ModuleType
from typing import TYPE_CHECKING, NamedTuple
import weakref

import numpy as np
import pyarrow as pa

from timenet.connectors import BaseConnector
from timenet.dataset import Record, Signal, Source, TimeFDataset
from timenet.dataset.axis import RegularAxis
from timenet.errors import TimeFFormatError, TimeFValidationError
from timenet.types import (
    Annotation,
    ClassificationTask,
    Label,
    LabelSet,
    ScalarPredictionTask,
    Split,
    Task,
    TimeSeriesSpec,
    ureg,
)
from timenet_connectors.download import Artifact, download_files, ensure_archive


if TYPE_CHECKING:
    # Only for the types. At run time, `_h5py` imports h5py on first use (docs/connectors.md).
    import h5py


RECORD_URL = "https://zenodo.org/records/4916206"
"""Zenodo record 4916206, version 1.0.0, the record's only version."""
RECORD_DOI = "10.5281/zenodo.4916206"
_CSV_FILE = "exams.csv"
_CSV_SHA256 = "f9f82f4c07c1793e2285915c69f1113f5a7db8a0f2cf2f4b90474502f406bc35"
# Zenodo publishes only md5s, so these SHA-256s were computed on a first download whose bytes matched them.
_PART_SHA256 = (
    "4941375309c741ab676071f6bbd77c9ee739e0212535122bbb7e3f72e7ff7bf5",
    "ae0fccfb10cfdc938d883991587f190af837b2f742bc76b1d7852044768a3e41",
    "bdf054b71a92b279fbf630d0ad05199577c425a4083511457705a0391ad13bf4",
    "ea3d0a24c67f92f01b1d91ba6ae0da26d15d17ce717a01ee727098c43032cf98",
    "b17532eb40eff6c7ee632b2ef55e8b89cd8a6c3418de89b858da69a0696cbef1",
    "fdb9cba13c4f8f5636360ae7cb455c0e97a4b26cc6899bd05d236ea81ca90e30",
    "33cd2b14fd156d8ada703a2c5f3cb17ce7cc70bc55939b115892752acb2027c6",
    "a10c91440337560583375a423395cb13cc2f6eccbc03303d7d0b646a67ce6fc0",
    "1d0070dda22f02bca30c04f35ca0c2cc92ca9460fb43fa1d5977a663635e4401",
    "7e9812179c45e1bb5adbbee5976ba0f5adf89f8b34356c9d77bdad06fc99555b",
    "22dba63939faf22602ed511a888bbd1a4e1c779a8d2634a6091300e5530077bd",
    "491ba8bfa3defc13cdc37af94dc9bfc7ddc90b03dd989288897795b2e3822053",
    "ce47ac6180ba15868dec6277124c0f0ef41dfad54bba564d3a204117d4f65811",
    "e39b90d72295a4a0923e3af89de09346e454256d24e15e2bf5a2c3e4ceec8bee",
    "e21fe30bf441bc09fb444719bc22ef72e504f719e71d919cc665ecc4b8f36658",
    "84ddf0a1ef567a3a0f308c6b2b8d240002a21be7105ba761d6d91f3299082e1a",
    "b66f46d9fd17162453891732039f4a62d4df66bba445610444a08d205a89b927",
    "af7fd3e5fa9e091ab1936e720c7fc55f116cca0722592ce2bdbff644de22e2e9",
)
"""The SHA-256 of ``exams_part0.zip`` to ``exams_part17.zip``, in that order."""
_PART_FILE = re.compile(r"exams_part(0|[1-9][0-9]*)\.hdf5")
"""The HDF5 file a zip holds, which ``exams.csv`` names in ``trace_file``. ``_part_file`` builds the name."""
_DECIMAL = re.compile(r"-?(0|[1-9][0-9]*)(\.[0-9]+)?")
"""A number as ``exams.csv`` writes it, such as ``40.160484``: no sign but minus, no exponent, no spaces."""
_DIGEST_BYTES = 16
"""The size of a trace digest. It detects a changed file, not an attack, so 128 bits are enough."""

_EXAM_ID_DATASET = "exam_id"
"""Each part's ``(N,)`` int64 exam ids. The rows are not in ``exams.csv``'s order."""
_TRACINGS_DATASET = "tracings"
"""Each part's ``(N, 4096, 12)`` float32 traces, in the order of ``exam_id``."""
_FILLER_EXAM_ID = 0
"""The id of the last row of every part: an all-zero trace that ``exams.csv`` does not list."""
_TRACE_SAMPLES = 4096
"""Every published trace's length: the authors' ``--new_len 4096``."""
_SAMPLE_RATE_HZ = 400
_KEPT_EXAMS = 345_109
"""The exams of the release whose trace is not zero everywhere, which get a Record. dataset.yaml states it."""
_KEPT_PATIENTS = 233_480
"""The patients of those exams. dataset.yaml states it."""
_SCAN_ROWS = 512
"""Traces read at a time while scanning: 96 MiB of float32."""

_MILLIVOLTS_PER_PUBLISHED_UNIT = np.float32(0.5)
"""The authors multiply each value by 2 (``ecg-age-prediction`` issue 5), so 0.5 gives millivolts.

Multiplying a float32 by 0.5 only lowers its exponent, so it is exact. README.md says why the connector keeps
the authors' factor of 2 and not our own fit of 1.953125.
"""
_SPEC = TimeSeriesSpec(spec_type="ecg", name="12-lead ECG", unit_value=ureg.millivolt, dtype="float32")
_TIME_AXIS = RegularAxis.from_rate_hz(_SAMPLE_RATE_HZ)
_SOURCE_NAME = "12-lead ECG"

# Lima et al. 2021 hold CODE-15% out to test their age model. Ribeiro et al. 2020 train their abnormality
# model on the CODE cohort, which CODE-15% is part of.
_AGE_SPLIT = Split.TEST
_ABNORMALITY_SPLIT = Split.TRAIN


@unique
class _Lead(StrEnum):
    """A lead, named and ordered as the release's traces hold them (the authors' ``all_leads``)."""

    DI = "DI"
    DII = "DII"
    DIII = "DIII"
    AVR = "AVR"
    AVL = "AVL"
    AVF = "AVF"
    V1 = "V1"
    V2 = "V2"
    V3 = "V3"
    V4 = "V4"
    V5 = "V5"
    V6 = "V6"


@unique
class _Abnormality(StrEnum):
    """One of the six abnormalities, by its ``exams.csv`` column, in the authors' model output order.

    ``exams.csv`` writes ST before AF. The order here is the ``automatic-ecg-diagnosis`` README's.
    """

    FIRST_DEGREE_AV_BLOCK = "1dAVb"
    RIGHT_BUNDLE_BRANCH_BLOCK = "RBBB"
    LEFT_BUNDLE_BRANCH_BLOCK = "LBBB"
    SINUS_BRADYCARDIA = "SB"
    ATRIAL_FIBRILLATION = "AF"
    SINUS_TACHYCARDIA = "ST"


# Each abnormality's name, as the Zenodo record and the authors' README write it.
_DESCRIPTIONS = {
    _Abnormality.FIRST_DEGREE_AV_BLOCK: "1st degree AV block.",
    _Abnormality.RIGHT_BUNDLE_BRANCH_BLOCK: "Right bundle branch block.",
    _Abnormality.LEFT_BUNDLE_BRANCH_BLOCK: "Left bundle branch block.",
    _Abnormality.SINUS_BRADYCARDIA: "Sinus bradycardia.",
    _Abnormality.ATRIAL_FIBRILLATION: "Atrial fibrillation.",
    _Abnormality.SINUS_TACHYCARDIA: "Sinus tachycardia.",
}


@unique
class _Column(StrEnum):
    """An ``exams.csv`` column that is not an abnormality."""

    EXAM_ID = "exam_id"
    AGE = "age"
    IS_MALE = "is_male"
    NN_PREDICTED_AGE = "nn_predicted_age"
    PATIENT_ID = "patient_id"
    DEATH = "death"
    TIMEY = "timey"
    NORMAL_ECG = "normal_ecg"
    TRACE_FILE = "trace_file"


_HEADER = (
    _Column.EXAM_ID,
    _Column.AGE,
    _Column.IS_MALE,
    _Column.NN_PREDICTED_AGE,
    _Abnormality.FIRST_DEGREE_AV_BLOCK,
    _Abnormality.RIGHT_BUNDLE_BRANCH_BLOCK,
    _Abnormality.LEFT_BUNDLE_BRANCH_BLOCK,
    _Abnormality.SINUS_BRADYCARDIA,
    _Abnormality.SINUS_TACHYCARDIA,
    _Abnormality.ATRIAL_FIBRILLATION,
    _Column.PATIENT_ID,
    _Column.DEATH,
    _Column.TIMEY,
    _Column.NORMAL_ECG,
    _Column.TRACE_FILE,
)
"""The release's columns, in file order."""


@unique
class _Sex(StrEnum):
    """A patient's sex, decoded from ``is_male``."""

    MALE = "male"
    FEMALE = "female"


@unique
class _MetadataKey(StrEnum):
    """A key of ``Record.metadata``. README.md documents them, and the tests pin them."""

    EXAM_ID = "exam_id"
    TRACE_FILE = "trace_file"
    N_PAD_BEFORE = "n_pad_before"
    N_PAD_AFTER = "n_pad_after"


@unique
class _Key(StrEnum):
    """The key of an annotation the connector writes."""

    SOURCE_URL = "source_url"
    SOURCE_DOI = "source_doi"
    SUBJECT_IDS = "subject_ids"
    SEX = "sex"
    DEATH = "death"
    FOLLOW_UP = "follow_up"
    NN_PREDICTED_AGE = "nn_predicted_age"


_SEX_NOTE = "The patient's sex, decoded from exams.csv's is_male."
_DEATH_NOTE = "Whether the patient died in the follow-up. exams.csv states it on at most one exam per patient."
_FOLLOW_UP_NOTE = (
    "exams.csv's timey: the years to the patient's death, or to the end of follow-up if they survived. Stated on "
    "the same exam as death."
)
_NN_PREDICTED_AGE_NOTE = (
    "exams.csv's nn_predicted_age: the age the authors' network predicts from the exam's trace (Lima et al. 2021), "
    "for the gap between ECG-age and age."
)
_NORMAL_ECG_LABEL = Label(
    name=_Column.NORMAL_ECG.value,
    description="Normal ECG.",
)
_ABNORMAL_UNSPECIFIED_LABEL = Label(
    name="abnormal_unspecified",
    description=("Abnormal ECG with none of the six listed abnormalities. The release does not name the finding."),
)
# normal_ecg uses the release's column of the same name. abnormal_unspecified means not marked normal,
# with all six diagnosis columns false. These two labels extend the authors' six-label task and each
# appears alone. See the README for the source definitions, counts, and limits of the published columns.
ABNORMALITIES = LabelSet(
    name="abnormality",
    labels=(
        *(Label(name=abnormality.value, description=_DESCRIPTIONS[abnormality]) for abnormality in _Abnormality),
        _NORMAL_ECG_LABEL,
        _ABNORMAL_UNSPECIFIED_LABEL,
    ),
    multi_label=True,
)
"""The six abnormalities in the authors' output order, then ``normal_ecg`` and ``abnormal_unspecified``.

Every abnormality task shares it. A target holds every abnormality that the exam is labelled with, only
``normal_ecg`` for a normal exam, or only ``abnormal_unspecified`` for an exam that is neither. No target is
empty.
"""


class _Padding(NamedTuple):
    """How many zero samples the authors added before and after a trace."""

    n_before: int
    n_after: int

    @property
    def n_values(self) -> int:
        """The number of samples that the trace holds without its padding."""
        return _TRACE_SAMPLES - self.n_before - self.n_after

    @property
    def values(self) -> slice:
        """The samples of the published trace that hold the recording."""
        return slice(self.n_before, _TRACE_SAMPLES - self.n_after)


class _Mortality(NamedTuple):
    """The ``death`` and ``timey`` fields, which ``exams.csv`` states together or not at all."""

    died: bool
    """Whether the patient died in the follow-up."""
    years: float
    """Years to death, or to the end of follow-up for a survivor."""


class _ExamRow(NamedTuple):
    """One validated row of ``exams.csv``."""

    exam_id: int
    age: int
    """The patient's age in years at the exam."""
    is_male: bool
    nn_predicted_age: float
    """The age the authors' network predicts from the trace, in years. Some are negative."""
    abnormalities: tuple[_Abnormality, ...]
    """The abnormalities the exam is labelled with, in the authors' output order."""
    patient_id: int
    mortality: _Mortality | None
    """What the exam states about the patient's death. At most one exam per patient states it."""
    normal_ecg: bool
    part: int
    """The number ``n`` of the ``exams_part<n>.hdf5`` file that holds the exam's trace."""

    @property
    def trace_file(self) -> str:
        """The name of the part that holds the exam's trace."""
        return _part_file(self.part)

    @property
    def sex(self) -> _Sex:
        """The patient's sex."""
        return _Sex.MALE if self.is_male else _Sex.FEMALE


class _Exam(NamedTuple):
    """An exam with a trace after the first pass: its row, where its trace is, and what was scanned."""

    row: _ExamRow
    path: Path
    index: int
    """The trace's row in its part."""
    padding: _Padding
    digest: bytes
    """The digest of the trace as scanned, to tell a reread of the same values from another."""


class _Scan(NamedTuple):
    """One row of a part, scanned."""

    exam_id: int
    index: int
    """The trace's row in its part."""
    padding: _Padding | None
    """``None`` for a trace that is zero in every lead and sample."""
    digest: bytes


def _part_file(part: int) -> str:
    """Return the name of the HDF5 file of part ``part``, such as ``exams_part3.hdf5``."""
    return f"exams_part{part}.hdf5"


def _part_zip(part: int) -> str:
    """Return the name of the zip of part ``part`` on Zenodo, such as ``exams_part3.zip``."""
    return f"exams_part{part}.zip"


def _download_url(name: str) -> str:
    """Return the Zenodo download URL of one file of the record."""
    return f"{RECORD_URL}/files/{name}?download=1"


def _trace_digest(trace: np.ndarray) -> bytes:
    """Return the digest of one trace's bytes, which a reread compares with the scan's."""
    return hashlib.blake2b(trace, digest_size=_DIGEST_BYTES).digest()


def _parse_bool(column: str, text: str) -> bool:
    """Read the release's ``True`` or ``False``, and nothing else.

    Returns:
        The boolean.

    Raises:
        ValueError: If the text is neither. The message names the column.
    """
    if text not in {"True", "False"}:
        raise ValueError(f"{column}: expected True or False, got {text!r}")
    return text == "True"


def _parse_int(column: str, text: str, *, minimum: int) -> int:
    """Read a plain decimal integer.

    Returns:
        The integer.

    Raises:
        ValueError: If the text is not digits, or is below ``minimum``. The message names the column.
    """
    if not (text.isascii() and text.isdigit()) or int(text) < minimum:
        raise ValueError(f"{column}: expected an integer of at least {minimum}, got {text!r}")
    return int(text)


def _parse_float(column: str, text: str, *, minimum: float | None) -> float:
    """Read a decimal number as ``exams.csv`` writes it.

    Returns:
        The number.

    Raises:
        ValueError: If the text is not such a number, or is below ``minimum``. The message names the column.
    """
    if not _DECIMAL.fullmatch(text):
        raise ValueError(f"{column}: expected a decimal number, got {text!r}")
    number = float(text)
    if minimum is not None and number < minimum:
        raise ValueError(f"{column}: expected a number of at least {minimum}, got {text!r}")
    return number


def _parse_mortality(death: str, timey: str) -> _Mortality | None:
    """Read ``death`` and ``timey``, which a row states together or leaves empty together.

    Returns:
        The mortality, or ``None`` if both fields are empty.

    Raises:
        ValueError: If only one of the two is stated, or one is malformed.
    """
    if not death and not timey:
        return None
    if not death or not timey:
        raise ValueError(f"{_Column.DEATH} and {_Column.TIMEY} must be both stated or both empty")
    return _Mortality(died=_parse_bool(_Column.DEATH, death), years=_parse_float(_Column.TIMEY, timey, minimum=0))


def _parse_row(fields: dict[str, str]) -> _ExamRow:
    """Read one row's fields as their types, and check the row agrees with itself.

    Returns:
        The row.

    Raises:
        ValueError: If a field is not one the release writes, ``death`` and ``timey`` are not both stated or
            both empty, ``normal_ecg`` and an abnormality are both set, or SB and ST are both set.
    """
    mortality = _parse_mortality(fields[_Column.DEATH], fields[_Column.TIMEY])
    labelled = tuple(abnormality for abnormality in _Abnormality if _parse_bool(abnormality, fields[abnormality]))
    normal_ecg = _parse_bool(_Column.NORMAL_ECG, fields[_Column.NORMAL_ECG])
    # Both rules hold on every row of the pinned release, so a row that breaks one is mislabelled.
    if normal_ecg and labelled:
        raise ValueError(f"normal_ecg is True, but the exam is labelled {[a.value for a in labelled]}")
    if {_Abnormality.SINUS_BRADYCARDIA, _Abnormality.SINUS_TACHYCARDIA} <= set(labelled):
        raise ValueError("SB and ST are both True, but a sinus rhythm can't be both slow and fast")
    part = _PART_FILE.fullmatch(fields[_Column.TRACE_FILE])
    if part is None:
        raise ValueError(f"trace_file {fields[_Column.TRACE_FILE]!r} is not an exams_part<n>.hdf5 file")
    return _ExamRow(
        exam_id=_parse_int(_Column.EXAM_ID, fields[_Column.EXAM_ID], minimum=1),
        age=_parse_int(_Column.AGE, fields[_Column.AGE], minimum=0),
        is_male=_parse_bool(_Column.IS_MALE, fields[_Column.IS_MALE]),
        nn_predicted_age=_parse_float(_Column.NN_PREDICTED_AGE, fields[_Column.NN_PREDICTED_AGE], minimum=None),
        abnormalities=labelled,
        patient_id=_parse_int(_Column.PATIENT_ID, fields[_Column.PATIENT_ID], minimum=0),
        mortality=mortality,
        normal_ecg=normal_ecg,
        part=int(part[1]),
    )


class _PatientRules:
    """Check what ``exams.csv`` says about a patient across the patient's exams."""

    def __init__(self) -> None:
        self._sex: dict[int, _Sex] = {}
        self._mortality_stated: set[int] = set()

    def check(self, row: _ExamRow) -> None:
        """Check one more exam of its patient.

        Raises:
            ValueError: If the patient has another sex on an earlier exam, or states mortality on a second exam.
        """
        first_sex = self._sex.setdefault(row.patient_id, row.sex)
        if first_sex is not row.sex:
            raise ValueError(f"patient {row.patient_id} is listed as both sexes")
        if row.mortality is not None:
            if row.patient_id in self._mortality_stated:
                raise ValueError(f"patient {row.patient_id}'s mortality is stated twice")
            self._mortality_stated.add(row.patient_id)


def _read_exams(path: Path) -> dict[int, _ExamRow]:
    """Read ``exams.csv`` and check the rules across its rows.

    Returns:
        Every row by exam id, in file order.

    Raises:
        TimeFFormatError: If the header is not the release's, the file has no rows, a row is malformed, an exam
            is listed twice, a patient is listed with two sexes, or a patient's mortality is stated on two exams.
    """  # ruff: ignore[docstring-missing-exception] (this function catches its ValueErrors as TimeFFormatError)
    rows: dict[int, _ExamRow] = {}
    patients = _PatientRules()
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.reader(handle)
        if next(reader, None) != [str(column) for column in _HEADER]:
            raise TimeFFormatError(f"{path}: expected the header {','.join(_HEADER)}")
        for line, fields in enumerate(reader, start=2):
            if len(fields) != len(_HEADER):
                raise TimeFFormatError(f"{path}:{line}: expected {len(_HEADER)} fields, got {len(fields)}")
            try:
                row = _parse_row(dict(zip(_HEADER, fields, strict=True)))
                if row.exam_id in rows:
                    raise ValueError(f"exam {row.exam_id} is listed twice")
                patients.check(row)
            except ValueError as exc:
                raise TimeFFormatError(f"{path}:{line}: {exc}") from exc
            rows[row.exam_id] = row
    if not rows:
        raise TimeFFormatError(f"{path}: lists no exams")
    return rows


def _h5py() -> ModuleType:
    """Import h5py, which reads the parts and which the connector's requirements declare.

    Returns:
        The module.

    Raises:
        ImportError: If h5py is missing.
    """
    try:
        import h5py  # ruff: ignore[import-outside-top-level] (optional connector dependency)
    except ImportError as exc:
        raise ImportError(
            "reading CODE-15% needs h5py, declared in the connector's requirements.txt. Run the build "
            "without --no-isolation, or install it yourself"
        ) from exc
    return h5py


def _zero_runs(traces: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Count each trace's samples, from the first and from the last, that are zero in every lead.

    Args:
        traces: ``(n, 4096, 12)`` traces.

    Returns:
        The leading and the trailing counts, one per trace. Both are 4096 for an all-zero trace.
    """
    nonzero = np.any(traces != 0, axis=2)
    any_nonzero = nonzero.any(axis=1)
    leading = np.where(any_nonzero, nonzero.argmax(axis=1), _TRACE_SAMPLES)
    trailing = np.where(any_nonzero, nonzero[:, ::-1].argmax(axis=1), _TRACE_SAMPLES)
    return leading, trailing


def _centred_padding(leading_zeros: int, trailing_zeros: int) -> _Padding:
    """Return the most padding the authors' ``preprocess_ecg`` could have added.

    It pads ``(4096 - n) // 2`` zeros before a trace of ``n`` samples and the rest after, so the padding after
    is the padding before or one more. Of the zeros at either end, only those this rule explains are padding.
    Any others are zeros of the published trace, and stay. The raw lengths are not published, so this is an
    inference. Only exact all-lead zeros are dropped, and the Record's metadata keeps the counts.

    Returns:
        The padding before and after the trace.
    """
    n_before = min(leading_zeros, trailing_zeros)
    return _Padding(n_before=n_before, n_after=min(trailing_zeros, n_before + 1))


class _PartDatasets(NamedTuple):
    """The two datasets of a part."""

    exam_ids: "h5py.Dataset"
    tracings: "h5py.Dataset"


def _check_datasets(handle: "h5py.File", path: Path) -> _PartDatasets:
    """Return a part's ``exam_id`` and ``tracings`` datasets, checked against the release's shapes and types.

    Returns:
        The two datasets.

    Raises:
        TimeFFormatError: If either is missing, or their shapes or types are not the release's.
    """
    if set(handle) != {_EXAM_ID_DATASET, _TRACINGS_DATASET}:
        raise TimeFFormatError(
            f"{path}: expected the datasets {_EXAM_ID_DATASET} and {_TRACINGS_DATASET}, got {set(handle)}"
        )
    exam_ids, tracings = handle[_EXAM_ID_DATASET], handle[_TRACINGS_DATASET]
    if exam_ids.dtype != np.int64 or tracings.dtype != np.float32:
        raise TimeFFormatError(f"{path}: expected int64 ids and float32 traces, got {exam_ids.dtype}, {tracings.dtype}")
    if exam_ids.ndim != 1 or tracings.shape != (exam_ids.shape[0], _TRACE_SAMPLES, len(_Lead)):
        raise TimeFFormatError(
            f"{path}: expected {_EXAM_ID_DATASET} (N,) and {_TRACINGS_DATASET} (N, {_TRACE_SAMPLES}, {len(_Lead)}), "
            f"got {exam_ids.shape} and {tracings.shape}"
        )
    return _PartDatasets(exam_ids=exam_ids, tracings=tracings)


def _check_filler(path: Path, exam_ids: np.ndarray, tracings: "h5py.Dataset") -> int:
    """Check that a part ends with the all-zero filler row, and only there.

    Returns:
        The number of exams, which is the number of rows before the filler.

    Raises:
        TimeFFormatError: If the last row is not the filler, or another row has the filler's exam id.
    """
    n_exams = exam_ids.shape[0] - 1
    if n_exams < 0 or exam_ids[-1] != _FILLER_EXAM_ID or np.any(tracings[n_exams] != 0):
        raise TimeFFormatError(f"{path}: the last row is not the all-zero filler with exam id {_FILLER_EXAM_ID}")
    if np.any(exam_ids[:n_exams] == _FILLER_EXAM_ID):
        raise TimeFFormatError(f"{path}: a row other than the last has the filler's exam id {_FILLER_EXAM_ID}")
    return n_exams


def _scan_block(path: Path, first_row: int, exam_ids: np.ndarray, block: np.ndarray) -> Iterator[_Scan]:
    """Scan a block of consecutive traces.

    Yields:
        One scan per trace.

    Raises:
        TimeFFormatError: If a trace holds a value that is not finite.
    """
    finite = np.isfinite(block).all(axis=(1, 2))
    if not finite.all():
        raise TimeFFormatError(
            f"{path}: exam {exam_ids[int(np.argmin(finite))]}'s trace holds a value that is not finite"
        )
    leading, trailing = _zero_runs(block)
    for offset in range(block.shape[0]):
        all_zero = leading[offset] == _TRACE_SAMPLES
        yield _Scan(
            exam_id=int(exam_ids[offset]),
            index=first_row + offset,
            padding=None if all_zero else _centred_padding(int(leading[offset]), int(trailing[offset])),
            digest=_trace_digest(block[offset]),
        )


def _scan_part(path: Path) -> Iterator[_Scan]:
    """Read a part once, block by block, and scan each trace.

    Yields:
        One scan per row but the filler, in the part's order.
    """
    with _h5py().File(path, "r") as handle:
        datasets = _check_datasets(handle, path)
        exam_ids = datasets.exam_ids[:]
        n_exams = _check_filler(path, exam_ids, datasets.tracings)
        for start in range(0, n_exams, _SCAN_ROWS):
            stop = min(start + _SCAN_ROWS, n_exams)
            yield from _scan_block(path, start, exam_ids[start:stop], datasets.tracings[start:stop])


def _scan_release(root: Path, rows: dict[int, _ExamRow]) -> Iterator[_Exam]:
    """Scan every part that ``exams.csv`` names and join each trace to its row by exam id.

    The parts' rows are not in ``exams.csv``'s order, so the join is by id, as the authors say in
    ``automatic-ecg-diagnosis`` issue 32. A trace that is zero everywhere has no Record, so it is left out.

    Yields:
        Each exam with a trace, by part number, then by row in the part.

    Raises:
        TimeFFormatError: If a part is missing, holds an exam that ``exams.csv`` does not list in that part,
            holds an exam twice, or if an exam of ``exams.csv`` has no trace.
    """
    parts = sorted({row.part for row in rows.values()})
    seen: set[int] = set()
    for part in parts:
        name = _part_file(part)
        path = root / name
        if not path.is_file():
            raise TimeFFormatError(f"{root}: {name} is missing")
        for scan in _scan_part(path):
            row = rows.get(scan.exam_id)
            if row is None or row.part != part:
                raise TimeFFormatError(
                    f"{path}: row {scan.index} holds exam {scan.exam_id}, which {_CSV_FILE} does not list in {name}"
                )
            if scan.exam_id in seen:
                raise TimeFFormatError(f"{path}: exam {scan.exam_id} has two traces")
            seen.add(scan.exam_id)
            if scan.padding is not None:
                yield _Exam(row=row, path=path, index=scan.index, padding=scan.padding, digest=scan.digest)
    missing = sorted(rows.keys() - seen)
    if missing:
        raise TimeFFormatError(f"{root}: {len(missing)} exams of {_CSV_FILE} have no trace, such as {missing[:5]}")


def _check_counts(root: Path, exams: list[_Exam]) -> None:
    """Check that the scan kept the release's number of exams and patients.

    The zip digests pin the files. This check also catches a change in how the connector reads them, such as
    a bug that takes a real trace for an all-zero one.

    Raises:
        TimeFFormatError: If the number of exams or patients with a Record is not the pinned one.
    """
    patients = {exam.row.patient_id for exam in exams}
    if len(exams) != _KEPT_EXAMS or len(patients) != _KEPT_PATIENTS:
        raise TimeFFormatError(
            f"{root}: expected {_KEPT_EXAMS} exams from {_KEPT_PATIENTS} patients whose trace is not zero "
            f"everywhere, got {len(exams)} exams from {len(patients)} patients"
        )


def _close(files: "dict[Path, h5py.File]") -> None:
    for handle in files.values():
        handle.close()
    files.clear()


class _TraceReader:
    """Reread one exam's trace for all its leads, keeping only the part last read open.

    A Record's signals load one after another, and the Records come by part, then by row. So one cached
    trace and one open part suffice, and a load holds at most one HDF5 file at a time.
    """

    def __init__(self) -> None:
        self._files: dict[Path, h5py.File] = {}
        # The finalizer holds the dict, not the reader, so the reader can be collected.
        weakref.finalize(self, _close, self._files)
        self._cached: _Exam | None = None
        self._millivolts = np.empty((0, len(_Lead)), dtype=np.float32)

    def lead(self, exam: _Exam, lead_column: int) -> pa.Array:
        """Return one lead of an exam, in millivolts and without padding.

        Returns:
            The lead's float32 values.
        """
        if self._cached is not exam:
            self._cached = None
            self._millivolts = self._read(exam)
            self._cached = exam
        return pa.array(np.ascontiguousarray(self._millivolts[:, lead_column]), type=pa.float32())

    def _read(self, exam: _Exam) -> np.ndarray:
        """Reread a trace, check it is the one scanned, and return its samples in millivolts.

        Returns:
            The ``(n, 12)`` float32 samples, without padding.

        Raises:
            TimeFFormatError: If the trace's digest differs from the scan's: the file changed.
        """
        handle = self._files.get(exam.path)
        if handle is None:
            _close(self._files)
            handle = self._files[exam.path] = _h5py().File(exam.path, "r")
        trace = handle[_TRACINGS_DATASET][exam.index]
        if _trace_digest(trace) != exam.digest:
            raise TimeFFormatError(
                f"{exam.path}: row {exam.index} (exam {exam.row.exam_id}) changed since it was scanned"
            )
        return trace[exam.padding.values] * _MILLIVOLTS_PER_PUBLISHED_UNIT


def _record(exam: _Exam, reader: _TraceReader) -> Record:
    """Build an exam's Record: one Source of 12 lazy leads, and the patient's sex and mortality.

    The age and the labels are the tasks' targets, so they stay off the Record.

    Returns:
        The Record, whose values stay in the part until read.
    """
    row, padding = exam.row, exam.padding
    signals = tuple(
        Signal.from_loader(
            name=lead.value,
            spec=_SPEC,
            time_axis=_TIME_AXIS,
            n_values=padding.n_values,
            loader=partial(reader.lead, exam, column),
        )
        for column, lead in enumerate(_Lead)
    )
    record = Record(
        sources=(Source(name=_SOURCE_NAME, signals=signals),),
        metadata={
            _MetadataKey.EXAM_ID: row.exam_id,
            _MetadataKey.TRACE_FILE: row.trace_file,
            _MetadataKey.N_PAD_BEFORE: padding.n_before,
            _MetadataKey.N_PAD_AFTER: padding.n_after,
        },
    )
    record.annotate(Annotation(key=_Key.SUBJECT_IDS, value=[str(row.patient_id)]))
    record.annotate(Annotation(key=_Key.SEX, value=row.sex.value, description=_SEX_NOTE))
    if row.mortality is not None:
        record.annotate(Annotation(key=_Key.DEATH, value=row.mortality.died, description=_DEATH_NOTE))
        record.annotate(
            Annotation(key=_Key.FOLLOW_UP, value=row.mortality.years, unit=ureg.year, description=_FOLLOW_UP_NOTE)
        )
    return record


def _age_task(record: Record, row: _ExamRow) -> ScalarPredictionTask:
    """Return an exam's age task: its patient's age, as Lima et al. 2021 predict it from the trace.

    The authors' published prediction is a task annotation, never model input.

    Returns:
        The scalar task, with the age in years as target.
    """
    task = ScalarPredictionTask(
        inputs=(record,),
        # A float with a unit keeps the type a regression metric needs.
        targets=(float(row.age),),
        unit=ureg.year,
        target_name="age",
        split=_AGE_SPLIT,
    )
    task.annotate(
        Annotation(
            key=_Key.NN_PREDICTED_AGE, value=row.nn_predicted_age, unit=ureg.year, description=_NN_PREDICTED_AGE_NOTE
        )
    )
    return task


def _abnormality_targets(row: _ExamRow) -> tuple[str, ...]:
    """Return the labels of an exam's abnormality task.

    Returns:
        Every abnormality the exam is labelled with, in the authors' output order. An exam marked normal has only
        ``normal_ecg``; an exam with all seven columns false has only ``abnormal_unspecified``.
    """
    if row.normal_ecg:
        return (_NORMAL_ECG_LABEL.name,)
    if not row.abnormalities:
        return (_ABNORMAL_UNSPECIFIED_LABEL.name,)
    return tuple(abnormality.value for abnormality in row.abnormalities)


def _abnormality_task(record: Record, row: _ExamRow) -> ClassificationTask:
    """Return the connector's eight-label task extending Ribeiro et al. 2020's six diagnosis classes.

    Returns:
        The multi-label classification task, whose targets ``_abnormality_targets`` gives.
    """
    return ClassificationTask(
        inputs=(record,),
        targets=_abnormality_targets(row),
        label_set=ABNORMALITIES,
        split=_ABNORMALITY_SPLIT,
    )


class Code15Connector(BaseConnector[Path]):
    """Build CODE-15% from ``exams.csv`` and the 18 zips of its Zenodo record."""

    async def download_async(self, cache_dir: Path) -> list[Path]:  # ruff: ignore[no-self-use] (the connector interface)
        """Fetch ``exams.csv`` and every zip, check their digests, and extract each zip once.

        Each zip is deleted once extracted.

        Returns:
            The one directory that holds ``exams.csv`` and the parts.
        """
        root = cache_dir / "code15"
        await download_files([Artifact(_download_url(_CSV_FILE), root / _CSV_FILE, sha256=_CSV_SHA256)])
        await asyncio.gather(
            *(
                ensure_archive(_download_url(_part_zip(part)), root, filename=_part_zip(part), sha256=sha)
                for part, sha in enumerate(_PART_SHA256)
            )
        )
        return [root]

    def convert(self, raw_refs: list[Path]) -> TimeFDataset:
        """Convert each exam with a trace into one lazy Record, and stream its age and abnormality tasks.

        Reads ``exams.csv`` and scans every part once, which checks each row and trace. The samples stay in
        the parts until written, so the files must stay available until then.

        Args:
            raw_refs: The one directory that holds ``exams.csv`` and the extracted parts.

        Returns:
            A dataset with the provenance annotations, the Records, and a stream of every task:
            each exam's age task, then each exam's abnormality task.

        Raises:
            TimeFValidationError: If ``raw_refs`` is not one directory.
            TimeFFormatError: If a file is missing or malformed.
        """
        if len(raw_refs) != 1 or not raw_refs[0].is_dir():
            raise TimeFValidationError(f"CODE-15% converts one directory, got {raw_refs}")
        root = raw_refs[0]
        if not (root / _CSV_FILE).is_file():
            raise TimeFFormatError(f"{root}: {_CSV_FILE} is missing")
        exams = list(_scan_release(root, _read_exams(root / _CSV_FILE)))
        _check_counts(root, exams)
        reader = _TraceReader()
        records = [_record(exam, reader) for exam in exams]

        dataset = TimeFDataset(metadata=self.metadata())
        dataset.annotate(Annotation(key=_Key.SOURCE_URL, value=RECORD_URL))
        dataset.annotate(Annotation(key=_Key.SOURCE_DOI, value=RECORD_DOI))
        for record in records:
            dataset.add_record(record=record)

        def tasks() -> Iterator[Task]:
            for exam, record in zip(exams, records, strict=True):
                yield _age_task(record, exam.row)
            for exam, record in zip(exams, records, strict=True):
                yield _abnormality_task(record, exam.row)

        dataset.set_task_stream((ScalarPredictionTask, ClassificationTask), tasks)
        return dataset


CONNECTOR = Code15Connector
