"""Read and validate CODE-15 CSV rows and lazily load checked HDF5 traces."""

from collections.abc import Iterator
import hashlib
from pathlib import Path
from types import ModuleType
from typing import TYPE_CHECKING, NamedTuple, Self
import weakref

import numpy as np
import pyarrow as pa
from pydantic import Field, ValidationInfo, field_validator, model_validator
from pydantic_core import PydanticCustomError

from timenet.errors import TimeNetConnectorError
from timenet.types.shared_loading import SharedChannelLoader
from timenet_connectors.demographics import Sex
from timenet_connectors.sources.csv import iter_csv_rows
from timenet_connectors.sources.reread import checked_reread
from timenet_connectors.sources.rows import SourceRow
from timenet_connectors.sources.signals import numeric_signal


if TYPE_CHECKING:
    # Only for the types. At run time, `_h5py` imports h5py on first use (docs/connectors.md).
    import h5py

from timenet_connectors.datasets.zenodo.code_15.vocabulary import (
    CSV_FILE,
    DECIMAL_PATTERN,
    DIGEST_BYTES,
    EXAM_ID_DATASET,
    FILLER_EXAM_ID,
    MILLIVOLTS_PER_PUBLISHED_UNIT,
    PART_FILENAME_PATTERN,
    RECORD_URL,
    SPEC,
    TRACE_SAMPLES,
    TRACINGS_DATASET,
    Abnormality,
    Lead,
)


_KEPT_EXAMS = 345_109


"""The exams of the release whose trace is not zero everywhere, which get a Record. dataset.yaml states it."""


_KEPT_PATIENTS = 233_480


"""The patients of those exams. dataset.yaml states it."""


_SCAN_ROWS = 512


"""Traces read at a time while scanning: 96 MiB of float32."""


class TracePadding(NamedTuple):
    """How many zero samples the authors added before and after a trace."""

    n_before: int
    n_after: int

    @property
    def n_values(self) -> int:
        """The number of samples that the trace holds without its padding."""
        return TRACE_SAMPLES - self.n_before - self.n_after

    @property
    def sample_slice(self) -> slice:
        """The samples of the published trace that hold the recording."""
        return slice(self.n_before, TRACE_SAMPLES - self.n_after)


class Mortality(NamedTuple):
    """The ``death`` and ``timey`` fields, which ``exams.csv`` states together or not at all."""

    died: bool
    """Whether the patient died in the follow-up."""
    years: float
    """Years to death, or to the end of follow-up for a survivor."""


class ExamRow(SourceRow):
    """Typed CODE-15 CSV fields, preserving the release's exact textual spellings."""

    exam_id: int = Field(ge=1)
    age: int = Field(ge=0)
    is_male: bool
    nn_predicted_age: float
    first_degree_av_block: bool = Field(alias="1dAVb")
    rbbb: bool = Field(alias="RBBB")
    lbbb: bool = Field(alias="LBBB")
    sb: bool = Field(alias="SB")
    st: bool = Field(alias="ST")
    af: bool = Field(alias="AF")
    patient_id: int = Field(ge=0)
    death: bool | None
    timey: float | None = Field(ge=0)
    normal_ecg: bool
    trace_file: str

    @field_validator("exam_id", "age", "patient_id", mode="before")
    @classmethod
    def _integer_spelling(cls, value: object) -> object:
        if isinstance(value, str) and not (value.isascii() and value.isdigit()):
            raise PydanticCustomError("integer_spelling", "expected an unsigned decimal integer")
        return int(value) if isinstance(value, str) else value

    @field_validator(
        "is_male", "first_degree_av_block", "rbbb", "lbbb", "sb", "st", "af", "normal_ecg", "death", mode="before"
    )
    @classmethod
    def _boolean_spelling(cls, value: object, info: ValidationInfo) -> object:
        if isinstance(value, str) and not value and info.field_name == "death":
            return None
        if isinstance(value, str) and value not in {"True", "False"}:
            raise PydanticCustomError("boolean_spelling", "expected True or False")
        return value == "True" if isinstance(value, str) else value

    @field_validator("nn_predicted_age", "timey", mode="before")
    @classmethod
    def _decimal_spelling(cls, value: object, info: ValidationInfo) -> object:
        if isinstance(value, str) and not value and info.field_name == "timey":
            return None
        if isinstance(value, str) and not DECIMAL_PATTERN.fullmatch(value):
            raise PydanticCustomError("decimal_spelling", "expected a decimal number")
        return float(value) if isinstance(value, str) else value

    @field_validator("trace_file")
    @classmethod
    def _part_filename(cls, value: str) -> str:
        if not PART_FILENAME_PATTERN.fullmatch(value):
            raise PydanticCustomError("part_filename", "expected an exams_part<n>.hdf5 file")
        return value

    @model_validator(mode="after")
    def _consistent_fields(self) -> Self:
        if (self.death is None) != (self.timey is None):
            raise PydanticCustomError("mortality_pair", "death and timey must be both stated or both empty")
        if self.normal_ecg and self.abnormalities:
            raise PydanticCustomError(
                "normal_abnormal",
                "normal_ecg is True, but the exam is labelled {labels}",
                {"labels": [a.value for a in self.abnormalities]},
            )
        if self.sb and self.st:
            raise PydanticCustomError(
                "sinus_labels", "SB and ST are both True, but a sinus rhythm can't be both slow and fast"
            )
        return self

    @property
    def abnormalities(self) -> tuple[Abnormality, ...]:
        """The labelled abnormalities in the authors' model output order (AF before ST)."""
        flags = (self.first_degree_av_block, self.rbbb, self.lbbb, self.sb, self.af, self.st)
        return tuple(label for label, present in zip(Abnormality, flags, strict=True) if present)

    @property
    def mortality(self) -> Mortality | None:
        """The paired mortality fields, or no follow-up information."""
        if self.death is None or self.timey is None:
            return None
        return Mortality(died=self.death, years=self.timey)

    @property
    def part(self) -> int:
        """The part number from the validated HDF5 filename."""
        return int(self.trace_file.removeprefix("exams_part").removesuffix(".hdf5"))

    @property
    def sex(self) -> Sex:
        """The patient's sex."""
        return Sex.MALE if self.is_male else Sex.FEMALE


class Exam(NamedTuple):
    """An exam with a trace after the first pass: its row, where its trace is, and what was scanned."""

    row: ExamRow
    path: Path
    index: int
    """The trace's row in its part."""
    padding: TracePadding
    digest: bytes
    """The digest of the trace as scanned, to tell a reread of the same values from another."""


class _Scan(NamedTuple):
    """One row of a part, scanned."""

    exam_id: int
    index: int
    """The trace's row in its part."""
    padding: TracePadding | None
    """``None`` for a trace that is zero in every lead and sample."""
    digest: bytes


def _part_file(part: int) -> str:
    """Return the name of the HDF5 file of part ``part``, such as ``exams_part3.hdf5``."""
    return f"exams_part{part}.hdf5"


def part_zip(part: int) -> str:
    """Return the name of the zip of part ``part`` on Zenodo, such as ``exams_part3.zip``."""
    return f"exams_part{part}.zip"


def download_url(name: str) -> str:
    """Return the Zenodo download URL of one file of the record."""
    return f"{RECORD_URL}/files/{name}?download=1"


def _trace_digest(trace: np.ndarray) -> bytes:
    """Return the digest of one trace's bytes, which a reread compares with the scan's."""
    return hashlib.blake2b(trace, digest_size=DIGEST_BYTES).digest()


class _PatientRules:
    """Check what ``exams.csv`` says about a patient across the patient's exams."""

    def __init__(self) -> None:
        self._sex: dict[int, Sex] = {}
        self._mortality_stated: set[int] = set()

    def check(self, row: ExamRow) -> None:
        """Check one more exam of its patient.

        Raises:
            TimeNetConnectorError: If the patient has another sex on an earlier exam, or states mortality on a second exam.
        """
        first_sex = self._sex.setdefault(row.patient_id, row.sex)
        if first_sex is not row.sex:
            raise TimeNetConnectorError(f"patient {row.patient_id} is listed as both sexes")
        if row.mortality is not None:
            if row.patient_id in self._mortality_stated:
                raise TimeNetConnectorError(f"patient {row.patient_id}'s mortality is stated twice")
            self._mortality_stated.add(row.patient_id)


def read_exams(path: Path) -> dict[int, ExamRow]:
    """Read ``exams.csv`` and check the rules across its rows.

    Returns:
        Every row by exam id, in file order.

    Raises:
        TimeNetConnectorError: If the header is not the release's, the file has no rows, a row is malformed, an exam
            is listed twice, a patient is listed with two sexes, or a patient's mortality is stated on two exams.
    """
    rows: dict[int, ExamRow] = {}
    patients = _PatientRules()
    for line, row in iter_csv_rows(path, ExamRow, unique="exam_id"):
        try:
            patients.check(row)
        except (ValueError, TimeNetConnectorError) as exc:
            raise TimeNetConnectorError(f"{path}:{line}: {exc}") from exc
        rows[row.exam_id] = row
    if not rows:
        raise TimeNetConnectorError(f"{path}: lists no exams")
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
    leading = np.where(any_nonzero, nonzero.argmax(axis=1), TRACE_SAMPLES)
    trailing = np.where(any_nonzero, nonzero[:, ::-1].argmax(axis=1), TRACE_SAMPLES)
    return leading, trailing


def _centred_padding(leading_zeros: int, trailing_zeros: int) -> TracePadding:
    """Return the most padding the authors' ``preprocess_ecg`` could have added.

    It pads ``(4096 - n) // 2`` zeros before a trace of ``n`` samples and the rest after, so the padding after
    is the padding before or one more. Of the zeros at either end, only those this rule explains are padding.
    Any others are zeros of the published trace, and stay. The raw lengths are not published, so this is an
    inference. Only exact all-lead zeros are dropped, and the Record's metadata keeps the counts.

    Returns:
        The padding before and after the trace.
    """
    n_before = min(leading_zeros, trailing_zeros)
    return TracePadding(n_before=n_before, n_after=min(trailing_zeros, n_before + 1))


class _PartDatasets(NamedTuple):
    """The two datasets of a part."""

    exam_ids: "h5py.Dataset"
    tracings: "h5py.Dataset"


def _check_datasets(handle: "h5py.File", path: Path) -> _PartDatasets:
    """Return a part's ``exam_id`` and ``tracings`` datasets, checked against the release's shapes and types.

    Returns:
        The two datasets.

    Raises:
        TimeNetConnectorError: If either is missing, or their shapes or types are not the release's.
    """
    if set(handle) != {EXAM_ID_DATASET, TRACINGS_DATASET}:
        raise TimeNetConnectorError(
            f"{path}: expected the datasets {EXAM_ID_DATASET} and {TRACINGS_DATASET}, got {set(handle)}"
        )
    exam_ids, tracings = handle[EXAM_ID_DATASET], handle[TRACINGS_DATASET]
    if exam_ids.dtype != np.int64 or tracings.dtype != np.float32:
        raise TimeNetConnectorError(
            f"{path}: expected int64 ids and float32 traces, got {exam_ids.dtype}, {tracings.dtype}"
        )
    if exam_ids.ndim != 1 or tracings.shape != (exam_ids.shape[0], TRACE_SAMPLES, len(Lead)):
        raise TimeNetConnectorError(
            f"{path}: expected {EXAM_ID_DATASET} (N,) and {TRACINGS_DATASET} (N, {TRACE_SAMPLES}, {len(Lead)}), "
            f"got {exam_ids.shape} and {tracings.shape}"
        )
    return _PartDatasets(exam_ids=exam_ids, tracings=tracings)


def _check_filler(path: Path, exam_ids: np.ndarray, tracings: "h5py.Dataset") -> int:
    """Check that a part ends with the all-zero filler row, and only there.

    Returns:
        The number of exams, which is the number of rows before the filler.

    Raises:
        TimeNetConnectorError: If the last row is not the filler, or another row has the filler's exam id.
    """
    n_exams = exam_ids.shape[0] - 1
    if n_exams < 0 or exam_ids[-1] != FILLER_EXAM_ID or np.any(tracings[n_exams] != 0):
        raise TimeNetConnectorError(f"{path}: the last row is not the all-zero filler with exam id {FILLER_EXAM_ID}")
    if np.any(exam_ids[:n_exams] == FILLER_EXAM_ID):
        raise TimeNetConnectorError(f"{path}: a row other than the last has the filler's exam id {FILLER_EXAM_ID}")
    return n_exams


def _scan_block(path: Path, first_row: int, exam_ids: np.ndarray, block: np.ndarray) -> Iterator[_Scan]:
    """Scan a block of consecutive traces.

    Yields:
        One scan per trace.

    Raises:
        TimeNetConnectorError: If a trace holds a value that is not finite.
    """
    finite = np.isfinite(block).all(axis=(1, 2))
    if not finite.all():
        raise TimeNetConnectorError(
            f"{path}: exam {exam_ids[int(np.argmin(finite))]}'s trace holds a value that is not finite"
        )
    leading, trailing = _zero_runs(block)
    for offset in range(block.shape[0]):
        all_zero = leading[offset] == TRACE_SAMPLES
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


def scan_release(root: Path, rows: dict[int, ExamRow]) -> Iterator[Exam]:
    """Scan every part that ``exams.csv`` names and join each trace to its row by exam id.

    The parts' rows are not in ``exams.csv``'s order, so the join is by id, as the authors say in
    ``automatic-ecg-diagnosis`` issue 32. A trace that is zero everywhere has no Record, so it is left out.

    Yields:
        Each exam with a trace, by part number, then by row in the part.

    Raises:
        TimeNetConnectorError: If a part is missing, holds an exam that ``exams.csv`` does not list in that part,
            holds an exam twice, or if an exam of ``exams.csv`` has no trace.
    """
    parts = sorted({row.part for row in rows.values()})
    seen: set[int] = set()
    for part in parts:
        name = _part_file(part)
        path = root / name
        if not path.is_file():
            raise TimeNetConnectorError(f"{root}: {name} is missing")
        for scan in _scan_part(path):
            row = rows.get(scan.exam_id)
            if row is None or row.part != part:
                raise TimeNetConnectorError(
                    f"{path}: row {scan.index} holds exam {scan.exam_id}, which {CSV_FILE} does not list in {name}"
                )
            if scan.exam_id in seen:
                raise TimeNetConnectorError(f"{path}: exam {scan.exam_id} has two traces")
            seen.add(scan.exam_id)
            if scan.padding is not None:
                yield Exam(row=row, path=path, index=scan.index, padding=scan.padding, digest=scan.digest)
    missing = sorted(rows.keys() - seen)
    if missing:
        raise TimeNetConnectorError(f"{root}: {len(missing)} exams of {CSV_FILE} have no trace, such as {missing[:5]}")


def check_counts(root: Path, exams: list[Exam]) -> None:
    """Check that the scan kept the release's number of exams and patients.

    The zip digests pin the files. This check also catches a change in how the connector reads them, such as
    a bug that takes a real trace for an all-zero one.

    Raises:
        TimeNetConnectorError: If the number of exams or patients with a Record is not the pinned one.
    """
    patients = {exam.row.patient_id for exam in exams}
    if len(exams) != _KEPT_EXAMS or len(patients) != _KEPT_PATIENTS:
        raise TimeNetConnectorError(
            f"{root}: expected {_KEPT_EXAMS} exams from {_KEPT_PATIENTS} patients whose trace is not zero "
            f"everywhere, got {len(exams)} exams from {len(patients)} patients"
        )


def _close(files: "dict[Path, h5py.File]") -> None:
    for handle in files.values():
        handle.close()
    files.clear()


class TraceReader:
    """Reread one exam's trace for all its leads, keeping only the part last read open.

    A Record's signals load one after another, and the Records come by part, then by row. So one cached
    trace and one open part suffice, and a load holds at most one HDF5 file at a time.
    """

    def __init__(self) -> None:
        self._files: dict[Path, h5py.File] = {}
        # The finalizer holds the dict, not the reader, so the reader can be collected.
        weakref.finalize(self, _close, self._files)
        self.shared = SharedChannelLoader(self._read_channels)

    def _read_channels(self, exam: Exam) -> tuple[pa.Array, ...]:
        try:
            values = self._read(exam)
        except (OSError, KeyError, IndexError, ValueError) as exc:
            raise TimeNetConnectorError(f"{exam.path}: exam {exam.row.exam_id}: cannot read trace: {exc}") from exc
        return tuple(
            numeric_signal(
                values[:, column],
                f"{exam.path}: exam {exam.row.exam_id}, {lead.value}",
                spec=SPEC,
                n_values=exam.padding.n_values,
            )
            for column, lead in enumerate(Lead)
        )

    def _read(self, exam: Exam) -> np.ndarray:
        """Reread a trace, check it is the one scanned, and return its samples in millivolts.

        Returns:
            The ``(n, 12)`` float32 samples, without padding.
        """
        handle = self._files.get(exam.path)
        if handle is None:
            _close(self._files)
            handle = self._files[exam.path] = _h5py().File(exam.path, "r")
        trace = checked_reread(
            lambda: handle[TRACINGS_DATASET][exam.index],
            fingerprint=_trace_digest,
            expected=exam.digest,
            context=f"{exam.path}: row {exam.index} (exam {exam.row.exam_id})",
        )
        return trace[exam.padding.sample_slice] * MILLIVOLTS_PER_PUBLISHED_UNIT
