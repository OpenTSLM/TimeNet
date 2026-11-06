"""Build reusable PTB-XL ECG records without tasks, at their original sampling rate."""

import ast
import csv
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path

from timenet.dataset import Record, Signal, Source, TimeFDataset
from timenet.dataset.axis import RegularAxis
from timenet.errors import TimeNetConnectorError
from timenet.types import Annotation, Split, TimeSeriesSpec, ureg
from timenet_connectors.bases.physionet import BasePhysioNetConnector
from timenet_connectors.demographics import Sex, age_annotation, sex_annotation, subject_ids_annotation
from timenet_connectors.download import ensure_archive, find_dir_containing


PTBXL_ZIP_URL = "s3://physionet-open/ptb-xl/ptb-xl-1.0.3.zip"
PTBXL_ZIP_SHA256 = "2d98687a9c69c8df03dbd69e0c569b3363981a0f667861c279c747d4aec29812"
_SAMPLING_RATE_HZ = 500
_LEAD_COUNT = 12
_VALIDATION_FOLD = 9
_TEST_FOLD = 10
_ECG = TimeSeriesSpec(spec_type="ecg", name="12-lead ECG", unit_value=ureg.millivolt)
# The columns that PTB-XL fills in every row. If one of them is empty, the row is broken.
_REQUIRED_COLUMNS = (
    "ecg_id",
    "filename_hr",
    "patient_id",
    "age",
    "sex",
    "recording_date",
    "device",
    "scp_codes",
    "strat_fold",
)
# A cell is empty if it holds one of these values after its surrounding whitespace is removed.
_EMPTY_CELLS = frozenset({"", "nan", "NaN"})
# PTB-XL's signal-quality columns, then its extra-beats and pacemaker columns.
_FINDINGS = ("baseline_drift", "static_noise", "burst_noise", "electrodes_problems", "extra_beats", "pacemaker")
# PTB-XL codes sex as 0 = male, 1 = female.
_SEX = {"0": Sex.MALE, "1": Sex.FEMALE}
# PTB-XL anonymizes the age of every patient older than 89 as 300, to comply with HIPAA. The PTB-XL
# page on PhysioNet says in "Data Preprocessing": "ages of more than 89 years appear in the range of
# 300 years".
_AGE_OVER_89 = 300
_SEX_DESCRIPTION = "The patient's sex, as PTB-XL states it."
_AGE_DESCRIPTION = (
    f"The patient's age in whole years, as PTB-XL states it. PTB-XL states {_AGE_OVER_89} for every "
    "patient older than 89."
)


@dataclass(frozen=True)
class PtbXlSource:
    """The extracted release root and its metadata table."""

    root: Path
    database_csv: Path


def _age_in_years(cells: dict[str, str]) -> int:
    """Return the age of a PTB-XL row in whole years.

    PTB-XL writes each age as a float, such as ``"56.0"``.

    Args:
        cells: The cells of the ``ptbxl_database.csv`` row, without surrounding whitespace.

    Returns:
        The age.

    Raises:
        TimeNetConnectorError: If the cell holds no whole number.
    """
    cell = cells["age"]
    try:
        years = float(cell)
    except ValueError:
        years = None
    if years is None or not years.is_integer():
        raise TimeNetConnectorError(f"age {cell!r} is not a whole number")
    return int(years)


def _parse_patient_id(cells: dict[str, str]) -> str:
    """Return the patient ID of a PTB-XL row as a whole number.

    PTB-XL writes each patient ID as a float. For example, this function returns ``"15709"`` for
    ``"15709.0"``.

    Args:
        cells: The cells of the ``ptbxl_database.csv`` row, without surrounding whitespace.

    Returns:
        The patient ID.

    Raises:
        TimeNetConnectorError: If the cell holds no whole number.
    """
    cell = cells["patient_id"]
    try:
        number = float(cell)
    except ValueError:
        number = None
    if number is None or not number.is_integer():
        raise TimeNetConnectorError(f"patient_id {cell!r} is not a whole number")
    return str(int(number))


class PtbXlConnector(BasePhysioNetConnector[PtbXlSource]):
    """Preserve PTB-XL waveforms and clinical metadata without downstream tasks."""

    async def download_async(self, cache_dir: Path) -> list[PtbXlSource]:  # ruff: ignore[no-self-use]
        """Return the extracted pinned PTB-XL release."""
        root = find_dir_containing(
            await ensure_archive(PTBXL_ZIP_URL, cache_dir, sha256=PTBXL_ZIP_SHA256), "ptbxl_database.csv"
        )
        return [PtbXlSource(root, root / "ptbxl_database.csv")]

    def convert(self, raw_refs: list[PtbXlSource]) -> TimeFDataset:
        """Return records without tasks, with lazy WFDB value loaders.

        Each record carries its PTB-XL ``ecg_id`` in its metadata. A child dataset selects the
        record by this ID.

        Raises:
            TimeNetConnectorError: If a row of ``ptbxl_database.csv`` or its recording does not hold
                what PTB-XL describes. Examples are a malformed cell, an empty required column, a
                patient ID that is not a whole number, a negative age, a sex code other than 0 or 1, or an
                ``ecg_id`` that an earlier row has. The message names the CSV row and the
                ``ecg_id``. For a repeated ``ecg_id``, the message also names the earlier row.
        """
        dataset = TimeFDataset(metadata=self.metadata())
        lead_loader = self._lead_loaders()
        source = raw_refs[0]
        # The CSV row where each ecg_id first appears.
        first_rows: dict[int, int] = {}
        with source.database_csv.open(newline="", encoding="utf-8") as stream:
            for line, row in enumerate(csv.DictReader(stream), 2):
                try:
                    annotations = self._annotations(row)
                    identity = int(row["ecg_id"])
                    if identity in first_rows:
                        raise TimeNetConnectorError(f"ecg_id {identity} repeats row {first_rows[identity]}")
                    first_rows[identity] = line
                    base = source.root / row["filename_hr"]
                    header = self._read_header(base)
                    if header.fs != _SAMPLING_RATE_HZ or len(header.sig_name) != _LEAD_COUNT:
                        raise TimeNetConnectorError("PTB-XL requires twelve leads at 500 Hz")
                    axis = RegularAxis.from_rate_hz(Fraction(str(header.fs)))
                    signals = tuple(
                        Signal.from_loader(
                            spec=_ECG,
                            name=name,
                            time_axis=axis,
                            loader=lead_loader(base, header, position),
                            n_values=int(header.sig_len),
                        )
                        for position, name in enumerate(header.sig_name)
                    )
                    record = dataset.add_record(
                        record=Record(
                            metadata={"ecg_id": identity},
                            sources=(Source(name="PTB-XL ECG", signals=signals),),
                        )
                    )
                    record.add_annotations(annotations)
                except (KeyError, ValueError, SyntaxError, OSError, TimeNetConnectorError) as exc:
                    raise TimeNetConnectorError(
                        f"invalid PTB-XL row {line} (ecg_id {row.get('ecg_id')}) in {source.database_csv}: {exc}"
                    ) from exc
        return dataset

    @staticmethod
    def _annotations(row: dict[str, str]) -> list[Annotation]:
        """Return the clinical annotations of one PTB-XL row.

        PTB-XL leaves a cell empty when a value is unknown, for example ``height`` and ``weight``. An empty
        cell in such a column adds no annotation. PTB-XL fills the required columns in every row, for
        example ``patient_id`` and ``sex``. If one of them is empty, the row is broken.

        Raises:
            TimeNetConnectorError: If a required column is empty, the sex code is not 0 or 1, the age is
                negative, or the age or the patient ID is not a whole number.
        """
        cells = {key: (value or "").strip() for key, value in row.items()}
        empty = [column for column in _REQUIRED_COLUMNS if cells.get(column, "") in _EMPTY_CELLS]
        if empty:
            names = ", ".join(empty)
            raise TimeNetConnectorError(
                f"required column {names} is empty" if len(empty) == 1 else f"required columns {names} are empty"
            )
        result = [age_annotation(_age_in_years(cells), description=_AGE_DESCRIPTION)]
        for key, unit in (("height", "centimeter"), ("weight", "kilogram")):
            if cells.get(key, "") not in _EMPTY_CELLS:
                result.append(Annotation(key=key, value=float(cells[key]), unit=unit))
        for key in ("report", "heart_axis"):
            if cells.get(key):
                result.append(Annotation(key=key, value=row[key]))
        result.extend(
            (
                Annotation(key="device", value=row["device"]),
                Annotation(key="recording_date", value=cells["recording_date"]),
            )
        )
        # The cell of a finding can name the leads that the finding affects. The annotation gets the
        # cell value without its surrounding whitespace.
        for key in _FINDINGS:
            if value := cells.get(key):
                result.append(Annotation(key=key, value=value))
        result.append(subject_ids_annotation([_parse_patient_id(cells)]))
        sex = _SEX.get(cells["sex"])
        if sex is None:
            raise TimeNetConnectorError(f"sex {row['sex']!r} is not 0 or 1")
        result.append(sex_annotation(sex, description=_SEX_DESCRIPTION))
        codes = ast.literal_eval(cells["scp_codes"])
        result.append(Annotation(key="scp_codes", value=sorted(codes), metadata={"likelihoods": codes}))
        fold = int(cells["strat_fold"])
        split = Split.TEST if fold == _TEST_FOLD else Split.VALIDATION if fold == _VALIDATION_FOLD else Split.TRAIN
        result.extend(
            (
                Annotation(key="strat_fold", value=fold),
                Annotation(key="split", value=split.value),
            )
        )
        return result


CONNECTOR = PtbXlConnector
