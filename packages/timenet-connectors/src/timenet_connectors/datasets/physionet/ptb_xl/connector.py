"""Build reusable, taskless PTB-XL ECG records at their original sampling rate."""

import ast
import csv
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path

from timenet.dataset import Record, Signal, Source, TimeFDataset
from timenet.dataset.axis import RegularAxis
from timenet.errors import TimeFFormatError, TimeFValidationError
from timenet.types import Annotation, TimeSeriesSpec, ureg
from timenet_connectors.bases.physionet import BasePhysioNetConnector
from timenet_connectors.download import ensure_archive, find_dir_containing


PTBXL_ZIP_URL = "s3://physionet-open/ptb-xl/ptb-xl-1.0.3.zip"
_SAMPLING_RATE_HZ = 500
_LEAD_COUNT = 12
_VALIDATION_FOLD = 9
_TEST_FOLD = 10
_ECG = TimeSeriesSpec(spec_type="ecg", name="12-lead ECG", unit_value=ureg.millivolt)
# PTB-XL's signal-quality columns, then its extra-beats and pacemaker columns.
_FINDINGS = ("baseline_drift", "static_noise", "burst_noise", "electrodes_problems", "extra_beats", "pacemaker")
# PTB-XL codes sex as 0 = male, 1 = female.
_SEX = {"0": "male", "1": "female"}
# PTB-XL anonymizes the age of every patient older than 89 as 300, in compliance with HIPAA (PhysioNet
# PTB-XL page, Data Preprocessing: "ages of more than 89 years appear in the range of 300 years").
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


def _age_in_years(row: dict[str, str]) -> int | None:
    """Return the age of a PTB-XL row in whole years.

    PTB-XL writes each age as a float, such as ``"56.0"``.

    Args:
        row: The ``ptbxl_database.csv`` row.

    Returns:
        The age, or ``None`` if the cell is empty.

    Raises:
        TimeFFormatError: If the cell holds no whole number.
    """
    if row.get("age") in {None, "", "nan", "NaN"}:
        return None
    cell = row["age"]
    try:
        years = float(cell)
    except ValueError:
        years = None
    if years is None or not years.is_integer():
        raise TimeFFormatError(f"ptbxl_database.csv ecg_id {row['ecg_id']}: age {cell!r} is not a whole number")
    return int(years)


class PtbXlConnector(BasePhysioNetConnector[PtbXlSource]):
    """Preserve PTB-XL waveforms and clinical metadata without downstream tasks."""

    async def download_async(self, cache_dir: Path) -> list[PtbXlSource]:  # noqa: PLR6301
        """Return the extracted pinned PTB-XL release."""
        root = find_dir_containing(await ensure_archive(PTBXL_ZIP_URL, cache_dir), "ptbxl_database.csv")
        return [PtbXlSource(root, root / "ptbxl_database.csv")]

    def convert(self, raw_refs: list[PtbXlSource]) -> TimeFDataset:
        """Return taskless records with lazy WFDB value loaders.

        Each record carries its PTB-XL ``ecg_id`` in its metadata, which children select it by.

        Raises:
            TimeFValidationError: If a source row or waveform header is invalid.
            TimeFFormatError: If a row's sex code is not 0 or 1, or its age is not a whole number.
        """  # noqa: DOC502 - TimeFFormatError is raised by _annotations
        dataset = TimeFDataset(metadata=self.metadata())
        source = raw_refs[0]
        with source.database_csv.open(newline="", encoding="utf-8") as stream:
            for line, row in enumerate(csv.DictReader(stream), 2):
                try:
                    identity = int(row["ecg_id"])
                    base = source.root / row["filename_hr"]
                    header = self._read_header(base)
                    if header.fs != _SAMPLING_RATE_HZ or len(header.sig_name) != _LEAD_COUNT:
                        raise TimeFValidationError("PTB-XL requires twelve leads at 500 Hz")
                    axis = RegularAxis.from_rate_hz(Fraction(str(header.fs)))
                    signals = tuple(
                        Signal.from_loader(
                            spec=_ECG,
                            name=name,
                            time_axis=axis,
                            loader=self._lead_loader(base, header, position),
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
                    record.add_annotations(self._annotations(row))
                except (KeyError, ValueError, SyntaxError, OSError) as exc:
                    raise TimeFValidationError(f"invalid PTB-XL row {line} in {source.database_csv}: {exc}") from exc
        return dataset

    @staticmethod
    def _annotations(row: dict[str, str]) -> list[Annotation]:
        """Return the clinical annotations of one PTB-XL row.

        Raises:
            TimeFFormatError: If the sex code is not 0 or 1, or the age is not a whole number.
        """
        result = []
        if (age := _age_in_years(row)) is not None:
            result.append(Annotation(key="age", value=age, unit="year", description=_AGE_DESCRIPTION))
        for key, unit in (("height", "centimeter"), ("weight", "kilogram")):
            if row.get(key) not in {None, "", "nan", "NaN"}:
                result.append(Annotation(key=key, value=float(row[key]), unit=unit))
        for key in ("report", "heart_axis", "device"):
            if row.get(key):
                result.append(Annotation(key=key, value=row[key]))
        # A cell counts only when it holds more than whitespace. A finding's cell can name the leads it affects.
        for key in ("recording_date", *_FINDINGS):
            if value := (row.get(key) or "").strip():
                result.append(Annotation(key=key, value=value))
        if row.get("patient_id"):
            result.append(Annotation(key="subject_ids", value=[str(int(float(row["patient_id"])))]))
        if row.get("sex"):
            sex = _SEX.get(row["sex"])
            if sex is None:
                raise TimeFFormatError(f"ptbxl_database.csv ecg_id {row['ecg_id']}: sex {row['sex']!r} is not 0 or 1")
            result.append(Annotation(key="sex", value=sex, description=_SEX_DESCRIPTION))
        if row.get("scp_codes"):
            codes = ast.literal_eval(row["scp_codes"])
            result.append(Annotation(key="scp_codes", value=sorted(codes), metadata={"likelihoods": codes}))
        if row.get("strat_fold"):
            fold = int(row["strat_fold"])
            split = "test" if fold == _TEST_FOLD else "validation" if fold == _VALIDATION_FOLD else "train"
            result.extend(
                (
                    Annotation(key="strat_fold", value=fold),
                    Annotation(key="split", value=split),
                )
            )
        return result


CONNECTOR = PtbXlConnector
