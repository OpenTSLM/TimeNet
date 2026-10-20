"""Build reusable, taskless PTB-XL ECG records at their original sampling rate."""

import ast
import csv
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path

from timenet.dataset import Record, Signal, Source, TimeFDataset
from timenet.dataset.axis import RegularAxis
from timenet.errors import TimeFValidationError
from timenet.types import Annotation, TimeSeriesSpec, ureg
from timenet_connectors.bases.physionet import BasePhysioNetConnector
from timenet_connectors.download import ensure_archive, find_dir_containing


PTBXL_ZIP_URL = "s3://physionet-open/ptb-xl/ptb-xl-1.0.3.zip"
_SAMPLING_RATE_HZ = 500
_LEAD_COUNT = 12
_VALIDATION_FOLD = 9
_TEST_FOLD = 10
_ECG = TimeSeriesSpec(spec_type="ecg", name="12-lead ECG", unit_value=ureg.millivolt)


@dataclass(frozen=True)
class PtbXlSource:
    """The extracted release root and its metadata table."""

    root: Path
    database_csv: Path


class PtbXlConnector(BasePhysioNetConnector[PtbXlSource]):
    """Preserve PTB-XL waveforms and clinical metadata without downstream tasks."""

    async def download_async(self, cache_dir: Path) -> list[PtbXlSource]:  # noqa: PLR6301
        """Return the extracted pinned PTB-XL release."""
        root = find_dir_containing(await ensure_archive(PTBXL_ZIP_URL, cache_dir), "ptbxl_database.csv")
        return [PtbXlSource(root, root / "ptbxl_database.csv")]

    def convert(self, raw_refs: list[PtbXlSource]) -> TimeFDataset:
        """Return taskless records with lazy WFDB value loaders.

        Raises:
            TimeFValidationError: If a source row or waveform header is invalid.
        """
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
                            id=f"ecg-{identity}-{name}",
                            loader=self._lead_loader(base, header, position),
                            n_values=int(header.sig_len),
                        )
                        for position, name in enumerate(header.sig_name)
                    )
                    record = dataset.add_record(
                        record=Record(
                            record_id=f"ptbxl-{identity}",
                            sources=(
                                Source(
                                    id=f"ptbxl-{identity}-source",
                                    name="PTB-XL ECG",
                                    signals=signals,
                                ),
                            ),
                        )
                    )
                    record.add_annotations(self._annotations(identity, row))
                except (KeyError, ValueError, SyntaxError, OSError) as exc:
                    raise TimeFValidationError(f"invalid PTB-XL row {line} in {source.database_csv}: {exc}") from exc
        return dataset

    @staticmethod
    def _annotations(identity: int, row: dict[str, str]) -> list[Annotation]:
        """Return clinical annotations with stable content IDs."""
        result = []
        for key, unit in (("age", "year"), ("height", "centimeter"), ("weight", "kilogram")):
            if row.get(key) not in {None, "", "nan", "NaN"}:
                result.append(Annotation(key=key, value=float(row[key]), unit=unit, id=f"ptbxl-{identity}-{key}"))
        for key in ("report", "heart_axis", "device"):
            if row.get(key):
                result.append(Annotation(key=key, value=row[key], id=f"ptbxl-{identity}-{key}"))
        if row.get("patient_id"):
            result.append(
                Annotation(
                    key="subject_ids", value=[str(int(float(row["patient_id"])))], id=f"ptbxl-{identity}-subject"
                )
            )
        if row.get("sex"):
            result.append(Annotation(key="sex", value=int(row["sex"]), id=f"ptbxl-{identity}-sex"))
        if row.get("scp_codes"):
            codes = ast.literal_eval(row["scp_codes"])
            result.append(
                Annotation(
                    key="scp_codes", value=sorted(codes), metadata={"likelihoods": codes}, id=f"ptbxl-{identity}-codes"
                )
            )
        if row.get("strat_fold"):
            fold = int(row["strat_fold"])
            split = "test" if fold == _TEST_FOLD else "validation" if fold == _VALIDATION_FOLD else "train"
            result.extend(
                (
                    Annotation(key="strat_fold", value=fold, id=f"ptbxl-{identity}-fold"),
                    Annotation(key="split", value=split, id=f"ptbxl-{identity}-split"),
                )
            )
        return result


CONNECTOR = PtbXlConnector
