"""Convert PTB-XL into a taskless base layer of reusable ECG records."""

import ast
import csv
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from typing import TypeGuard

from timenet.composition import BuildContext
from timenet.dataset import Record, Signal, Source, TimeFDataset
from timenet.dataset.axis import RegularAxis
from timenet.types import Annotation, TimeSeriesSpec, ureg
from timenet_connectors.bases.physionet import BasePhysioNetConnector
from timenet_connectors.download import ensure_archive, find_dir_containing


PTBXL_ZIP_URL = "s3://physionet-open/ptb-xl/ptb-xl-1.0.3.zip"

ECG_SPEC = TimeSeriesSpec(
    spec_type="ecg",
    name="12-lead ECG",
    unit_value=ureg.millivolt,
)
_VALIDATION_FOLD = 9
_TEST_FOLD = 10


@dataclass(frozen=True)
class PtbXlSource:
    """Paths to the extracted PTB-XL metadata and high-rate records."""

    root: Path
    database_csv: Path


def record_id(ecg_id: int) -> str:
    """Return the stable TimeF Record ID for a PTB-XL ECG."""
    return f"ptbxl-{ecg_id}"


def _present(value: str | None) -> TypeGuard[str]:
    return value is not None and value.strip() not in {"", "nan", "NaN"}


class PtbXlConnector(BasePhysioNetConnector[PtbXlSource]):
    """Connector for all PTB-XL records, without downstream benchmark tasks."""

    async def download_async(  # noqa: PLR6301 - BaseConnector override
        self, cache_dir: Path
    ) -> list[PtbXlSource]:
        """Download and extract the pinned PTB-XL release.

        Args:
            cache_dir: Directory that holds the downloaded archive.

        Returns:
            One source that names the release root and database CSV.
        """
        extracted = await ensure_archive(PTBXL_ZIP_URL, cache_dir)
        root = find_dir_containing(extracted, "ptbxl_database.csv")
        return [PtbXlSource(root=root, database_csv=root / "ptbxl_database.csv")]

    def convert(
        self,
        raw_refs: list[PtbXlSource],
        context: BuildContext | None = None,
    ) -> TimeFDataset:
        """Build one taskless Record for every PTB-XL database row.

        Args:
            raw_refs: The single source from :meth:`download_async`.
            context: Unused because PTB-XL is a root dataset.

        Returns:
            The taskless PTB-XL dataset.
        """
        _ = context
        source = raw_refs[0]
        dataset = TimeFDataset(metadata=self.metadata())
        with source.database_csv.open(newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                dataset.add_record(record=self._record(source.root, row))
        return dataset

    def _record(self, root: Path, row: dict[str, str]) -> Record:
        """Build one Record with lazy lead loaders and clinical annotations.

        Returns:
            The populated Record.
        """
        ecg_id = int(row["ecg_id"])
        record_base = root / row["filename_hr"]
        header = self._read_header(record_base)
        axis = RegularAxis.from_rate_hz(Fraction(str(header.fs)))
        source_id = f"ptbxl-{ecg_id}-source"
        signals = tuple(
            Signal.from_loader(
                spec=ECG_SPEC,
                name=name,
                time_axis=axis,
                loader=self._lead_loader(record_base, header, lead_index),
                source_id=source_id,
                id=f"ecg-{ecg_id}-{name}",
                n_values=int(header.sig_len),
            )
            for lead_index, name in enumerate(header.sig_name)
        )
        patient_id = row.get("patient_id")
        record = Record(
            record_id=record_id(ecg_id),
            subject_ids=() if not _present(patient_id) else (str(int(float(patient_id))),),
            sources=(Source(id=source_id, name="PTB-XL ECG", signals=signals),),
        )
        record.add_annotations(self._annotations(ecg_id, row))
        return record

    @staticmethod
    def _annotations(ecg_id: int, row: dict[str, str]) -> list[Annotation]:
        """Return normalized PTB-XL metadata annotations for one record."""
        annotations: list[Annotation] = []

        def add(key: str, value: object, *, unit: str | None = None) -> None:
            annotations.append(
                Annotation(
                    key=key,
                    value=value,
                    unit=unit,
                    id=f"ptbxl-{ecg_id}-{key}",
                )
            )

        for key, unit in (("age", "year"), ("height", "centimeter"), ("weight", "kilogram")):
            raw = row.get(key)
            if _present(raw):
                add(key, float(raw), unit=unit)
        sex = row.get("sex")
        if _present(sex):
            add("sex", int(float(sex)))
        for key in ("report", "heart_axis", "device"):
            raw = row.get(key)
            if _present(raw):
                add(key, raw)
        codes = row.get("scp_codes")
        if _present(codes):
            parsed = ast.literal_eval(codes)
            add("scp_codes", sorted(str(code) for code in parsed))
        fold = row.get("strat_fold")
        if _present(fold):
            fold_number = int(float(fold))
            split = (
                "test" if fold_number == _TEST_FOLD else "validation" if fold_number == _VALIDATION_FOLD else "train"
            )
            add("strat_fold", fold_number)
            add("split", split)
        return annotations


CONNECTOR = PtbXlConnector
