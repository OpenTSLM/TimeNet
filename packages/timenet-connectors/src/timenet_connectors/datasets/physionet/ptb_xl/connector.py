"""Convert PTB-XL into a taskless base layer of reusable ECG records."""

import ast
import csv
from fractions import Fraction
from pathlib import Path
from typing import Annotated, Any

from pydantic import BaseModel, BeforeValidator, ConfigDict, Field
from pydantic.dataclasses import dataclass

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


def _missing_as_none(value: Any) -> Any:
    return None if value is None or str(value).strip() in {"", "nan", "NaN"} else value


def _parse_codes(value: Any) -> Any:
    return ast.literal_eval(value) if isinstance(value, str) else value


_OptionalFloat = Annotated[float | None, BeforeValidator(_missing_as_none)]
_OptionalInt = Annotated[int | None, BeforeValidator(_missing_as_none)]
_OptionalText = Annotated[str | None, BeforeValidator(_missing_as_none)]
_ScpCodes = Annotated[dict[str, float], BeforeValidator(_parse_codes)]


class _PtbXlRow(BaseModel):
    """One validated row of ``ptbxl_database.csv``."""

    model_config = ConfigDict(extra="ignore", frozen=True)

    ecg_id: int
    filename_hr: str = Field(min_length=1)
    patient_id: _OptionalFloat = None
    age: _OptionalFloat = None
    height: _OptionalFloat = None
    weight: _OptionalFloat = None
    sex: _OptionalInt = None
    report: _OptionalText = None
    heart_axis: _OptionalText = None
    device: _OptionalText = None
    scp_codes: _ScpCodes
    strat_fold: _OptionalInt = None


def record_id(ecg_id: int) -> str:
    """Return the stable TimeF Record ID for a PTB-XL ECG."""
    return f"ptbxl-{ecg_id}"


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
            for raw in csv.DictReader(handle):
                dataset.add_record(record=self._record(source.root, _PtbXlRow.model_validate(raw)))
        return dataset

    def _record(self, root: Path, row: _PtbXlRow) -> Record:
        """Build one Record with lazy lead loaders and clinical annotations.

        Returns:
            The populated Record.
        """
        ecg_id = row.ecg_id
        record_base = root / row.filename_hr
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
        record = Record(
            record_id=record_id(ecg_id),
            subject_ids=() if row.patient_id is None else (str(int(row.patient_id)),),
            sources=(Source(id=source_id, name="PTB-XL ECG", signals=signals),),
        )
        record.add_annotations(self._annotations(ecg_id, row))
        return record

    @staticmethod
    def _annotations(ecg_id: int, row: _PtbXlRow) -> list[Annotation]:
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

        for key, value, unit in (
            ("age", row.age, "year"),
            ("height", row.height, "centimeter"),
            ("weight", row.weight, "kilogram"),
        ):
            if value is not None:
                add(key, value, unit=unit)
        if row.sex is not None:
            add("sex", row.sex)
        for key in ("report", "heart_axis", "device"):
            if (value := getattr(row, key)) is not None:
                add(key, value)
        add("scp_codes", sorted(row.scp_codes))
        if row.strat_fold is not None:
            split = (
                "test"
                if row.strat_fold == _TEST_FOLD
                else "validation"
                if row.strat_fold == _VALIDATION_FOLD
                else "train"
            )
            add("strat_fold", row.strat_fold)
            add("split", split)
        return annotations


CONNECTOR = PtbXlConnector
