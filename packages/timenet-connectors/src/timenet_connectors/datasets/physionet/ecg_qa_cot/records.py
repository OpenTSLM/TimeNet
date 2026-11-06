"""Build the ECG-QA CoT records from the PTB-XL parent, and the annotations that the tasks reference."""

from collections.abc import Mapping
from dataclasses import dataclass

from timenet.dataset import Record, TimeFDataset
from timenet.errors import TimeNetConnectorError
from timenet.reader import TimeFReader
from timenet.types import (
    Annotation,
    Split,
    read_value,
)
from timenet_connectors.datasets.physionet.ecg_qa_cot.labels import (
    KEPT_ANNOTATIONS,
    OTHER_FINDINGS,
    PTBXL_SEXES,
    QUALITY_ISSUES,
)
from timenet_connectors.datasets.physionet.ecg_qa_cot.source import WholeNumber
from timenet_connectors.sources.rows import SourceRow, validate_row


class _PatientRow(SourceRow):
    """A PTB-XL subject identifier written as decimal text."""

    patient_id: WholeNumber


@dataclass(frozen=True)
class _PtbxlRecording:
    """One recording's PTB-XL values, read once from the annotations of its imported record."""

    patient_id: int
    age: int
    sex: str
    annotations: Mapping[str, object]
    """The imported annotation values by key, for the device, recording-date, signal-quality, extra-beats
    and pacemaker values."""


def _ptbxl_recording(ecg_id: int, record: Record) -> _PtbxlRecording:
    """Read a recording's PTB-XL values from the annotations of its imported record.

    Args:
        ecg_id: The recording id.
        record: The imported PTB-XL record, with its annotations.

    Returns:
        The recording's PTB-XL values.

    Raises:
        TimeNetConnectorError: If the record has no patient, age, or sex, or has a sex other than
            male or female.
    """
    values = {annotation.key: read_value(annotation.value) for annotation in record.annotations}
    patients = values.get("subject_ids")
    if not isinstance(patients, list) or not patients:
        raise TimeNetConnectorError(f"PTB-XL ecg_id {ecg_id}: patient_id is empty")
    # PTB-XL states the age as an int in years and the sex as "male" or "female". The connector
    # keeps both values as they are.
    age = values.get("age")
    if not isinstance(age, int):
        raise TimeNetConnectorError(f"PTB-XL ecg_id {ecg_id}: age is empty")
    sex = values.get("sex")
    if not isinstance(sex, str):
        raise TimeNetConnectorError(f"PTB-XL ecg_id {ecg_id}: sex is empty")
    if sex not in PTBXL_SEXES:
        raise TimeNetConnectorError(f"PTB-XL ecg_id {ecg_id}: sex {sex!r} is not male or female")
    return _PtbxlRecording(
        patient_id=int(
            validate_row({"patient_id": patients[0]}, _PatientRow, location=f"PTB-XL ecg_id {ecg_id}").patient_id
        ),
        age=age,
        sex=sex,
        annotations=values,
    )


def _text(values: Mapping[str, object], key: str) -> str:
    """Return a text annotation value without surrounding whitespace, or ``""`` if the record has none.

    Args:
        values: The imported annotation values by key.
        key: The annotation key.

    Returns:
        The text.
    """
    value = values.get(key)
    return value.strip() if isinstance(value, str) else ""


def _clinical_context(recording: _PtbxlRecording) -> str:
    """Build a recording's clinical context from its PTB-XL values.

    The context states the patient's age and sex, the recording device, and each signal-quality issue,
    extra beats, or pacemaker for which the imported record has an annotation.

    Args:
        recording: The recording's PTB-XL values.

    Returns:
        The clinical-context text.
    """
    values = recording.annotations
    parts = [f"{recording.age}-year-old", f"{recording.sex} patient", "12-lead ECG"]
    if _text(values, "recording_date"):
        parts.append("clinical recording")
    if device := _text(values, "device"):
        parts.append(f"recorded with {device}")
    issues = [phrase for key, phrase in QUALITY_ISSUES if _text(values, key)]
    parts.append(f"Signal quality: {', '.join(issues)}" if issues else "Signal quality: adequate for analysis")
    parts.extend(phrase for key, phrase in OTHER_FINDINGS if _text(values, key))
    return f"{parts[0]} {parts[1]}. " + ". ".join(parts[2:]) + "."


def add_records(
    dataset: TimeFDataset, ptbxl: TimeFReader, split_of_ecg: Mapping[int, Split]
) -> tuple[dict[int, Record], dict[int, str]]:
    """Import the PTB-XL records that the CoT rows name, and keep only what ECG-QA needs.

    Each record keeps PTB-XL's subject IDs, sex, and age, gains its split, and records the patient ID
    in its metadata. The parent finds a record by the ``ecg_id`` in its metadata.

    Args:
        dataset: The dataset that imports the records.
        ptbxl: The opened PTB-XL parent.
        split_of_ecg: The split of each recording that the CoT rows name.

    Returns:
        The records and the clinical context of each recording, both keyed by PTB-XL ECG ID.

    Raises:
        TimeNetConnectorError: If the PTB-XL parent has no recording for an ECG that the rows name, or
            an imported record has no patient, age, or sex, or a sex other than male or female.
    """
    # PTB-XL keeps each recording's upstream ecg_id in the record's metadata.
    parent_ids = {
        record.metadata["ecg_id"]: record.id
        for record in ptbxl.iter_records(with_annotations=False)
        if record.metadata.get("ecg_id") in split_of_ecg
    }
    missing = sorted(split_of_ecg.keys() - parent_ids.keys())
    if missing:
        raise TimeNetConnectorError(f"the PTB-XL parent has no recording with ecg_id {missing}")
    imported = dataset.import_from(ptbxl, records=[parent_ids[ecg_id] for ecg_id in sorted(split_of_ecg)])
    records: dict[int, Record] = {}
    contexts: dict[int, str] = {}
    for ecg_id, split in sorted(split_of_ecg.items()):
        record = imported.records[parent_ids[ecg_id]]
        recording = _ptbxl_recording(ecg_id, record)
        record.metadata["patient_id"] = recording.patient_id
        # Keep the subject IDs, the sex, and the age from PTB-XL, which have no span. Clear the
        # other imported annotations. The child adds its own split.
        record.annotations = tuple(
            annotation for annotation in record.annotations if annotation.key in KEPT_ANNOTATIONS
        )
        record.add_annotations([Annotation(key="split", value=split.value)])
        records[ecg_id] = record
        contexts[ecg_id] = _clinical_context(recording)
    return records, contexts
