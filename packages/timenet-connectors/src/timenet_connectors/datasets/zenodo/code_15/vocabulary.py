"""CODE-15 release constants, lead contracts and diagnostic vocabulary."""

from enum import StrEnum, unique
import re

import numpy as np

from timenet.dataset.axis import RegularAxis
from timenet.types import (
    Label,
    LabelSet,
    Split,
    TimeSeriesSpec,
    ureg,
)


RECORD_URL = "https://zenodo.org/records/4916206"


"""Zenodo record 4916206, version 1.0.0, the record's only version."""


RECORD_DOI = "10.5281/zenodo.4916206"


CSV_FILE = "exams.csv"


CSV_SHA256 = "f9f82f4c07c1793e2285915c69f1113f5a7db8a0f2cf2f4b90474502f406bc35"


# Zenodo publishes only md5s, so these SHA-256s were computed on a first download whose bytes matched them.
PART_SHA256 = (
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


PART_FILENAME_PATTERN = re.compile(r"exams_part(0|[1-9][0-9]*)\.hdf5")


"""The HDF5 file a zip holds, which ``exams.csv`` names in ``trace_file``. ``_part_file`` builds the name."""


DECIMAL_PATTERN = re.compile(r"-?(0|[1-9][0-9]*)(\.[0-9]+)?")


"""A number as ``exams.csv`` writes it, such as ``40.160484``: no sign but minus, no exponent, no spaces."""


DIGEST_BYTES = 16


"""The size of a trace digest. It detects a changed file, not an attack, so 128 bits are enough."""


EXAM_ID_DATASET = "exam_id"


"""Each part's ``(N,)`` int64 exam ids. The rows are not in ``exams.csv``'s order."""


TRACINGS_DATASET = "tracings"


"""Each part's ``(N, 4096, 12)`` float32 traces, in the order of ``exam_id``."""


FILLER_EXAM_ID = 0


"""The id of the last row of every part: an all-zero trace that ``exams.csv`` does not list."""


TRACE_SAMPLES = 4096


"""Every published trace's length: the authors' ``--new_len 4096``."""


_SAMPLE_RATE_HZ = 400


MILLIVOLTS_PER_PUBLISHED_UNIT = np.float32(0.5)


"""The authors multiply each value by 2 (``ecg-age-prediction`` issue 5), so 0.5 gives millivolts.

Multiplying a float32 by 0.5 only lowers its exponent, so it is exact. README.md says why the connector keeps
the authors' factor of 2 and not our own fit of 1.953125.
"""


SPEC = TimeSeriesSpec(spec_type="ecg", name="12-lead ECG", unit_value=ureg.millivolt, dtype="float32")


TIME_AXIS = RegularAxis.from_rate_hz(_SAMPLE_RATE_HZ)


SOURCE_NAME = "12-lead ECG"


# Lima et al. 2021 hold CODE-15% out to test their age model. Ribeiro et al. 2020 train their abnormality
# model on the CODE cohort, which CODE-15% is part of.
AGE_SPLIT = Split.TEST


ABNORMALITY_SPLIT = Split.TRAIN


@unique
class Lead(StrEnum):
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
class Abnormality(StrEnum):
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
    Abnormality.FIRST_DEGREE_AV_BLOCK: "1st degree AV block.",
    Abnormality.RIGHT_BUNDLE_BRANCH_BLOCK: "Right bundle branch block.",
    Abnormality.LEFT_BUNDLE_BRANCH_BLOCK: "Left bundle branch block.",
    Abnormality.SINUS_BRADYCARDIA: "Sinus bradycardia.",
    Abnormality.ATRIAL_FIBRILLATION: "Atrial fibrillation.",
    Abnormality.SINUS_TACHYCARDIA: "Sinus tachycardia.",
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


@unique
class MetadataKey(StrEnum):
    """A key of ``Record.metadata``. README.md documents them, and the tests pin them."""

    EXAM_ID = "exam_id"
    TRACE_FILE = "trace_file"
    N_PAD_BEFORE = "n_pad_before"
    N_PAD_AFTER = "n_pad_after"


@unique
class AnnotationKey(StrEnum):
    """The key of an annotation the connector writes."""

    SOURCE_URL = "source_url"
    SOURCE_DOI = "source_doi"
    SUBJECT_IDS = "subject_ids"
    SEX = "sex"
    DEATH = "death"
    FOLLOW_UP = "follow_up"
    NN_PREDICTED_AGE = "nn_predicted_age"


SEX_NOTE = "The patient's sex, decoded from exams.csv's is_male."


DEATH_NOTE = "Whether the patient died in the follow-up. exams.csv states it on at most one exam per patient."


FOLLOW_UP_NOTE = (
    "exams.csv's timey: the years to the patient's death, or to the end of follow-up if they survived. Stated on "
    "the same exam as death."
)


NN_PREDICTED_AGE_NOTE = (
    "exams.csv's nn_predicted_age: the age the authors' network predicts from the exam's trace (Lima et al. 2021), "
    "for the gap between ECG-age and age."
)


NORMAL_ECG_LABEL = Label(
    name=_Column.NORMAL_ECG.value,
    description="Normal ECG.",
)


ABNORMAL_UNSPECIFIED_LABEL = Label(
    name="abnormal_unspecified",
    description=("Abnormal ECG with none of the six listed abnormalities. The release does not name the finding."),
)


# normal_ecg uses the release's column of the same name. abnormal_unspecified means not marked normal,
# with all six diagnosis columns false. These two labels extend the authors' six-label task and each
# appears alone. See the README for the source definitions, counts, and limits of the published columns.
ABNORMALITIES = LabelSet(
    name="abnormality",
    labels=(
        *(Label(name=abnormality.value, description=_DESCRIPTIONS[abnormality]) for abnormality in Abnormality),
        NORMAL_ECG_LABEL,
        ABNORMAL_UNSPECIFIED_LABEL,
    ),
    multi_label=True,
)


"""The six abnormalities in the authors' output order, then ``normal_ecg`` and ``abnormal_unspecified``.

Every abnormality task shares it. A target holds every abnormality that the exam is labelled with, only
``normal_ecg`` for a normal exam, or only ``abnormal_unspecified`` for an exam that is neither. No target is
empty.
"""
