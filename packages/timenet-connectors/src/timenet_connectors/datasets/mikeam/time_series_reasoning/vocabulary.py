"""Time Series Reasoning release files, categories, annotation keys and signal contracts."""

from enum import StrEnum, unique
from typing import TypeVar

from timenet.dataset.axis import OrdinalAxis
from timenet.types import Split, TimeSeriesSpec


CategoryT = TypeVar("CategoryT", bound=StrEnum)


@unique
class File(StrEnum):
    """A file the connector reads, named by its path in the Hub repository.

    All six are JSON Lines, ``TS_Dataset_MCQ/*.json`` included. The release also holds
    ``TS_Dataset_MCQ/all.json``, which no config of the authors loads, and ``*_llava.json`` files for
    their image model. The connector reads neither.
    """

    SERIES = "TS_Dataset.jsonl"
    DESCRIPTION_TRAIN = "TS_Dataset_MCQ/train.json"
    DESCRIPTION_VAL = "TS_Dataset_MCQ/val.json"
    DESCRIPTION_TEST = "TS_Dataset_MCQ/test.json"
    QUESTIONS = "MCQ_1_TS.jsonl"
    COUNTERFACTUALS = "MCQ_2_TS.jsonl"

    @property
    def part(self) -> str:
        """The file's part name, the key the Hub base downloads it under."""
        return self.name.lower()


# The authors name the description splits by file. The question files hold no split: the authors' configs
# read train, val and test files for them that the release does not include.
SPLITS = {
    File.DESCRIPTION_TRAIN: Split.TRAIN,
    File.DESCRIPTION_VAL: Split.VALIDATION,
    File.DESCRIPTION_TEST: Split.TEST,
}


DESCRIPTION_FILES = tuple(SPLITS)


@unique
class AnnotationKey(StrEnum):
    """The key of an annotation the connector writes. A series' texts keep their published field names."""

    DESCRIPTION = "description"
    DESCRIPTION_SHORT = "description_short"
    DESCRIPTION_TINY = "description_tiny"
    CHARACTERISTICS = "characteristics"
    GENERATOR = "generator"
    """The authors' Python code for the Record's series: ``generator``, or ``new_generator`` for a changed
    series."""
    METADATA = "metadata"
    """The series' published metadata as the authors' prompt prints it: ``str()`` of the dict, in
    published key order."""
    COUNTERFACTUAL = "counterfactual"
    """A changed series' change, as a "What if ...?" question."""
    SOURCE_REPOSITORY = "source_repository"
    SOURCE_REVISION = "source_revision"


SERIES_TEXTS = (
    AnnotationKey.DESCRIPTION,
    AnnotationKey.DESCRIPTION_SHORT,
    AnnotationKey.DESCRIPTION_TINY,
    AnnotationKey.CHARACTERISTICS,
    AnnotationKey.GENERATOR,
)


"""The texts every row repeats for its series, each under its published field name."""


@unique
class TaskMeta(StrEnum):
    """A key of the task metadata. Metadata is provenance, never model input."""

    FILE = "file"
    LINE = "line"
    """The task's line in its file, counted from 1."""
    SCENARIO_CATEGORY = "scenario_category"
    ANSWER_INDEX = "answer_index"
    """A question's published position of the right option."""
    ANSWER_TEXT = "answer_text"
    """A question's right option, the text its target letter stands for."""
    CATEGORY = "category"
    SOURCE_ISSUES = "source_issues"


@unique
class SourceIssue(StrEnum):
    """A problem in a published question row, which the connector keeps as published."""

    BLANK_QUESTION = "blank_question"
    BLANK_OPTION = "blank_option"
    REPEATED_OPTION = "repeated_option"
    """Two options have the same text."""
    ANSWER_TEXT_REPEATED = "answer_text_repeated"
    """The answer's text is also a wrong option's, so only ``answer_index`` tells which one the authors
    count as right."""


@unique
class ScenarioCategory(StrEnum):
    """A description row's published ``scenario_category``: one of the release's 10 domains.

    The train and validation files hold the first 7, and the test file holds the last 3, which the
    authors hold out (the paper's Appendix B.1).
    """

    SALES_AND_MARKET_TRENDS = "Sales and Market Trends"
    ENVIRONMENTAL_AND_CLIMATE_DATA = "Environmental and Climate Data"
    ENERGY_AND_RESOURCE_CONSUMPTION = "Energy and Resource Consumption"
    RECREATIONAL_AND_ENTERTAINMENT_TRENDS = "Recreational and Entertainment Trends"
    TECHNOLOGICAL_AND_DIGITAL_TRENDS = "Technological and Digital Trends"
    TRANSPORT_AND_TRAFFIC_TRENDS = "Transport and Traffic Trends"
    WILDLIFE_AND_NATURE_OBSERVATION = "Wildlife and Nature Observation"
    HEALTH_AND_MEDICAL_DATA = "Health and Medical Data"
    AGRICULTURAL_AND_FOOD_PRODUCTION = "Agricultural and Food Production"
    EDUCATIONAL_AND_PUBLIC_SERVICES = "Educational and Public Services"


@unique
class QuestionCategory(StrEnum):
    """A two-series question's published ``category``: what the question checks."""

    DIFFERENCE = "difference"
    SIMILARITY = "similarity"


@unique
class ValueType(StrEnum):
    """The dtype a series is stored as, chosen so no published value changes.

    ``int32``, TimeF's widest integer dtype, for a series written in integers only; ``float64`` otherwise.
    """

    INT32 = "int32"
    FLOAT64 = "float64"

    @property
    def spec(self) -> TimeSeriesSpec:
        """The spec of a series of this type. One ``spec_type`` has one dtype, so each has its own."""
        return _SPECS[self]


_SPECS = {
    ValueType.FLOAT64: TimeSeriesSpec(
        spec_type="tsr_series", name="Synthetic series", unit_value=None, dtype="float64"
    ),
    ValueType.INT32: TimeSeriesSpec(
        spec_type="tsr_integer_series", name="Synthetic integer series", unit_value=None, dtype="int32"
    ),
}


AXIS = OrdinalAxis()


SERIES_NAME = "series"


"""The name of every Record's one Source and its one Signal."""


# The authors' loader, MultimodalMCQDataset.__getitem__ in behavioral-data/TSandLanguage at d4e748cb
# (src/models/tasks/multimodal.py), with format_abc_mcq: the config's context_prefix, then the row's
# context_columns joined by newlines, then one line per option, "A) ...". The label is the right letter.
DESCRIPTION_PREFIX = "What description best relates to this time series?"


"""The ``context_prefix`` of ``ts2desc_mcq.yaml``, whose ``context_columns`` is empty."""


QUESTION_PREFIX = "Answer the following question about the time series"


"""The ``context_prefix`` of ``qa_mcq.yaml`` and ``qa_counterfactual.yaml``, whose ``context_columns`` are
``description_tiny``, ``metadata`` and ``question``."""


_OPTION_COUNT = 4


"""Every question offers four options; the authors' configs set ``num_classes: 4``."""


LETTERS = "ABCD"


@unique
class SeriesField(StrEnum):
    """The numeric field selected when rereading a source row."""

    ORIGINAL = "series"
    COUNTERFACTUAL = "new_series"
