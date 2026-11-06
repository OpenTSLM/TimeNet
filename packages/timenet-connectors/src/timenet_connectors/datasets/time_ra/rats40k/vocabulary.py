"""Published vocabulary, source contracts, and signal specifications."""

from enum import StrEnum, unique
from pathlib import Path
import re

from timenet.dataset.axis import OrdinalAxis
from timenet.types import Split, TimeSeriesSpec


@unique
class Variant(StrEnum):
    """The release's two files: univariate and multivariate series, each with its own prompt."""

    UNI = "uni"
    MULTI = "multi"

    @property
    def source_file(self) -> str:
        """The file's name in the Hub repository, such as ``RATs-Uni-TSImage_Reason.json``."""
        return f"RATs-{self.value.capitalize()}-TSImage_Reason.json"

    @property
    def label_set_name(self) -> str:
        """The name of this variant's label set, such as ``uni_category``."""
        return f"{self.value}_category"

    @property
    def template_key(self) -> str:
        """The key of the dataset annotation that holds this variant's full evaluation prompt."""
        return f"{self.value}_request_template"


@unique
class SourceSplit(StrEnum):
    """A top-level key of a file: the authors' split."""

    TRAIN = "TSAD_train"
    TEST = "TSAD_test"

    @property
    def timef_split(self) -> Split:
        """The TimeNet split of the key's rows: ``TSAD_train`` is ``train``."""
        return Split(self.value.removeprefix("TSAD_"))


@unique
class SourceField(StrEnum):
    """A field of a published row."""

    OBSERVATION = "Observation"
    DOMAIN = "Source"
    FIGURE_PATH = "FigurePath"
    THOUGHT = "Thought"
    ACTION = "Action"
    ACTION_ID = "ActionID"
    LABEL = "Label"
    RANK = "Rank"
    """Published on test rows. May be null or absent."""


@unique
class SourceLabel(StrEnum):
    """The binary label of a row."""

    NORMAL = "Normal"
    ANOMALY = "Anomaly"


@unique
class MetadataKey(StrEnum):
    """Keys stored in record, source or task metadata."""

    VARIANT = "variant"
    SPLIT = "split"
    ROW_KEY = "key"
    CHANNEL_ORDER = "channel_order"
    TARGET_IN_VOCABULARY = "target_in_vocabulary"
    FIGURE_PATH = "figure_path"
    RANK = "rank"


@unique
class AnnotationKey(StrEnum):
    """Keys of attached annotations."""

    SOURCE_REPOSITORY = "source_repository"
    SOURCE_REVISION = "source_revision"
    SOURCE_DOMAIN = "source_domain"
    LABEL = "label"
    ACTION_ID = "action_id"


UNI_CHANNEL = "series"


"""The name of the one channel of a univariate row."""


PROMPT_DIRECTORY = Path(__file__).parent / "prompts"


# The authors' spelling, "oberservation", included.
DOMAIN_CLAUSE_PATTERN = re.compile(r" The oberservation is from (?:the )?domain of \{our_source\}\.")


OBSERVATION_LINE = "Observation: {our_observation}\n"


ANSWER_FORMAT = "Thought: \\boxed1{{{thought}}}\nAction: \\boxed2{{{action}}}"


"""The two boxed fields that end each prompt, as a ``str.format`` template.

The prompt shows them empty. :func:`authors_answer` fills them with a task's rationale and target.
"""


CATEGORY_PATTERN = re.compile(r"^\d+\. ([^:\n]+):", re.MULTILINE)


"""A category the prompt defines: ``12. Sudden Spike Anomaly: ...``."""


NON_SPACE = re.compile(r"\S")


# The release gives neither units nor a sampling cadence.
SPEC = TimeSeriesSpec(spec_type="rats40k_series", name="RATS40K series", unit_value=None, dtype="float64")


AXIS = OrdinalAxis()


LABEL_DESCRIPTION = "The release's Label, Normal or Anomaly, as published. It can disagree with the target."


ACTION_ID_DESCRIPTION = "The release's ActionID, as published; it doesn't always follow the prompt's numbering."


OUTSIDE_PROMPT_DESCRIPTION = "A published target name that is not one of this variant's prompt categories."
