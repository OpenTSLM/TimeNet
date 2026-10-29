"""The names RCAEval writes into a dataset, so writing and reading them share one spelling."""

from enum import StrEnum, unique


@unique
class AnnotationKey(StrEnum):
    """The key of an annotation the connector builds."""

    SOURCE_URL = "source_url"
    SOURCE_DOI = "source_doi"
    INJECT_TIME = "inject_time"
    """A Record's fault injection time, from ``inject_time.txt``, and an input of each of its tasks."""
    SYSTEM = "system"
    """A Record's microservice system, such as ``Online Boutique``, and an input of each of its tasks."""
    FAULT = "fault"
    """A task's injected fault, such as ``cpu``, by which the authors report their scores."""
    ROOT_CAUSE_LOG = "root_cause_log"
    """A task's ``root_cause.txt``, the root cause log line, which 8 RE3-SS cases hold."""


@unique
class SourceName(StrEnum):
    """The name of a Record's Source: the file its Signals come from."""

    METRICS = "metrics"
    LOGS = "logs"
    TRACES = "traces"
    LOGTS = "logts"
    TRACETS_ERR = "tracets_err"
    TRACETS_LAT = "tracets_lat"


COLUMN_ORDER = "column_order"
"""The Source metadata key that lists its columns in file order, which TimeF doesn't keep on read."""
CLOCKS = "clocks"
"""The metrics Source metadata key that maps each of a file's time columns to the columns it times."""
SOURCE_FILE = "file"
"""The Source metadata key that names the case's file the Source holds, such as ``simple_metrics.csv``."""
