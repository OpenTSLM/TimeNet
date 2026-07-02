"""TimeF value types and type system: versions, units, enums, specs, annotations, tasks, metadata."""

from timenet.types.annotations import (
    ANNOTATION_BASES,
    Annotation,
    AnnotationDescriptor,
    AnnotationType,
    IntervalAnnotation,
    PointAnnotation,
    StaticAnnotation,
    annotation_type_of,
    value_type_of,
)
from timenet.types.domains import Domain
from timenet.types.licenses import License
from timenet.types.metadata import DatasetMetadata, DatasetSchema
from timenet.types.specs import DataSource, TimeSeriesSpec
from timenet.types.tasks import (
    TASKS,
    CaptioningTask,
    ClassificationTask,
    ForecastingTask,
    LabelingTask,
    QATask,
    ReasoningTask,
    Task,
    TaskType,
)
from timenet.types.units import ureg
from timenet.types.version import Version
from timenet.types.views import View


__all__ = [
    "ANNOTATION_BASES",
    "TASKS",
    "Annotation",
    "AnnotationDescriptor",
    "AnnotationType",
    "CaptioningTask",
    "ClassificationTask",
    "DataSource",
    "DatasetMetadata",
    "DatasetSchema",
    "Domain",
    "ForecastingTask",
    "IntervalAnnotation",
    "LabelingTask",
    "License",
    "PointAnnotation",
    "QATask",
    "ReasoningTask",
    "StaticAnnotation",
    "Task",
    "TaskType",
    "TimeSeriesSpec",
    "Version",
    "View",
    "annotation_type_of",
    "ureg",
    "value_type_of",
]
