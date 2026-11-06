"""TSAQA: true-or-false, multiple-choice and puzzle questions about time series, across six analysis tasks."""

from timenet_connectors.datasets.tsaqa.tsaqa_benchmark.connector import (
    CONNECTOR as CONNECTOR,
    TSAQAConnector as TSAQAConnector,
)
from timenet_connectors.datasets.tsaqa.tsaqa_benchmark.issues import without_source_issues as without_source_issues
from timenet_connectors.datasets.tsaqa.tsaqa_benchmark.labels import SourceIssue as SourceIssue
