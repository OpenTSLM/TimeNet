"""TSAQA: true-or-false, multiple-choice and puzzle questions about time series, across six analysis tasks."""

from timenet_connectors.datasets.tsaqa.tsaqa_benchmark.connector import (
    CONNECTOR as CONNECTOR,
    SourceIssue as SourceIssue,
    TSAQAConnector as TSAQAConnector,
    without_source_issues as without_source_issues,
)
