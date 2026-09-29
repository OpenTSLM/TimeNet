"""Semantic input kinds shared by task declarations and Signal specifications."""

from enum import StrEnum, unique


@unique
class InputModality(StrEnum):
    """A kind of input supplied to a task."""

    TEXT = "text"
    TIME_SERIES = "time_series"
    IMAGE = "image"
    AUDIO = "audio"
    NO_INPUT = "no_input"
