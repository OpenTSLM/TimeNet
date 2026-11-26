"""Validated round-three source questions and the released baseline prompt.

Prompt wording adapted from TimeSeriesExam (MIT, Copyright 2024 Auton Lab, CMU).
Optional concept, hint, and example-series augmentations are deliberately not enabled.
"""

from collections.abc import Iterator
from typing import Literal, Self

import numpy as np
from numpy.typing import NDArray
from pydantic import BaseModel, ConfigDict, Field, PositiveInt, TypeAdapter, model_validator


_RESPONSE_INSTRUCTIONS = (
    "{format_hint}. \n"
    "\n"
    "        Here is an example question answer pair to help you understand the format better:\n"
    "\n"
    "        EXAMPLE QUESTION: \n"
    "        \n"
    "        What is the most likely autocorrelation at lag 1 for the given time series?\n"
    "        \n"
    "        Choose From Following Options: \n"
    "        \n"
    "        A) High positive autocorrelation\n"
    "B) No autocorrelation\n"
    "C) Negative autocorrelation\n"
    "Now, answer the question.\n"
    "\n"
    "        EXAMPLE RESPONSE: \n"
    "        \n"
    "        Based on the given time series, the data points appear to fluctuate randomly around the mean with no clear pattern of persistence or trend. This suggests that the time series does not exhibit a strong relationship between consecutive data points.\n"
    "\n"
    "Given the options:\n"
    "\n"
    "A) High positive autocorrelation\n"
    "B) No autocorrelation\n"
    "C) Negative autocorrelation\n"
    "\n"
    "The most likely autocorrelation at lag 1 for the given time series is:\n"
    "\n"
    "B) No autocorrelation\n"
    "        \n"
    "        Now, answer the given question and also explain your thought process: "
)


class ExamQuestion(BaseModel):
    """One immutable source question; numeric options remain numeric until rendering."""

    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)

    id: PositiveInt
    tid: PositiveInt
    question: str
    options: tuple[str | float, ...] = Field(min_length=2)
    answer: str | float
    question_type: Literal["binary", "multiple_choice", "multiple-choice"]
    difficulty: str
    format_hint: str
    relevant_concepts: tuple[str, ...]
    question_hint: str
    category: str
    subcategory: str
    ts: tuple[float, ...] | None = None
    ts1: tuple[float, ...] | None = None
    ts2: tuple[float, ...] | None = None

    @model_validator(mode="after")
    def validate_source(self) -> Self:
        """Require one signal or an ordered pair and one unambiguous reference option."""
        single = self.ts is not None and self.ts1 is None and self.ts2 is None
        paired = self.ts is None and self.ts1 is not None and self.ts2 is not None
        if not (single or paired):
            raise ValueError(f"question {self.id} must contain ts or both ts1 and ts2")
        if self.options.count(self.answer) != 1:
            raise ValueError(f"question {self.id} must have exactly one correct option")
        if any(values is not None and not values for values in (self.ts, self.ts1, self.ts2)):
            raise ValueError(f"question {self.id} contains an empty signal")
        return self

    def render_prompt(self) -> str:
        """Return the upstream baseline query with its complete response-format example."""
        options = "\n".join(f"{chr(65 + index)}) {option}" for index, option in enumerate(self.options))
        question = f"{self.question}\n        \n        Choose From Following Options: \n        \n        {options}\n"
        return question + _RESPONSE_INSTRUCTIONS.format(format_hint=self.format_hint)

    def render_answer(self) -> str:
        """Resolve the typed option before formatting its letter and content, as upstream does."""
        index = self.options.index(self.answer)
        return f"{chr(65 + index)}) {self.answer}"

    def iter_signals(self) -> Iterator[tuple[str, NDArray[np.float64]]]:
        """Yield the published signal order without rounding, normalization, or resampling."""
        if self.ts is not None:
            yield "Time Series", np.asarray(self.ts, dtype=np.float64)
        else:
            yield "Time Series 1", np.asarray(self.ts1, dtype=np.float64)
            yield "Time Series 2", np.asarray(self.ts2, dtype=np.float64)


QUESTIONS = TypeAdapter(tuple[ExamQuestion, ...])
