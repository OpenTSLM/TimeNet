"""Parse OpenSQA JSONL rows without retaining the two-gigabyte release."""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from functools import lru_cache
import json
import math
from pathlib import Path
import re
from typing import Any, cast

from timenet.errors import TimeFFormatError
from timenet_connectors.datasets.bash_lab.opensqa.release import ReleaseFile


_GYROSCOPE = "Gyroscope:"
_ACCELEROMETER = "Accelerometer:"
_SUMMARY = "Summary:"
_EXPECTED_ROLES = ("system", "user", "assistant")
_SAMPLES = 60
_SENSOR_AXES = 3
_LEADING_Q = re.compile(
    r"^\s*(?:[-*]\s*)?(?:\*\*)?Q\s*(\d{1,2})(?:\*\*)?\s*[:.)-]\s*(.*)$",
    re.IGNORECASE,
)
_NUMBERED = re.compile(r"^\s*(?:\*\*)?(\d{1,2})[.)]\s*(.*)$")
_Q_PREFIX = re.compile(r"^(?:\*\*)?Q(?:\*\*)?\s*:\s*(.*)$", re.IGNORECASE)
_ANSWER_PREFIX = re.compile(
    r"^\s*(?:[-*]\s*)?(?:\d{1,2}[.)]\s*)?(?:\*\*)?"
    r"(?:Answer|A)\s*\d{0,2}\s*:(?:\*\*)?\s*",
    re.IGNORECASE,
)


@dataclass(frozen=True, slots=True)
class OpenSqaFile:
    """One local file and the release facts used to validate it."""

    release: ReleaseFile
    path: Path


@dataclass(frozen=True, slots=True)
class OpenSqaSource:
    """The pinned OpenSQA training files downloaded as one source."""

    files: tuple[OpenSqaFile, ...]
    revision: str


@dataclass(frozen=True, slots=True)
class ParsedRow:
    """One source row's sensor arrays and generated language."""

    summary: str
    gyroscope: tuple[tuple[float, float, float], ...]
    accelerometer: tuple[tuple[float, float, float], ...]
    caption: str
    assistant: str


def parse_row(line: bytes, *, path: Path, index: int) -> ParsedRow:
    """Decode and validate one OpenSQA source row.

    Args:
        line: One JSONL line.
        path: Source path used in validation messages.
        index: Zero-based source row.

    Returns:
        The validated sensor arrays and generated text.

    Raises:
        TimeFFormatError: If the row does not match the pinned release shape.
    """
    location = f"{path}:{index + 1}"
    try:
        raw: Any = json.loads(line)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise TimeFFormatError(f"{location}: invalid OpenSQA JSON row") from exc
    if not isinstance(raw, dict):
        raise TimeFFormatError(f"{location}: expected a JSON object")
    messages = raw.get("messages")
    if not isinstance(messages, list) or len(messages) != len(_EXPECTED_ROLES):
        raise TimeFFormatError(f"{location}: messages must contain system, user, and assistant entries")
    if any(
        not isinstance(item, dict) or cast("dict[str, Any]", item).get("role") != role
        for item, role in zip(messages, _EXPECTED_ROLES, strict=True)
    ):
        raise TimeFFormatError(f"{location}: messages must be ordered system, user, assistant")
    user = messages[1].get("content")
    assistant = messages[2].get("content")
    if not isinstance(user, str) or not isinstance(assistant, str):
        raise TimeFFormatError(f"{location}: user and assistant content must be text")
    summary, gyroscope, accelerometer, caption = _parse_user(user, location)
    return ParsedRow(
        summary=summary,
        gyroscope=gyroscope,
        accelerometer=accelerometer,
        caption=caption,
        assistant=assistant,
    )


def iter_rows(source: OpenSqaFile) -> Iterator[tuple[int, int, ParsedRow]]:
    """Yield row index, byte offset, and parsed row from one JSONL file.

    Yields:
        Each row with the byte offset used by lazy signal loaders.
    """
    with source.path.open("rb") as handle:
        index = 0
        while True:
            offset = handle.tell()
            line = handle.readline()
            if not line:
                break
            yield index, offset, parse_row(line, path=source.path, index=index)
            index += 1


@lru_cache(maxsize=1)
def row_at(path: Path, offset: int, index: int) -> ParsedRow:
    """Read one row for its adjacent signal loaders.

    Returns:
        The parsed row at the byte offset.

    Raises:
        TimeFFormatError: If the offset does not identify a row.
    """
    with path.open("rb") as handle:
        handle.seek(offset)
        line = handle.readline()
    if not line:
        raise TimeFFormatError(f"{path}: no OpenSQA row at byte offset {offset}")
    return parse_row(line, path=path, index=index)


def qa_pairs(text: str, *, maximum: int) -> tuple[tuple[str, str], ...]:
    """Recover complete generated QA pairs while dropping truncated questions.

    OpenSQA has no structured QA columns. Its model generations use several
    Markdown conventions and sometimes end after a question. Numbered lines
    count as question starts only when they contain a question mark; this
    avoids treating ordinary numbered answer lists as new questions.

    Args:
        text: Assistant completion from one source row.
        maximum: Expected upper bound: ten for v1 and five for v2.

    Returns:
        Complete question-answer pairs in source order.
    """
    lines = text.splitlines()
    starts = [
        (line_index, question) for line_index, line in enumerate(lines) if (question := _question(line)) is not None
    ]
    pairs: list[tuple[str, str]] = []
    for position, (line_index, raw_question) in enumerate(starts[:maximum]):
        end = starts[position + 1][0] if position + 1 < len(starts) else len(lines)
        body_lines = lines[line_index + 1 : end]
        answer_line = next(
            (item for item, line in enumerate(body_lines) if _ANSWER_PREFIX.match(line)),
            None,
        )
        if answer_line is not None:
            continuation = " ".join(line.strip() for line in body_lines[:answer_line] if line.strip())
            question = f"{raw_question} {continuation}" if continuation else raw_question
            first_answer_line = _ANSWER_PREFIX.sub("", body_lines[answer_line], count=1)
            body_lines = [first_answer_line, *body_lines[answer_line + 1 :]]
        else:
            question = raw_question
        answer = "\n".join(body_lines).strip().strip("*").strip()
        question = question.strip().strip("*").strip()
        if question and answer:
            pairs.append((question, answer))
    return tuple(pairs)


def _parse_user(
    text: str,
    location: str,
) -> tuple[
    str,
    tuple[tuple[float, float, float], ...],
    tuple[tuple[float, float, float], ...],
    str,
]:
    summary_at = text.find(_SUMMARY)
    gyroscope_at = text.find(_GYROSCOPE)
    accelerometer_at = text.find(_ACCELEROMETER, gyroscope_at + len(_GYROSCOPE))
    if not (0 <= summary_at < gyroscope_at < accelerometer_at):
        raise TimeFFormatError(f"{location}: user content lacks ordered Summary, Gyroscope, and Accelerometer fields")
    summary = text[summary_at + len(_SUMMARY) : gyroscope_at].strip()
    gyroscope_text = text[gyroscope_at + len(_GYROSCOPE) : accelerometer_at].strip()
    accelerometer_text = text[accelerometer_at + len(_ACCELEROMETER) :].lstrip()
    try:
        gyroscope_raw: Any = json.loads(gyroscope_text)
        accelerometer_raw, end = json.JSONDecoder().raw_decode(accelerometer_text)
    except json.JSONDecodeError as exc:
        raise TimeFFormatError(f"{location}: invalid embedded sensor JSON") from exc
    caption = accelerometer_text[end:].strip()
    if not summary or not caption:
        raise TimeFFormatError(f"{location}: summary and embedded SensorCaps completion must be non-empty")
    return (
        summary,
        _sensor(gyroscope_raw, name="gyroscope", location=location),
        _sensor(accelerometer_raw, name="accelerometer", location=location),
        caption,
    )


def _sensor(raw: Any, *, name: str, location: str) -> tuple[tuple[float, float, float], ...]:
    if not isinstance(raw, list) or len(raw) != _SAMPLES:
        raise TimeFFormatError(f"{location}: {name} must contain 60 three-axis samples")
    values: list[tuple[float, float, float]] = []
    for sample in raw:
        if not isinstance(sample, list) or len(sample) != _SENSOR_AXES:
            raise TimeFFormatError(f"{location}: {name} must contain 60 three-axis samples")
        if any(isinstance(value, bool) or not isinstance(value, int | float) for value in sample):
            raise TimeFFormatError(f"{location}: {name} samples must be numeric")
        point = (float(sample[0]), float(sample[1]), float(sample[2]))
        if not all(math.isfinite(value) for value in point):
            raise TimeFFormatError(f"{location}: {name} samples must be finite")
        values.append(point)
    return tuple(values)


def _question(line: str) -> str | None:
    stripped = line.strip()
    leading = _LEADING_Q.match(stripped)
    if leading:
        return leading.group(2).strip().strip("*").strip()
    numbered = _NUMBERED.match(stripped)
    if not numbered:
        return None
    body = numbered.group(2).strip()
    prefixed = _Q_PREFIX.match(body)
    if prefixed:
        return prefixed.group(1).strip().strip("*").strip()
    if body.lower().startswith("**q:"):
        return body[4:].strip().strip("*").strip()
    if "?" in body:
        return body.strip().strip("*").strip()
    return None
