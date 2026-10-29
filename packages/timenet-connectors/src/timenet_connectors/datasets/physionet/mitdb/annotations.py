"""Read a record's reference annotations, ``<record>.atr``, into its beats, episodes, events and comments.

The cardiologists' reference annotations are one ``.atr`` file per record, in WFDB's MIT format, read with
``wfdb.rdann``. Each annotation has a sample, a code, a subtype, a signal number (``chan``), a number
(``num``) and an optional auxiliary text. The release sets ``chan`` and ``num`` to 0 throughout.

A rhythm change and a signal quality change each hold until the next change of their kind, or until the end of
the record, as the authors' ``dbnotes.c`` counts their durations. Before the first change the record states
none; ``dbnotes.c`` doesn't count that time either.

The archive also holds older annotation files: ``102-0.atr``, and the ``.at_`` and ``203.at-`` files, from
before the corrections the authors' introduction lists. The connector reads only ``<record>.atr``.
"""

from collections.abc import Sequence
from dataclasses import dataclass
from itertools import pairwise
from pathlib import Path
from typing import Any

import numpy as np

from timenet.errors import TimeFFormatError
from timenet_connectors.datasets.physionet.mitdb.release import REFERENCE_ANNOTATOR, SIGNALS_PER_RECORD
from timenet_connectors.datasets.physionet.mitdb.vocabulary import (
    EVENT_CODES,
    BeatCode,
    NonBeatCode,
    RhythmLabel,
    SignalQuality,
)


# Sets of the values, since a membership test on the enum itself takes members only before Python 3.12.
_BEAT_CODES = frozenset(BeatCode)
_RHYTHM_LABELS = frozenset(RhythmLabel)

_UNREADABLE = 0xFF
"""The subtype byte that marks every signal unreadable; the release writes it as -1."""
_NOISY_BIT = 0x01
"""Shifted left by the signal number, the bit that marks the signal noisy."""
_UNREADABLE_BIT = 0x10
"""Shifted left by the signal number, the bit that, with the noisy bit, marks the signal unreadable."""


@dataclass(frozen=True)
class Beat:
    """A beat label at its fiducial point."""

    sample: int
    code: BeatCode
    subtype: int
    """The WFDB subtype, which the release sets on some beats and doesn't document."""


@dataclass(frozen=True)
class Event:
    """A non-beat point label, other than a rhythm, signal quality or comment."""

    sample: int
    code: NonBeatCode
    subtype: int


@dataclass(frozen=True)
class Comment:
    """A comment at a point."""

    sample: int
    text: str


@dataclass(frozen=True)
class RhythmEpisode:
    """The half-open samples ``[start, end)`` that one rhythm label covers."""

    start: int
    end: int
    label: RhythmLabel


@dataclass(frozen=True)
class QualityEpisode:
    """The half-open samples ``[start, end)`` that one signal quality change covers."""

    start: int
    end: int
    subtype: int
    """The subtype as written: a bit mask of noisy and unreadable signals, or -1 for all unreadable."""
    qualities: tuple[SignalQuality, ...]
    """Each signal's quality, in signal order."""


@dataclass(frozen=True)
class ReferenceAnnotations:
    """One record's reference annotations, by kind, each in sample order."""

    beats: tuple[Beat, ...]
    rhythm_episodes: tuple[RhythmEpisode, ...]
    quality_episodes: tuple[QualityEpisode, ...]
    events: tuple[Event, ...]
    comments: tuple[Comment, ...]


def decode_quality(subtype: int, context: str) -> tuple[SignalQuality, ...]:
    """Decode a signal quality change's subtype into each signal's quality, as ``dbnotes.c`` does.

    Signal ``i`` is noisy when bit ``0x01 << i`` is set, and unreadable when bit ``0x10 << i`` is set too. The
    byte ``0xff``, written as -1, marks every signal unreadable.

    Args:
        subtype: The subtype as written.
        context: The file and sample, for the error.

    Returns:
        Each signal's quality, in signal order.

    Raises:
        TimeFFormatError: If the subtype sets a bit for a signal the record doesn't have, or marks a signal
            unreadable but not noisy.
    """
    mask = subtype & 0xFF
    if mask == _UNREADABLE:
        return (SignalQuality.UNREADABLE,) * SIGNALS_PER_RECORD
    known = 0
    qualities = []
    for signal in range(SIGNALS_PER_RECORD):
        noisy, unreadable = _NOISY_BIT << signal, _UNREADABLE_BIT << signal
        known |= noisy | unreadable
        if mask & unreadable and not mask & noisy:
            raise TimeFFormatError(f"{context}: subtype {subtype} marks signal {signal} unreadable but not noisy")
        if mask & unreadable:
            qualities.append(SignalQuality.UNREADABLE)
        elif mask & noisy:
            qualities.append(SignalQuality.NOISY)
        else:
            qualities.append(SignalQuality.CLEAN)
    if mask & ~known:
        raise TimeFFormatError(f"{context}: subtype {subtype} names a signal the record doesn't have")
    return tuple(qualities)


def _episode_bounds(starts: Sequence[int], sig_len: int, context: str) -> list[tuple[int, int]]:
    """Give each change the samples up to the next change, or to the end of the record.

    Returns:
        One half-open ``(start, end)`` per change, in order.

    Raises:
        TimeFFormatError: If two changes of one kind share a sample, so one covers nothing.
    """
    bounds = list(pairwise([*starts, sig_len]))
    for start, end in bounds:
        if end <= start:
            raise TimeFFormatError(f"{context}: two changes of one kind at sample {start}")
    return bounds


def read_reference(base: Path, sig_len: int, wfdb: Any) -> ReferenceAnnotations:
    """Read and check one record's reference annotations.

    Args:
        base: The record's path without an extension.
        sig_len: The record's samples per signal, which every annotation must fall within.
        wfdb: The ``wfdb`` module.

    Returns:
        The record's annotations, by kind.

    Raises:
        TimeFFormatError: If an annotation names a signal or a number, falls outside the record, is out of
            order, has a code the release doesn't write, or has auxiliary text or a subtype its code doesn't
            take; or if a rhythm label or a signal quality subtype isn't one the authors decode.
    """
    path = f"{base}.{REFERENCE_ANNOTATOR}"
    ann = wfdb.rdann(str(base), REFERENCE_ANNOTATOR)
    samples = np.asarray(ann.sample, dtype=np.int64)
    if (np.asarray(ann.chan) != 0).any() or (np.asarray(ann.num) != 0).any():
        raise TimeFFormatError(f"{path}: an annotation names a signal or a number; the release sets both to 0")
    if len(samples) and (samples[0] < 0 or samples[-1] >= sig_len or (np.diff(samples) < 0).any()):
        raise TimeFFormatError(f"{path}: annotations must be in order within the record's {sig_len} samples")
    beats, events, comments = [], [], []
    rhythm_starts: list[tuple[int, RhythmLabel]] = []
    quality_starts: list[tuple[int, int, tuple[SignalQuality, ...]]] = []
    for sample, symbol, subtype, aux in zip(
        samples.tolist(), ann.symbol, np.asarray(ann.subtype).tolist(), ann.aux_note, strict=True
    ):
        context = f"{path} at sample {sample}"
        # An auxiliary text is padded with a NUL to an even length; WFDB's C library reads up to it.
        text = aux.rstrip("\x00")
        takes_text = symbol in {NonBeatCode.RHYTHM_CHANGE, NonBeatCode.COMMENT}
        if bool(text) != takes_text:
            raise TimeFFormatError(f"{context}: code {symbol!r} with auxiliary text {aux!r}")
        if takes_text and subtype != 0:
            raise TimeFFormatError(f"{context}: code {symbol!r} with subtype {subtype}")
        if symbol in _BEAT_CODES:
            beats.append(Beat(sample=sample, code=BeatCode(symbol), subtype=subtype))
        elif symbol == NonBeatCode.RHYTHM_CHANGE:
            if text not in _RHYTHM_LABELS:
                raise TimeFFormatError(f"{context}: {text!r} is not a rhythm label the authors name")
            rhythm_starts.append((sample, RhythmLabel(text)))
        elif symbol == NonBeatCode.SIGNAL_QUALITY_CHANGE:
            quality_starts.append((sample, subtype, decode_quality(subtype, context)))
        elif symbol == NonBeatCode.COMMENT:
            comments.append(Comment(sample=sample, text=text))
        elif symbol in EVENT_CODES:
            events.append(Event(sample=sample, code=NonBeatCode(symbol), subtype=subtype))
        else:
            raise TimeFFormatError(f"{context}: code {symbol!r} is not one the release writes")
    rhythm_bounds = _episode_bounds([start for start, _ in rhythm_starts], sig_len, f"{path} rhythm")
    quality_bounds = _episode_bounds([start for start, _, _ in quality_starts], sig_len, f"{path} signal quality")
    return ReferenceAnnotations(
        beats=tuple(beats),
        rhythm_episodes=tuple(
            RhythmEpisode(start=start, end=end, label=label)
            for (start, end), (_, label) in zip(rhythm_bounds, rhythm_starts, strict=True)
        ),
        quality_episodes=tuple(
            QualityEpisode(start=start, end=end, subtype=subtype, qualities=qualities)
            for (start, end), (_, subtype, qualities) in zip(quality_bounds, quality_starts, strict=True)
        ),
        events=tuple(events),
        comments=tuple(comments),
    )
