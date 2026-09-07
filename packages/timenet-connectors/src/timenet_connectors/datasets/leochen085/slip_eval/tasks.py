"""The classification task one window answers, and the annotations it references.

This module reads no file and walks nothing. It turns a window's facts into a task and into the
annotations that task points at, and it owns the ids both ends are built from: a task's
``target_schema`` has to equal the id of the annotation holding its folder's class set, so one
function builds both.
"""

from __future__ import annotations

import hashlib
from typing import Protocol

from timenet.types import Annotation, ClassificationTask
from timenet_connectors.datasets.leochen085.slip_eval.keys import SlipEvalKey


class Window(Protocol):
    """What this module needs to know about a window, so the import stays one way.

    ``connector.py`` owns the value; this module only reads these fields off it, and never imports
    it. A helper module that needed the concrete type would make the two import each other.
    """

    @property
    def folder(self) -> str:
        """The directory the window came from."""

    @property
    def split(self) -> str:
        """``train`` or ``test``."""

    @property
    def label(self) -> str:
        """The readable class."""

    @property
    def prompt(self) -> str:
        """The folder's own prompt template, verbatim."""

    @property
    def index_of_label(self) -> int | None:
        """The release's own class index, where the folder ships an integer."""


def _label_id(folder: str, label: str) -> str:
    """Give the id of the annotation holding one class of one folder's vocabulary.

    The digest is taken over the label text rather than with the builtin ``hash``, which is salted
    per interpreter and would give two builds of one release two sets of ids.

    Args:
        folder: The folder's directory name.
        label: The readable class.

    Returns:
        The same id in every build, so every task with that class references the one annotation.
    """
    digest = hashlib.blake2b(label.encode("utf-8"), digest_size=8).hexdigest()
    return f"{folder}-label-{digest}"


def _vocabulary_id(folder: str) -> str:
    """Give the id of the annotation holding one folder's closed set of classes.

    A task states this same string as its ``target_schema``, so the two ends are built from one
    function and cannot drift.

    Args:
        folder: The folder's directory name.

    Returns:
        The id, which a task also states as its ``target_schema``.
    """
    return f"vocabulary-{folder}"


def _benchmark_id(folder: str) -> str:
    """Give the id of the annotation naming one folder.

    Args:
        folder: The folder's directory name.

    Returns:
        A stable id, so every task from that folder references the one annotation.
    """
    return f"benchmark-{folder}"


def _split_id(split: str) -> str:
    """Give the id of the annotation naming one split.

    Args:
        split: ``train`` or ``test``.

    Returns:
        A stable id, so every task in that split references the one annotation.
    """
    return f"split-{split}"


def _task_for(window: Window, record_id: str) -> ClassificationTask:
    """Build the classification task one window answers.

    The folder and the split are carried by the task and not by the record. A record built from the
    three PPG folders belongs to all three, and can be in ``train`` for one diagnosis and ``test``
    for another, so neither fact is a property of the record.

    Args:
        window: The window's facts.
        record_id: The record the window became, or was merged into.

    Returns:
        The task, whose answer is a registered annotation rather than an inline copy.
    """
    # A streamed task never reaches Record.task_ids, so nothing resolves its id and it states none.
    return ClassificationTask(
        prompt=window.prompt,
        target_schema=_vocabulary_id(window.folder),
        target_annotation_ids=(_label_id(window.folder, window.label),),
        input_annotation_ids=(_benchmark_id(window.folder), _split_id(window.split)),
        record_ids=(record_id,),
    )


def _vocabularies(found: dict[str, dict[str, int | None]]) -> list[Annotation]:
    """Give one annotation per folder, holding that folder's whole closed set of classes.

    The order is the release's own, for the folders whose ``label`` column is an integer: a class
    sits at the position that column gives it, so the index the release states survives as the
    position in this list. Where the column is a string the release states no order, and the classes
    are sorted so two builds agree.

    Args:
        found: The classes seen per folder, each mapped to the index the release gave it or ``None``.

    Returns:
        One annotation per folder, its id the ``target_schema`` its tasks state.
    """
    annotations = []
    for folder, labels in sorted(found.items()):
        indexed = all(index is not None for index in labels.values())
        order = sorted(labels, key=lambda k: labels[k] or 0) if indexed else sorted(labels)
        annotations.append(Annotation(key=SlipEvalKey.VOCABULARY, value=order, id=_vocabulary_id(folder)))
    return annotations
