"""The allowed labels for a classification target."""

from collections import Counter
from collections.abc import Hashable, Iterable, Sequence
from dataclasses import dataclass
import json

from timenet.errors import TimeFValidationError
from timenet.types._fields import as_tuple
from timenet.types._model import TimeFModel
from timenet.types.annotations import Annotation, SupportsAnnotate
from timenet.types.wire import NonEmptyString


class Label(TimeFModel):
    """One label in a label set, such as ``awake`` in a sleep-stage task."""

    name: NonEmptyString
    """The text that a target holds for this label."""
    description: NonEmptyString | None = None
    """What the label means, such as ``The subject is awake.``."""


def _as_label(item: object) -> object:
    """Turn a plain string into a label without a description. Leave every other item as it is.

    Returns:
        The label, or the item. :meth:`LabelSet._validate` checks its type.

    Raises:
        TimeFValidationError: If the string is empty.
    """
    if not isinstance(item, str):
        return item
    if not item:
        raise TimeFValidationError("a label set cannot have an empty label")
    return Label(name=item)


@dataclass(init=False, eq=False)
class LabelSet(SupportsAnnotate):
    """The allowed labels for a classification target.

    A target must contain one label unless ``multi_label=True``. By default the label order does not matter.
    Use ``ordered=True`` for ordinal labels or for options whose order matters.

    Tasks that ask the same kind of question share one label set::

        SLEEP_STAGE = LabelSet(
            [Label(name="awake", description="The subject is awake."), "non_rem_sleep", "rem_sleep"],
            annotations=(Annotation(key="name", value="sleep_stage"),),
        )
        ClassificationTask(inputs=(record,), targets=("awake",), label_set=SLEEP_STAGE)

    A plain string is a label without a description, so ``LabelSet(["yes", "no"])`` is enough when no label
    needs one.

    A label set has no name. Two sets are the same set if they have the same labels (with the same
    descriptions), the same ``multi_label`` and the same ``ordered``. They must also have the same annotations
    (with the same values). Then ``==`` is true for them. If ``ordered`` is ``False``, the labels can be in any
    order. If ``ordered`` is ``True``, the labels must be in the same order. The annotations can be in any order.
    A dataset stores equal sets once, with the label order and the annotation order of the first of them that
    it writes. Two sets with the same labels and different annotations are two sets::

        LabelSet(["yes", "no"], annotations=(Annotation(key="name", value="afib_present"),))
        LabelSet(["yes", "no"], annotations=(Annotation(key="name", value="signal_noisy"),))
    """

    labels: tuple[Label, ...]
    """The allowed labels in order. For an unordered set, a dataset keeps the order of the first equal set that
    it writes."""
    multi_label: bool = False
    """If ``False``, a target holds exactly one label from the set.

    If ``True``, a target holds all labels from the set that apply.
    An empty target means that no label applies."""
    ordered: bool = False
    """If ``True``, the label order is part of the identity of the set. If ``False``, the order does not matter.

    An ordered set needs labels in a fixed order, so pass a list or a tuple, not a Python set."""
    annotations: tuple[Annotation, ...] = ()
    """Annotations that describe the set, such as ``Annotation(key="name", value="sleep_stage")``.

    An annotation cannot have a span. The set holds its own occurrence of each annotation.
    The annotations are part of the identity of the set."""

    def __init__(
        self,
        labels: Iterable[Label | str],
        *,
        multi_label: bool = False,
        ordered: bool = False,
        annotations: Iterable[Annotation] = (),
    ) -> None:
        """Build a label set, make sure that it is valid, and attach the annotations.

        Args:
            labels: The allowed labels in order. A plain string is a label without a description.
            multi_label: Whether a target can hold more than one label.
            ordered: Whether the label order is part of the identity of the set.
            annotations: Annotations that describe the set. They are part of the identity of the set.

        Raises:
            TimeFValidationError: If the set is not valid (see :meth:`_validate`).
        """
        if ordered and isinstance(labels, (set, frozenset)):
            raise TimeFValidationError("an ordered label set needs labels in a fixed order, got a set")
        self.labels = tuple(_as_label(label) for label in as_tuple(labels, item_type=Label))  # ty: ignore[invalid-assignment]
        self.multi_label = multi_label
        self.ordered = ordered
        self.annotations = as_tuple(annotations, item_type=Annotation)
        self._validate()
        self.annotations = tuple(annotation._new_occurrence() for annotation in self.annotations)

    def _validate(self) -> None:
        """Make sure that the set is valid. The writer calls this again, because a set can change after it is built.

        Raises:
            TimeFValidationError: If a label is not a Label, the set has no label or repeats a name,
                ``multi_label`` or ``ordered`` is not a bool, or an annotation is not an Annotation or has a span.
        """
        for label in self.labels:
            if not isinstance(label, Label):
                raise TimeFValidationError(f"a label set takes Label objects, got {type(label).__name__}")
        if not self.labels:
            raise TimeFValidationError("a label set must list at least one label")
        if len(set(self.names)) != len(self.labels):
            raise TimeFValidationError(f"{self} lists a label more than once")
        if not isinstance(self.multi_label, bool):
            raise TimeFValidationError(f"{self} multi_label must be a bool, got {type(self.multi_label).__name__}")
        if not isinstance(self.ordered, bool):
            raise TimeFValidationError(f"{self} ordered must be a bool, got {type(self.ordered).__name__}")
        for annotation in self.annotations:
            self._check_annotation(annotation)

    def _check_annotation(self, annotation: object) -> None:
        """Make sure that an annotation can describe the set.

        Raises:
            TimeFValidationError: If the annotation is not an Annotation or has a span.
        """
        if not isinstance(annotation, Annotation):
            raise TimeFValidationError(f"{self} takes Annotation objects, got {type(annotation).__name__}")
        if annotation.span is not None:
            raise TimeFValidationError(
                f"annotation {annotation.key!r} has a span, but {self} takes annotations without a span"
            )

    def annotate(self, annotation: Annotation) -> Annotation:
        """Attach one annotation occurrence and return it.

        Returns:
            The attached occurrence.

        Raises:
            TimeFValidationError: If the annotation has a span. A label set has no timeline.
        """  # ruff: ignore[docstring-extraneous-exception] - _check_annotation raises it
        self._check_annotation(annotation)
        attached = annotation._new_occurrence()
        self.annotations = (*self.annotations, attached)
        return attached

    @property
    def names(self) -> tuple[str, ...]:
        """The name of each label, in label order."""
        return tuple(label.name for label in self.labels)

    def __str__(self) -> str:
        """Return the labels of the set, for error messages."""
        return f"label set {list(self.names)}"

    def __eq__(self, other: object) -> bool:
        """Return whether the sets have the same content key (see :meth:`_content_key`)."""
        if not isinstance(other, LabelSet):
            return NotImplemented
        return self._content_key() == other._content_key()

    __hash__ = None

    def _content_key(self) -> tuple[Hashable, ...]:
        """Return the key that makes two sets equal.

        The key ignores label order unless the set is ordered. It ignores the order of the annotations.
        It compares every field of each annotation.

        Returns:
            The hashable key.
        """
        labels = ((label.name, label.description) for label in self.labels)
        annotations = Counter(_occurrence_key(annotation) for annotation in self.annotations)
        return (
            self.multi_label,
            self.ordered,
            tuple(labels) if self.ordered else frozenset(labels),
            frozenset(annotations.items()),
        )

    def check_targets(self, targets: Sequence[str]) -> None:
        """Make sure that the target labels follow the rules of this set.

        For a set with the labels ``W``, ``N1`` and ``R``, the rules are:

        - Each label must be in the set. ``("X",)`` fails.
        - If ``multi_label`` is ``False``, the target holds exactly one label. ``("W",)`` passes.
          ``()`` and ``("W", "R")`` fail.
        - No label can repeat. ``("W", "W")`` fails, also when ``multi_label`` is ``True``.

        Args:
            targets: The target labels, in the order that the task stores them.

        Raises:
            TimeFValidationError: If a label is not in the set or a label repeats.
                Also if a single-label target does not hold exactly one label.
        """
        names = set(self.names)
        unknown = sorted({label for label in targets if label not in names})
        if unknown:
            raise TimeFValidationError(f"labels not in {self}: {unknown}")
        if not self.multi_label and len(targets) != 1:
            raise TimeFValidationError(f"{len(targets)} labels, but {self} takes exactly one")
        repeated = sorted({label for label in targets if targets.count(label) > 1})
        if repeated:
            raise TimeFValidationError(f"repeated labels of {self}: {repeated}")


def _occurrence_key(annotation: Annotation) -> tuple[Hashable, ...]:
    """Return a key that is equal for two occurrences if every field of them is equal.

    Returns:
        The hashable key.
    """
    confidence = annotation.confidence
    return (
        annotation._content_key,
        annotation.source,
        float.hex(confidence) if isinstance(confidence, float) else confidence,
        json.dumps(annotation.occurrence_metadata, sort_keys=True),
    )
