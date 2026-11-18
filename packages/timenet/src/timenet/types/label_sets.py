"""The allowed labels for a classification target."""

from collections.abc import Hashable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import TypeVar

from timenet.errors import TimeFValidationError
from timenet.types._model import TimeFModel
from timenet.types.annotations import Annotation, SupportsAnnotate
from timenet.types.wire import NonEmptyString


class Label(TimeFModel):
    """One label in a label set, such as ``awake`` in a sleep-stage task."""

    name: NonEmptyString
    """The text that a target holds for this label."""
    description: NonEmptyString | None = None
    """What the label means, such as ``The subject is awake.``."""


_Item = TypeVar("_Item")


def _as_tuple(value: object, item_type: type[_Item]) -> tuple[_Item, ...]:
    """Return the items of a label set field as a tuple.

    Returns:
        The items. :meth:`LabelSet._validate` checks their type.

    Raises:
        TimeFValidationError: If the value is one item, a string, a mapping, or not iterable.
    """
    if isinstance(value, (item_type, str, Mapping)) or not isinstance(value, Iterable):
        raise TimeFValidationError(
            f"a label set takes a list of {item_type.__name__} items, got {type(value).__name__} {value!r}"
        )
    return tuple(value)


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

    A target must contain one label unless ``multi_label=True``.

    Tasks that ask the same kind of question share one label set::

        SLEEP_STAGE = LabelSet(
            [Label(name="awake", description="The subject is awake."), "non_rem_sleep", "rem_sleep"],
            annotations=(Annotation(key="name", value="sleep_stage"),),
        )
        ClassificationTask(inputs=(record,), targets=("awake",), label_set=SLEEP_STAGE)

    A plain string is a label without a description, so ``LabelSet(["yes", "no"])`` is enough when no label
    needs one.

    A label set has no name. Two sets with the same labels (in any order, with the same descriptions)
    and the same ``multi_label`` are the same set, so ``==`` is true for them. A dataset stores them once,
    with the label order of the first of them that it writes. Annotations are not part of that identity.
    """

    labels: tuple[Label, ...]
    """The allowed labels in order. A dataset keeps the order of the first equal set that it writes."""
    multi_label: bool = False
    """If ``False``, a target holds exactly one label from the set.

    If ``True``, a target holds all labels from the set that apply.
    An empty target means that no label applies."""
    annotations: tuple[Annotation, ...] = field(default=(), compare=False)
    """Annotations that describe the set, such as ``Annotation(key="name", value="sleep_stage")``.

    An annotation cannot have a span. The set holds its own occurrence of each annotation.
    When a dataset holds several equal sets, it stores each distinct annotation occurrence of them once."""

    def __init__(
        self,
        labels: Iterable[Label | str],
        *,
        multi_label: bool = False,
        annotations: Iterable[Annotation] = (),
    ) -> None:
        """Build a label set, make sure that it is valid, and attach the annotations.

        Args:
            labels: The allowed labels in order. A plain string is a label without a description.
            multi_label: Whether a target can hold more than one label.
            annotations: Annotations that describe the set.

        Raises:
            TimeFValidationError: If the set is not valid (see :meth:`_validate`).
        """  # ruff: ignore[docstring-extraneous-exception] - _validate raises it
        self.labels = tuple(_as_label(label) for label in _as_tuple(labels, Label))  # ty: ignore[invalid-assignment]
        self.multi_label = multi_label
        self.annotations = _as_tuple(annotations, Annotation)
        self._validate()
        self.annotations = tuple(annotation._new_occurrence() for annotation in self.annotations)

    def _validate(self) -> None:
        """Make sure that the set is valid. The writer calls this again, because a set can change after it is built.

        Raises:
            TimeFValidationError: If a label is not a Label, the set has no label or repeats a name,
                ``multi_label`` is not a bool, or an annotation is not an Annotation or has a span.
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
        """Return whether the sets have the same labels, in any order, and the same ``multi_label``."""
        if not isinstance(other, LabelSet):
            return NotImplemented
        return self._content_key() == other._content_key()

    __hash__ = None

    def _content_key(self) -> tuple[Hashable, ...]:
        """Return the key that makes two sets equal.

        The key ignores label order and annotations.

        Returns:
            The hashable key.
        """
        return (self.multi_label, frozenset((label.name, label.description) for label in self.labels))

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
