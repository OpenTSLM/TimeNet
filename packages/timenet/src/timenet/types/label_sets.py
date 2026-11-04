"""The closed set of labels that a classification target takes."""

from collections.abc import Iterator, Sequence
from enum import StrEnum, unique
from typing import Any, Self

from pydantic import field_validator, model_validator

from timenet.errors import TimeFValidationError
from timenet.types._model import TimeFModel
from timenet.types.wire import NonEmptyString


@unique
class LabelMode(StrEnum):
    """How many labels of its label set a classification target holds."""

    SINGLE = "single"
    """Exactly one label."""
    MULTI = "multi"
    """Every label that applies, and only those. A label left out does not apply."""


class Label(TimeFModel):
    """One label of a label set, such as ``N1`` for a sleep stage."""

    name: NonEmptyString
    """The text that a target holds for this label."""
    description: NonEmptyString | None = None
    """What the label means, such as ``Light sleep, stage 1.``."""


class LabelSet(TimeFModel):
    """The closed set of labels that a classification target takes, and how many it holds.

    Tasks that ask the same kind of question share one label set::

        TRUE_OR_FALSE = LabelSet(
            name="true_or_false",
            labels=(Label(name="T", description="True."), Label(name="F", description="False.")),
        )
        ClassificationTask(inputs=(record,), targets=("T",), label_set=TRUE_OR_FALSE)

    A plain string in ``labels`` is a label without a description, so ``labels=("T", "F")`` also
    works. Within a dataset, a name stands for one set.
    """

    name: NonEmptyString
    """Identifies the set within a dataset, such as ``true_or_false``."""
    labels: tuple[Label, ...]
    """Every allowed label, in a stable order."""
    mode: LabelMode = LabelMode.SINGLE
    """Whether a target holds one label of the set or every label that applies."""

    @field_validator("labels", mode="before")
    @classmethod
    def _plain_names(cls, value: Any) -> Any:
        # A plain string is a label without a description.
        if isinstance(value, Sequence) and not isinstance(value, str):
            return tuple(Label(name=item) if isinstance(item, str) else item for item in value)
        return value

    @model_validator(mode="after")
    def _valid_set(self) -> Self:
        if not self.labels:
            raise ValueError(f"label set {self.name!r} must list at least one label")
        if len(set(self.names)) != len(self.labels):
            raise ValueError(f"label set {self.name!r} lists a label more than once")
        return self

    @property
    def names(self) -> tuple[str, ...]:
        """The name of each label, in label order."""
        return tuple(label.name for label in self.labels)

    def description(self, name: str) -> str | None:
        """Return the meaning of one label.

        Returns:
            The description, or ``None`` if the label has none.

        Raises:
            KeyError: If the set does not list the label.
        """
        for label in self.labels:
            if label.name == name:
                return label.description
        raise KeyError(f"label set {self.name!r} does not list {name!r}")

    def check_targets(self, targets: Sequence[str]) -> None:
        """Make sure that a classification target fits this set and its mode.

        Args:
            targets: The target labels, in the order that the task stores them.

        Raises:
            TimeFValidationError: If a label is not in the set, a label repeats, or a single-label
                target does not hold exactly one label.
        """
        names = set(self.names)
        unknown = sorted({label for label in targets if label not in names})
        if unknown:
            raise TimeFValidationError(f"labels not in label set {self.name!r}: {unknown}")
        if self.mode is LabelMode.SINGLE and len(targets) != 1:
            raise TimeFValidationError(f"{len(targets)} labels, but label set {self.name!r} takes exactly one")
        repeated = sorted({label for label in targets if targets.count(label) > 1})
        if repeated:
            raise TimeFValidationError(f"repeated labels of label set {self.name!r}: {repeated}")


class LabelSetRegistry:
    """The label sets of a dataset by name, so that one name stands for one set.

    The sets keep the order in which tasks first claimed them.
    """

    def __init__(self) -> None:
        """Start with no label sets."""
        self._by_name: dict[str, LabelSet] = {}

    def claim(self, label_set: LabelSet) -> LabelSet:
        """Store a label set under its name, or return the equal set that the name already has.

        Returns:
            The set that the registry holds under the name.

        Raises:
            TimeFValidationError: If a different set already has the same name.
        """
        known = self._by_name.setdefault(label_set.name, label_set)
        if known is not label_set and known != label_set:
            raise TimeFValidationError(
                f"label set {label_set.name!r} differs from another set with that name. "
                "A name stands for one label set in a dataset"
            )
        return known

    def copy(self) -> "LabelSetRegistry":
        """Return a registry with the same sets, which a caller can change without changing this one.

        Returns:
            The copy.
        """
        copy = LabelSetRegistry()
        copy._by_name = dict(self._by_name)
        return copy

    def __iter__(self) -> Iterator[LabelSet]:
        """Iterate over the sets in the order in which tasks first claimed them.

        Returns:
            The iterator.
        """
        return iter(self._by_name.values())

    def __len__(self) -> int:
        """Return the number of sets."""
        return len(self._by_name)
