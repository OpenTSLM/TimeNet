"""The closed set of labels that a classification target takes."""

from collections.abc import Mapping, Sequence
from enum import StrEnum, unique
from typing import Any, Self

from pydantic import model_validator

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


class LabelSet(TimeFModel):
    """The closed set of labels that a classification target takes, and how many it holds.

    Tasks that ask the same kind of question share one label set. The writer stores each set once,
    and the reader gives every task that uses it the same object::

        TRUE_OR_FALSE = LabelSet(
            name="true_or_false",
            labels=("T", "F"),
            label_descriptions={"T": "True.", "F": "False."},
        )
        ClassificationTask(inputs=(record,), targets=("T",), label_set=TRUE_OR_FALSE)

    Within a dataset, a name stands for one set. Two different sets with the same name are an error.

    A label set cannot change after construction, so tasks can share one object safely.
    """

    name: NonEmptyString
    """Identifies the set within a dataset, such as ``true_or_false``."""
    labels: tuple[NonEmptyString, ...]
    """Every allowed label, in a stable order."""
    label_descriptions: tuple[NonEmptyString | None, ...] = ()
    """The meaning of each label, in label order, with ``None`` where a label has none. The
    constructor also takes a mapping from label to meaning, such as ``{"T": "True."}``."""
    mode: LabelMode = LabelMode.SINGLE
    """Whether a target holds one label of the set or every label that applies."""

    @model_validator(mode="before")
    @classmethod
    def _descriptions_in_label_order(cls, data: Any) -> Any:
        # A mapping from label to meaning becomes one entry per label, in label order.
        if not isinstance(data, dict):
            return data
        labels = tuple(data.get("labels") or ())
        descriptions = data.get("label_descriptions")
        if descriptions is None or (isinstance(descriptions, Sequence) and not descriptions):
            return {**data, "label_descriptions": (None,) * len(labels)}
        if isinstance(descriptions, Mapping):
            unknown = sorted(set(descriptions) - set(labels))
            if unknown:
                raise ValueError(f"label set {data.get('name')!r} describes labels it does not list: {unknown}")
            return {**data, "label_descriptions": tuple(descriptions.get(label) for label in labels)}
        return data

    @model_validator(mode="after")
    def _valid_set(self) -> Self:
        if not self.labels:
            raise ValueError(f"label set {self.name!r} must list at least one label")
        if len(set(self.labels)) != len(self.labels):
            raise ValueError(f"label set {self.name!r} lists a label more than once")
        if len(self.label_descriptions) != len(self.labels):
            raise ValueError(
                f"label set {self.name!r} has {len(self.label_descriptions)} descriptions for {len(self.labels)} labels"
            )
        return self

    def description(self, label: str) -> str | None:
        """Return the meaning of one label.

        Returns:
            The description, or ``None`` if the label has none.

        Raises:
            KeyError: If the set does not list the label.
        """
        if label not in self.labels:
            raise KeyError(f"label set {self.name!r} does not list {label!r}")
        return self.label_descriptions[self.labels.index(label)]

    def check_targets(self, targets: Sequence[str], task: str) -> None:
        """Make sure that a classification target fits this set and its mode.

        Args:
            targets: The target labels, in the order the task stores them.
            task: The task, as error messages name it.

        Raises:
            TimeFValidationError: If a label is not in the set, a label repeats, or a single-label
                target does not hold exactly one label.
        """
        unknown = sorted({label for label in targets if label not in self.labels})
        if unknown:
            raise TimeFValidationError(f"{task} has labels that label set {self.name!r} does not list: {unknown}")
        if self.mode is LabelMode.SINGLE and len(targets) != 1:
            raise TimeFValidationError(
                f"{task} has {len(targets)} labels, but label set {self.name!r} takes exactly one"
            )
        if len(set(targets)) != len(targets):
            raise TimeFValidationError(f"{task} repeats a label of label set {self.name!r}")
