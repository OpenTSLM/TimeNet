"""The allowed labels for a classification target."""

from collections.abc import Iterator, Sequence
from typing import Self

from pydantic import StrictBool, model_validator

from timenet.errors import TimeFValidationError
from timenet.types._model import TimeFModel
from timenet.types.wire import NonEmptyString


class Label(TimeFModel):
    """One label in a label set, such as ``awake`` in a sleep-stage task."""

    name: NonEmptyString
    """The text that a target holds for this label."""
    description: NonEmptyString | None = None
    """What the label means, such as ``The subject is awake.``."""


class LabelSet(TimeFModel):
    """The allowed labels for a classification target.

    A target must contain one label unless ``multi_label=True``.

    Tasks that ask the same kind of question share one label set::

        SLEEP_STAGE = LabelSet(
            name="sleep_stage",
            labels=(
                Label(name="awake", description="The subject is awake."),
                Label(name="non_rem_sleep"),
                Label(name="rem_sleep"),
            ),
        )
        ClassificationTask(inputs=(record,), targets=("awake",), label_set=SLEEP_STAGE)

    Each name identifies one label set in a dataset.
    """

    name: NonEmptyString
    """The name of the set within a dataset, such as ``sleep_stage``."""
    labels: tuple[Label, ...]
    """The allowed labels in a fixed order."""
    multi_label: StrictBool = False
    """If ``False``, a target holds exactly one label from the set.

    If ``True``, a target holds all labels from the set that apply.
    An empty target means that no label applies."""

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
            raise TimeFValidationError(f"labels not in label set {self.name!r}: {unknown}")
        if not self.multi_label and len(targets) != 1:
            raise TimeFValidationError(f"{len(targets)} labels, but label set {self.name!r} takes exactly one")
        repeated = sorted({label for label in targets if targets.count(label) > 1})
        if repeated:
            raise TimeFValidationError(f"repeated labels of label set {self.name!r}: {repeated}")


class LabelSetRegistry:
    """The label sets of a dataset, with one set per name.

    The sets stay in the order in which tasks first added them.
    """

    def __init__(self) -> None:
        """Start with no label sets."""
        self._by_name: dict[str, LabelSet] = {}

    def register_or_reuse_matching_label_set(self, label_set: LabelSet) -> LabelSet:
        """Store a label set by name.

        If the registry already holds an equal set with this name, return that set.

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
        """Return a separate registry with the same label sets.

        The dataset uses the copy to check a batch before adding its tasks.
        If a task fails, the dataset discards the copy and keeps its original registry.

        Returns:
            The copy.
        """
        copy = LabelSetRegistry()
        copy._by_name = dict(self._by_name)
        return copy

    def __iter__(self) -> Iterator[LabelSet]:
        """Iterate over the sets in the order in which tasks first added them.

        Returns:
            The iterator.
        """
        return iter(self._by_name.values())
