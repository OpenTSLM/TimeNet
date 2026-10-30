"""Recursive data sources that group signals into a recording hierarchy."""

from collections import Counter
from collections.abc import Iterable, Iterator
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from timenet.dataset.span_validation import check_span_within_window
from timenet.dataset.time_series import Signal
from timenet.errors import TimeFValidationError
from timenet.json import JsonMapping, validate_json_mapping
from timenet.types import Annotation, SupportsAnnotate


if TYPE_CHECKING:
    from timenet.dataset.selection import SignalSelection


def check_sibling_names(sources: Iterable["Source"], owner: str) -> None:
    """Reject sibling Sources that share a name.

    Sibling Sources are the root Sources of one Record, or the child Sources of one Source. Their
    names are distinct, so an import finds the copy of a Source by its name.

    Args:
        sources: The sibling Sources.
        owner: The Record or Source that holds them, as the error message names it.

    Raises:
        TimeFValidationError: If two of the Sources have the same name.
    """
    shared = sorted(name for name, count in Counter(source.name for source in sources).items() if count > 1)
    if shared:
        raise TimeFValidationError(
            f"{owner} holds more than one Source named {shared[0]!r}; sibling Sources must have distinct names"
        )


@dataclass(kw_only=True)
class Source(SupportsAnnotate):
    """A device, sensor, or subsystem that contains child sources and signals."""

    name: str
    """Human-readable source name."""
    sources: tuple["Source", ...] = ()
    """Direct child sources, each with a distinct name. Their order has no semantic meaning."""
    signals: tuple[Signal, ...] = ()
    """Signals produced directly by this source."""
    annotations: tuple[Annotation, ...] = ()
    """Annotations attached to this source."""
    metadata: JsonMapping = field(default_factory=dict)
    """Optional JSON-compatible source metadata."""

    def __post_init__(self) -> None:
        """Validate the source's name, the names of its child sources, and the spans of its annotations.

        Raises:
            TimeFValidationError: If the source has no name, or two of its child sources have the same
                name.
        """
        if not self.name:
            raise TimeFValidationError("Source.name must be non-empty")
        check_sibling_names(self.sources, f"source {self.name!r}")
        self.metadata = validate_json_mapping(self.metadata)

        for annotation in self.annotations:
            if annotation.span is not None:
                check_span_within_window(
                    f"annotation {annotation.key!r}",
                    annotation.span,
                    tuple(self.walk_signals()),
                    f"source {self.name!r}",
                )

    def walk_sources(self) -> Iterator["Source"]:
        """Yield this source and all descendants in deterministic depth-first order.

        Yields:
            This source followed by its descendants, sorted by name at each level.

        Raises:
            TimeFValidationError: If the graph contains a cycle or one source has two parents.
        """  # noqa: DOC502 - raised by the nested traversal helper
        seen: set[int] = set()
        active: set[int] = set()

        def walk(source: Source) -> Iterator[Source]:
            identity = id(source)
            if identity in active:
                raise TimeFValidationError(f"source hierarchy contains a cycle at source {source.name!r}")
            if identity in seen:
                raise TimeFValidationError(f"source {source.name!r} is attached to more than one parent")
            seen.add(identity)
            active.add(identity)
            yield source
            for child in sorted(source.sources, key=lambda item: item.name):
                yield from walk(child)
            active.remove(identity)

        yield from walk(self)

    def walk_signals(self) -> Iterator[Signal]:
        """Yield every descendant signal in deterministic source and signal order.

        Yields:
            Signals from this source and every descendant source.
        """
        for source in self.walk_sources():
            yield from sorted(source.signals, key=lambda signal: signal.name)

    def annotate(self, annotation: Annotation) -> Annotation:
        """Attach one annotation and return it.

        Returns:
            The attached annotation.
        """
        if annotation.span is not None:
            signals = tuple(self.walk_signals())
            check_span_within_window(
                f"annotation {annotation.key!r}", annotation.span, signals, f"source {self.name!r}"
            )
        attached = annotation._new_occurrence()
        self.annotations = (*self.annotations, attached)
        return attached

    def select(
        self,
        *,
        signals: tuple[Signal, ...] | None = None,
        signal_names: tuple[str, ...] | None = None,
    ) -> "SignalSelection":
        """Select descendant signals by object or by human-readable name.

        Returns:
            A selection that can be annotated as one declarative operation.
        """
        from timenet.dataset.selection import SignalSelection  # noqa: PLC0415

        return SignalSelection.from_source(self, signals=signals, signal_names=signal_names)
