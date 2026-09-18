"""Recursive data sources that group signals into a recording hierarchy."""

from collections.abc import Iterator
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from timenet.dataset.time_series import Signal
from timenet.errors import TimeFValidationError
from timenet.types import Annotation, new_id


if TYPE_CHECKING:
    from timenet.dataset.selection import SignalSelection


@dataclass(kw_only=True)
class Source:
    """A device, sensor, or subsystem that contains child sources and signals."""

    name: str
    """Human-readable source name."""
    id: str = field(default_factory=new_id)
    """Stable source identifier."""
    sources: tuple["Source", ...] = ()
    """Direct child sources. Their order has no semantic meaning."""
    signals: tuple[Signal, ...] = ()
    """Signals produced directly by this source."""
    annotations: tuple[Annotation, ...] = ()
    """Annotations attached to this source."""
    metadata: dict[str, object] = field(default_factory=dict)
    """Optional JSON-compatible source metadata."""

    def __post_init__(self) -> None:
        """Validate the source's local identity and direct children.

        Raises:
            TimeFValidationError: If the source has no name or repeats a direct child ID.
        """
        if not self.name:
            raise TimeFValidationError("Source.name must be non-empty")
        source_ids = [source.id for source in self.sources]
        signal_ids = [signal.id for signal in self.signals]
        if len(source_ids) != len(set(source_ids)):
            raise TimeFValidationError(f"source {self.id!r} contains duplicate child source IDs")
        if len(signal_ids) != len(set(signal_ids)):
            raise TimeFValidationError(f"source {self.id!r} contains duplicate signal IDs")

    def walk_sources(self) -> Iterator["Source"]:
        """Yield this source and all descendants in deterministic depth-first order.

        Yields:
            This source followed by its descendants, sorted by name and ID at each level.

        Raises:
            TimeFValidationError: If the graph contains a cycle or one source has two parents.
        """  # noqa: DOC502 - raised by the nested traversal helper
        seen: set[int] = set()
        active: set[int] = set()

        def walk(source: Source) -> Iterator[Source]:
            identity = id(source)
            if identity in active:
                raise TimeFValidationError(f"source hierarchy contains a cycle at source {source.id!r}")
            if identity in seen:
                raise TimeFValidationError(f"source {source.id!r} is attached to more than one parent")
            seen.add(identity)
            active.add(identity)
            yield source
            for child in sorted(source.sources, key=lambda item: (item.name, item.id)):
                yield from walk(child)
            active.remove(identity)

        yield from walk(self)

    def walk_signals(self) -> Iterator[Signal]:
        """Yield every descendant signal in deterministic source and signal order.

        Yields:
            Signals from this source and every descendant source.
        """
        for source in self.walk_sources():
            yield from sorted(source.signals, key=lambda signal: (signal.name, signal.id))

    def annotate(self, annotation: Annotation) -> Annotation:
        """Attach one annotation and return it.

        Returns:
            The attached annotation.
        """
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
