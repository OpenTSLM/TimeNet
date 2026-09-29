"""Declarative selections over a source hierarchy."""

from dataclasses import dataclass

from timenet.dataset.source import Source
from timenet.dataset.time_series import Signal
from timenet.errors import TimeFValidationError
from timenet.types import Annotation


@dataclass(frozen=True)
class SignalSelection:
    """A validated set of signals selected from one source subtree."""

    source: Source
    """Source whose descendant Signals can be selected."""
    signals: tuple[Signal, ...]
    """Selected descendant signals in caller order."""

    @classmethod
    def from_source(
        cls,
        source: Source,
        *,
        signals: tuple[Signal, ...] | None,
        signal_names: tuple[str, ...] | None,
    ) -> "SignalSelection":
        """Resolve an object or name selection against a source subtree.

        Returns:
            The validated selection.

        Raises:
            TimeFValidationError: If selectors are missing, mixed, duplicated, ambiguous, or unknown.
        """
        if (signals is None) == (signal_names is None):
            raise TimeFValidationError("select() requires exactly one of signals= or signal_names=")
        available = tuple(source.walk_signals())
        by_id = {signal.id: signal for signal in available}
        if signals is not None:
            selected = tuple(signals)
            unknown = [signal.id for signal in selected if by_id.get(signal.id) is not signal]
            if unknown:
                raise TimeFValidationError(f"signals {unknown} do not belong to source {source.id!r}")
        else:
            names = tuple(signal_names or ())
            matches: dict[str, list[Signal]] = {}
            for signal in available:
                matches.setdefault(signal.name, []).append(signal)
            ambiguous = [name for name in names if len(matches.get(name, ())) > 1]
            unknown = [name for name in names if name not in matches]
            if ambiguous:
                raise TimeFValidationError(
                    f"signal_names {ambiguous} are ambiguous below source {source.id!r}; select objects instead"
                )
            if unknown:
                raise TimeFValidationError(f"signal_names {unknown} do not exist below source {source.id!r}")
            selected = tuple(matches[name][0] for name in names)
        ids = [signal.id for signal in selected]
        if not ids:
            raise TimeFValidationError("select() requires at least one signal")
        if len(ids) != len(set(ids)):
            raise TimeFValidationError("select() does not accept duplicate signals")
        return cls(source=source, signals=selected)

    def annotate(self, annotation: Annotation) -> tuple[Annotation, ...]:
        """Attach one annotation content item to the selected signals.

        Timed and static annotations create one occurrence per selected Signal.
        Each occurrence shares the original content.
        All selected Signals are validated before any annotation is attached.

        Returns:
            The attached occurrences in selection order.
        """
        for signal in self.signals:
            signal._validate_annotation(annotation)
        return tuple(signal._attach_annotation(annotation) for signal in self.signals)
