"""Canonical text rendering for declarative modelling tasks."""

from dataclasses import dataclass
import json

from timenet.dataset import IrregularAxis, OrdinalAxis, Record, RegularAxis, Signal, Source
from timenet.types import Annotation, Span, StepInterval, StepPoint, Task, TimeInterval, TimePoint


@dataclass(frozen=True)
class TextTrainingExample:
    """The canonical text prompt and expected output for one task."""

    prompt: str
    """Task instruction and input hierarchy rendered as text."""
    target: str | None
    """Ordered target items rendered as text, or ``None`` when targets are annotation-only."""


class TextPromptRenderer:
    """Render one fixed text representation of a declarative task.

    The renderer deliberately has no formatting options. Training code should see the same text for
    the same task regardless of which caller renders it. Signal values remain lazy; the renderer
    describes their identity, specification, axis, and length without calling ``to_arrow()``.
    """

    def render(self, *, task: Task) -> TextTrainingExample:
        """Render a task without loading signal values.

        Returns:
            The canonical prompt and target text.
        """
        prompt_lines = [f"Task: {task.prompt or ''}"]
        self._append_annotations(prompt_lines, "Task annotations", task.annotations)
        self._append_annotations(prompt_lines, "Input annotations", task.input_annotations)
        prompt_lines.append("Inputs:")
        if task.inputs:
            for position, record in enumerate(task.inputs, start=1):
                prompt_lines.extend(self._render_record(record, prefix=f"{position}. "))
        else:
            prompt_lines.append("  (none)")

        target_lines: list[str] = []
        if task.targets is not None:
            for position, target in enumerate(task.targets, start=1):
                target_lines.extend(self._render_target(target, position=position))
        self._append_annotations(target_lines, "Target annotations", task.target_annotations)
        target = None if task.targets is None and not task.target_annotations else "\n".join(target_lines)
        return TextTrainingExample(prompt="\n".join(prompt_lines), target=target)

    def _render_record(self, record: Record, *, prefix: str = "") -> list[str]:
        lines = [f"  {prefix}Record {record.id}"]
        self._append_annotations(lines, "    Annotations", record.annotations)
        for source in sorted(record.sources, key=lambda item: (item.name, item.id)):
            lines.extend(self._render_source(source, indent="    "))
        return lines

    def _render_source(self, source: Source, *, indent: str) -> list[str]:
        lines = [f"{indent}Source {source.name} [{source.id}]"]
        self._append_annotations(lines, f"{indent}  Annotations", source.annotations)
        for signal in sorted(source.signals, key=lambda item: (item.name, item.id)):
            lines.extend(self._render_signal(signal, indent=f"{indent}  "))
        for child in sorted(source.sources, key=lambda item: (item.name, item.id)):
            lines.extend(self._render_source(child, indent=f"{indent}  "))
        return lines

    def _render_signal(self, signal: Signal, *, indent: str) -> list[str]:
        spec = signal.spec
        lines = [
            f"{indent}Signal {signal.name} [{signal.id}]",
            (
                f"{indent}  Spec {spec.spec_type}: {spec.name}; unit={spec.unit_value}; "
                f"dtype={spec.dtype}; values={signal.n_values}"
            ),
            f"{indent}  Axis {self._render_axis(signal)}",
        ]
        self._append_annotations(lines, f"{indent}  Annotations", signal.annotations)
        return lines

    @staticmethod
    def _render_axis(signal: Signal) -> str:
        axis = signal.time_axis
        if isinstance(axis, RegularAxis):
            return f"regular [{axis.axis_id}]; period_us={axis.period_us}; start_index={axis.start_index}"
        if isinstance(axis, IrregularAxis):
            return f"irregular [{axis.axis_id}]; first_us={axis.first_us}; last_us={axis.last_us}"
        if isinstance(axis, OrdinalAxis):
            return f"ordinal [{axis.axis_id}]"
        raise AssertionError(f"unsupported axis {type(axis).__name__}")

    def _render_target(self, target: object, *, position: int) -> list[str]:
        if isinstance(target, Record):
            lines = [f"{position}. Record target"]
            lines.extend(self._render_record(target))
            return lines
        if isinstance(target, Signal):
            return [f"{position}. Signal target", *self._render_signal(target, indent="  ")]
        if isinstance(target, Span):
            return [f"{position}. {self._render_span(target)}"]
        return [f"{position}. {_json_value(target)}"]

    @staticmethod
    def _append_annotations(lines: list[str], heading: str, annotations: tuple[Annotation, ...]) -> None:
        if not annotations:
            return
        lines.append(f"{heading}:")
        indent = heading[: len(heading) - len(heading.lstrip())]
        lines.extend(f"{indent}  - {_render_annotation(annotation)}" for annotation in annotations)

    @staticmethod
    def _render_span(span: Span) -> str:
        if isinstance(span, TimePoint):
            return f"time point at {_seconds(span.start_us)} s{_time_scope(span.time_series_ids)}"
        if isinstance(span, TimeInterval):
            return (
                f"time interval [{_seconds(span.start_us)}, {_seconds(span.end_us)}) s"
                f"{_time_scope(span.time_series_ids)}"
            )
        if isinstance(span, StepPoint):
            return f"step point at {span.start} on Signal {span.time_series_id}"
        if isinstance(span, StepInterval):
            return f"step interval [{span.start}, {span.stop}) on Signal {span.time_series_id}"
        raise AssertionError(f"unsupported span {type(span).__name__}")


def _render_annotation(annotation: Annotation) -> str:
    value = "" if annotation.value is None else f"={_json_value(annotation.value)}"
    unit = "" if annotation.unit is None else f" {annotation.unit}"
    span = "" if annotation.span is None else f" @ {TextPromptRenderer._render_span(annotation.span)}"
    return f"{annotation.name}{value}{unit}{span}"


def _json_value(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _seconds(microseconds: int) -> str:
    whole, remainder = divmod(microseconds, 1_000_000)
    if remainder == 0:
        return str(whole)
    return f"{microseconds / 1_000_000:.6f}".rstrip("0").rstrip(".")


def _time_scope(signal_ids: tuple[str, ...] | None) -> str:
    return "" if signal_ids is None else f" on Signals {', '.join(signal_ids)}"
