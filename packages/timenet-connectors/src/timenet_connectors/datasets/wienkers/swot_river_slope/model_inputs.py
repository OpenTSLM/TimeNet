"""Leak-free model input projection for the SWOT forecast tasks."""

from timenet.errors import TimeFValidationError
from timenet.json import JsonValue
from timenet.types import Split, Task, TimeInterval, read_value


_FEATURES = frozenset(
    {
        "wse_m",
        "wse_u_m",
        "fall_m_per_10km",
        "fall_u_m_per_10km",
        "fall_z",
        "reach_q",
    }
)
_CONTEXT = (
    "issue_utc",
    "length_m",
    "longitude",
    "latitude",
    "stratum",
    "baseline_nonneg_m_per_10km",
)


def model_inputs(task: Task) -> dict[str, JsonValue]:
    """Export the prompt, whitelisted context and in-scope pre-issue time series."""
    if task.split not in {Split.TRAIN, Split.VALIDATION}:
        raise TimeFValidationError("Only released train and validation issues can be serialised.")
    if len(task.inputs) != 1:
        raise TimeFValidationError("A SWOT model input requires one input record.")
    (record,) = task.inputs
    scope = task.scope
    annotated = {item.key: read_value(item.value) for item in task.input_annotations}
    issue_offset = annotated.get("issue_offset_us")
    if not isinstance(scope, TimeInterval) or scope.time_series is None or type(issue_offset) is not int:
        raise TimeFValidationError("A SWOT model input requires a scoped task and explicit issue time.")
    selected = {id(signal) for signal in scope.time_series}
    channels: list[JsonValue] = []
    for source in sorted(record.sources, key=lambda source: source.name):
        for signal in sorted(source.signals, key=lambda signal: signal.name):
            if id(signal) not in selected or signal.name not in _FEATURES:
                continue
            elapsed = signal.time_offsets_us()
            mask = (elapsed >= scope.start_us) & (elapsed < min(issue_offset, scope.end_us))
            values = signal.to_arrow().to_pylist()
            idx = mask.nonzero()[0].tolist()
            channels.append(
                {
                    "source": source.name,
                    "feature": signal.name,
                    "offset_from_issue_us": (elapsed[mask] - issue_offset).tolist(),
                    "values": [values[i] for i in idx],
                }
            )
    context = {key: record.metadata[key] for key in ("central_reach_id", "endpoint") if key in record.metadata}
    for key in _CONTEXT:
        value = annotated.get(key)
        if isinstance(value, (str, int, float)):
            context[key] = value
    return {
        "prompt": task.prompt.read() if task.prompt else None,
        "context": context,
        "channels": channels,
    }
