"""Load the pinned VerbalTS release facts from ``release.yaml``."""

from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from typing import NotRequired, TypedDict, cast

import yaml

from timenet.dataset.axis import OrdinalAxis, RegularAxis, TimeAxis
from timenet.types import TimeSeriesSpec, ureg


class _AttributeConfig(TypedDict):
    key: str
    description: str
    labels: NotRequired[list[str | int]]
    codes: NotRequired[list[int]]


class _SpecConfig(TypedDict):
    type: str
    name: str
    unit: str
    dtype: str


class _AxisConfig(TypedDict):
    kind: str
    period_seconds: NotRequired[int]


class _ComponentConfig(TypedDict):
    spec: _SpecConfig
    axis: _AxisConfig
    attributes: dict[str, str]
    channels: NotRequired[list[str]]
    channel_from: NotRequired[str]
    stride: NotRequired[int]


class _ReleaseConfig(TypedDict):
    attributes: dict[str, _AttributeConfig]
    components: dict[str, _ComponentConfig]


@dataclass(frozen=True)
class Attribute:
    """An attribute's annotation key, description, and code values."""

    key: str
    """Annotation key shared by components with the same meaning."""
    description: str
    """Description stored in the dataset schema."""
    labels: tuple[str | int, ...] | None = None
    """Labels in code order, or ``None`` to keep integer codes."""
    codes: tuple[int, ...] | None = None
    """Valid integer codes in release order."""

    @property
    def vocabulary(self) -> tuple[str | int, ...] | None:
        """Return labels or integer codes in release order, if known.

        Returns:
            The known labels or codes, or ``None``.
        """
        return self.labels if self.labels is not None else self.codes


def _load_release() -> _ReleaseConfig:
    """Load the trusted release configuration shipped with the connector.

    Returns:
        The release configuration.
    """
    path = Path(__file__).with_name("release.yaml")
    return cast(_ReleaseConfig, yaml.safe_load(path.read_text(encoding="utf-8")))


def _attribute(config: _AttributeConfig) -> Attribute:
    """Convert one attribute definition to the connector's immutable value.

    Returns:
        The attribute definition.
    """
    labels = config.get("labels")
    codes = config.get("codes")
    return Attribute(
        key=config["key"],
        description=config["description"],
        labels=None if labels is None else tuple(labels),
        codes=None if codes is None else tuple(codes),
    )


def _axis(config: _AxisConfig) -> TimeAxis:
    """Convert one configured axis.

    Returns:
        The TimeF axis.
    """
    if config["kind"] == "ordinal":
        return OrdinalAxis()
    return RegularAxis.from_rate_hz(Fraction(1, config["period_seconds"]))


_RELEASE = _load_release()
_ATTRIBUTES = {name: _attribute(config) for name, config in _RELEASE["attributes"].items()}

COMPONENTS: tuple[str, ...] = tuple(_RELEASE["components"])
"""Component order from the VerbalTS paper."""

CODEBOOKS: dict[str, dict[str, Attribute]] = {
    component: {source_name: _ATTRIBUTES[attribute] for source_name, attribute in config["attributes"].items()}
    for component, config in _RELEASE["components"].items()
}
"""Attributes by component and ``meta.json`` column order."""

CHANNELS: dict[str, tuple[str, ...]] = {
    component: tuple(config["channels"]) for component, config in _RELEASE["components"].items() if "channels" in config
}
"""Fixed channel names in array order."""

CHANNEL_FROM: dict[str, str] = {
    component: config["channel_from"]
    for component, config in _RELEASE["components"].items()
    if "channel_from" in config
}
"""Annotation key that names each variable-selected component's channel."""

STRIDE: dict[str, int] = {
    component: config["stride"] for component, config in _RELEASE["components"].items() if "stride" in config
}
"""Sliding-window strides in steps."""

SPEC_BY_COMPONENT: dict[str, TimeSeriesSpec] = {
    component: TimeSeriesSpec(
        spec_type=config["spec"]["type"],
        name=config["spec"]["name"],
        unit_value=ureg.Unit(config["spec"]["unit"]),
        dtype=config["spec"]["dtype"],
    )
    for component, config in _RELEASE["components"].items()
}
"""Signal spec for each component."""

AXIS_BY_COMPONENT: dict[str, TimeAxis] = {
    component: _axis(config["axis"]) for component, config in _RELEASE["components"].items()
}
"""Axis for each component, without absolute timestamps."""
