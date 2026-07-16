"""The :class:`ManifestFiles` block: the dataset's file pointers, relative to the version directory."""

from dataclasses import dataclass


@dataclass(frozen=True)
class ManifestFiles:
    """Relative paths to every artifact of a dataset version. Readers use this list, not a glob."""

    samples: str
    annotations: str
    time_series_index: str
    tasks: tuple[str, ...] = ()
    time_series: tuple[str, ...] = ()
