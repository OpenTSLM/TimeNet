"""The :class:`ManifestFiles` block: the dataset's file pointers, relative to the version directory."""

from dataclasses import dataclass


@dataclass(frozen=True)
class ManifestFiles:
    """Relative paths to every artifact of a dataset version. Readers use this list, not a glob.

    Every artifact is a list of parts, so any of them can shard later without a manifest-format change.
    Today the writer emits a single part for ``samples`` / ``annotations`` / ``time_series_index``;
    ``tasks`` and ``time_series`` already carry several.
    """

    samples: tuple[str, ...]
    annotations: tuple[str, ...]
    time_series_index: tuple[str, ...]
    tasks: tuple[str, ...] = ()
    time_series: tuple[str, ...] = ()

    def all_parts(self) -> tuple[str, ...]:
        """Return every file part across all artifacts, in a stable order.

        Returns:
            The concatenation of the ``samples``, ``annotations``, ``time_series_index``, ``tasks``,
            and ``time_series`` parts.
        """
        return (*self.samples, *self.annotations, *self.time_series_index, *self.tasks, *self.time_series)
