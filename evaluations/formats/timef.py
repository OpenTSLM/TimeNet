"""The TimeF format: the version this dataset's own connector builds."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from evaluations.formats.base import FormatName
from timenet.engine import store_dataset
from timenet.reader import TimeFReader
from timenet.registry import DatasetVersion
from timenet_connectors.discovery import resolve


# The dataset under test, named by id. Nothing here imports a connector module.
DATASET_ID = "physionet/sleep-edfx"


class TimeFFormat:
    """TimeF as a consumer gets it, through the connector that publishes this dataset.

    This format states nothing about the release. The connector already says what a sample is,
    which channels a recording holds and what the scoring means, and the engine already derives
    the schema and commits the version. A second reading written here would measure that reading
    rather than TimeF.
    """

    name = FormatName.TIMEF

    def write(self, source: Path, out: Path) -> Path:  # noqa: PLR6301 - implements the Format Protocol
        """Convert the release with its connector, and store what the connector returns.

        ``download`` resolves the release inside ``source``. It is idempotent, so it extracts on a
        first run and finds the same files on every run after.

        Args:
            source: The directory the release was extracted into.
            out: Directory to write into. The version lands beneath it.

        Returns:
            The version directory.
        """
        out.mkdir(parents=True, exist_ok=True)
        connector = resolve(DATASET_ID)()

        return store_dataset(connector.convert(connector.download(source)), out)

    def read_all(self, path: Path) -> list[np.ndarray]:  # noqa: PLR6301 - implements the Format Protocol
        """Read every value back from the TimeF version.

        Args:
            path: The version directory written by :meth:`write`.

        Returns:
            One array for each time series, in sample order.
        """
        with TimeFReader(DatasetVersion.open_local(path)) as reader:
            dataset = reader.read()

            return [series.to_numpy() for sample in dataset.samples for series in sample.time_series]
