"""Convert pinned SensorCaps chat rows into lazy IMU Signals and streamed caption tasks.

The connector downloads the original JSONL files at a pinned commit through the HuggingFace base. Its
``convert`` checks the number of rows in each file against the pinned inventory, and adds the dataset
annotations: the source pin and the timing interpretation. These values do not come from the files.

``SensorCapsLoader`` in ``loader.py`` does the rest: ``parse.py`` reads each row, and ``records.py``
builds its Record with lazy values. The loader re-reads the files for the tasks, so the corpus is
never held in memory.
"""

from collections.abc import Mapping
from typing import ClassVar

from timenet.dataset import TimeFDataset
from timenet.errors import TimeFValidationError
from timenet.types import Annotation
from timenet_connectors.bases.huggingface import BaseHuggingFaceConnector, HubFileRef
from timenet_connectors.datasets.bash_lab.sensorcaps import release
from timenet_connectors.datasets.bash_lab.sensorcaps.loader import SensorCapsLoader
from timenet_connectors.datasets.bash_lab.sensorcaps.records import (
    DECLARED_SAMPLING_RATE_HZ,
    SOURCE_SAMPLING_RATE_HZ,
    SOURCE_WINDOW_SAMPLES,
    TIME_AXIS_RECONSTRUCTION,
)


TIMING_ANNOTATIONS = tuple(
    Annotation(key=key, value=value)
    for key, value in (
        ("declared_sampling_rate_hz", DECLARED_SAMPLING_RATE_HZ),
        ("source_sampling_rate_hz", SOURCE_SAMPLING_RATE_HZ),
        ("source_window_samples", SOURCE_WINDOW_SAMPLES),
        ("time_axis_reconstruction", TIME_AXIS_RECONSTRUCTION),
    )
)
"""The timing interpretation of the full dataset.

It does not come from the files, so the connector adds it and not the loader.
"""


class SensorCapsConnector(BaseHuggingFaceConnector):
    """Build SensorCaps with lazy values and streamed captions."""

    HF_REPO = release.HF_REPO
    SOURCE_REVISION = release.SOURCE_REVISION
    SOURCE_ROWS: ClassVar[Mapping[str, int]] = release.SOURCE_ROWS
    SOURCE_FILES: ClassVar[Mapping[str, str]] = {
        partition: release.source_file(partition) for partition in release.SOURCE_ROWS
    }

    def convert(self, raw_refs: list[HubFileRef]) -> TimeFDataset:
        """Index local rows into six-channel Records and a re-iterable caption stream.

        Returns:
            A dataset retaining its hierarchy, small annotations, and file-backed loaders.

        Raises:
            TimeFValidationError: If references or source rows violate the SensorCaps mapping.
        """
        loader = SensorCapsLoader.parse_from(raw_refs)
        for file, count in loader.row_counts.items():
            # Every pinned count is nonzero, so this also rejects an empty partition.
            expected = self.SOURCE_ROWS[file.part]
            if count != expected:
                raise TimeFValidationError(f"{file.path} ({file.part}): expected {expected} rows, found {count}")
        dataset = TimeFDataset(metadata=self.metadata())
        # The pin comes from the class attributes that download reads, so the stored pin is the
        # one that the connector downloaded.
        dataset.annotate(Annotation(key="source_repository", value=self.HF_REPO))
        dataset.annotate(Annotation(key="source_revision", value=self.SOURCE_REVISION))
        for annotation in TIMING_ANNOTATIONS:
            dataset.annotate(annotation)
        return loader.populate(dataset)


CONNECTOR = SensorCapsConnector
