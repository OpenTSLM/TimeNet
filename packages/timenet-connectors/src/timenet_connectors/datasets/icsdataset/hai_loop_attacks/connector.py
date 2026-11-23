"""The HAI loop-attacks connector: 300-second windows of one control loop, labelled attack or normal.

The series come from the ``icsdataset/hai`` parent, which holds every HAI data file as published.
This connector downloads only the examples table (:mod:`.examples`) from Hugging Face, pinned by
commit and SHA-256 (:mod:`.sources`). It imports the series that the examples use from the parent:
for each of the 17 files that hold examples, a copy of its record with only those signals. Each
example becomes a ClassificationTask scoped to its window and series, and each attack example also
a TemporalLocalizationTask for the flagged runs of its loop inside the window.

The table derives from HAI, so it is licensed CC BY-SA 4.0 like HAI and is not part of this MIT
package.
"""

from collections.abc import Iterable, Mapping
import hashlib
from pathlib import Path

from timenet.connectors.context import BuildContext
from timenet.dataset import Record, Signal, TimeFDataset
from timenet.errors import TimeNetConnectorError, TimeNetDownloadError
from timenet.reader import TimeFReader
from timenet.types import Annotation, Task
from timenet.types.reference_keys import ParentAlias
from timenet_connectors.bases.huggingface import BaseHuggingFaceConnector
from timenet_connectors.datasets.icsdataset.hai_loop_attacks.examples import (
    Example,
    FileKey,
    FileRecord,
    SeriesTag,
    read_examples,
    series_by_version,
)
from timenet_connectors.datasets.icsdataset.hai_loop_attacks.sources import (
    EXAMPLES_FILENAME,
    EXAMPLES_REPO,
    EXAMPLES_REVISION,
    EXAMPLES_SHA256,
)


HAI = ParentAlias("hai")
"""The alias of the ``icsdataset/hai`` parent on the card."""


def check_examples_table(path: Path) -> None:
    """Check that the downloaded table is the one the connector pins.

    Raises:
        TimeNetDownloadError: If the SHA-256 of the file differs from ``EXAMPLES_SHA256``.
    """
    with path.open("rb") as handle:
        digest = hashlib.file_digest(handle, "sha256").hexdigest()
    if digest != EXAMPLES_SHA256:
        raise TimeNetDownloadError(f"{path} has SHA-256 {digest}, not the pinned {EXAMPLES_SHA256}")


class FileRecords:
    """The records the dataset imports from the parent, one per data file that holds examples."""

    def __init__(self, records: Mapping[FileKey, FileRecord]) -> None:
        self._records = dict(records)

    @classmethod
    def import_from(
        cls,
        dataset: TimeFDataset,
        parent: TimeFReader,
        series: Mapping[str, Mapping[str, SeriesTag]],
        files: Iterable[FileKey],
    ) -> "FileRecords":
        """Import the series of each file from the parent, each file as a copy of its record.

        The parent keeps the version and the file of each record in its metadata, and names each
        signal after its column. The copy of a record holds only the signals of the loop series of
        its version, which get the tag, loop, role and match of their series as metadata.

        Args:
            dataset: The dataset that imports.
            parent: The opened ``icsdataset/hai`` parent.
            series: The series of each version, by 22.04 tag.
            files: The files that hold examples.

        Returns:
            The imported records.

        Raises:
            TimeNetConnectorError: If the parent has no record of a file, or the record has no column
                that a series needs.
        """
        wanted = set(files)
        found: dict[FileKey, Record] = {}
        for record in parent.iter_records(with_annotations=False):
            key = FileKey(str(record.metadata.get("version")), str(record.metadata.get("file")))
            if key in wanted:
                found[key] = record
        missing = sorted(f"{key.version}/{key.stem}" for key in wanted - found.keys())
        if missing:
            raise TimeNetConnectorError(f"the HAI parent has no record of the files {missing}")
        chosen: dict[int, tuple[FileKey, SeriesTag]] = {}
        for key, record in sorted(found.items(), key=lambda item: (item[0].version, item[0].stem)):
            by_name = {signal.name: signal.id for signal in record.signals}
            for one in series[key.version].values():
                if one.source_tag not in by_name:
                    raise TimeNetConnectorError(f"hai-{key.version}/{key.stem} has no column {one.source_tag!r}")
                chosen[by_name[one.source_tag]] = (key, one)
        imported = dataset.import_from(parent, alias=HAI, signals=list(chosen))
        signals: dict[FileKey, dict[str, Signal]] = {}
        for parent_id, (key, one) in chosen.items():
            signal = imported.signals[parent_id]
            # The copy is this dataset's own object, so its metadata can change without the parent's.
            signal.metadata.update(one.signal_metadata())
            signals.setdefault(key, {})[one.tag] = signal
        records: dict[FileKey, FileRecord] = {}
        for record in dataset.records:
            key = FileKey(str(record.metadata["version"]), str(record.metadata["file"]))
            records[key] = FileRecord(record=record, signals=signals[key])
        return cls(records)

    def of_example(self, example: Example) -> FileRecord:
        """Return the record of an example's file.

        Returns:
            The record.

        Raises:
            TimeNetConnectorError: If no record holds the example's file.
        """
        found = self._records.get(example.key)
        if found is None:
            raise TimeNetConnectorError(f"{example.example_id}: no record holds the file {example.file!r}")
        return found


def selection_annotations(examples: list[Example]) -> list[Annotation]:
    """Return the dataset annotations that name the examples of the evaluation protocol.

    Each of these flags depends on the label, so no task carries it. The dataset lists the examples
    instead.

    Returns:
        The annotations.
    """
    return [
        Annotation(
            "training_sample",
            value=[one.example_id for one in examples if one.train_use],
            description=(
                "The example ids of the class-balanced training sample: every train attack example with a visible"
                " change, plus two normal examples from attack windows and two from normal windows for each."
            ),
        ),
        Annotation(
            "scored_windows",
            value=sorted({one.window_id for one in examples if one.scored}),
            description="The window ids of the one window per attack, picked at random, that scoring counts.",
        ),
        Annotation(
            "reacting_examples",
            value=[one.example_id for one in examples if one.reacting],
            description=(
                "The example ids of normal boiler-loop examples whose series start or stop swinging when another"
                " loop of the same process is attacked. A mark, not a label."
            ),
        ),
    ]


class HaiLoopAttacksConnector(
    BaseHuggingFaceConnector,
    hf_repo=EXAMPLES_REPO,
    source_revision=EXAMPLES_REVISION,
    source_files={"examples": EXAMPLES_FILENAME},
):
    """Connector for 300-second control-loop windows of the HAI security dataset."""

    async def download_async(self, cache_dir: Path) -> dict[str, Path]:
        """Download the pinned examples table and check its SHA-256.

        Args:
            cache_dir: Directory where the connector caches Hub files.

        Returns:
            The local path of the table, under ``"examples"``.
        """
        files = await super().download_async(cache_dir)
        check_examples_table(files["examples"])
        return files

    def compose(self, files: Mapping[str, Path], /, context: BuildContext) -> TimeFDataset:
        """Import the loop series from the parent and create the tasks of every example.

        Args:
            files: The table from :meth:`download_async`.
            context: The opened ``icsdataset/hai`` parent, under the alias ``hai``.

        Returns:
            The populated dataset.
        """
        examples = read_examples(files["examples"])
        dataset = TimeFDataset(metadata=self.metadata(), annotations=selection_annotations(examples))
        records = FileRecords.import_from(
            dataset, context.parents[HAI], series_by_version(examples), {one.key for one in examples}
        )
        tasks: list[Task] = []
        for example in examples:
            file_record = records.of_example(example)
            classify = example.create_classification_task(file_record)
            tasks.append(classify)
            if example.attack is not None:
                tasks.append(example.create_localization_task(file_record, from_task=classify))
        dataset.add_tasks(tasks=tasks)
        return dataset


CONNECTOR = HaiLoopAttacksConnector
