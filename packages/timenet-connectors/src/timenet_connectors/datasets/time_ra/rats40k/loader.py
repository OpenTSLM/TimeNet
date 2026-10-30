"""Load RATS40K's two files into one lazy Record and one category task per usable row."""

from collections import Counter, defaultdict
from collections.abc import Iterable, Iterator, Mapping
from typing import Self

from timenet.dataset import Record
from timenet.errors import TimeFValidationError
from timenet.types import Annotation, ClassificationTask, Split
from timenet_connectors.bases.huggingface import HubFileRef
from timenet_connectors.bases.loader import DatasetLoader
from timenet_connectors.datasets.time_ra.rats40k.keys import category_vocabulary, request_template
from timenet_connectors.datasets.time_ra.rats40k.records import ChannelLoader, build_record
from timenet_connectors.datasets.time_ra.rats40k.release import PINNED_INVENTORY, PinnedInventory, SourceSplit, Variant
from timenet_connectors.datasets.time_ra.rats40k.request import CATEGORY_NAMES, TEMPLATES
from timenet_connectors.datasets.time_ra.rats40k.rows import Row, read_rows
from timenet_connectors.datasets.time_ra.rats40k.tasks import category_task


class RATS40KDatasetLoader(DatasetLoader):
    """RATS40K's files, read and checked: one lazy Record per usable row, and one category task each.

    RATS40K defines one task (:func:`category_task`): name a row's anomaly category, with the published
    ``Thought`` as rationale. After ``populate``, ``TimeFDataset.get_split`` selects one split.

    The loader keeps each row's small fields, not its values, so tasks stream from memory and the signal
    loaders reread only their own entry. The source files must stay available until writing ends.
    """

    TASK_TYPES = (ClassificationTask,)

    def __init__(self, variants: tuple[Variant, ...], rows: tuple[tuple[Row, Record], ...]) -> None:
        self._variants = variants
        self._rows = rows

    @classmethod
    def parse_from(
        cls, refs: Iterable[HubFileRef], *, inventory: Mapping[Variant, PinnedInventory] = PINNED_INVENTORY
    ) -> Self:
        """Read and check each file once, in order, and build one lazy Record per usable row.

        ``refs`` are the references the connector's base already checked: each variant at most once.
        Only explicit null entries are skipped. A malformed row, or a file whose rows and null keys
        differ from ``inventory``, raises ``TimeFValidationError``.

        Args:
            refs: The downloaded files, one per variant.
            inventory: Each variant's pinned rows and null keys.

        Returns:
            The loaded release.

        Raises:
            TimeFValidationError: If a reference names no RATS40K variant.
        """
        variants: list[Variant] = []
        rows: list[tuple[Row, Record]] = []
        # One loader for the Records of this load, so a row's channels share one reread.
        channels = ChannelLoader()
        for file in refs:
            if file.part not in set(Variant):
                raise TimeFValidationError(f"{file.path}: {file.part!r} is not a RATS40K variant")
            variant = Variant(file.part)
            counts: Counter[Split] = Counter()
            nulls: defaultdict[Split, set[str]] = defaultdict(set)
            for ref, row in read_rows(file, variant):
                if row is None:
                    nulls[ref.split].add(ref.key)
                    continue
                rows.append((row, build_record(row, channels)))
                counts[ref.split] += 1
            # Every split of the file, so a row or null entry the inventory doesn't list fails the check.
            splits = [source_split.timenet_split for source_split in SourceSplit]
            found_rows = {split: counts[split] for split in splits}
            found_nulls = {split: frozenset(nulls[split]) for split in splits}
            inventory[variant].check(file.path, found_rows, found_nulls)
            variants.append(variant)
        return cls(tuple(variants), tuple(rows))

    def dataset_annotations(self) -> tuple[Annotation, ...]:
        """Return each loaded variant's category vocabulary and request template.

        A task names its vocabulary's key as its ``target_schema``. A name's index in the vocabulary is
        the authors' code, which ``ActionID`` doesn't always follow. The connector adds the provenance.

        Returns:
            Two annotations per variant.
        """
        annotations: list[Annotation] = []
        for variant in self._variants:
            annotations.append(Annotation(key=category_vocabulary(variant), value=list(CATEGORY_NAMES[variant])))
            annotations.append(Annotation(key=request_template(variant), value=TEMPLATES[variant]))
        return tuple(annotations)

    def records(self) -> Iterator[Record]:
        """Return one Record per usable row, in file order, whose values stay in the files until written."""
        return (record for _, record in self._rows)

    def tasks(self) -> Iterator[ClassificationTask]:
        """Return one task per usable row, in file order."""
        return (category_task(row, record) for row, record in self._rows)
