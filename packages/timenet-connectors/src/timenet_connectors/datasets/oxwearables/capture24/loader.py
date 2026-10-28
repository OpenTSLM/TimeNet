"""Load CAPTURE-24's recordings into one lazy Record per participant and one task per window and scheme."""

from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Self

from timenet.dataset import Record
from timenet.errors import TimeFValidationError
from timenet.types import Annotation, Split
from timenet_connectors.bases.loader import DatasetLoader
from timenet_connectors.datasets.oxwearables.capture24.keys import AnnotationKey, scheme_mapping, scheme_vocabulary
from timenet_connectors.datasets.oxwearables.capture24.participants import read_participants, read_timeline
from timenet_connectors.datasets.oxwearables.capture24.records import AccelerationLoader, build_record
from timenet_connectors.datasets.oxwearables.capture24.release import (
    DICTIONARY_FILE,
    PINNED_COUNTS,
    PinnedCounts,
    split_of,
)
from timenet_connectors.datasets.oxwearables.capture24.tasks import Capture24WindowTask
from timenet_connectors.datasets.oxwearables.capture24.vocabulary import (
    BENCHMARK_SCHEMES,
    LabelDictionary,
    LabelScheme,
)
from timenet_connectors.datasets.oxwearables.capture24.windows import Windows, make_windows


@dataclass(frozen=True)
class _Loaded:
    """One participant, parsed: their Record, their split, and their windows."""

    record: Record
    split: Split
    windows: Windows


class Capture24DatasetLoader(DatasetLoader[Capture24WindowTask]):
    """CAPTURE-24's recordings, read and checked: one lazy Record per participant, and its window tasks.

    CAPTURE-24 defines one task (:class:`Capture24WindowTask`): classify a 10 s window's activity, in each
    scheme of :data:`~timenet_connectors.datasets.oxwearables.capture24.vocabulary.BENCHMARK_SCHEMES`.
    Select one split with ``tasks_of``.

    The loader keeps each participant's segments, label runs and windows, not their samples. The signals
    reread their recording, so the files must stay available until writing ends.
    """

    TASK_KINDS = (Capture24WindowTask,)

    def __init__(self, dictionary: LabelDictionary, participants: tuple[_Loaded, ...]) -> None:
        self._dictionary = dictionary
        self._participants = participants

    @classmethod
    def parse_from(cls, refs: Iterable[Path], *, counts: PinnedCounts = PINNED_COUNTS) -> Self:
        """Read and check every recording's clock and labels once, and build one lazy Record per participant.

        Args:
            refs: The one extracted ``capture24`` directory.
            counts: What the directory must hold.

        Returns:
            The loaded release.

        Raises:
            TimeFValidationError: If ``refs`` isn't one directory, or its counts differ from ``counts``.
        """
        roots = list(refs)
        if len(roots) != 1:
            raise TimeFValidationError(f"CAPTURE-24 loads from one directory, got {roots}")
        (root,) = roots
        dictionary = LabelDictionary.read(root / DICTIONARY_FILE)
        # One loader for the Records of this load, so a recording's signals share one read.
        acceleration = AccelerationLoader()
        loaded = []
        for participant in read_participants(root):
            timeline = read_timeline(participant.path, dictionary)
            windows = make_windows(timeline.offsets_us, timeline.labels, dictionary, BENCHMARK_SCHEMES)
            record = build_record(participant, timeline, dictionary, acceleration)
            loaded.append(_Loaded(record=record, split=split_of(participant.number), windows=windows))
        found = PinnedCounts(
            participants=len(loaded),
            segments=sum(len(one.record.sources) for one in loaded),
            windows=sum(len(one.windows) for one in loaded),
        )
        counts.check(root, found)
        return cls(dictionary, tuple(loaded))

    def dataset_annotations(self) -> tuple[Annotation, ...]:
        """Return the raw labels, and each scheme's classes and class of each raw label.

        A task names its scheme's vocabulary key as its ``target_schema``. The mappings let a reader turn
        a raw ``activity_label`` into any scheme's class, including the four that have no tasks.

        Returns:
            One raw-label vocabulary, then a vocabulary and a mapping per scheme.
        """
        annotations = [Annotation(key=AnnotationKey.ACTIVITY_LABEL_VOCABULARY, value=list(self._dictionary.raw_labels))]
        for scheme in LabelScheme:
            annotations.append(
                Annotation(key=scheme_vocabulary(scheme), value=list(self._dictionary.vocabulary(scheme)))
            )
            annotations.append(Annotation(key=scheme_mapping(scheme), value=list(self._dictionary.classes[scheme])))
        return tuple(annotations)

    def records(self) -> Iterator[Record]:
        """Return one Record per participant, in id order, whose values stay in the files until written."""
        return (one.record for one in self._participants)

    def tasks(self) -> Iterator[Capture24WindowTask]:
        """Return one task per window and benchmarked scheme, by participant, then window, then scheme.

        Yields:
            The tasks.
        """
        for one in self._participants:
            vocabularies = {scheme: self._dictionary.vocabulary(scheme) for scheme in BENCHMARK_SCHEMES}
            for index, start_us in enumerate(one.windows.start_us.tolist()):
                for scheme in BENCHMARK_SCHEMES:
                    yield Capture24WindowTask(
                        record=one.record,
                        participant_split=one.split,
                        scheme=scheme,
                        start_us=start_us,
                        target=vocabularies[scheme][one.windows.classes[scheme][index]],
                    )
