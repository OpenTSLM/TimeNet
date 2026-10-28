"""Load CAPTURE-24's recordings into one lazy Record per participant and one task per window and scheme."""

from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Self

from timenet.dataset import Record
from timenet.errors import TimeFValidationError
from timenet.types import Annotation
from timenet_connectors.bases.loader import DatasetLoader
from timenet_connectors.datasets.oxwearables.capture24.benchmark import BENCHMARK_SCHEMES
from timenet_connectors.datasets.oxwearables.capture24.keys import AnnotationKey, scheme_mapping, scheme_vocabulary
from timenet_connectors.datasets.oxwearables.capture24.participants import Participant, read_participants
from timenet_connectors.datasets.oxwearables.capture24.recordings import read_recording
from timenet_connectors.datasets.oxwearables.capture24.records import AccelerationLoader, build_record
from timenet_connectors.datasets.oxwearables.capture24.release import (
    DICTIONARY_FILE,
    PINNED_COUNTS,
    PinnedCounts,
)
from timenet_connectors.datasets.oxwearables.capture24.tasks import Capture24WindowTask
from timenet_connectors.datasets.oxwearables.capture24.vocabulary import LabelDictionary, LabelScheme
from timenet_connectors.datasets.oxwearables.capture24.windows import Windows, make_windows


@dataclass(frozen=True)
class _ParsedParticipant:
    """One participant after parsing: who they are, their lazy Record, and their windows."""

    participant: Participant
    record: Record
    windows: Windows


class Capture24DatasetLoader(DatasetLoader[Capture24WindowTask]):
    """CAPTURE-24's recordings, read and checked: one lazy Record per participant, and its window tasks.

    CAPTURE-24 defines one task (:class:`Capture24WindowTask`): classify a 10 s window's activity, in each
    scheme of :data:`~timenet_connectors.datasets.oxwearables.capture24.benchmark.BENCHMARK_SCHEMES`.
    Select one split with ``tasks_of``.

    The loader keeps each participant's segments, label runs and windows, not their samples. The signals
    reread their recording, so the files must stay available until writing ends.
    """

    TASK_KINDS = (Capture24WindowTask,)

    def __init__(self, dictionary: LabelDictionary, participants: tuple[_ParsedParticipant, ...]) -> None:
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
        acceleration_loader = AccelerationLoader()
        parsed = []
        for participant in read_participants(root):
            # The samples are used here and dropped, so only one participant's are ever held.
            recording, samples = read_recording(participant.recording_path, dictionary)
            windows = make_windows(samples.offsets_us, samples.label_codes, dictionary, BENCHMARK_SCHEMES)
            record = build_record(participant, recording, dictionary, acceleration_loader)
            parsed.append(_ParsedParticipant(participant=participant, record=record, windows=windows))
        found = PinnedCounts(
            participants=len(parsed),
            segments=sum(len(item.record.sources) for item in parsed),
            windows=sum(len(item.windows) for item in parsed),
        )
        counts.check(root, found)
        return cls(dictionary, tuple(parsed))

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
            annotations.append(Annotation(key=scheme_mapping(scheme), value=list(self._dictionary.classes(scheme))))
        return tuple(annotations)

    def records(self) -> Iterator[Record]:
        """Return one Record per participant, in id order, whose values stay in the files until written."""
        return (parsed.record for parsed in self._participants)

    def tasks(self) -> Iterator[Capture24WindowTask]:
        """Return one task per window and benchmarked scheme, by participant, then window, then scheme.

        Yields:
            The tasks.
        """
        vocabularies = {scheme: self._dictionary.vocabulary(scheme) for scheme in BENCHMARK_SCHEMES}
        for parsed in self._participants:
            split = parsed.participant.split
            for window, start_us in enumerate(parsed.windows.start_us.tolist()):
                for scheme in BENCHMARK_SCHEMES:
                    yield Capture24WindowTask(
                        record=parsed.record,
                        participant_split=split,
                        scheme=scheme,
                        start_us=start_us,
                        target=vocabularies[scheme][parsed.windows.class_codes[scheme][window]],
                    )
