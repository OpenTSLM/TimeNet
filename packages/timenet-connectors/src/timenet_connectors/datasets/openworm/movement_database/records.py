"""One Record per video: its lazy skeleton signal, its start time and ventral side, and its provenance."""

from fractions import Fraction
from functools import partial

import numpy as np
import pyarrow as pa

from timenet.dataset import Record, Signal, Source
from timenet.dataset.axis import RegularAxis
from timenet.types import Annotation, TimeOrigin, TimeSeriesSpec, ureg
from timenet_connectors.datasets.openworm.movement_database.index import IndexedVideo
from timenet_connectors.datasets.openworm.movement_database.keys import AnnotationKey
from timenet_connectors.datasets.openworm.movement_database.release import SKELETON_POINTS
from timenet_connectors.datasets.openworm.movement_database.videos import Video, read_skeletons
from timenet_connectors.metadata import RecordMetadata


SOURCE_NAME = "Tierpsy Tracker"
SIGNAL_NAME = "skeleton"

# One timestep is the frame's skeleton, as the file stores it: 49 points from head to tail, then x and y,
# float32 in micrometers. A frame the tracker gave no skeleton is NaN in all 98 values, so it is a missing
# timestep: null.
_SPEC = TimeSeriesSpec(
    spec_type="worm_skeleton",
    name="Worm skeleton",
    unit_value=ureg.micrometer,
    dtype="float32",
    value_shape=(SKELETON_POINTS, 2),
    dimension_names=("point", "coordinate"),
    nullable=True,
)
_TENSOR_TYPE = pa.fixed_shape_tensor(pa.float32(), (SKELETON_POINTS, 2))


class OpenWormRecordMetadata(RecordMetadata):
    """The movement database's Record metadata: which video, which Zenodo record, and the authors' fold.

    Each video follows a different worm, and the source names no worm, so ``subject_id`` stays unset. The base
    name spells out the gene and allele, so this metadata names the class: it is provenance, never model input.
    """

    base_name: str
    zenodo_record: int
    fold: int
    """The video's fold in the authors' ``SWDB_fold_dict.p``. Fold 0 is their test set."""


def record_metadata(record: Record) -> OpenWormRecordMetadata:
    """Read a movement database Record's typed metadata back.

    Returns:
        The video's metadata.

    Raises:
        TimeFValidationError: If the Record's metadata isn't this dataset's.
    """  # noqa: DOC502 - raised by from_json
    return OpenWormRecordMetadata.from_json(record.metadata)


def load_skeletons(video: Video) -> pa.Array:
    """Load a video's skeletons as one tensor per frame, null where the frame has no skeleton.

    Returns:
        The ``(49, 2)`` float32 tensors, one per frame.
    """
    skeletons = read_skeletons(video)
    missing = np.isnan(skeletons[:, 0, 0])
    storage = pa.FixedSizeListArray.from_arrays(
        pa.array(skeletons.reshape(-1)), SKELETON_POINTS * 2, mask=pa.array(missing)
    )
    return pa.ExtensionArray.from_storage(_TENSOR_TYPE, storage)


def build_record(indexed: IndexedVideo, video: Video, fold: int) -> Record:
    """Build a video's Record: its lazy skeleton signal, its start time, ventral side and provenance.

    The strain, sex, gene and allele are the class, so none is an annotation.

    Args:
        indexed: The video's Zenodo record.
        video: The video's features file.
        fold: The video's fold in the authors' benchmark.

    Returns:
        The Record, whose values stay in the file until read.
    """
    signal = Signal.from_loader(
        name=SIGNAL_NAME,
        spec=_SPEC,
        # The video's frame cadence, starting at the frame tracking started on, so each skeleton keeps
        # its time from the start of the video.
        time_axis=RegularAxis(period_us=Fraction(video.period_us)).at_index(video.first_frame),
        n_values=video.n_frames,
        loader=partial(load_skeletons, video),
    )
    metadata = OpenWormRecordMetadata(base_name=indexed.base_name, zenodo_record=indexed.record_id, fold=fold)
    record = Record(
        sources=(Source(name=SOURCE_NAME, signals=(signal,)),),
        start_time=TimeOrigin(video.experiment.timestamp),
        # Provenance only, never model input.
        metadata=metadata.to_json(),
    )
    record.annotate(Annotation(key=AnnotationKey.VENTRAL_SIDE, value=video.experiment.ventral_side.value))
    return record
