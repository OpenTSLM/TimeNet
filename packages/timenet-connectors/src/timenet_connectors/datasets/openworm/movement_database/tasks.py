"""The movement database's one benchmark task: name a video's strain from the worm's posture over time."""

from timenet.dataset import Record
from timenet.types import ClassificationTask
from timenet_connectors.datasets.openworm.movement_database.benchmark import split_of
from timenet_connectors.datasets.openworm.movement_database.keys import AnnotationKey


def strain_task(record: Record, fold: int, target: str) -> ClassificationTask:
    """Return one video's class, which the authors' classifier predicts from the whole video's skeletons.

    The authors classify a whole video, joining all its trajectory pieces, so the task has no scope. Their
    model reads 48 tangent angles, or 6 eigenworms, computed from the skeletons after resampling them to
    25 fps; README.md says how. The benchmark asks no question in words, so the task has no prompt.

    Args:
        record: The video's Record.
        fold: The video's fold. The authors' test fold is test, the others train.
        target: The class: the strain, then the sex chromosomes, such as ``N2_XO``.

    Returns:
        The classification task, with the video's class as target.
    """
    return ClassificationTask(
        inputs=(record,),
        targets=(target,),
        target_schema=AnnotationKey.STRAIN_VOCABULARY,
        split=split_of(fold),
    )
