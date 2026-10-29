"""What the authors' strain benchmark fixes: its videos, their folds, its classes, and its test fold.

These come from the authors' repository, ``ver228/worm-ts-classification``, and from the paper (Javer et al.,
ECCV Workshops 2018), not from Zenodo. The repository publishes two pickles: ``SWDB_fold_dict.p`` maps each
video's base name to one of three folds, and ``SWDB_straindict.p`` maps each class to its index.
"""

from pathlib import Path

# The authors publish their folds and classes as pickles; _PlainUnpickler reads them without loading any class.
import pickle  # noqa: S403
from typing import Any

from timenet.errors import TimeFFormatError
from timenet.types import Split
from timenet_connectors.datasets.openworm.movement_database.vocabulary import Sex


AUTHORS_REPOSITORY = "https://github.com/ver228/worm-ts-classification"
AUTHORS_COMMIT = "6d4d6b6f04e05a93baea7a5a7550a7a180b60c94"
"""The last commit of the authors' repository, from 24 April 2019."""

_RAW_ROOT = f"https://raw.githubusercontent.com/ver228/worm-ts-classification/{AUTHORS_COMMIT}"
FOLDS_URL = f"{_RAW_ROOT}/worm_ts_classification/flow/SWDB_fold_dict.p"
FOLDS_SHA256 = "3e5a700806b3f710caeedbe5e4e62f1b81b51577b8a99b5fa506d31dc37f0aa8"
CLASSES_URL = f"{_RAW_ROOT}/worm_ts_classification/flow/SWDB_straindict.p"
CLASSES_SHA256 = "f7b9e9d1f8b34dc6b3af4e6c1ac73dc5de61681ed61c620cff8520a3ace4eb07"

N_FOLDS = 3
"""``flow_helper.save_folds_dict`` deals each strain's videos into three folds in turn."""
TEST_FOLD = 0
"""The fold the authors test on: ``SkelEmbeddingsFlow``'s ``fold_n_test`` defaults to 0, and their ``Trainer``
never sets it. The paper: "Two thirds of the data are used for training and the rest for testing"."""


def split_of(fold: int) -> Split:
    """Return a video's benchmark split: its fold is the test fold, or one of the two it trains on.

    Returns:
        ``Split.TEST`` for :data:`TEST_FOLD`, else ``Split.TRAIN``.
    """
    return Split.TEST if fold == TEST_FOLD else Split.TRAIN


def class_of(strain: str, sex: Sex) -> str:
    """Return the authors' class of a video: its strain and its sex chromosomes, such as ``N2_XO``.

    The paper: "All the videos for a given strain are given the same class label, except N2 where
    hermaphrodites and males are considered as separated classes".

    Returns:
        The class, as ``SWDB_straindict.p`` names it.
    """
    return f"{strain}_{sex.karyotype}"


class _PlainUnpickler(pickle.Unpickler):  # noqa: S301 - it loads no class, so it runs no code
    """Unpickle only builtin values: the two files hold a dict of str to int, and need no class."""

    def find_class(self, module: str, name: str) -> Any:  # noqa: PLR6301 - pickle's interface
        """Refuse every class, so the file can't run code.

        Raises:
            TimeFFormatError: Always.
        """
        raise TimeFFormatError(f"the authors' pickle names {module}.{name}; it should hold only a dict")


def _read_str_to_int(path: Path) -> dict[str, int]:
    """Read one of the authors' pickles, which must be a dict of str to int.

    Returns:
        The dict, in the file's order.

    Raises:
        TimeFFormatError: If the file isn't a pickled dict of str to int.
    """
    with path.open("rb") as handle:
        try:
            value = _PlainUnpickler(handle).load()
        except (pickle.UnpicklingError, EOFError) as exc:
            raise TimeFFormatError(f"{path}: not a pickle: {exc}") from exc
    if not isinstance(value, dict) or not all(
        isinstance(key, str) and type(item) is int for key, item in value.items()
    ):
        raise TimeFFormatError(f"{path}: expected a dict of str to int")
    return value


def read_folds(path: Path) -> dict[str, int]:
    """Read ``SWDB_fold_dict.p``: each benchmark video's base name, and its fold.

    Returns:
        The fold of each base name, in the file's order.

    Raises:
        TimeFFormatError: If a fold isn't one of the :data:`N_FOLDS`.
    """
    folds = _read_str_to_int(path)
    bad = sorted({fold for fold in folds.values() if not 0 <= fold < N_FOLDS})
    if bad:
        raise TimeFFormatError(f"{path}: folds {bad} are outside 0 to {N_FOLDS - 1}")
    return folds


def read_classes(path: Path) -> tuple[str, ...]:
    """Read ``SWDB_straindict.p``: each class, and the index the authors' model gives it.

    Returns:
        The classes in index order, which is also their sorted order.

    Raises:
        TimeFFormatError: If the indices aren't 0 to n - 1, or don't follow the classes' sorted order, as
            ``flow_helper.save_strain_dict`` numbers them.
    """
    indices = _read_str_to_int(path)
    classes = tuple(sorted(indices, key=indices.__getitem__))
    if [indices[name] for name in classes] != list(range(len(classes))):
        raise TimeFFormatError(f"{path}: the class indices aren't 0 to {len(classes) - 1}")
    if list(classes) != sorted(classes):
        raise TimeFFormatError(f"{path}: the class indices don't follow the classes' sorted order")
    return classes
