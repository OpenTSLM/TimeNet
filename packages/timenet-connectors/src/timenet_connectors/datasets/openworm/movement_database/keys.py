"""The names the movement database writes into a dataset, so writing and reading them share one spelling."""

from enum import StrEnum, unique


@unique
class AnnotationKey(StrEnum):
    """The key of an annotation the connector builds."""

    SOURCE_URL = "source_url"
    """The Zenodo community the videos come from."""
    BENCHMARK_URL = "benchmark_url"
    """The authors' repository, at the commit whose folds and classes the tasks follow."""
    STRAIN_VOCABULARY = "strain_vocabulary"
    """The authors' 365 classes, in their index order; every task names it as its ``target_schema``."""
    VENTRAL_SIDE = "ventral_side"
    """A Record's ventral side, which tells the skeleton's ventral bends from its dorsal ones."""
