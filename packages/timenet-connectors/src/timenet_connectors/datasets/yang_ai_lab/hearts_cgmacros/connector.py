"""Expose the CGMacros-derived HEARTS records as a taskless layer."""

from timenet_connectors.datasets.yang_ai_lab.hearts.corpus import HeartsCorpusConnector


class HeartsCgmacrosConnector(HeartsCorpusConnector):
    """Connector for task-specific CGMacros records distributed by HEARTS."""

    CORPUS = "cgmacros"
    values_backend = "zarr"  # meal-classification records include image tensors


CONNECTOR = HeartsCgmacrosConnector
