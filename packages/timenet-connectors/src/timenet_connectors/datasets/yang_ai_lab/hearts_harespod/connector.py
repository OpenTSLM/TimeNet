"""Expose the HARESPOD-derived HEARTS records as a taskless layer."""

from timenet_connectors.datasets.yang_ai_lab.hearts.corpus import HeartsCorpusConnector


class HeartsHarespodConnector(HeartsCorpusConnector):
    """Connector for task-specific HARESPOD records distributed by HEARTS."""

    CORPUS = "harespod"


CONNECTOR = HeartsHarespodConnector
