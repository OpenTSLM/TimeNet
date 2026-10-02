"""Expose the Coswara-derived HEARTS records as a taskless layer."""

from timenet_connectors.datasets.yang_ai_lab.hearts.corpus import HeartsCorpusConnector


class HeartsCoswaraConnector(HeartsCorpusConnector):
    """Connector for task-specific Coswara records distributed by HEARTS."""

    CORPUS = "coswara"


CONNECTOR = HeartsCoswaraConnector
