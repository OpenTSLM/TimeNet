"""Expose the COUGHVID-derived HEARTS records as a taskless layer."""

from timenet_connectors.datasets.yang_ai_lab.hearts.corpus import HeartsCorpusConnector


class HeartsCoughvidConnector(HeartsCorpusConnector):
    """Connector for task-specific COUGHVID records distributed by HEARTS."""

    CORPUS = "coughvid"


CONNECTOR = HeartsCoughvidConnector
