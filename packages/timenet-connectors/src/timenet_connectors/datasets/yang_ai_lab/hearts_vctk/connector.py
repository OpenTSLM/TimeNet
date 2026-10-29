"""Expose the VCTK-derived HEARTS records as a taskless layer."""

from timenet_connectors.datasets.yang_ai_lab.hearts.corpus import HeartsCorpusConnector


class HeartsVctkConnector(HeartsCorpusConnector):
    """Connector for task-specific VCTK records distributed by HEARTS."""

    CORPUS = "vctk"


CONNECTOR = HeartsVctkConnector
