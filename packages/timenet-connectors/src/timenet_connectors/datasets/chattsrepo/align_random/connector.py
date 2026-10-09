"""Connector for ChatTS's variable-length alignment configuration."""

from timenet_connectors.datasets.chattsrepo.chatts_core.connector import ChatTSConnector


class ChatTSAlignRandomConnector(ChatTSConnector):
    """Build the ChatTS ``align_random`` configuration."""

    HF_CONFIG = "align_random"


CONNECTOR = ChatTSAlignRandomConnector
