"""Connector for ChatTS's instruction-following configuration."""

from timenet_connectors.datasets.chattsrepo.chatts_core.connector import ChatTSConnector


class ChatTSIftConnector(ChatTSConnector):
    """Build the ChatTS ``ift`` configuration."""

    HF_CONFIG = "ift"


CONNECTOR = ChatTSIftConnector
