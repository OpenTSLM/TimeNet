"""Connector for ChatTS's development configuration."""

from timenet_connectors.datasets.chattsrepo.chatts_core.connector import ChatTSConnector


class ChatTSDevConnector(ChatTSConnector):
    """Build the ChatTS ``dev`` configuration."""

    HF_CONFIG = "dev"


CONNECTOR = ChatTSDevConnector
