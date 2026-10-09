"""Connector for ChatTS's fixed-length alignment configuration."""

from timenet_connectors.datasets.chattsrepo.chatts_core.connector import ChatTSConnector


class ChatTSAlign256Connector(ChatTSConnector):
    """Build the ChatTS ``align_256`` configuration."""

    HF_CONFIG = "align_256"


CONNECTOR = ChatTSAlign256Connector
