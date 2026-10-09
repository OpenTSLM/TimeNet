"""Connector for ChatTS's development configuration."""

from timenet.types import Split
from timenet_connectors.datasets.chattsrepo.chatts_core.connector import ChatTSConnector


class ChatTSDevConnector(ChatTSConnector):
    """Build the ChatTS ``dev`` configuration."""

    HF_CONFIG = "dev"
    TASK_SPLIT = Split.VALIDATION


CONNECTOR = ChatTSDevConnector
