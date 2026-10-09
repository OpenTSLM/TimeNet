"""Connector for ChatTS's supervised fine-tuning configuration."""

from timenet_connectors.datasets.chattsrepo.chatts_core.connector import ChatTSConnector


class ChatTSSftConnector(ChatTSConnector):
    """Build the ChatTS ``sft`` configuration."""

    HF_CONFIG = "sft"


CONNECTOR = ChatTSSftConnector
