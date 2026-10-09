"""Connector for the complete ChatTS training corpus."""

from timenet_connectors.datasets.chattsrepo.chatts_core.connector import ChatTSConnector


class ChatTSTrainingDatasetConnector(ChatTSConnector):
    """Build the complete ChatTS corpus from all five source configurations."""

    HF_CONFIGS = ("align_256", "align_random", "sft", "ift", "dev")


CONNECTOR = ChatTSTrainingDatasetConnector
