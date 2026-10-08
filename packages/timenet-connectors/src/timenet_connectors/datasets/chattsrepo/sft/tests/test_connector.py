"""Tests for the ChatTS supervised fine-tuning connector."""

from timenet_connectors.datasets.chattsrepo.chatts_core.tests.helpers import assert_configuration_conversion
from timenet_connectors.datasets.chattsrepo.sft import ChatTSSftConnector


def test_convert_preserves_sft_configuration():
    """Convert fixture rows with supervised fine-tuning provenance."""
    assert_configuration_conversion(ChatTSSftConnector(), "sft")
