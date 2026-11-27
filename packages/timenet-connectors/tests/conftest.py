import pytest

from timenet_connectors import discovery


@pytest.fixture(autouse=True)
def _clear_discovery_cache():
    """Read the datasets roots again in each test, because a test can add or remove a root."""
    discovery._get_connector_cards_by_id.cache_clear()
    yield
    discovery._get_connector_cards_by_id.cache_clear()
