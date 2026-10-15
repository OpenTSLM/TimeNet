from timenet.types import Domain, License


def test_domain_values():
    assert Domain.CARDIOLOGY == "cardiology"
    assert Domain.HEALTH == "health"
    assert Domain.GENERAL == "general"


def test_enums_are_str():
    assert isinstance(License.MIT, str)
    assert isinstance(Domain.HEALTH, str)
