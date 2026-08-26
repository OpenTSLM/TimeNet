from timenet.types import Domain, License


def test_domain_values():
    assert Domain.CARDIOLOGY == "cardiology"
    assert Domain.HEALTH == "health"
    assert Domain.GENERAL == "general"


def test_license_values_are_stable_identifiers():
    # Use SPDX identifiers when they exist. Use a stable source-specific identifier otherwise.
    assert License.MIT == "MIT"
    assert License.APACHE_2_0 == "Apache-2.0"
    assert License.CC_BY_4_0 == "CC-BY-4.0"
    assert License.PHYSIONET_CREDENTIALED_HEALTH_DATA_1_5_0 == "PhysioNet-Credentialed-Health-Data-1.5.0"


def test_enums_are_str():
    assert isinstance(License.MIT, str)
    assert isinstance(Domain.HEALTH, str)
