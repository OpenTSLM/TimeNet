from timenet.types import Domain, License, View


def test_view_values():
    assert View.FULL == "full"
    assert View.SINGLE_CHANNEL == "single_channel"
    assert View.SUBSET == "subset"
    assert View.WINDOW == "window"


def test_domain_values():
    assert Domain.CARDIOLOGY == "cardiology"
    assert Domain.HEALTH == "health"
    assert Domain.GENERAL == "general"


def test_license_values_are_spdx_style():
    # SPDX identifiers, not display strings like "CC BY 4.0".
    assert License.MIT == "MIT"
    assert License.APACHE_2_0 == "Apache-2.0"
    assert License.CC_BY_4_0 == "CC-BY-4.0"


def test_enums_are_str():
    assert isinstance(View.FULL, str)
    assert isinstance(License.MIT, str)
    assert isinstance(Domain.HEALTH, str)
