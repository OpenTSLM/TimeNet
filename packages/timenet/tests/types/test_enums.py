from packaging.licenses import canonicalize_license_expression
import pytest

from timenet.types import Domain, License


def test_domain_values():
    assert Domain.CARDIOLOGY == "cardiology"
    assert Domain.HEALTH == "health"
    assert Domain.GENERAL == "general"


@pytest.mark.parametrize("license", [lic for lic in License if lic is not License.OTHER])
def test_license_values_are_canonical_spdx_ids(license):
    # A known SPDX ID in its canonical spelling, not a display string like "CC BY 4.0".
    assert canonicalize_license_expression(license.value) == license.value


def test_enums_are_str():
    assert isinstance(License.MIT, str)
    assert isinstance(Domain.HEALTH, str)
