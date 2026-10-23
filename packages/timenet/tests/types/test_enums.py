from packaging.licenses import canonicalize_license_expression
import pytest

from timenet.types import Domain, License


def test_domain_values():
    assert Domain.CARDIOLOGY == "cardiology"
    assert Domain.HEALTH == "health"
    assert Domain.GENERAL == "general"


# Values outside the SPDX license list, which `canonicalize_license_expression` cannot resolve.
NON_SPDX_LICENSES = {License.OTHER, License.NONE}


@pytest.mark.parametrize("license", [lic for lic in License if lic not in NON_SPDX_LICENSES])
def test_license_values_are_canonical_spdx_ids(license):
    # A known SPDX ID in its canonical spelling, not a display string like "CC BY 4.0".
    assert canonicalize_license_expression(license.value) == license.value


def test_enums_are_str():
    assert isinstance(License.MIT, str)
    assert isinstance(Domain.HEALTH, str)
