"""The :class:`License` enum: the legal license of a dataset's source data.

Values are SPDX-style identifiers that mirror GitHub's license keyword list, not display strings.
This lets them round-trip through the manifest without ambiguity.
"""

from enum import StrEnum, unique


@unique
class License(StrEnum):
    """The legal license of the source data. Values are SPDX-style identifiers."""

    MIT = "MIT"
    APACHE_2_0 = "Apache-2.0"
    BSD_2_CLAUSE = "BSD-2-Clause"
    BSD_3_CLAUSE = "BSD-3-Clause"
    GPL_3_0 = "GPL-3.0"
    LGPL_3_0 = "LGPL-3.0"
    MPL_2_0 = "MPL-2.0"
    CC0_1_0 = "CC0-1.0"
    CC_BY_4_0 = "CC-BY-4.0"
    CC_BY_SA_4_0 = "CC-BY-SA-4.0"
    CC_BY_NC_4_0 = "CC-BY-NC-4.0"
    ODC_BY_1_0 = "ODC-By-1.0"
    ODBL_1_0 = "ODbL-1.0"
    OTHER = "other"
    """Escape hatch for a license outside this list; requires a ``license_url`` on the card."""
