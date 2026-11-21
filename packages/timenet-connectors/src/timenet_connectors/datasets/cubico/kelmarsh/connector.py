"""The Kelmarsh connector: one Record for each turbine and UTC day, with labels from the status log.

The release has one zip archive for each year, with the files of the six turbines. This connector
reads 2019 to 2022. The later years use another table layout.
"""

from timenet_connectors.datasets.cubico.kelmarsh.farm import GreenbyteFarmConnector


class KelmarshConnector(GreenbyteFarmConnector):
    """Connector for the Kelmarsh wind farm release on Zenodo."""

    FILES_URL = "https://zenodo.org/records/16807551/files"
    STATIC_FILE = ("Kelmarsh_WT_static.csv", "a249919471580244d41a547e407ef371d45f2d5347a59bc23cf3c980f607ee93")
    # Zenodo states an MD5 hash for each file. These SHA-256 hashes are of the files with those MD5 hashes.
    YEAR_ARCHIVES = {  # ruff: ignore[mutable-class-default]
        2019: (("Kelmarsh_SCADA_2019_3085.zip", "c2f11578b9a1678be198fd7883a7dfe1b923c5b0565f806bdda5dd9d0097f4dc"),),
        2020: (("Kelmarsh_SCADA_2020_3086.zip", "1cf3ad08aaed0952055b3668e39f73bbea8e09654d28b104ad2c4041b256452a"),),
        2021: (("Kelmarsh_SCADA_2021_4456.zip", "9b9aed27db92ede492d83d59083c2d4463d74716746606adcb08b2b3bb393ea6"),),
        2022: (("Kelmarsh_SCADA_2022_4457.zip", "c897b1178f1a12e18b1b718c9eada52b58775a1345945265014817be3b2be79b"),),
    }


CONNECTOR = KelmarshConnector
