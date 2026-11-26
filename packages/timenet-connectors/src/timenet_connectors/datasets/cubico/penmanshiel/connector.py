"""The Penmanshiel connector: one Record for each turbine and UTC day, with labels from the status log.

The release has two zip archives for each year: one for turbines 01 to 10 and one for turbines 11
to 15. The farm has no turbine 03. This connector reads 2019 to 2022. The later years use another
table layout. The files have the format of the Kelmarsh release, so the Kelmarsh modules read them.
"""

from timenet_connectors.datasets.cubico.kelmarsh.farm import GreenbyteFarmConnector


class PenmanshielConnector(GreenbyteFarmConnector):
    """Connector for the Penmanshiel wind farm release on Zenodo."""

    FILES_URL = "https://zenodo.org/records/16807304/files"
    STATIC_FILE = ("Penmanshiel_WT_static.csv", "2a742dfd5c8486bf54c1afb647bd803954f4c56aea31635e5aa93828f56ebd7b")
    # Zenodo states an MD5 hash for each file. These SHA-256 hashes are of the files with those MD5 hashes.
    YEAR_ARCHIVES = {  # ruff: ignore[mutable-class-default]
        2019: (
            (
                "Penmanshiel_SCADA_2019_WT01-10_3112.zip",
                "c611f4d04fa9f42ce093db643dcb11d96a995810ca52a9b5fda76dc8f26281d7",
            ),
            (
                "Penmanshiel_SCADA_2019_WT11-15_3117.zip",
                "54523b9bffe717566f8ef056af0836e91ae2aed3e1a4083533f659a154a60eaf",
            ),
        ),
        2020: (
            (
                "Penmanshiel_SCADA_2020_WT01-10_3109.zip",
                "f2bbf7f4efa24c9bdcc98cb987b17830272d77067a43958cd62ee5e6c490efe1",
            ),
            (
                "Penmanshiel_SCADA_2020_WT11-15_3118.zip",
                "6c41960df290326831634aab2162c020a340f6ad439ed6eba2daa381ba707f31",
            ),
        ),
        2021: (
            (
                "Penmanshiel_SCADA_2021_WT01-10_4460.zip",
                "aa66084e5d06ee41242ff53c2eae990edf2ae271072ee70c67def448becfb97d",
            ),
            (
                "Penmanshiel_SCADA_2021_WT11-15_4461.zip",
                "7896dd7d314963efe4db589b6294a129d38a35773fc7b83c44ee23de73802cf6",
            ),
        ),
        2022: (
            (
                "Penmanshiel_SCADA_2022_WT01-10_4462.zip",
                "94df920978c8eded7f3c6e300943108b27454312b3172639053c0ba21a7dafcd",
            ),
            (
                "Penmanshiel_SCADA_2022_WT11-15_4463.zip",
                "d88e0e9d2af2409e23ef1c7a6de893f3bde53520068dcf36a92f402422ee03ad",
            ),
        ),
    }


CONNECTOR = PenmanshielConnector
