"""The SwissMetNet stations of the national weather series and the population that weights them."""

STATION_CANTONS: dict[str, tuple[str, ...]] = {
    "SMA": ("ZH", "SH"),  # Zürich / Fluntern
    "BUS": ("AG", "SO"),  # Buchs / Aarau
    "BAS": ("BS", "BL", "JU"),  # Basel / Binningen
    "BER": ("BE", "FR"),  # Bern / Zollikofen
    "NEU": ("NE",),  # Neuchâtel
    "PUY": ("VD",),  # Pully (Lausanne)
    "GVE": ("GE",),  # Genève / Cointrin
    "LUZ": ("LU", "ZG", "SZ", "OW", "NW", "UR"),  # Luzern
    "STG": ("SG", "AR", "AI", "TG"),  # St. Gallen
    "CHU": ("GR", "GL"),  # Chur
    "SIO": ("VS",),  # Sion
    "LUG": ("TI",),  # Lugano
}
"""Each station stands for the cantons listed with it."""

CANTON_POPULATION_2023: dict[str, int] = {
    "ZH": 1605508, "BE": 1063533, "LU": 432744, "UR": 37931, "SZ": 167403, "OW": 39272,
    "NW": 45016, "GL": 42056, "ZG": 132556, "FR": 341537, "SO": 286844, "BS": 200031,
    "BL": 298837, "SH": 87111, "AR": 56495, "AI": 16585, "SG": 535114, "GR": 204888,
    "AG": 726894, "TG": 295220, "TI": 357720, "VD": 845870, "VS": 365844, "NE": 178291,
    "GE": 524410, "JU": 74548,
}  # fmt: skip
"""Permanent resident population per canton at the end of 2023, Swiss Federal Statistical Office
(BFS) STATPOP, table px-x-0102010000_101. A test checks these values against the downloaded table."""
