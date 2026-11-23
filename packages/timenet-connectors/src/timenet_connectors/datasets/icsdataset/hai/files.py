"""The pinned HAI files: every data file and label file of versions 20.07, 21.03, 22.04 and 23.05.

Each file is pinned by URL and SHA-256 at one commit of the HAI repository. The SHA-256 of a Git
LFS file (22.04 and 23.05) is the ``oid`` of its LFS pointer at that commit.
"""

from dataclasses import dataclass
from pathlib import PurePosixPath


HAI_COMMIT = "2a814cebc9a66b06c9e5cd545e2d72e65d383737"
"""The commit of https://github.com/icsdataset/hai that every file comes from."""
_RAW_URL = f"https://raw.githubusercontent.com/icsdataset/hai/{HAI_COMMIT}"
# 22.04 and 23.05 sit in Git LFS. Their raw link returns a 133-byte pointer, not the data.
_LFS_URL = f"https://media.githubusercontent.com/media/icsdataset/hai/{HAI_COMMIT}"


@dataclass(frozen=True)
class VersionFormat:
    """How one HAI version publishes its data files."""

    url: str
    delimiter: str
    time_column: str
    flag_columns: tuple[str, ...]
    """HAI's own attack flags in the data files. 23.05 keeps them in separate label files."""


FORMATS = {
    "20.07": VersionFormat(_RAW_URL, ";", "time", ("attack", "attack_P1", "attack_P2", "attack_P3")),
    "21.03": VersionFormat(_RAW_URL, ",", "time", ("attack", "attack_P1", "attack_P2", "attack_P3")),
    "22.04": VersionFormat(_LFS_URL, ",", "timestamp", ("Attack",)),
    "23.05": VersionFormat(_LFS_URL, ",", "timestamp", ()),
}


@dataclass(frozen=True)
class SourceFile:
    """One pinned HAI file."""

    version: str
    filename: str
    sha256: str

    @property
    def stem(self) -> str:
        """The file name without its extensions, such as ``test1`` for ``test1.csv.gz``."""
        return self.filename.split(".", 1)[0]

    @property
    def path(self) -> str:
        """The path of the file in the HAI repository, such as ``hai-21.03/test1.csv.gz``."""
        return str(PurePosixPath(f"hai-{self.version}") / self.filename)

    @property
    def url(self) -> str:
        """The pinned download URL."""
        return f"{FORMATS[self.version].url}/{self.path}"

    @property
    def format(self) -> VersionFormat:
        """How the file's version publishes its data."""
        return FORMATS[self.version]


DATA_FILES = (
    SourceFile("20.07", "test1.csv.gz", "68cc18af44efcd756e9eaaa81790855836e8aee3f2ce9c4d9dd95a0ada36c2e4"),
    SourceFile("20.07", "test2.csv.gz", "427841d8240d5fa557fb28ed58da21bb93d0a3c2b30e43b82ab2d816330185e2"),
    SourceFile("20.07", "train1.csv.gz", "376e8288a5e3cf69bc5c506f75d6fb9cebb30ff43326c60ed675ccea37700844"),
    SourceFile("20.07", "train2.csv.gz", "30caafaeba9736238fe746a780c0baf3aa71ee7bcf494fb3daf721128ad7fdde"),
    SourceFile("21.03", "test1.csv.gz", "532de41bed372206941609d05b763958f59abecfd63c8c123649f823452e445e"),
    SourceFile("21.03", "test2.csv.gz", "c5572e0ed19906107ceb56c4b4f40e165153fc7ba87b3506230d53d2478c29bb"),
    SourceFile("21.03", "test3.csv.gz", "bff835894baff0c73871bc01ece4f5cf759d455a049e808b012128767420caaf"),
    SourceFile("21.03", "test4.csv.gz", "51380fd28fd985b94a10cae3d8b80926becb3af9f233f2d3a38ab8be9482a482"),
    SourceFile("21.03", "test5.csv.gz", "f0d59308b167ed95d0e81fb3e27c906682c9b21d3dba0ab4c4d696565df35179"),
    SourceFile("21.03", "train1.csv.gz", "fdd0162315b32817ba665e2cc8990eb09c47c4edc29683a86b11dc9df062b3a9"),
    SourceFile("21.03", "train2.csv.gz", "c330af06a24f5fe3df864dddc0e9cbbbfd27db2e005bd32dc8ab3eb2ab356337"),
    SourceFile("21.03", "train3.csv.gz", "de09120c617bad5f4f3cab86705137fe7ced4b1123fb67ffdf67fd645f3d1b8f"),
    SourceFile("22.04", "test1.csv", "19627f5dac40c3a5039e7468cfd6c6e4ccd37a85055671e8c349b6bce5f50ed1"),
    SourceFile("22.04", "test2.csv", "b2ba5d6665deee1c86870a3970417b51a3e640a2cc18b9f8de2d6d8a21979923"),
    SourceFile("22.04", "test3.csv", "f253f1c78a4a4654d843f8f670a24caa7a72ac91e4418b4505009822b3c55940"),
    SourceFile("22.04", "test4.csv", "2d2e99ed3364d75d27ab84f0a6523496f7295decf363c90b8a1b9b84e1070376"),
    SourceFile("22.04", "train1.csv", "63383690ffc5344ba68a019c7000854a4d341a22d743ac035f5207072a197b33"),
    SourceFile("22.04", "train2.csv", "f496991bbef4a6ef5881505aa242fe05cc500b4d5c4cea0d2cf1cf79e4e99285"),
    SourceFile("22.04", "train3.csv", "4910c49dc07cce238e05c449972a30b7588139ef92d36067f32aadd820e63de7"),
    SourceFile("22.04", "train4.csv", "81a1c3a8e1726c92c9253602b8628b72d0508a9c260ff21109433ad2b82687bf"),
    SourceFile("22.04", "train5.csv", "298468467991e73689b61f3cd3975ca21685ef8ba5b5ced4d6a7874f304d9cf7"),
    SourceFile("22.04", "train6.csv", "717b37b7b88b40885cb6994c9fe00d08a941ec0c3c73801e378f84e846ae35d4"),
    SourceFile("23.05", "hai-test1.csv", "78c7f1d4de1f2ab9ccc2f8c719f80f831033543adb0c81d0d78f84f40838d4be"),
    SourceFile("23.05", "hai-test2.csv", "b2b8dd295aefd87e39260fe43cb4c73ee86d6264b0ac4b0761e7efb0c2b545c3"),
    SourceFile("23.05", "hai-train1.csv", "53007b0ba604fbf338e7ac2e08cd81d874b5d1388f3aecb213ddcba5bf2bec4a"),
    SourceFile("23.05", "hai-train2.csv", "0e520e82bf78a661ab19ce4967f3c766bd809820f457a9c90c365102d4534c56"),
    SourceFile("23.05", "hai-train3.csv", "bfcec2dc05adea103e7491546b0e28268faaa26d3cc717d10f4595c94b81e85d"),
    SourceFile("23.05", "hai-train4.csv", "56658c83657d42a65db982b864362e0d0ffeb96d1f7b357d5e76e3a5c522d940"),
)
"""The 28 data files, as published: one row per second."""

LABEL_FILES = {
    "hai-test1": SourceFile(
        "23.05", "label-test1.csv", "eaf69edb9c5834bc393afd7bf658b5e408d34fd7bfc3261f80516765fb818fbc"
    ),
    "hai-test2": SourceFile(
        "23.05", "label-test2.csv", "8090c44981176e39b0f01a7126a80248ac0b93355c00f9db4d4e2f2106452b92"
    ),
}
"""HAI's attack labels of the two 23.05 test files, by the stem of the data file they label."""
