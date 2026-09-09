"""The pinned Google Drive file table for the VerbalTS corpus, and the guard that checks a download.

Only :func:`check_drive_body` opens a file here. Nothing in this module decodes an array.

Drive states no revision and no digest of its own. So every file is pinned by its Drive id, its
byte count and its SHA-256. The 60 digests come from one full download of the release on 2026-09-06.

The download layer checks a digest only for a file it fetched, so a warm cache skips that check.
:func:`check_drive_body` re-reads every file after the download instead. It checks the byte count,
the NPY magic number and the digest.
"""

import hashlib
from pathlib import Path

from timenet.errors import TimeFFormatError


DRIVE_URL = "https://drive.usercontent.google.com/download?id={file_id}&export=download&confirm=t"
"""Direct-download template. ``confirm=t`` is what makes Drive serve the bytes for a large file
instead of the virus-scan interstitial."""

COMPONENTS: tuple[str, ...] = (
    "synthetic_u",
    "synthetic_m",
    "Weather",
    "BlindWays",
    "ETTm1",
    "istanbul_traffic",
)
"""The six components, in the order the VerbalTS paper introduces them in Section 5.1."""

SPLITS: tuple[str, ...] = ("train", "valid", "test")
"""The three splits every component ships."""

NPY_MAGIC = b"\x93NUMPY"

# (component, filename, drive file id, bytes, sha256)
FILES: tuple[tuple[str, str, str, int, str], ...] = (
    (
        "synthetic_u",
        "meta.json",
        "1CPlh0DRzg58BRoa47-bBavXTWpugfBdG",
        393,
        "432a42314388ed09815cf2d0fe5cafa7e53574b9f9312291811777bd90ce5fb5",
    ),
    (
        "synthetic_u",
        "train_ts.npy",
        "1qxpPGNLNj_aMFB-Z7PXwtmzNRA1Cngt_",
        24576128,
        "3834ab621d5f7094a482f97368a91216d9f759061e8e7523288f9af938319782",
    ),
    (
        "synthetic_u",
        "train_attrs_idx.npy",
        "12X6G5OsnHb-yR1RbzjfozfMrpa0rE4u8",
        576128,
        "1778030eca4dd00d7fc85726d931221640909348b6b5d427db78189f40c43912",
    ),
    (
        "synthetic_u",
        "train_text_caps.npy",
        "1cTWRIZFPaQPf0onGPJsdSDZE_g8LXln8",
        24480128,
        "53f31bcfa17fb2be157c24f19645966dfd7c16cb56a77d2b289f5913c96f8eda",
    ),
    (
        "synthetic_u",
        "valid_ts.npy",
        "1qf70pPW1M6mwPEsi-bKUOQ0F6pk1reas",
        4096128,
        "6b17e30428314f564751145e948b3564694860e3ce972bcfb14afb80e1763e80",
    ),
    (
        "synthetic_u",
        "valid_attrs_idx.npy",
        "1qb1jNKW6uekNQ1t0s9QBEf0oFVdVf6rd",
        96128,
        "b3c26d850732c2b6fc2c7e63218723f8a57b81fa05f5e9a4cfae88b9cf8b7921",
    ),
    (
        "synthetic_u",
        "valid_text_caps.npy",
        "18_TMFjTnGNtCn5ZlWMDcG0WaFgcI-dA1",
        4064128,
        "c5c3ac56552aabd3cce94e0fabb153d58ec9648b87274c8a675fecee2823791f",
    ),
    (
        "synthetic_u",
        "test_ts.npy",
        "1u0GKR6J6thb8ODxNmL7ftdIkdYhOa6Mr",
        4096128,
        "01a582b6868eb0818fbba3d3752e76f41b372b0fca0e7aa0e977dfd90c99f74d",
    ),
    (
        "synthetic_u",
        "test_attrs_idx.npy",
        "1MuanrJFqOrcTrKm_XhozXQHO2uKrNsj5",
        96128,
        "96f0ad388131915ef52c1602a1d03cc3539806c2cd7edde0205a06c7bd8db448",
    ),
    (
        "synthetic_u",
        "test_text_caps.npy",
        "18MbizU4fJ-hW6qV1AB1c4i9tzPgKpf9b",
        4048128,
        "7da1708906ac7b3a14eaa700bb8053796c0d9419083a74b89b4af065e80c02b0",
    ),
    (
        "synthetic_m",
        "meta.json",
        "1bRGQ-z38ZmuhDa6pnu8N-Y9Mi-Jr4bxY",
        427,
        "f2c3a2ee2b73265ec8c3f260852126e636be2412428e552d056388d9a2923002",
    ),
    (
        "synthetic_m",
        "train_ts.npy",
        "1V876tbygZrXIGOWCPLF3Hi3cc3sXZ11v",
        49152128,
        "7db31624462f8f206e3c27b2030aae67d7a1e42919fc96f35ea40442f43ee00d",
    ),
    (
        "synthetic_m",
        "train_attrs_idx.npy",
        "1t6JZpGSrqOb7m5GBSEzf22FEvd1njw5w",
        768128,
        "dbf910aef0ddc77484be8f7878e39c99cf383440e4684e250199f6cea8ff7ed2",
    ),
    (
        "synthetic_m",
        "train_text_caps.npy",
        "1NYDNp2puENF3HL81X2z_T1JP3l79dVTk",
        34752128,
        "38f2c4aa3cb4af286a17e65c585a5d1a99297470ad7ff3f3be820216fcb95c7b",
    ),
    (
        "synthetic_m",
        "valid_ts.npy",
        "1eS6pzPQ04RJKrBqldkIXWgrpMfb2p92n",
        8192128,
        "806c0ab62ba3c5f0bca42594a08cfcca9b190c149e32c7e168e3eaa87909309b",
    ),
    (
        "synthetic_m",
        "valid_attrs_idx.npy",
        "1L_l9CvhMkhwFaNJoPKA1kgUL_k0CF10e",
        128128,
        "5de3ae0c6c183b76c2c20ffbcbba56bd30d034a575ffdfc611c55ebfcbea2719",
    ),
    (
        "synthetic_m",
        "valid_text_caps.npy",
        "1hA_5S5fTGgxIUbGKm3jAU20oIbEYGKtX",
        5744128,
        "8abf293b7760d729e2fb83095e48a5d24d56f8ddcc69a37c9ed0ff79bb42005b",
    ),
    (
        "synthetic_m",
        "test_ts.npy",
        "1LCbVCGAP5BYNyJx4gUBnU6S0n56lVWEF",
        8192128,
        "75520a702374a521cbc21b2bbc0f3ee544f56e5add626be97b180c41d25b20ac",
    ),
    (
        "synthetic_m",
        "test_attrs_idx.npy",
        "1YCxy4J_iUGQtbUTMAQG40LzwfudPpob5",
        128128,
        "94f37a2d7ed60913fe39c32305974b7c51fd2883886d94f9798933e1f10504cc",
    ),
    (
        "synthetic_m",
        "test_text_caps.npy",
        "13NR5RDrKXp4ZVHxDUqadsJiK3_sGTF6v",
        5808128,
        "78cfaff486d2c46477f888b5b9de9ffecb53d8aaa633a000a6b520ac0c10e1c8",
    ),
    (
        "Weather",
        "meta.json",
        "14b618dmJ8IhGseW4OPlfmg1W6YN0Hnr8",
        455,
        "a97f186918e0e4ceb9618d2ea0fd121163a6710784c6eee6ffc744206cd36b5d",
    ),
    (
        "Weather",
        "train_ts.npy",
        "1qx-naeoJ83KIyFZtBoKkjLPBRsK5ep-D",
        61641344,
        "9c92fb7eecefc63cdfdc9d36f9cee715959da7d4d4888e6641f6a5d595ae52c9",
    ),
    (
        "Weather",
        "train_attrs_idx.npy",
        "1kW1U2rlC0WvCmwohHDa9j_-hm5jTyg9G",
        570880,
        "8fbb859de49e6c92c6149fe3374faf438a9154bbbb669787c6aba6c402e47f22",
    ),
    (
        "Weather",
        "train_text_caps.npy",
        "1YXl9zJ1xlaxVck-P0nNtlH7gs6G4fIJo",
        82310720,
        "4f6a431295be22b61edf0f8a787196c9fceaaad3a74e02271000fc63d5484bb4",
    ),
    (
        "Weather",
        "valid_ts.npy",
        "18-aGNSxowQ9LAcOMr3Wg4p_F55-Ua9EN",
        8830208,
        "dcd7ab55b4cefb1dd3ca7c90407536ae676ea0351b18bef16c030f8919b0a5ea",
    ),
    (
        "Weather",
        "valid_attrs_idx.npy",
        "1dLiotei4GYBgSMsqwCpN7Guv5kYRfFAR",
        81888,
        "ded2d52e10cf9a180b861a72d52d6d31681062f026795ec667012eb39fd94e9f",
    ),
    (
        "Weather",
        "valid_text_caps.npy",
        "18ISSIz0BIz6JTFgW2tQvKT_ILa-0CR_t",
        11633408,
        "127f122dfb09bbb4688ba1adbf4331bb6b40a80ba1967e89d5a437be6db64c09",
    ),
    (
        "Weather",
        "test_ts.npy",
        "18SOdqwhZnVicMJwrwNrF-sCWX5gMe85Y",
        8757632,
        "66420d6137aeb4622fbbff19665e198ccd2bd70b95c51e5003ae47d92bd60dba",
    ),
    (
        "Weather",
        "test_attrs_idx.npy",
        "12BqyqPtQ1drEzWMYXSSrEJEzACguGkmh",
        81216,
        "575bba8f0d66eff05ab661d1a695dcb43e3d893c79497d7e1af23babb91f0978",
    ),
    (
        "Weather",
        "test_text_caps.npy",
        "1rrEm8saMf_buGG64OALzR_Y2yvzwjLdH",
        10599488,
        "1d2bbf8ae5ab1657dbc57807669627b695099ec6b88126b060fdf1409089840f",
    ),
    (
        "BlindWays",
        "meta.json",
        "1-iJbFDBC_yw9WBxI56JZO8qIZuiroumR",
        331,
        "110526eb7e9803d8d682e1a1b3324febec1992c52e5457def7518471a83075d5",
    ),
    (
        "BlindWays",
        "train_ts.npy",
        "19DpoqjLtna2sbgQ4Ufj20BenAN91knvT",
        284428928,
        "bf9baab60f600f6c6ec71177984a9b6bb198ef91a574c3c420a1860738bf0f5f",
    ),
    (
        "BlindWays",
        "train_attrs_idx.npy",
        "1IuBRkO9cWdTv-mpwNqfL2kFKPDCQwUwZ",
        13296,
        "fb06926daed0c3b6d8479490a12e8038961c04347791b36bac267d1d2137878a",
    ),
    (
        "BlindWays",
        "train_text_caps.npy",
        "1w1vH2TN1kHmcuGWVgnST8lCbAhNGhb02",
        1349848,
        "6d7c3052d32f2b2f76b7c0dbd034abb3e834e48cf2d3b6c306e2a02c5a8d7358",
    ),
    (
        "BlindWays",
        "valid_ts.npy",
        "132aG3622qv8e7qVXiFYFH7LTNPRTj9lq",
        35596928,
        "aa1950759cfb11d0fe2f6190d57813c10e0dd69f44cda53aca3a59d01da9972c",
    ),
    (
        "BlindWays",
        "valid_attrs_idx.npy",
        "130kOnkWYud2WnIFCyITSmW7mDyxxkoXz",
        1776,
        "88407106b32d5c34a792ead923693dc477cbe8e5959a03c62078229f32693087",
    ),
    (
        "BlindWays",
        "valid_text_caps.npy",
        "1HoFDxwkom6t-NM1GznRLUFBka5cbqX9b",
        240324,
        "7d85d2721eb6cd43706d765708b7e6cfdecf0f34c4a4d336641fa1f8f8d1fabd",
    ),
    (
        "BlindWays",
        "test_ts.npy",
        "1N5MsAo4JSumJC0aSaOhw4IO1MFAwyOxe",
        35596928,
        "df7f67c34cefbcef59a93d882396e8b1f2e30459e112e1776e3710900781453e",
    ),
    (
        "BlindWays",
        "test_attrs_idx.npy",
        "1wJkCiAOFGoBxlCRioUJr1pc8l8SCHjni",
        1776,
        "6464cd8255870634f217642b863cd2646d62b270733c570a342f467f842a30eb",
    ),
    (
        "BlindWays",
        "test_text_caps.npy",
        "1JMUEtSGrxdPduRZp4WGQg-aB2B8HqpFq",
        135676,
        "1b99b9d6f4f6e6178af54d7abb90c791e8f39cfa8c98c759546e2c39c42e6d83",
    ),
    (
        "ETTm1",
        "meta.json",
        "1PB4vwskyre7FCTbEZvOGhlvqfcSh3QBl",
        432,
        "507ecbf731d59d0ba3516ed1baae286a5ad6b09fa5f098f7b45f88a250fd5e4d",
    ),
    (
        "ETTm1",
        "train_ts.npy",
        "1FU9FjGLGZIKaIF3RKYwVv3RPLQn2lMC-",
        12492608,
        "c3ca03d2ec32ea30b84c13d72d63f19dff4dd020102eb90743a6ac95d2dff948",
    ),
    (
        "ETTm1",
        "train_attrs_idx.npy",
        "1voHkTCkPbkG1SO-wk2sT5X5Ix6mknxCG",
        520648,
        "82db6e83fbc767066ef9022d7ee656f3564c780ff3e30aa3fb413eab1cb953b7",
    ),
    (
        "ETTm1",
        "train_text_caps.npy",
        "1ts7FLh58sYq_Pk3VR0wfj9YFJSPEyM8_",
        25505608,
        "244ec8a9d3caa070b2b4b3aec9f3145afe62de88a5eafbfe0b8e0e73e3056922",
    ),
    (
        "ETTm1",
        "valid_ts.npy",
        "1YBNAeRvRnf0FDSIyYLFDJ-j1jOjWnyLz",
        1565888,
        "bbb8d48668780ef64a66d5f8f85b3b99daeb5fa31f70c99947ff0afa775e4292",
    ),
    (
        "ETTm1",
        "valid_attrs_idx.npy",
        "1d_8E-HBfrQ0_H11tcFSz1bQ13JWCIGk-",
        65368,
        "d9cc455acfe7d98cb2e74b8bce3612335ebc2b4e6280c5c307306769ec1de053",
    ),
    (
        "ETTm1",
        "valid_text_caps.npy",
        "1M7EKmHkC0EwB91QVyqniPxQ94fTlmt8p",
        3092504,
        "bfb833c11dcad43eb58244106ac3f03df55364cd7c7ac0f30b9f76c405e6f2ce",
    ),
    (
        "ETTm1",
        "test_ts.npy",
        "1K33MPY9RV46gl636rEDwDM8xijxPWEC-",
        1565888,
        "d4a423f4b80b2073214aacc0b9704ea921d2a30d94b54dfffa3005c30991751a",
    ),
    (
        "ETTm1",
        "test_attrs_idx.npy",
        "1m72hCOc4tj6zWaCpDU6ukT2HWDzadEaQ",
        65368,
        "f5cfb7e07dc6933968094e98cf87e4e2be605fa9cd42a7e3969bae04875ddae2",
    ),
    (
        "ETTm1",
        "test_text_caps.npy",
        "1PGfF2WqACoewcp-tt8u1k_E9O3fgxGaY",
        3066408,
        "162e17b169fef013d1939999ccd3459ec7c63f1d9acf1655fb9ac22f51c0b03c",
    ),
    (
        "istanbul_traffic",
        "meta.json",
        "1xEqajAgFW4sbpNot0TRU99-xlROhZtmy",
        430,
        "77c5b4734b2b361fd58b1aec30e061e1ef0a50c98ae32bad566ff22ecc6519dc",
    ),
    (
        "istanbul_traffic",
        "train_ts.npy",
        "1Kg7r4UdFDP-Z29_vE2yZvW2ZwOkaBbtp",
        9421184,
        "04d7e0a5d79186f2ce9f9346213bb35adb9026800b7708e83719ce30179a61c3",
    ),
    (
        "istanbul_traffic",
        "train_attrs_idx.npy",
        "1ibY-G9covX1x1N6QZFz2GKdl1KIkWkAP",
        327248,
        "45d44e262f52271e29b6971f2f6ef2f01b6286be6dbb313a4c2d5dd071f3ba14",
    ),
    (
        "istanbul_traffic",
        "train_text_caps.npy",
        "19azAHZ7q8j7yqxiXmvvOH0zqL2ndPDU-",
        15342056,
        "3dd6683854dcc0a388c3f2f8e99ab368988169b40b2182a2fb2102756634e816",
    ),
    (
        "istanbul_traffic",
        "valid_ts.npy",
        "1Fthl8BtyNN7gXq7cbkzdzQ7yDViwbAsy",
        1178624,
        "bb71a824a0648f46a0298a3d5266a8f98b870a009344cdec98ea8b000f4665fe",
    ),
    (
        "istanbul_traffic",
        "valid_attrs_idx.npy",
        "1S8Ai2gAhOimYRmyT3xr4urpuDkAq9A9U",
        41048,
        "b84a8a526f3011392630308a43ba97777556c377d52052e56c033fc68a87419a",
    ),
    (
        "istanbul_traffic",
        "valid_text_caps.npy",
        "1DAmPActhIdyYtOGgQRjO6kw6tcoOG661",
        1825160,
        "e4a51f5dce92a9097d2e8819874c8828c6064f304f4fc9f98275b09b55c8fb58",
    ),
    (
        "istanbul_traffic",
        "test_ts.npy",
        "1KdBRcLsyZBTJQ19cHwVbJLhfQIAgJRMn",
        1178624,
        "57468d2ed5d670086348c8ca0993cb656008e4fd449d84e47f1c4c472d706e1b",
    ),
    (
        "istanbul_traffic",
        "test_attrs_idx.npy",
        "18y8gVqlQ7ADduhAQZkDF30NTwY4HjqOD",
        41048,
        "5dba1bf78b305d30754006aaafc1ac0306f39ea5a8881511cc38f994a8886658",
    ),
    (
        "istanbul_traffic",
        "test_text_caps.npy",
        "1kSo60iFYO28eVbZIAKdQN-lmjiYi8HRh",
        1833344,
        "6b0a72b509d2867d942f896c373a7000b40b23daad0a4efe884f149ac99af2a2",
    ),
)


_HASH_CHUNK_BYTES = 1 << 22


def check_drive_body(path: Path, expected_bytes: int, expected_sha256: str) -> None:
    """Check one downloaded file against its pinned size, NPY magic and SHA-256.

    Drive answers a large-file download without ``confirm=t`` with a small HTML page under HTTP 200.
    The download layer sees a success and writes that page to disk. The size check catches it, and
    the NPY magic number catches any other body that is not an array. The digest catches what those
    two cannot see: bytes replaced in place at the same Drive id and the same length.

    The digest is checked here and not left to the download layer. That layer skips a file that is
    already on disk, so a warm cache never re-reads it.

    Args:
        path: The downloaded file.
        expected_bytes: The pinned size of that file.
        expected_sha256: The pinned SHA-256 of that file.

    Raises:
        TimeFFormatError: If the file is missing, has the wrong size, is an ``.npy`` file without
            NPY magic, or does not match the pinned digest.
    """
    if not path.is_file():
        raise TimeFFormatError(f"{path} is missing after the download, so its pinned bytes cannot be checked")
    actual = path.stat().st_size
    if actual != expected_bytes:
        raise TimeFFormatError(
            f"{path.name} is {actual} bytes, expected {expected_bytes}. Google Drive returns its "
            f"virus-scan interstitial as HTTP 200 text/html when 'confirm=t' is dropped; delete "
            f"the file and run the download again"
        )
    if path.suffix == ".npy":
        with path.open("rb") as handle:
            if handle.read(len(NPY_MAGIC)) != NPY_MAGIC:
                raise TimeFFormatError(f"{path.name} has the right size but is not an NPY file")
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(_HASH_CHUNK_BYTES):
            digest.update(chunk)
    if digest.hexdigest() != expected_sha256:
        raise TimeFFormatError(
            f"{path.name} has the pinned size but SHA-256 {digest.hexdigest()}, expected "
            f"{expected_sha256}. The Drive file was replaced since the table was pinned; do not "
            f"build from these bytes"
        )
