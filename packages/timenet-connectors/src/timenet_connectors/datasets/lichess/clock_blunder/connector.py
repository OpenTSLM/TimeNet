"""The Lichess Clock-Blunder connector: real games as clock and evaluation series, and a blunder question.

The Lichess open database ships every rated game as PGN. Since 2017 each move carries the mover's clock
(``[%clk 0:04:59]``), and analysed games also carry the Stockfish evaluation after the move
(``[%eval 0.27]`` in pawns, or ``[%eval #-3]`` for a forced mate). This connector reads a fixed byte range
at the start of the April 2017 dump, keeps the complete games that have both on every ply, and turns each
game into a few windows. A window holds the last plies before one player's move, and its task asks
whether that player blunders within their next three moves.

A blunder follows Lichess's own rule: the move drops the mover's winning chances by at least 0.3, where
winning chances are ``2 / (1 + exp(-0.00368208 * centipawns)) - 1`` and a forced mate counts as +1 or -1.
"""

from collections.abc import Iterator, Sequence
from dataclasses import dataclass
import hashlib
import io
import math
from pathlib import Path
import re
from typing import Self

import numpy as np

from timenet.connectors import BaseConnector
from timenet.dataset import Record, Signal, Source, TimeFDataset
from timenet.dataset.axis import OrdinalAxis
from timenet.errors import TimeNetConnectorError
from timenet.types import Annotation, ClassificationTask, Label, LabelSet, Split, TimeSeriesSpec, ureg
from timenet_connectors.download import Artifact, download_files


DUMP_URL = "https://database.lichess.org/standard/lichess_db_standard_rated_2017-04.pgn.zst"
PREFIX_BYTES = 64 * 1024 * 1024
"""Only this many bytes at the start of the 3.5 GB dump are read, so every build sees the same games."""
PREFIX_SHA256 = "d53e00e3cf7098057f136c2937d97f0cdd45963c52362565cbd2008814c99d1b"

WINDOW_PLIES = 40
"""A window holds at most this many plies before the decision point."""
MIN_DECISION_PLY = 12
"""No window ends inside the first six moves of each side: the opening is mostly preparation."""
HORIZON_MOVES = 3
"""The task asks about the mover's next three moves, so their plies p, p + 2 and p + 4."""
WINDOWS_PER_GAME = 4
TRAIN_BUCKETS = 8
"""Of the ten buckets a game id hashes into, the first eight are train, then one validation, one test."""
BLUNDER_DROP = 0.3
"""Lichess's blunder threshold on the -1 to +1 winning-chances scale."""
MATE_PAWNS = 100.0
"""A forced mate, stored in pawns. Its winning chances round to exactly +1 or -1."""
EVAL_CLIP_PAWNS = 10.0
"""The stored evaluation is clipped here, so one mate does not dwarf the rest of the window."""

BLUNDER = LabelSet(
    name="blunder_within_three_moves",
    labels=(
        Label(name="no", description="None of the player's next three moves is a blunder."),
        Label(name="yes", description="At least one of the player's next three moves is a blunder."),
    ),
)
CLOCK = TimeSeriesSpec(spec_type="chess_clock", name="Chess clock", unit_value=ureg.second, dtype="float64")
EVALUATION = TimeSeriesSpec(
    spec_type="engine_evaluation", name="Engine evaluation", unit_value=ureg.dimensionless, dtype="float64"
)
PER_PLY = OrdinalAxis()

_HEADER = re.compile(r'^\[(\w+) "(.*)"\]$', re.MULTILINE)
_COMMENT = re.compile(r"\{([^}]*)\}")
_EVAL = re.compile(r"\[%eval (#?-?[\d.]+)\]")
_CLOCK = re.compile(r"\[%clk (\d+):(\d\d):(\d\d)\]")
_RESULT = re.compile(r"(1-0|0-1|1/2-1/2)\s*$")
_TIME_CONTROL = re.compile(r"^(\d+)\+(\d+)$")


def winning_chances(pawns: float) -> float:
    """Return Lichess's winning chances for White, from -1 to +1.

    Args:
        pawns: The evaluation in pawns from White's view. A mate is ``±MATE_PAWNS``.

    Returns:
        The winning chances, -1 for a lost position and +1 for a won one.
    """
    if abs(pawns) >= MATE_PAWNS:
        return math.copysign(1.0, pawns)
    return 2.0 / (1.0 + math.exp(-0.00368208 * pawns * 100.0)) - 1.0


def parse_eval(text: str) -> float:
    """Parse a PGN ``%eval`` value into pawns from White's view, a mate becoming ``±MATE_PAWNS``.

    Returns:
        The evaluation in pawns.

    Raises:
        TimeNetConnectorError: If the value is not a number or a mate.
    """
    try:
        if text.startswith("#"):
            # "#-3" is a mate for Black. "#0" never appears after a move, but treat it as decided.
            return -MATE_PAWNS if text[1:].startswith("-") else MATE_PAWNS
        return float(text)
    except ValueError as exc:
        raise TimeNetConnectorError(f"evaluation {text!r} is not a number or a mate") from exc


@dataclass(frozen=True)
class Game:
    """One complete game: its tags and, for every ply, the mover's clock and the evaluation after it."""

    game_id: str
    time_control: str
    white_elo: int
    black_elo: int
    result: str
    clocks: np.ndarray
    """The mover's clock in seconds after each ply. Even plies are White's moves."""
    evals: np.ndarray
    """The evaluation in pawns from White's view after each ply."""

    @property
    def n_plies(self) -> int:
        """The number of plies."""
        return len(self.clocks)

    @classmethod
    def from_pgn(cls, text: str) -> Self | None:
        """Parse one PGN game, or return ``None`` when it lacks something the dataset needs.

        A kept game is rated with a ``base+increment`` time control, finished with a result, and has
        exactly one clock and one evaluation on every ply.

        Returns:
            The game, or ``None``.
        """
        tags = dict(_HEADER.findall(text))
        site, control = tags.get("Site", ""), tags.get("TimeControl", "")
        if not site.startswith("https://lichess.org/") or not _TIME_CONTROL.match(control):
            return None
        movetext = text.partition("\n\n")[2]  # the tag pairs end at the first blank line
        if not _RESULT.search(movetext):
            return None  # a game cut by the end of the byte range has no result token
        comments = _COMMENT.findall(movetext)
        evals = [found for c in comments if (found := _EVAL.search(c))]
        clocks = [found for c in comments if (found := _CLOCK.search(c))]
        if not comments or len(evals) != len(comments) or len(clocks) != len(comments):
            return None
        # One comment per ply: count the move tokens outside the comments to be sure none is missing.
        bare = _COMMENT.sub(" ", movetext)
        n_moves = len([t for t in bare.split() if not t.endswith(".") and not _RESULT.match(t)])
        if n_moves != len(comments):
            return None
        try:
            white_elo, black_elo = int(tags["WhiteElo"]), int(tags["BlackElo"])
        except (KeyError, ValueError):
            return None
        return cls(
            game_id=site.rsplit("/", 1)[-1],
            time_control=control,
            white_elo=white_elo,
            black_elo=black_elo,
            result=tags.get("Result", ""),
            clocks=np.array([int(h) * 3600 + int(m) * 60 + int(s) for h, m, s in (c.groups() for c in clocks)], float),
            evals=np.array([parse_eval(e.group(1)) for e in evals], float),
        )

    def blunders(self) -> np.ndarray:
        """Return, for each ply, whether that move is a blunder by the player who made it.

        The position before the first move counts as level.

        Returns:
            One boolean per ply.
        """
        chances = np.array([winning_chances(e) for e in self.evals])
        before = np.concatenate(([0.0], chances[:-1]))
        mover_sign = np.where(np.arange(self.n_plies) % 2 == 0, 1.0, -1.0)
        return mover_sign * (before - chances) >= BLUNDER_DROP

    def series_until(self, ply: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return both clocks and the evaluation for the plies before ``ply``, at most ``WINDOW_PLIES``.

        A clock only changes on its owner's moves, so between them it keeps its last value. Before a
        player's first move, their clock is their first recorded reading. That is not always the base
        time of the control: a player who berserks in a Lichess tournament starts with half of it.

        Returns:
            White's clock, Black's clock and the clipped evaluation, one value per ply of the window.
        """
        white, black = np.empty(ply), np.empty(ply)
        last_white, last_black = float(self.clocks[0]), float(self.clocks[1])
        for index in range(ply):
            if index % 2 == 0:
                last_white = self.clocks[index]
            else:
                last_black = self.clocks[index]
            white[index], black[index] = last_white, last_black
        start = max(0, ply - WINDOW_PLIES)
        evals = np.clip(self.evals[start:ply], -EVAL_CLIP_PAWNS, EVAL_CLIP_PAWNS)
        return white[start:], black[start:], evals

    def decision_plies(self) -> list[int]:
        """Pick up to ``WINDOWS_PER_GAME`` decision plies, seeded by the game id.

        A decision ply leaves room for the mover's next three moves, so every answer is known.

        Returns:
            The chosen plies, in increasing order.
        """
        last = self.n_plies - 2 * (HORIZON_MOVES - 1)
        candidates = list(range(MIN_DECISION_PLY, last))
        if not candidates:
            return []
        seed = int.from_bytes(hashlib.sha256(self.game_id.encode()).digest()[:8], "big")
        rng = np.random.default_rng(seed)
        count = min(WINDOWS_PER_GAME, len(candidates))
        return sorted(int(p) for p in rng.choice(candidates, size=count, replace=False))

    @property
    def split(self) -> Split:
        """The split of every window of this game: 80% train, 10% validation, 10% test, by game id."""
        bucket = hashlib.sha256(self.game_id.encode()).digest()[-1] % 10
        return Split.TRAIN if bucket < TRAIN_BUCKETS else Split.VALIDATION if bucket == TRAIN_BUCKETS else Split.TEST


def _clock_text(seconds: float) -> str:
    minutes, rest = divmod(round(seconds), 60)
    return f"{minutes}:{rest:02d}"


def rationale(game: Game, ply: int, answer: bool) -> str:
    """Write the template rationale of a window from its own numbers and its known answer.

    It is generated supervision: it states what the series show, then the answer. It never claims the
    clock caused the outcome.

    Returns:
        The rationale text, ending with ``Answer: yes`` or ``Answer: no``.
    """
    side = "White" if ply % 2 == 0 else "Black"
    own = game.clocks[ply % 2 : ply : 2]
    other = game.clocks[1 - ply % 2 : ply : 2]
    increment = int(game.time_control.split("+")[1])
    # The time a move took is the clock before it, plus the increment, minus the clock after it.
    spent = -np.diff(own) + increment
    recent = spent[-5:]
    eval_now = float(np.clip(game.evals[ply - 1], -EVAL_CLIP_PAWNS, EVAL_CLIP_PAWNS))
    mover_eval = eval_now if side == "White" else -eval_now
    swing = float(np.ptp(np.clip(game.evals[max(0, ply - 10) : ply], -EVAL_CLIP_PAWNS, EVAL_CLIP_PAWNS)))
    own_left, other_left = float(own[-1]), float(other[-1])
    moves_left_at_pace = own_left / max(float(np.mean(recent)), 1.0)
    parts = [
        f"{side} is to move with {_clock_text(own_left)} left against {_clock_text(other_left)}.",
        f"Over the last {len(recent)} moves {side} spent {np.mean(recent):.1f} seconds per move on average,"
        f" enough for about {moves_left_at_pace:.0f} more moves at that pace.",
        f"The evaluation is {mover_eval:+.1f} pawns from {side}'s side, and it swung {swing:.1f} pawns over"
        " the last ten plies.",
    ]
    if answer:
        parts.append(f"{side} does blunder within the next three moves.")
    else:
        parts.append(f"None of {side}'s next three moves is a blunder.")
    return " ".join(parts) + f" Answer: {'yes' if answer else 'no'}"


def prompt(game: Game, ply: int) -> str:
    """Write the question of a window.

    Returns:
        The prompt text.
    """
    side = "White" if ply % 2 == 0 else "Black"
    return (
        f"Rated Lichess game, time control {game.time_control} (seconds + increment), White rated "
        f"{game.white_elo}, Black rated {game.black_elo}. The series give White's clock and Black's clock in "
        "seconds and the engine evaluation in pawns from White's view, one value per ply, for the plies "
        f"before move {ply // 2 + 1}. {side} is to move. Will {side} blunder within their next three moves, "
        "that is, make a move that loses at least 0.3 winning chances on Lichess's -1 to +1 scale? "
        "Answer yes or no."
    )


@dataclass(frozen=True)
class DumpPrefix:
    """The downloaded byte range at the start of the compressed dump."""

    path: Path

    def iter_games(self) -> Iterator[Game]:
        """Decompress the prefix and yield every game the dataset keeps, in dump order.

        The byte range ends inside a zstd frame, so decoding stops at the cut, with or without an error.
        Everything decoded before it is kept, and the last, cut game has no result token, so it is dropped.

        Yields:
            Each kept game.

        Raises:
            ImportError: If ``zstandard``, declared in this connector's requirements, is not installed.
        """
        try:
            import zstandard  # ruff: ignore[import-outside-top-level]
        except ImportError as exc:
            raise ImportError(
                "reading the Lichess dump needs zstandard, declared in this connector's "
                "requirements.txt. Run the build without --no-isolation, or install it yourself"
            ) from exc

        decoded = io.BytesIO()
        with self.path.open("rb") as raw, zstandard.ZstdDecompressor().stream_reader(raw) as reader:
            try:
                while chunk := reader.read(1 << 20):
                    decoded.write(chunk)
            except zstandard.ZstdError:
                pass  # the cut at the end of the byte range
        text = decoded.getvalue().decode("utf-8", errors="replace")
        for block in re.split(r"\n\n(?=\[Event )", text):
            if (game := Game.from_pgn(block.strip())) is not None:
                yield game


def create_window(game: Game, ply: int) -> tuple[Record, ClassificationTask]:
    """Create the Record and the task of one window.

    Returns:
        The Record and the task, not yet added to a dataset.
    """
    white, black, evals = game.series_until(ply)
    record = Record(
        sources=(
            Source(
                name="Clocks",
                signals=(
                    Signal.from_values(white, spec=CLOCK, name="white_clock", time_axis=PER_PLY),
                    Signal.from_values(black, spec=CLOCK, name="black_clock", time_axis=PER_PLY),
                ),
            ),
            Source(
                name="Engine",
                signals=(Signal.from_values(evals, spec=EVALUATION, name="evaluation_pawns", time_axis=PER_PLY),),
            ),
        ),
        annotations=(
            Annotation(key="time_control", value=game.time_control),
            Annotation(key="white_elo", value=game.white_elo),
            Annotation(key="black_elo", value=game.black_elo),
            Annotation(key="side_to_move", value="white" if ply % 2 == 0 else "black"),
            Annotation(key="ply", value=ply),
        ),
        metadata={"lichess_game_id": game.game_id},
    )
    answer = bool(game.blunders()[ply : ply + 2 * HORIZON_MOVES : 2].any())
    task = ClassificationTask(
        inputs=(record,),
        targets=("yes" if answer else "no",),
        label_set=BLUNDER,
        prompt=prompt(game, ply),
        rationale=rationale(game, ply, answer),
        split=game.split,
    )
    return record, task


class ClockBlunderConnector(BaseConnector[DumpPrefix]):
    """Connector for windows of real Lichess games with clocks and evaluations."""

    async def download_async(self, cache_dir: Path) -> list[DumpPrefix]:  # ruff: ignore[no-self-use]
        """Download the fixed byte range at the start of the April 2017 dump once.

        Args:
            cache_dir: The directory that holds the byte range.

        Returns:
            The one byte range.
        """
        artifact = Artifact(
            DUMP_URL,
            cache_dir / "lichess_db_standard_rated_2017-04.prefix.pgn.zst",
            headers={"Range": f"bytes=0-{PREFIX_BYTES - 1}"},
            sha256=PREFIX_SHA256,
        )
        (path,) = await download_files([artifact])
        return [DumpPrefix(path)]

    def convert(self, raw_refs: Sequence[DumpPrefix]) -> TimeFDataset:
        """Create one Record and one ClassificationTask for each window of each kept game.

        Args:
            raw_refs: The byte range from :meth:`download_async`.

        Returns:
            The populated dataset.

        Raises:
            TimeNetConnectorError: If the byte range holds no usable game.
        """
        dataset = TimeFDataset(metadata=self.metadata())
        windows = 0
        for prefix in raw_refs:
            for game in prefix.iter_games():
                for ply in game.decision_plies():
                    record, task = create_window(game, ply)
                    dataset.add_record(record=record)
                    dataset.add_task(task=task)
                    windows += 1
        if windows == 0:
            raise TimeNetConnectorError("the dump prefix held no complete game with clocks and evaluations")
        return dataset


CONNECTOR = ClockBlunderConnector
