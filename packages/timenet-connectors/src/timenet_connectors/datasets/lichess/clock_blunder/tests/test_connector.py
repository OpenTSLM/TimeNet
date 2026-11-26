from pathlib import Path

import numpy as np
import pytest
import zstandard

from timenet.dataset import TimeFDataset
from timenet.errors import TimeNetConnectorError
from timenet.reader import TimeFReader
from timenet.registry import open_local_version
from timenet.testing import assert_datasets_equal
from timenet.types import ClassificationTask, Split
from timenet.types.text import read_value
from timenet_connectors.datasets.lichess.clock_blunder.connector import (
    BLUNDER,
    DUMP_URL,
    MIN_DECISION_PLY,
    PREFIX_BYTES,
    PREFIX_SHA256,
    WINDOW_PLIES,
    WINDOWS_PER_GAME,
    ClockBlunderConnector,
    DumpPrefix,
    Game,
    create_window,
    parse_eval,
    prompt,
    winning_chances,
)


_MOVES = ("Nf3", "Nf6", "Ng1", "Ng8")
"""Both knights go out and back, so the moves are legal chess at any length."""


def _pgn(  # ruff: ignore[too-many-arguments]
    game_id: str,
    evals: list[str | None],
    clocks: list[int],
    *,
    time_control: str = "300+0",
    result: str = "1-0",
    white_elo: str = "1500",
    site: str = "https://lichess.org/",
) -> str:
    # Synthetic games in the shape of the Lichess dump: tag pairs, a blank line, then one comment
    # with the evaluation and the clock after each move.
    tags = {
        "Event": "Rated Blitz game",
        "Site": site + game_id,
        "White": "Manvith",
        "Black": "Twisha",
        "Result": result,
        "WhiteElo": white_elo,
        "BlackElo": "1480",
        "TimeControl": time_control,
        "Termination": "Normal",
    }
    moves = []
    for ply, (evaluation, clock) in enumerate(zip(evals, clocks, strict=True)):
        number = f"{ply // 2 + 1}{'.' if ply % 2 == 0 else '...'}"
        notes = ([] if evaluation is None else [f"[%eval {evaluation}]"]) + [
            f"[%clk {clock // 3600}:{clock % 3600 // 60:02d}:{clock % 60:02d}]"
        ]
        moves.append(f"{number} {_MOVES[ply % 4]} {{ {' '.join(notes)} }}")
    header = "\n".join(f'[{key} "{value}"]' for key, value in tags.items())
    return f"{header}\n\n{' '.join([*moves, result])}\n"


def _clocks(n_plies: int, *, start: int = 300, white_step: int = 4, black_step: int = 2) -> list[int]:
    # White spends white_step seconds on each move after the first, Black black_step.
    return [start - (white_step if ply % 2 == 0 else black_step) * (ply // 2) for ply in range(n_plies)]


# 26 plies with an increment. White walks into a forced mate on ply 18 and Black lets it go on ply 21.
# Those are the only two blunders: 0.3 pawns is 0.055 winning chances, a mate for Black is -1.
_MATE_EVALS: list[str | None] = [*["0.3"] * 18, "#-3", "#-2", "#-2", *["0.0"] * 5]
_MATE_PGN = _pgn("mateGame", _MATE_EVALS, _clocks(26), time_control="300+3", result="1/2-1/2")
# The same game until Black's move on ply 21, where Black now keeps the mate.
_MATE_KEPT_PGN = _pgn("mateGame", [*_MATE_EVALS[:21], *["#-2"] * 5], _clocks(26), time_control="300+3", result="0-1")
# 20 quiet plies. Both players berserked: half the base time on the clock from the first move.
_BERSERK_PGN = _pgn("berserk2", ["0.2"] * 20, _clocks(20, start=150, white_step=3, black_step=4))
_LONG_PGN = _pgn("longGame", ["0.2"] * 60, _clocks(60, white_step=1, black_step=1), time_control="600+0")
# 16 plies: no ply at or after MIN_DECISION_PLY leaves the mover three more moves.
_SHORT_PGN = _pgn("shortGam", ["0.2"] * 16, _clocks(16))
_NO_EVALS_PGN = _pgn("noEvals1", [None] * 20, _clocks(20))
# The byte range of the real dump ends inside a game: no result token, and half a comment.
_CUT_OFF_PGN = _pgn("cutOff01", ["0.2"] * 20, _clocks(20))[: -len("0:04:42] } 1-0\n")]

_GAMES_IN_THE_DUMP = (_MATE_PGN, _NO_EVALS_PGN, _BERSERK_PGN, _SHORT_PGN, _LONG_PGN, _CUT_OFF_PGN)


def _game(pgn: str) -> Game:
    game = Game.from_pgn(pgn.strip())
    assert game is not None
    return game


def _write_dump(directory: Path, *pgns: str) -> DumpPrefix:
    # The dump puts one blank line after each game.
    path = directory / "dump.pgn.zst"
    path.write_bytes(zstandard.ZstdCompressor().compress("".join(pgn + "\n" for pgn in pgns).encode()))
    return DumpPrefix(path)


def _convert(directory: Path) -> TimeFDataset:
    return ClockBlunderConnector().convert([_write_dump(directory, *_GAMES_IN_THE_DUMP)])


def _tasks_of_game(dataset: TimeFDataset, game_id: str) -> list[ClassificationTask]:
    return [
        task for task in dataset.tasks_of(ClassificationTask) if task.inputs[0].metadata["lichess_game_id"] == game_id
    ]


def _texts(task: ClassificationTask) -> tuple[str, str, str]:
    # The prompt, the rationale and the answer. Every window has all three.
    assert task.prompt is not None
    assert task.rationale is not None
    assert task.targets is not None
    return str(read_value(task.prompt)), str(read_value(task.rationale)), str(read_value(task.targets[0]))


def test_metadata():
    metadata = ClockBlunderConnector().metadata()
    assert metadata.dataset_id == "lichess/clock-blunder"
    assert str(metadata.license) == "CC0-1.0"
    assert str(metadata.dataset_version) == "1.0.0"


def test_source_is_pinned():
    # Literal pins, so moving to another dump or byte range is a deliberate test edit.
    assert DUMP_URL == "https://database.lichess.org/standard/lichess_db_standard_rated_2017-04.pgn.zst"
    assert PREFIX_BYTES == 67_108_864
    assert PREFIX_SHA256 == "d53e00e3cf7098057f136c2937d97f0cdd45963c52362565cbd2008814c99d1b"


@pytest.mark.parametrize(
    ("pawns", "chances"),
    # 2 / (1 + exp(-0.00368208 * centipawns)) - 1, worked out by hand as tanh(0.00184104 * centipawns).
    [(0.0, 0.0), (1.0, 0.182052), (-3.0, -0.502251), (10.0, 0.950895), (100.0, 1.0), (-100.0, -1.0)],
)
def test_winning_chances_follow_the_lichess_formula(pawns, chances):
    assert winning_chances(pawns) == pytest.approx(chances, abs=1e-6)


@pytest.mark.parametrize(("text", "pawns"), [("0.27", 0.27), ("-1.5", -1.5), ("#4", 100.0), ("#-3", -100.0)])
def test_an_evaluation_is_pawns_and_a_mate_is_a_hundred_of_them(text, pawns):
    assert parse_eval(text) == pytest.approx(pawns)


def test_an_evaluation_that_is_not_a_number_raises():
    with pytest.raises(TimeNetConnectorError, match=r"evaluation '1\.2\.3' is not a number or a mate"):
        parse_eval("1.2.3")


def test_a_blunder_is_a_drop_of_0_3_winning_chances_for_the_mover():
    # Winning chances for White: 1.7 pawns is 0.3031, 1.6 pawns is 0.2863, 0.1 pawns is 0.0184,
    # 9 pawns is 0.9296.
    evals: list[str | None] = ["0.0", "1.7", "0.1", "0.0", "-1.6", "-1.6", "#-3", "#-1", "#-1", "-9.0"]
    expected = [
        False,  # White, level to level
        True,  # Black gives White 0.3031
        False,  # White drops 0.3031 - 0.0184 = 0.2847
        False,  # Black gains 0.0184
        False,  # White drops 0.2863, just short
        False,  # Black, no change
        True,  # White walks into a mate: 1 - 0.2863 = 0.7137
        False,  # Black keeps the mate
        False,  # White is mated anyway
        False,  # Black loses the mate but only 1 - 0.9296 = 0.0704
    ]
    assert _game(_pgn("edgeGame", evals, _clocks(10))).blunders().tolist() == expected


def test_a_game_keeps_its_tags_and_one_clock_and_evaluation_per_ply():
    game = _game(_MATE_PGN)
    assert (game.game_id, game.time_control, game.white_elo, game.black_elo) == ("mateGame", "300+3", 1500, 1480)
    assert game.result == "1/2-1/2"
    assert game.n_plies == 26
    np.testing.assert_array_equal(game.clocks, np.array(_clocks(26), dtype=np.float64))
    np.testing.assert_array_equal(game.evals, [*[0.3] * 18, -100.0, -100.0, -100.0, *[0.0] * 5])
    assert np.flatnonzero(game.blunders()).tolist() == [18, 21]


@pytest.mark.parametrize(
    "pgn",
    [
        _NO_EVALS_PGN,
        _pgn("oneNoEvl", [*["0.2"] * 19, None], _clocks(20)),
        _CUT_OFF_PGN,
        _pgn("cutOff02", ["0.2"] * 20, _clocks(20))[: -len(" 1-0\n")],
        _pgn("unfinish", ["0.2"] * 20, _clocks(20), result="*"),
        _pgn("corresp1", ["0.2"] * 20, _clocks(20), time_control="-"),
        _pgn("unrated1", ["0.2"] * 20, _clocks(20), white_elo="?"),
        _BERSERK_PGN.replace(" 3. Nf3 { [%eval 0.2] [%clk 0:02:24] }", " 3. Nf3"),
        _pgn("otherSit", ["0.2"] * 20, _clocks(20), site="https://example.org/"),
    ],
    ids=[
        "no evaluations",
        "one ply without an evaluation",
        "cut off inside a comment",
        "cut off between the last move and the result",
        "not finished",
        "no clock in the time control",
        "an unknown rating",
        "a move without a comment",
        "not a Lichess game",
    ],
)
def test_a_game_the_dataset_cannot_use_is_skipped(pgn):
    assert Game.from_pgn(pgn.strip()) is None


def test_each_clock_keeps_its_last_value_between_its_owner_moves():
    white, black, evals = _game(_MATE_PGN).series_until(5)
    # White moved on plies 0, 2 and 4, Black on plies 1 and 3. Black's clock on ply 0 is its first reading.
    np.testing.assert_array_equal(white, [300, 300, 296, 296, 292])
    np.testing.assert_array_equal(black, [300, 300, 300, 298, 298])
    np.testing.assert_array_equal(evals, [0.3] * 5)


def test_a_berserked_clock_starts_at_its_first_reading_not_at_the_base_time():
    white, black, _ = _game(_BERSERK_PGN).series_until(4)
    np.testing.assert_array_equal(white, [150, 150, 147, 147])
    np.testing.assert_array_equal(black, [150, 150, 150, 146])


def test_a_window_holds_the_last_plies_before_the_decision_and_a_clipped_evaluation():
    game = _game(_LONG_PGN)
    white, black, evals = game.series_until(51)
    assert len(white) == len(black) == len(evals) == WINDOW_PLIES
    # Plies 11 to 50. White's last move was ply 50, its 26th; Black's was ply 49, its 25th.
    assert (white[0], white[-1]) == (300 - 5, 300 - 25)
    assert (black[0], black[-1]) == (300 - 5, 300 - 24)
    # A mate is stored as the clip, not as the 100 pawns the label uses.
    _, _, mate_evals = _game(_MATE_PGN).series_until(21)
    np.testing.assert_array_equal(mate_evals, [*[0.3] * 18, -10.0, -10.0, -10.0])


def test_decision_plies_are_seeded_by_the_game_and_leave_room_for_three_moves():
    for pgn in (_MATE_PGN, _BERSERK_PGN, _LONG_PGN):
        game = _game(pgn)
        plies = game.decision_plies()
        assert plies == _game(pgn).decision_plies()
        assert plies == sorted(set(plies))
        assert len(plies) == WINDOWS_PER_GAME
        # The mover's third move is ply + 4, and the last ply of the game is n_plies - 1.
        assert all(MIN_DECISION_PLY <= ply <= game.n_plies - 5 for ply in plies)
    # 20 plies leave exactly plies 12 to 15, so all four are chosen whatever the seed.
    assert _game(_BERSERK_PGN).decision_plies() == [12, 13, 14, 15]
    # A literal pin: another choice here moves every window of a published dataset version.
    assert _game(_MATE_PGN).decision_plies() == [13, 14, 15, 18]
    assert _game(_LONG_PGN).decision_plies() == [19, 21, 34, 48]


@pytest.mark.parametrize(("n_plies", "expected"), [(16, []), (17, [12]), (18, [12, 13])])
def test_a_game_too_short_for_three_more_moves_gives_fewer_windows(n_plies, expected):
    assert _game(_pgn("shortGam", ["0.2"] * n_plies, _clocks(n_plies))).decision_plies() == expected


def test_the_answer_counts_the_movers_next_three_moves_only():
    game = _game(_MATE_PGN)
    answers = {ply: _texts(create_window(game, ply)[1])[2] for ply in range(12, 22)}
    assert answers == {
        12: "no",  # White: plies 12, 14, 16. The blunder on ply 18 is White's fourth move.
        13: "no",  # Black: plies 13, 15, 17.
        14: "yes",  # White: ply 18 is the third move.
        15: "no",  # Black: plies 15, 17, 19. White's blunder on ply 18 is not Black's.
        16: "yes",
        17: "yes",  # Black: ply 21 is the third move, and the last ply with room is 21.
        18: "yes",  # White blunders on the move itself.
        19: "yes",
        20: "no",  # White: plies 20, 22, 24. Black's blunder on ply 21 is not White's.
        21: "yes",  # Black: plies 21, 23 and 25, the last ply of the game.
    }


def test_the_split_depends_only_on_the_game_id():
    # Literal pins: a change here moves games of a published dataset version between splits.
    assert _game(_LONG_PGN).split == Split.TRAIN
    assert _game(_BERSERK_PGN).split == Split.VALIDATION
    assert _game(_MATE_PGN).split == Split.TEST
    assert _game(_MATE_KEPT_PGN).split == Split.TEST


def test_iter_games_reads_the_compressed_dump_and_keeps_the_complete_games(tmp_path):
    games = list(_write_dump(tmp_path, *_GAMES_IN_THE_DUMP).iter_games())
    # The short game is a complete game. It only gives no window.
    assert [game.game_id for game in games] == ["mateGame", "berserk2", "shortGam", "longGame"]


def test_iter_games_stops_quietly_where_the_byte_range_cuts_the_stream(tmp_path):
    # Enough games for several zstd blocks, so a cut loses the last block and not the whole stream.
    pgns = [_pgn(f"game{index:04d}", ["0.2"] * 20, _clocks(20)) for index in range(600)]
    path = _write_dump(tmp_path, *pgns).path
    path.write_bytes(path.read_bytes()[:-8])
    games = [game.game_id for game in DumpPrefix(path).iter_games()]
    assert 0 < len(games) < len(pgns)
    assert games == [f"game{index:04d}" for index in range(len(games))]


def test_each_window_becomes_a_record_and_a_classification_task(tmp_path):
    dataset = _convert(tmp_path)
    tasks = dataset.tasks_of(ClassificationTask)
    # Four windows from each of the three games that are long enough, in dump order.
    assert [task.inputs[0].metadata["lichess_game_id"] for task in tasks] == (
        ["mateGame"] * 4 + ["berserk2"] * 4 + ["longGame"] * 4
    )
    assert len(dataset.records) == len(tasks)
    for game_id, split in (("mateGame", Split.TEST), ("berserk2", Split.VALIDATION), ("longGame", Split.TRAIN)):
        assert {task.split for task in _tasks_of_game(dataset, game_id)} == {split}

    # The last window of the first game: White is to move on ply 18, and walks into the mate.
    task = _tasks_of_game(dataset, "mateGame")[-1]
    (record,) = task.inputs
    signals = {signal.name: signal.to_numpy() for signal in record.signals}
    assert set(signals) == {"white_clock", "black_clock", "evaluation_pawns"}
    assert {values.dtype for values in signals.values()} == {np.dtype(np.float64)}
    np.testing.assert_array_equal(signals["white_clock"], [300 - 4 * (ply // 2) for ply in range(18)])
    np.testing.assert_array_equal(signals["black_clock"], [300, *[300 - 2 * ((ply - 1) // 2) for ply in range(1, 18)]])
    np.testing.assert_array_equal(signals["evaluation_pawns"], [0.3] * 18)
    assert {annotation.key: read_value(annotation.value) for annotation in record.annotations} == {
        "time_control": "300+3",
        "white_elo": 1500,
        "black_elo": 1480,
        "side_to_move": "white",
        "ply": 18,
    }
    assert task.label_set == BLUNDER
    assert task.annotations == ()
    prompt_text, rationale_text, answer = _texts(task)
    assert answer == "yes"
    # The texts of one window in full: a model is trained on these words.
    assert prompt_text == (
        "Rated Lichess game, time control 300+3 (seconds + increment), White rated 1500, Black rated 1480. "
        "The series give White's clock and Black's clock in seconds and the engine evaluation in pawns from "
        "White's view, one value per ply, for the plies before move 10. White is to move. Will White blunder "
        "within their next three moves, that is, make a move that loses at least 0.3 winning chances on "
        "Lichess's -1 to +1 scale? Answer yes or no."
    )
    # White has 300 - 4 * 8 = 268 seconds and spent 4 + 3 seconds on each move, so 268 / 7 = 38 moves.
    assert rationale_text == (
        "White is to move with 4:28 left against 4:44. Over the last 5 moves White spent 7.0 seconds per "
        "move on average, enough for about 38 more moves at that pace. The evaluation is +0.3 pawns from "
        "White's side, and it swung 0.0 pawns over the last ten plies. White does blunder within the next "
        "three moves. Answer: yes"
    )


def test_the_names_of_the_players_are_not_kept(tmp_path):
    for task in _convert(tmp_path).tasks_of(ClassificationTask):
        (record,) = task.inputs
        texts = [*_texts(task), *map(str, record.metadata.values())]
        texts += [str(read_value(annotation.value)) for annotation in record.annotations]
        assert not any(name in text for text in texts for name in ("Manvith", "Twisha"))


def test_the_prompt_and_the_series_never_contain_the_answer():
    # Two games that are the same until the decision on ply 21 and differ only after it.
    blunders, keeps_the_mate = _game(_MATE_PGN), _game(_MATE_KEPT_PGN)
    for ply in (17, 19, 21):
        (record, task), (other_record, other_task) = create_window(blunders, ply), create_window(keeps_the_mate, ply)
        assert (_texts(task)[2], _texts(other_task)[2]) == ("yes", "no")
        assert _texts(task)[0] == _texts(other_task)[0] == prompt(blunders, ply)
        assert "Answer:" not in _texts(task)[0]
        for signal, other_signal in zip(record.signals, other_record.signals, strict=True):
            np.testing.assert_array_equal(signal.to_numpy(), other_signal.to_numpy())
        assert record.annotations == other_record.annotations


def test_the_rationale_ends_with_the_answer(tmp_path):
    tasks = _convert(tmp_path).tasks_of(ClassificationTask)
    assert {_texts(task)[2] for task in tasks} == {"yes", "no"}
    for task in tasks:
        _, rationale_text, answer = _texts(task)
        assert rationale_text.endswith(f" Answer: {answer}")


def test_the_rationale_reads_the_evaluation_from_the_side_of_the_mover():
    # Black to move on ply 21 with a forced mate, stored as the clip of 10 pawns. Black has
    # 300 - 2 * 9 = 282 seconds and spent 2 + 3 seconds on each move, so 282 / 5 = 56 moves.
    _, task = create_window(_game(_MATE_KEPT_PGN), 21)
    assert _texts(task)[1] == (
        "Black is to move with 4:42 left against 4:20. Over the last 5 moves Black spent 5.0 seconds per "
        "move on average, enough for about 56 more moves at that pace. The evaluation is +10.0 pawns from "
        "Black's side, and it swung 10.3 pawns over the last ten plies. None of Black's next three moves is "
        "a blunder. Answer: no"
    )


def test_a_dump_without_a_usable_game_raises(tmp_path):
    prefix = _write_dump(tmp_path, _NO_EVALS_PGN, _SHORT_PGN, _CUT_OFF_PGN)
    with pytest.raises(TimeNetConnectorError, match="held no complete game with clocks and evaluations"):
        ClockBlunderConnector().convert([prefix])


def test_the_dataset_round_trips_through_storage(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    dataset = _convert(source)
    version_dir = dataset.write(path=tmp_path / "registry")
    with TimeFReader(open_local_version(version_dir)) as reader:
        reader.verify()
        assert_datasets_equal(dataset, reader.read())
