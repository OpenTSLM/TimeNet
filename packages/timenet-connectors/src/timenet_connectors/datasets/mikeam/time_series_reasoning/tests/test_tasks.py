import random

from timenet.types import AnswerTask
from timenet_connectors.bases.huggingface import HubFileRef
from timenet_connectors.datasets.mikeam.time_series_reasoning.loader import TimeSeriesReasoningLoader
from timenet_connectors.datasets.mikeam.time_series_reasoning.release import SourceFile
from timenet_connectors.datasets.mikeam.time_series_reasoning.request import authors_series
from timenet_connectors.datasets.mikeam.time_series_reasoning.rows import CounterfactualRow, DescriptionRow, QuestionRow
from timenet_connectors.datasets.mikeam.time_series_reasoning.tests.release_fixtures import fixture_path, published_rows


def _loader() -> TimeSeriesReasoningLoader:
    refs = [HubFileRef(fixture_path(file), file.part) for file in SourceFile]
    return TimeSeriesReasoningLoader.parse_from(refs, pinned=False)


# The authors' configs, configs/tasks/llms_and_ts/*.yaml at d4e748cb, keyed by the rows they load.
_CONFIGS = {
    DescriptionRow: {
        "ts_column": "series",
        "label_column": "description",
        "context_columns": [],
        "context_prefix": "What description best relates to this time series?",
        "shuffle_labels": True,
        "format_abc_mcq": True,
    },
    QuestionRow: {
        "ts_column": "series",
        "context_prefix": "Answer the following question about the time series",
        "context_columns": ["description_tiny", "metadata", "question"],
        "shuffle_labels": False,
        "format_abc_mcq": True,
    },
    CounterfactualRow: {
        "ts_column": "series",
        "context_prefix": "Answer the following question about the time series",
        "context_columns": ["description_tiny", "metadata", "question"],
        "contrastive_column": "new_series",
        "shuffle_labels": False,
        "format_abc_mcq": True,
    },
}


def _authors_item(data: dict, config: dict, shuffle) -> dict:
    """``MultimodalMCQDataset.__getitem__`` (src/models/tasks/multimodal.py, lines 79-125), line for line.

    ``shuffle`` stands in for the module-level ``random.shuffle`` the authors call unseeded, so a test
    can give it the draw the connector made. The defaults are the class's.
    """
    options_column = config.get("options_column", "options")
    label_column = config.get("label_column", "source")
    contrastive_column = config.get("contrastive_column")
    options = data[options_column]

    context = "\n".join([str(data[col]) for col in config["context_columns"]])
    context = config["context_prefix"] + context

    if config["shuffle_labels"]:
        shuffle(options)

    if "answer_index" in data:  # noqa: SIM108 - the authors' code, line for line
        label_index = data["answer_index"]
    else:
        label_index = options.index(data[label_column])

    if config["format_abc_mcq"]:
        full_options = [f"{chr(ord('A') + i)}) {x}" for i, x in enumerate(options)]
        options = [f"{chr(ord('A') + i)}" for i, x in enumerate(options)]
        context = context + "\n" + "\n".join(full_options)
        answer = options[label_index]
    else:
        answer = chr(ord("A") + label_index)

    if contrastive_column:  # noqa: SIM108 - the authors' code, line for line
        ts = data[config["ts_column"]] + data[contrastive_column]
    else:
        ts = data[config["ts_column"]]

    return {"context": context, "ts": ts, "label": answer, "label_index": label_index}


def _provenance(task: AnswerTask) -> tuple[SourceFile, int]:
    line = task.metadata["line"]
    assert isinstance(line, int)
    return SourceFile(task.metadata["file"]), line


def test_every_task_is_the_item_the_authors_loader_builds():
    loader = _loader()
    by_method = {
        DescriptionRow: list(loader.description_tasks()),
        QuestionRow: list(loader.question_tasks()),
        CounterfactualRow: list(loader.counterfactual_tasks()),
    }
    # tasks() yields from each method in turn.
    assert [_provenance(task) for task in loader.tasks()] == [
        _provenance(task) for tasks in by_method.values() for task in tasks
    ]
    assert sum(map(len, by_method.values())) == 12
    for row_type, tasks in by_method.items():
        for task in tasks:
            file, line = _provenance(task)
            # The connector draws the description order from this seed. Any draw is one the authors' loader can make.
            draw = random.Random(f"{file.value}:{line}").shuffle  # noqa: S311 - a shuffle, not cryptography
            item = _authors_item(published_rows(file)[line - 1], _CONFIGS[row_type], draw)
            assert isinstance(task, AnswerTask)
            assert task.prompt == item["context"]
            assert task.targets == (item["label"],)
            assert authors_series(task) == item["ts"]


def test_the_description_order_is_fixed_and_puts_the_answer_anywhere():
    descriptions = list(_loader().description_tasks())
    # Published, every answer is the first option. The seeded draw is pinned here: a change to it, or to
    # Python's shuffle, shows as a changed target.
    assert [task.targets for task in descriptions] == [(letter,) for letter in "DBDCABA"]


def test_question_tasks_keep_the_answer_and_the_source_issues_in_metadata():
    loader = _loader()
    metadata = {
        (_provenance(task)[0].name, _provenance(task)[1]): task.metadata
        for tasks in (loader.question_tasks(), loader.counterfactual_tasks())
        for task in tasks
    }
    # MCQ_1_TS.jsonl line 14221: the answer's text is also a wrong option's. The letter keeps the
    # authors' choice.
    question = published_rows(SourceFile.QUESTIONS)[0]
    assert metadata["QUESTIONS", 1] == {
        "file": "MCQ_1_TS.jsonl",
        "line": 1,
        "answer_index": 0,
        "answer_text": question["options"][0],
        "source_issues": ["repeated_option", "answer_text_repeated"],
    }
    # MCQ_2_TS.jsonl line 45584: the question and every option are blank.
    assert metadata["COUNTERFACTUALS", 3] == {
        "file": "MCQ_2_TS.jsonl",
        "line": 3,
        "answer_index": 0,
        "answer_text": "",
        "source_issues": ["blank_question", "blank_option", "repeated_option", "answer_text_repeated"],
        "category": "difference",
    }
    assert sum(1 for value in metadata.values() if value["source_issues"]) == 2
