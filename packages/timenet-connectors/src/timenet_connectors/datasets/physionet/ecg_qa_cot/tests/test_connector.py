from collections.abc import Iterable
import csv
from pathlib import Path
import shutil
import sys

import numpy as np
import pytest
import wfdb

from timenet.connectors import BaseConnector
from timenet.dataset import TimeFDataset
from timenet.engine import store_dataset
from timenet.errors import TimeFFormatError
from timenet.reader import TimeFReader
from timenet.registry import DatasetVersion
from timenet.types import AnswerTask
from timenet_connectors.bases.physionet import BasePhysioNetConnector
from timenet_connectors.datasets.physionet.ecg_qa_cot import connector
from timenet_connectors.datasets.physionet.ecg_qa_cot.connector import EcgQaCotConnector, EcgQaCotSource
from timenet_connectors.download import Artifact


_CSV_COLUMNS = [
    "ecg_id",
    "question",
    "answer",
    "rationale",
    "template_id",
    "question_type",
    "clinical_context",
    "prompt",
]
_LEADS = ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"]


def _prompt(question: str, options: list[str]) -> str:
    # The shape the release's prompt column uses. The heading is the format marker the connector
    # parses; the surrounding wording is synthetic, so the repo ships no dataset prose.
    bullets = "".join(f"- {option}\n" for option in options)
    return (
        f"Synthetic preamble: a 12-lead recording follows.\n\n"
        f"Question: {question}\n\n"
        f"This question has one of two possible answers:\n{bullets}\n"
        f"Synthetic instruction: answer with one of the two.\n"
    )


# Synthetic rows, hand-written for the tests and not derived from the real ECG-QA / PTB-XL data, so the
# repo ships no dataset bytes. The distribution is what the assertions need: three questions over two
# recordings (ecg_id 1 twice, 2 once), template 0 seen twice and template 1 once, and every answer
# distinct from its rationale. Every prompt offers two candidates drawn from its template's wider
# vocabulary, and the two template-0 rows offer different pairs.
_ROWS = [
    {
        "ecg_id": 1,
        "question": "Synthetic question: does this recording show a normal rhythm?",
        "answer": "no",
        "rationale": "Synthetic rationale one: the leads were reviewed before the label was assigned.",
        "template_id": 0,
        "question_type": "single-verify",
        "clinical_context": "12-lead ECG recording.",
        "options": ["no", "yes"],
    },
    {
        "ecg_id": 1,
        "question": "Synthetic question: which rhythm best describes this recording?",
        "answer": "sinus rhythm",
        "rationale": "Synthetic rationale two: the morphology was compared against the options.",
        "template_id": 1,
        "question_type": "single-query",
        "clinical_context": "12-lead ECG recording.",
        "options": ["atrial fibrillation", "sinus rhythm"],
    },
    {
        "ecg_id": 2,
        "question": "Synthetic question: is the finding present in this recording?",
        "answer": "not sure",
        "rationale": "Synthetic rationale three: the evidence was inconclusive for this lead set.",
        "template_id": 0,
        "question_type": "single-verify",
        "clinical_context": "12-lead ECG recording.",
        "options": ["not sure", "yes"],
    },
]

# template_id -> question_type and its answer classes. Template 0's classes are asserted verbatim.
_ANSWER_TEMPLATES = [
    {
        "template_id": 0,
        "question_type": "single-verify",
        "classes": "['yes', 'no', 'not sure']",
    },
    {
        "template_id": 1,
        "question_type": "single-query",
        "classes": "['sinus rhythm', 'atrial fibrillation', 'not sure']",
    },
]


@pytest.fixture(scope="session")
def records_dir(tmp_path_factory) -> Path:
    # Write two valid format-16 WFDB records (00001_hr, 00002_hr) with synthetic signals. wfdb.wrsamp
    # derives the gain/baseline it stores in each .hea, so wfdb.rdsamp and the connector's direct int16
    # reader decode the same physical values. Built once per session, before any test nulls out wfdb.
    out = tmp_path_factory.mktemp("ecg_qa_cot_records")
    rng = np.random.default_rng(0)
    for ecg_id in (1, 2):
        signal = rng.standard_normal((100, len(_LEADS)))
        wfdb.wrsamp(
            f"{ecg_id:05d}_hr",
            fs=500,
            units=["mV"] * len(_LEADS),
            sig_name=list(_LEADS),
            p_signal=signal,
            fmt=["16"] * len(_LEADS),
            write_dir=str(out),
        )
    return out


def _rows():
    return _ROWS


def _write_answers(tmp_path: Path) -> Path:
    csv_path = tmp_path / "answers_for_each_template.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["template_id", "question_type", "classes"])
        writer.writeheader()
        writer.writerows(_ANSWER_TEMPLATES)
    return csv_path


def _source(tmp_path: Path, records_dir: Path, split: str = "train", rows: list[dict] | None = None) -> EcgQaCotSource:
    # Recreate what download() produces: a records500/<bucket>/ tree copied from the synthesized records,
    # an answers CSV, and a CoT CSV. All fixture rows go in one split.
    rows = _rows() if rows is None else rows
    records_root = tmp_path / "records500"
    for row in rows:
        ecg_id = int(row["ecg_id"])
        bucket = records_root / f"{ecg_id // 1000 * 1000:05d}"
        bucket.mkdir(parents=True, exist_ok=True)
        for ext in (".hea", ".dat"):
            shutil.copy(records_dir / f"{ecg_id:05d}_hr{ext}", bucket / f"{ecg_id:05d}_hr{ext}")
    csv_path = tmp_path / f"ecg_qa_cot_{split}.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=_CSV_COLUMNS)
        writer.writeheader()
        for row in rows:
            written = row if "prompt" in row else {**row, "prompt": _prompt(row["question"], row["options"])}
            writer.writerow({column: written.get(column, "") for column in _CSV_COLUMNS})
    return EcgQaCotSource(
        records_root=records_root,
        answers_path=_write_answers(tmp_path),
        cot_csvs=((split, csv_path),),
    )


def _convert(tmp_path: Path, records_dir: Path) -> TimeFDataset:
    return EcgQaCotConnector().convert([_source(tmp_path, records_dir)])


def test_is_a_connector():
    assert isinstance(EcgQaCotConnector(), BaseConnector)
    assert isinstance(EcgQaCotConnector(), BasePhysioNetConnector)


def test_metadata():
    assert EcgQaCotConnector().metadata().dataset_id == "physionet/ecg-qa-cot"
    assert str(EcgQaCotConnector().metadata().license) == "CC-BY-4.0"


def test_download_locates_all_three_sources_in_their_extracted_trees(tmp_path, monkeypatch):
    # No network: each fetch is replaced by a tree holding the one file the real layout is found by.
    # Under test is what download() does with them - records500 under the PTB-XL root, and the split
    # to CSV mapping under the CoT root, where the `validation` split reads `ecg_qa_cot_val.csv`.
    cache_dir = tmp_path / "cache"
    ptbxl_root = cache_dir / "ptbxl-extract" / "ptb-xl-1.0.3"
    (ptbxl_root / "records500" / "00000").mkdir(parents=True)
    (ptbxl_root / "ptbxl_database.csv").write_text("ecg_id\n1\n", encoding="utf-8")
    cot_root = cache_dir / "cot-extract" / "ecg_qa_cot"
    cot_root.mkdir(parents=True)
    for name in ("ecg_qa_cot_train.csv", "ecg_qa_cot_val.csv", "ecg_qa_cot_test.csv"):
        (cot_root / name).write_text("ecg_id\n", encoding="utf-8")
    extracted = {connector.PTBXL_ZIP_URL: ptbxl_root.parent, connector.ECG_QA_COT_URL: cot_root.parent}
    fetched: list[str] = []

    async def fake_ensure_archive(url: str, target: str | Path) -> Path:
        fetched.append(url)
        return extracted[url]

    async def fake_download_files(artifacts: Iterable[Artifact]) -> list[Path]:
        dests = []
        for artifact in artifacts:
            fetched.append(artifact.url)
            artifact.dest.write_text("template_id,question_type,classes\n", encoding="utf-8")
            dests.append(artifact.dest)
        return dests

    monkeypatch.setattr(connector, "ensure_archive", fake_ensure_archive)
    monkeypatch.setattr(connector, "download_files", fake_download_files)

    (source,) = EcgQaCotConnector().download(cache_dir)

    assert fetched[0] == connector.PTBXL_ZIP_URL  # awaited first; the other two go concurrently
    assert set(fetched) == {connector.PTBXL_ZIP_URL, connector.ECG_QA_COT_URL, connector.ECG_QA_TEMPLATE_ANSWERS_URL}
    assert source.records_root == ptbxl_root / "records500"
    assert source.answers_path == cache_dir / "answers_for_each_template.csv"
    assert source.answers_path.exists()
    assert source.cot_csvs == (
        ("train", cot_root / "ecg_qa_cot_train.csv"),
        ("validation", cot_root / "ecg_qa_cot_val.csv"),
        ("test", cot_root / "ecg_qa_cot_test.csv"),
    )


def test_one_record_per_ecg_not_per_question(tmp_path, records_dir):
    dataset = _convert(tmp_path, records_dir)
    assert isinstance(dataset, TimeFDataset)
    # 3 rows over 2 recordings -> 2 records, not one per question
    assert {record.record_id for record in dataset.records} == {"ptbxl-1", "ptbxl-2"}


def test_twelve_leads_per_record(tmp_path, records_dir):
    assert all(len(record.time_series) == 12 for record in _convert(tmp_path, records_dir).records)


def test_each_question_is_an_answer_task_on_its_recording(tmp_path, records_dir):
    dataset = _convert(tmp_path, records_dir)
    rows = _rows()
    tasks = list(dataset.iter_tasks())
    assert len(tasks) == len(rows)
    assert all(isinstance(task, AnswerTask) for task in tasks)
    assert {task.target for task in tasks} == {row["answer"] for row in rows}
    assert {task.rationale for task in tasks} == {row["rationale"] for row in rows}
    assert {task.record_ids[0] for task in tasks} == {"ptbxl-1", "ptbxl-2"}  # each task points at its recording
    assert all(task.target != task.rationale for task in tasks)  # answer is the short label, not the CoT


def test_recording_carries_its_split(tmp_path, records_dir):
    for record in _convert(tmp_path, records_dir).records:
        assert [ann.value for ann in record.annotations if ann.key == "split"] == ["train"]


def test_question_metadata_is_deduped_registered_annotations(tmp_path, records_dir):
    dataset = _convert(tmp_path, records_dir)
    keys = {ann.key for ann in dataset.registered_annotations}
    assert {"question_type", "template_id", "template_answer_options", "answer_options", "clinical_context"} <= keys
    # template 0 appears in two rows but is registered once
    template_values = sorted(ann.value for ann in dataset.registered_annotations if ann.key == "template_id")
    assert template_values == [0, 1]


def test_task_annotation_refs_all_resolve_to_registered(tmp_path, records_dir):
    dataset = _convert(tmp_path, records_dir)
    registered = {ann.id for ann in dataset.registered_annotations}
    for task in dataset.iter_tasks():
        assert set(task.input_annotation_ids) <= registered


def test_template_answer_options_come_from_template(tmp_path, records_dir):
    dataset = _convert(tmp_path, records_dir)
    options = {ann.id: ann.value for ann in dataset.registered_annotations if ann.key == "template_answer_options"}
    # template 0 in the fixture answers CSV
    assert options["ecgqa-template-options-0"] == ["yes", "no", "not sure"]


def _options_of(dataset, task, key: str) -> list[str]:
    # Unpacks a single match, so a task referencing no annotation under `key`, or two, fails here.
    registered = {ann.id: ann for ann in dataset.registered_annotations}
    (value,) = [registered[ref].value for ref in task.input_annotation_ids if registered[ref].key == key]
    return list(value)


def _tasks_by_target(dataset):
    return {task.target: task for task in dataset.iter_tasks()}


def test_each_task_carries_its_own_candidate_pair(tmp_path, records_dir):
    dataset = _convert(tmp_path, records_dir)
    pairs = {target: _options_of(dataset, task, "answer_options") for target, task in _tasks_by_target(dataset).items()}
    assert pairs == {
        "no": ["no", "yes"],
        "sinus rhythm": ["atrial fibrillation", "sinus rhythm"],
        "not sure": ["not sure", "yes"],
    }


def test_the_pair_is_narrower_than_its_template_vocabulary(tmp_path, records_dir):
    # The whole point of reading the prompt: template 0 offers three answers, the question offers two.
    dataset = _convert(tmp_path, records_dir)
    task = _tasks_by_target(dataset)["no"]
    pair = _options_of(dataset, task, "answer_options")
    vocabulary = _options_of(dataset, task, "template_answer_options")
    assert pair == ["no", "yes"]
    assert vocabulary == ["yes", "no", "not sure"]
    assert set(pair) < set(vocabulary)


def test_two_questions_on_one_template_keep_different_pairs(tmp_path, records_dir):
    # Both rows use template 0, so a per-template pair would collapse them into one wrong choice set.
    dataset = _convert(tmp_path, records_dir)
    by_target = _tasks_by_target(dataset)
    first, second = by_target["no"], by_target["not sure"]
    assert _options_of(dataset, first, "template_answer_options") == _options_of(
        dataset, second, "template_answer_options"
    )
    assert _options_of(dataset, first, "answer_options") != _options_of(dataset, second, "answer_options")


def test_the_gold_answer_is_inside_the_pair(tmp_path, records_dir):
    dataset = _convert(tmp_path, records_dir)
    for task in dataset.iter_tasks():
        assert task.target in _options_of(dataset, task, "answer_options")


def test_questions_sharing_a_pair_share_one_annotation(tmp_path, records_dir):
    # Value-derived ids, so the same pair in the same order is stored once however many ask it.
    rows = [_rows()[0], {**_rows()[2], "options": ["no", "yes"], "answer": "no"}]
    dataset = EcgQaCotConnector().convert([_source(tmp_path, records_dir, rows=rows)])
    pairs = [ann for ann in dataset.registered_annotations if ann.key == "answer_options"]
    assert len(pairs) == 1
    assert {task.input_annotation_ids.count(pairs[0].id) for task in dataset.iter_tasks()} == {1}


def test_pair_order_is_the_order_the_prompt_lists_them(tmp_path, records_dir):
    # Sorting the pair would erase which candidate the benchmark showed first.
    rows = [{**_rows()[0], "options": ["yes", "no"]}]
    dataset = EcgQaCotConnector().convert([_source(tmp_path, records_dir, rows=rows)])
    assert [ann.value for ann in dataset.registered_annotations if ann.key == "answer_options"] == [["yes", "no"]]


@pytest.mark.parametrize(
    "prompt",
    [
        "Synthetic question with no candidate heading at all.\n",
        "This question has one of two possible answers:\n- only one\n",
        "This question has one of two possible answers:\n- yes\n- no\n- not sure\n",
    ],
)
def test_a_prompt_that_states_no_two_way_choice_is_rejected(tmp_path, records_dir, prompt):
    rows = [{**_rows()[0], "prompt": prompt}]
    with pytest.raises(TimeFFormatError):
        EcgQaCotConnector().convert([_source(tmp_path, records_dir, rows=rows)])


def test_series_values_match_fixture_record(tmp_path, records_dir):
    # Every lead, not just the first: the fixture's gains and baselines differ per lead, so this
    # catches a permuted column or a wrong-index gain that a single-lead check would pass.
    dataset = _convert(tmp_path, records_dir)
    signal, fields = wfdb.rdsamp(str(records_dir / "00001_hr"))
    record = next(r for r in dataset.records if r.record_id == "ptbxl-1")
    assert [series.signal for series in record.time_series] == list(fields["sig_name"])
    for index, series in enumerate(record.time_series):
        np.testing.assert_allclose(series.to_numpy(), signal[:, index], rtol=1e-6)


def test_convert_needs_no_wfdb_for_format16(tmp_path, records_dir, monkeypatch):
    # Format-16 records are read straight from the .hea/.dat, so convert and lead loading work with no
    # wfdb installed. wfdb is only a fallback for other signal formats.
    monkeypatch.setitem(sys.modules, "wfdb", None)  # `import wfdb` -> ImportError if anything reaches for it
    dataset = EcgQaCotConnector().convert([_source(tmp_path, records_dir)])
    assert dataset.records
    assert len(dataset.records[0].time_series[0].to_numpy())  # runs the direct loader, no wfdb


def test_convert_round_trips_through_the_writer(tmp_path, records_dir):
    dataset = _convert(tmp_path / "src", records_dir)
    dataset.derive_schema()
    version_dir = store_dataset(dataset, tmp_path / "out")
    with TimeFReader(DatasetVersion.open_local(version_dir)) as reader:
        restored = reader.read()
    assert {record.record_id for record in restored.records} == {"ptbxl-1", "ptbxl-2"}
    assert len(restored.tasks) == 3  # streamed to disk and back
    assert {ann.key for ann in restored.registered_annotations} >= {
        "question_type",
        "template_id",
        "template_answer_options",
        "answer_options",
        "clinical_context",
    }


def test_read_header_uses_the_direct_path_for_plain_single_dat_16(tmp_path):
    (tmp_path / "plain.hea").write_text(
        "plain 2 500 100\nplain.dat 16 1000(0)/mV 16 0 0 0 0 I\nplain.dat 16 1000(0)/mV 16 0 0 0 0 II\n"
    )
    assert BasePhysioNetConnector._read_header(tmp_path / "plain").direct_read is True


def test_read_header_falls_back_when_the_record_is_not_plain_16(tmp_path):
    # 16x2 (two WFDB samples per frame) breaks the (-1, n_sig) reshape, so the record must route to wfdb; a
    # multi-word signal description is kept whole rather than truncated to its last token.
    (tmp_path / "framed.hea").write_text("framed 1 500 100\nframed.dat 16x2 1000(0)/mV 16 0 0 0 0 my lead II\n")
    header = BasePhysioNetConnector._read_header(tmp_path / "framed")
    assert header.direct_read is False
    assert header.sig_name == ["my lead II"]


def test_read_header_falls_back_when_signals_span_multiple_dat_files(tmp_path):
    (tmp_path / "multi.hea").write_text(
        "multi 2 500 100\na.dat 16 1000(0)/mV 16 0 0 0 0 I\nb.dat 16 1000(0)/mV 16 0 0 0 0 II\n"
    )
    assert BasePhysioNetConnector._read_header(tmp_path / "multi").direct_read is False
