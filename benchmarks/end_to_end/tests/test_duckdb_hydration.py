from benchmarks.duckdb_hydration import run_benchmark


def test_hydration_benchmark_exercises_batch_and_per_record_paths(tmp_path):
    result = run_benchmark(records=12, selected=3, repeats=1, root=tmp_path)

    assert result.records == 12
    assert result.selected == 3
    assert result.batch_seconds > 0
    assert result.per_record_seconds > 0
