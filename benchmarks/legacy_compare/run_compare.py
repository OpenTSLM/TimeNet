"""Run the v0.1 versus DuckDB control-plane comparison over the locally cached datasets.

For each dataset: time the v0.1 writer under the ``main`` environment, convert and time this stack's
writer, then run the same read workload against the v0.1 version under ``main`` and against the
converted version under this stack. Prints a Markdown table and writes every raw JSON result.

    uv run python -m benchmarks.legacy_compare.run_compare --main-python PATH --out DIR [dataset ...]
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess  # noqa: S404 - runs this repo's own scripts under two interpreters
import sys
from time import perf_counter


HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
STORAGE = Path(os.environ.get("TIMENET_STORAGE", Path.home() / ".cache/timenet/storage")).expanduser()


def _version_dirs() -> dict[str, Path]:
    found: dict[str, Path] = {}
    for manifest in sorted(STORAGE.glob("*/*/*/manifest.json")):
        version_dir = manifest.parent
        found[f"{version_dir.parents[1].name}/{version_dir.parent.name}"] = version_dir
    return found


def _run(python: str, script: str, *args: str, cwd: Path) -> dict:
    started = perf_counter()
    completed = subprocess.run(  # noqa: S603 - fixed argv built from our own paths
        [python, str(HERE / script), *args],
        cwd=cwd,
        capture_output=True,
        text=True,
        env={**os.environ, "PYTHONPATH": str(REPO), "PYTHONWARNINGS": "ignore"},
        check=False,
    )
    if completed.returncode != 0:
        return {
            "error": completed.stderr.strip().splitlines()[-1] if completed.stderr.strip() else "failed",
            "wall": perf_counter() - started,
        }
    return {**json.loads(completed.stdout.strip().splitlines()[-1]), "wall": perf_counter() - started}


def main() -> None:
    """Run the comparison from the command line."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--main-python", required=True, help="Python of the main-branch environment.")
    parser.add_argument("--out", required=True, help="Directory for converted versions and raw results.")
    parser.add_argument("--signals", type=int, default=2000)
    parser.add_argument("--records", type=int, default=200)
    parser.add_argument("datasets", nargs="*", help="Dataset ids to run; default is every cached one, smallest first.")
    arguments = parser.parse_args()
    out = Path(arguments.out)
    out.mkdir(parents=True, exist_ok=True)
    available = _version_dirs()
    selected = arguments.datasets or sorted(
        available, key=lambda name: sum(f.stat().st_size for f in available[name].rglob("*") if f.is_file())
    )
    results: dict[str, dict] = {}
    for name in selected:
        version_dir = available[name]
        converted_root = out / "converted" / name.replace("/", "__")
        print(f"== {name}", file=sys.stderr, flush=True)
        result = {
            "main_write": _run(arguments.main_python, "write_bench_main.py", str(version_dir), cwd=REPO),
            "new_write": _run(
                sys.executable, "write_bench_new.py", str(version_dir), "--out", str(converted_root), cwd=REPO
            ),
        }
        converted = next(iter(converted_root.glob("*/*/*/manifest.json")), None)
        result["main_read"] = _run(
            arguments.main_python,
            "read_bench.py",
            str(version_dir),
            "--signals",
            str(arguments.signals),
            "--records",
            str(arguments.records),
            cwd=REPO,
        )
        result["new_read"] = (
            _run(
                sys.executable,
                "read_bench.py",
                str(converted.parent),
                "--signals",
                str(arguments.signals),
                "--records",
                str(arguments.records),
                cwd=REPO,
            )
            if converted
            else {"error": "no converted version"}
        )
        results[name] = result
        (out / "results.json").write_text(json.dumps(results, indent=2))
        print(json.dumps(result), file=sys.stderr, flush=True)
    print(_table(results))
    (out / "results.md").write_text(_table(results))


def _cell(result: dict, key: str, scale: float = 1.0, digits: int = 2) -> str:
    if "error" in result:
        return "error"
    value = result.get(key)
    return "n/a" if value is None else f"{value * scale:.{digits}f}"


def _table(results: dict[str, dict]) -> str:
    lines = [
        "| Dataset | Records | Tasks | Write v0.1 (s) | Write new (s) | Full read v0.1 (s) | Full read new (s) | Values v0.1 (s) | Values new (s) | One record v0.1 (ms) | One record new (ms) |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for name, result in results.items():
        read = result["main_read"] if "error" not in result["main_read"] else result["new_read"]
        lines.append(
            f"| {name} | {read.get('records', 'n/a')} | {read.get('tasks', 'n/a')} "
            f"| {_cell(result['main_write'], 'write_seconds')} | {_cell(result['new_write'], 'write_seconds')} "
            f"| {_cell(result['main_read'], 'read_seconds')} | {_cell(result['new_read'], 'read_seconds')} "
            f"| {_cell(result['main_read'], 'values_seconds')} | {_cell(result['new_read'], 'values_seconds')} "
            f"| {_cell(result['main_read'], 'single_ms_per_record')} | {_cell(result['new_read'], 'single_ms_per_record')} |"
        )
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    main()
