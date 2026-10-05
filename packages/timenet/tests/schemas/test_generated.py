from pathlib import Path
import subprocess
import sys


def test_checked_in_schemas_match_pydantic_models():
    root = Path(__file__).resolve().parents[4]
    result = subprocess.run(
        [sys.executable, str(root / "scripts" / "gen_schemas.py"), "--check"],
        cwd=root,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
