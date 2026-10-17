"""Build and smoke-test the viewer extra in a clean isolated environment."""

# Only fixed developer commands and the wheel produced in this process are executed.
# ruff: noqa: S404, S603, S607

from pathlib import Path
import subprocess
from tempfile import TemporaryDirectory


def main() -> None:
    """Check that the wheel includes CLI dependencies and all local browser assets."""
    with TemporaryDirectory() as directory:
        subprocess.run(["uv", "build", "--package", "timenet", "--wheel", "--out-dir", directory], check=True)
        wheel = next(Path(directory).glob("timenet-*.whl"))
        environment = ["uv", "run", "--isolated", "--no-project", "--with", f"{wheel}[viewer]"]
        subprocess.run([*environment, "timenet", "view", "--help"], check=True)
        code = (
            "from importlib.resources import files; "
            "from timenet.viewer.app import create_app; "
            "root = files('timenet.viewer').joinpath('static'); "
            "assert all(root.joinpath(name).is_file() for name in "
            "('index.html', 'app.css', 'app.js', 'plotly-basic.min.js', 'PLOTLY-LICENSE'))"
        )
        subprocess.run([*environment, "python", "-c", code], check=True)


if __name__ == "__main__":
    main()
