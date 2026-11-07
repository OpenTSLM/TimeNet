import subprocess
import sys


def test_the_planner_imports_nothing_from_timenet_connectors():
    code = "import sys, timenet.coordinator.plan; assert 'timenet_connectors' not in sys.modules"
    subprocess.run([sys.executable, "-c", code], check=True)
