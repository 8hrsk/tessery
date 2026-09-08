import json
import subprocess
import sys
from pathlib import Path


def test_portable_soak_reports_scope_and_preserves_existing_report(tmp_path):
    tool = Path(__file__).resolve().parents[2] / "tools/soak_portable.py"
    output = tmp_path / "soak.json"
    command = [sys.executable, str(tool), "--seconds", "1", "--output", str(output)]
    result = subprocess.run(command, capture_output=True, text=True, timeout=20)
    assert result.returncode == 0, result.stderr
    report = json.loads(output.read_text())
    assert report["status"] == "passed"
    assert not report["metal_qualified"] and not report["model_inference_qualified"]
    assert report["iterations"] > 0 and report["workload_seconds"] >= 1
    assert report["python_growth_bytes"] <= report["python_growth_budget_bytes"]
    previous = output.read_bytes()
    result = subprocess.run(command, capture_output=True, text=True, timeout=20)
    assert result.returncode != 0
    assert output.read_bytes() == previous
