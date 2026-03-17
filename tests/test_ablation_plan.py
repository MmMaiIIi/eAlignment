import json
import subprocess
import sys
from pathlib import Path


def test_plan_ablations_smoke(tmp_path: Path) -> None:
    md_path = tmp_path / "ablation_plan.md"
    json_path = tmp_path / "ablation_plan.json"
    cmd = [
        sys.executable,
        "scripts/plan_ablations.py",
        "--config",
        "configs/eval/ablation_matrix.yaml",
        "--output-md",
        str(md_path),
        "--output-json",
        str(json_path),
        "--check-paths",
    ]
    subprocess.run(cmd, check=True)

    assert md_path.exists()
    assert json_path.exists()

    plan = json.loads(json_path.read_text(encoding="utf-8"))
    assert plan["name"] == "stage5_ablation_matrix"
    assert len(plan["experiments"]) >= 5
