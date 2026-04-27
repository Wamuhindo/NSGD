#!/usr/bin/env python3
import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent
PYTHON = ROOT / ".venv" / "bin" / "python"
SIMULATE = ROOT / "AutoscalerFaasScalarVectorial" / "simulate.py"

EXPERIMENTS = [
    ("A", "input_sanity_1_A.json", "DAG_sanity_1_A.json"),
    ("A1->A2", "input_sanity_2_A1_A2.json", "DAG_sanity_2_A1_A2.json"),
    ("A1->A2->A3", "input_sanity_3_A1_A2_A3.json", "DAG_sanity_3_A1_A2_A3.json"),
    ("A1->A2->A3->A4", "input_sanity_4_A1_A2_A3_A4.json", "DAG_sanity_4_A1_A2_A3_A4.json"),
]


def find_latest_run_dir(experiment_name, arrival_rate):
    pattern = f"{experiment_name}_arr{arrival_rate}_*"
    candidates = sorted((ROOT / "logs").glob(pattern), key=lambda p: p.stat().st_mtime)
    if not candidates:
        raise FileNotFoundError(f"No log directory found for pattern logs/{pattern}")
    return candidates[-1]


def load_first_result(run_dir):
    aggregated_path = run_dir / "aggregated_results.json"
    with aggregated_path.open() as f:
        aggregated = json.load(f)
    if not aggregated["runs"]:
        raise RuntimeError(f"No completed runs in {aggregated_path}")
    return aggregated["runs"][0]


def main():
    python = PYTHON if PYTHON.exists() else Path(sys.executable)
    rows = []

    for label, input_name, dag_name in EXPERIMENTS:
        input_path = ROOT / input_name
        dag_path = ROOT / dag_name
        with input_path.open() as f:
            config = json.load(f)

        cmd = [
            str(python),
            str(SIMULATE),
            "--input",
            str(input_path),
            "--dag",
            str(dag_path),
        ]

        print(f"\n=== Running {label} ===")
        subprocess.run(cmd, cwd=ROOT, check=True)

        run_dir = find_latest_run_dir(config["experiment_name"], config["arrival_rate"])
        result = load_first_result(run_dir)
        rows.append({
            "graph": label,
            "cost": result["graph_cost_avg"],
            "time_cost": result["graph_time_avg_cost"],
            "requests": result["graph_reqs_total"],
            "external": result["external_arrivals"],
            "internal": result["internal_arrivals"],
            "cold": result["graph_prob_cold"],
            "reject": result["graph_prob_reject"],
            "log_dir": str(run_dir),
        })

    baseline = rows[0]["time_cost"]

    print("\nSanity comparison")
    print("=" * 132)
    print(f"{'graph':<18} {'time_cost':>12} {'delta':>12} {'event_cost':>12} {'reqs':>10} {'external':>10} {'internal':>10} {'p_cold':>10} {'p_reject':>10}")
    print("-" * 132)
    for row in rows:
        delta = row["time_cost"] - baseline
        print(
            f"{row['graph']:<18} "
            f"{row['time_cost']:>12.4f} "
            f"{delta:>12.4f} "
            f"{row['cost']:>12.4f} "
            f"{row['requests']:>10} "
            f"{row['external']:>10} "
            f"{row['internal']:>10} "
            f"{row['cold']:>10.4f} "
            f"{row['reject']:>10.4f}"
        )

    print("\nLog directories")
    for row in rows:
        print(f"{row['graph']}: {row['log_dir']}")


if __name__ == "__main__":
    main()
