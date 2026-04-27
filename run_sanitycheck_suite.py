#!/usr/bin/env python3
import csv
import json
import subprocess
import sys
from pathlib import Path

import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parent
PYTHON = ROOT / ".venv" / "bin" / "python"
SIMULATE = ROOT / "AutoscalerFaasScalarVectorial" / "simulate.py"
SANITY_DIR = ROOT / "sanitycheck"
FIG_DIR = ROOT / "results_figures"

CHAIN_LENGTHS = range(1, 11)

BASE_ARRIVAL_RATE = 10.0
BASE_WARM_SERVICE_RATE = 1.0
BASE_COLD_SERVICE_RATE = 100.0
BASE_COLD_START_RATE = 0.1
BASE_EXPIRATION_RATE = 0.01
MAX_CONCURRENCY = 100000
TAU = 1000
K = 2
SEEDS = [1]

SCENARIOS = {
    "fixed_external": {
        "description": "Same external simulated-time horizon for every graph length.",
        "max_time": 500000,
        "stop_by_simulated_time": True,
        "max_simulated_time": 500,
    },
    "event_budget": {
        "description": "Same event/update budget for every graph length; external arrivals fall as graph length grows.",
        "max_time": 50000,
        "stop_by_simulated_time": False,
    },
}


def chain_label(length):
    if length == 1:
        return "A1"
    return "->".join(f"A{i}" for i in range(1, length + 1))


def build_dag(length):
    dag = {}
    for idx in range(1, length + 1):
        node = f"A{idx}"
        if idx < length:
            dag[node] = {
                "next": [f"A{idx + 1}"],
                "transition_probability": [1.0],
            }
        else:
            dag[node] = {
                "next": [],
                "transition_probability": [],
            }
    return {"DirectedAcyclicGraph": dag}


def node_config(length):
    multiplier = float(length)
    return {
        "warm_service": {
            "rate": BASE_WARM_SERVICE_RATE * multiplier,
            "type": "Exponential",
        },
        "cold_service": {
            "rate": BASE_COLD_SERVICE_RATE * multiplier,
            "type": "Exponential",
        },
        "cold_start": {
            "rate": BASE_COLD_START_RATE * multiplier,
            "type": "Exponential",
        },
        "expiration": {
            "rate": BASE_EXPIRATION_RATE * multiplier,
            "type": "Exponential",
        },
    }


def build_input(scenario_name, scenario, length):
    nodes = {f"A{i}": node_config(length) for i in range(1, length + 1)}
    theta_stock = 1.0 / length
    theta_exp = 10 * length

    config = {
        "description": (
            f"Sanitycheck {scenario_name}, chain length {length}. "
            "Per-node rates are multiplied by chain length so serial means "
            "sum to the single-node target."
        ),
        "experiment_name": f"sanitycheck_{scenario_name}_L{length:02d}",
        "arrival_rate": BASE_ARRIVAL_RATE,
        "nodes": nodes,
        "optimization": {"type": "sgd"},
        "theta": [[theta_stock, theta_stock, theta_exp]],
        "tau": TAU,
        "max_concurrency": MAX_CONCURRENCY,
        "max_time": scenario["max_time"],
        "log_dir": "logs/",
        "K": K,
        "seeds": SEEDS,
        "k_delta": 1,
        "k_gamma": [0, 0, 0],
        "prtb": [[-0.5, 0.5], [-0.5, 0.5], [-1, 1]],
        "learn_mask": [False, False, False],
        "accumulate_cost": True,
        "K_exp": 1000,
        "gamma_min": 1,
        "theta_stock_min": 0,
        "theta_idle_min": 0,
    }
    if scenario["stop_by_simulated_time"]:
        config["stop_by_simulated_time"] = True
        config["max_simulated_time"] = scenario["max_simulated_time"]
    return config


def write_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        json.dump(payload, f, indent=2)
        f.write("\n")


def generate_configs():
    generated = []
    for scenario_name, scenario in SCENARIOS.items():
        dag_dir = SANITY_DIR / scenario_name / "dags"
        input_dir = SANITY_DIR / scenario_name / "inputs"
        for length in CHAIN_LENGTHS:
            dag_path = dag_dir / f"DAG_chain_L{length:02d}.json"
            input_path = input_dir / f"input_chain_L{length:02d}.json"
            write_json(dag_path, build_dag(length))
            write_json(input_path, build_input(scenario_name, scenario, length))
            generated.append((scenario_name, length, input_path, dag_path))
    return generated


def find_latest_run_dir(experiment_name, arrival_rate):
    pattern = f"{experiment_name}_arr{arrival_rate}_*"
    candidates = sorted((ROOT / "logs").glob(pattern), key=lambda p: p.stat().st_mtime)
    if not candidates:
        raise FileNotFoundError(f"No log directory found for logs/{pattern}")
    return candidates[-1]


def load_first_result(run_dir):
    with (run_dir / "aggregated_results.json").open() as f:
        aggregated = json.load(f)
    if not aggregated["runs"]:
        raise RuntimeError(f"No completed runs in {run_dir}")
    return aggregated["runs"][0]


def run_experiment(python, scenario_name, length, input_path, dag_path):
    with input_path.open() as f:
        config = json.load(f)

    run_output_dir = FIG_DIR / "run_logs"
    run_output_dir.mkdir(parents=True, exist_ok=True)
    stdout_path = run_output_dir / f"{scenario_name}_L{length:02d}.log"

    cmd = [
        str(python),
        str(SIMULATE),
        "--input",
        str(input_path),
        "--dag",
        str(dag_path),
    ]

    print(f"Running {scenario_name} L={length:02d} ...", flush=True)
    with stdout_path.open("w") as stdout_file:
        subprocess.run(
            cmd,
            cwd=ROOT,
            check=True,
            stdout=stdout_file,
            stderr=subprocess.STDOUT,
        )

    run_dir = find_latest_run_dir(config["experiment_name"], config["arrival_rate"])
    result = load_first_result(run_dir)
    return {
        "scenario": scenario_name,
        "chain_length": length,
        "graph": chain_label(length),
        "graph_time_avg_cost": result["graph_time_avg_cost"],
        "graph_event_avg_cost": result["graph_cost_avg"],
        "graph_reqs_total": result["graph_reqs_total"],
        "external_arrivals": result["external_arrivals"],
        "internal_arrivals": result["internal_arrivals"],
        "graph_prob_cold": result["graph_prob_cold"],
        "graph_prob_reject": result["graph_prob_reject"],
        "simulated_time": result["simulated_time"],
        "run_dir": str(run_dir),
        "stdout_log": str(stdout_path),
    }


def save_results_csv(rows):
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = FIG_DIR / "sanitycheck_results.csv"
    fieldnames = [
        "scenario",
        "chain_length",
        "graph",
        "graph_time_avg_cost",
        "graph_event_avg_cost",
        "graph_reqs_total",
        "external_arrivals",
        "internal_arrivals",
        "graph_prob_cold",
        "graph_prob_reject",
        "simulated_time",
        "run_dir",
        "stdout_log",
    ]
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return csv_path


def plot_metric(rows, metric, ylabel, output_name):
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(8, 5))
    for scenario_name in SCENARIOS:
        scenario_rows = sorted(
            [row for row in rows if row["scenario"] == scenario_name],
            key=lambda row: row["chain_length"],
        )
        xs = [row["chain_length"] for row in scenario_rows]
        ys = [row[metric] for row in scenario_rows]
        plt.plot(xs, ys, marker="o", label=scenario_name)
    plt.xlabel("Chain length")
    plt.ylabel(ylabel)
    plt.xticks(list(CHAIN_LENGTHS))
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    output_path = FIG_DIR / output_name
    plt.savefig(output_path, dpi=160)
    plt.close()
    return output_path


def save_summary(rows):
    summary_path = FIG_DIR / "sanitycheck_summary.md"
    lines = [
        "# Sanitycheck Results",
        "",
        "| scenario | L | time avg cost | event avg cost | external | internal | p cold | p reject |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in sorted(rows, key=lambda item: (item["scenario"], item["chain_length"])):
        lines.append(
            "| {scenario} | {chain_length} | {graph_time_avg_cost:.4f} | "
            "{graph_event_avg_cost:.4f} | {external_arrivals} | {internal_arrivals} | "
            "{graph_prob_cold:.4f} | {graph_prob_reject:.4f} |".format(**row)
        )
    summary_path.write_text("\n".join(lines) + "\n")
    return summary_path


def main():
    python = PYTHON if PYTHON.exists() else Path(sys.executable)
    generated = generate_configs()

    rows = []
    for scenario_name, length, input_path, dag_path in generated:
        rows.append(run_experiment(python, scenario_name, length, input_path, dag_path))

    csv_path = save_results_csv(rows)
    cost_fig = plot_metric(
        rows,
        "graph_time_avg_cost",
        "Graph time-average cost",
        "sanitycheck_time_avg_cost.png",
    )
    event_cost_fig = plot_metric(
        rows,
        "graph_event_avg_cost",
        "Graph event-average cost",
        "sanitycheck_event_avg_cost.png",
    )
    external_fig = plot_metric(
        rows,
        "external_arrivals",
        "External arrivals",
        "sanitycheck_external_arrivals.png",
    )
    cold_fig = plot_metric(
        rows,
        "graph_prob_cold",
        "Cold-start probability",
        "sanitycheck_cold_probability.png",
    )
    summary_path = save_summary(rows)

    print("\nDone.")
    print(f"Generated configs: {SANITY_DIR}")
    print(f"CSV: {csv_path}")
    print(f"Summary: {summary_path}")
    print(f"Figures: {cost_fig}, {event_cost_fig}, {external_fig}, {cold_fig}")


if __name__ == "__main__":
    main()
