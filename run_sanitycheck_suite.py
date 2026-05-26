#!/usr/bin/env python3
import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path

import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parent
PYTHON = ROOT / ".venv" / "bin" / "python"
SIMULATE = ROOT / "AutoscalerFaasScalarVectorial" / "simulate.py"

DEFAULT_MAX_LENGTH = 5

MAX_TIME = 200000
BASE_ARRIVAL_RATE = 10
BASE_WARM_SERVICE_RATE = 0.5
BASE_COLD_SERVICE_RATE = 100.0
BASE_COLD_START_RATE = 0.1
BASE_EXPIRATION_RATE = 0.01
MAX_CONCURRENCY = 50
TAU = 100
K = 2
SEEDS = [1]

SCENARIOS = {
    "fixed_external": {
        "description": "Same external simulated-time horizon for every graph length.",
        "max_time": MAX_TIME,
        "stop_by_simulated_time": True,
        "max_simulated_time": 500,
    },
    "event_budget": {
        "description": "Same event/update budget for every graph length; external arrivals fall as graph length grows.",
        "max_time": MAX_TIME,
        "stop_by_simulated_time": False,
    },
}

GRAPH_RESULT_FIELDNAMES = [
    "scenario",
    "chain_length",
    "graph",
    "graph_time_avg_cost",
    "graph_event_avg_cost",
    "graph_response_time_count",
    "graph_response_time_avg",
    "graph_response_time_p50",
    "graph_response_time_p95",
    "graph_response_time_p99",
    "graph_response_time_max",
    "graph_reqs_total",
    "graph_reqs_cold",
    "graph_reqs_warm",
    "graph_reqs_init_free",
    "graph_reqs_init_reserved",
    "graph_reqs_queued",
    "graph_reqs_reject",
    "external_arrivals",
    "internal_arrivals",
    "graph_prob_cold",
    "graph_prob_warm",
    "graph_prob_queued",
    "graph_prob_reject",
    "graph_inst_count_avg",
    "graph_inst_running_count_avg",
    "graph_inst_idle_count_avg",
    "graph_inst_init_free_count_avg",
    "graph_inst_init_reserved_count_avg",
    "graph_inst_queued_jobs_count_avg",
    "graph_response_time_warm_count",
    "graph_response_time_warm_avg",
    "graph_response_time_cold_count",
    "graph_response_time_cold_avg",
    "graph_response_time_queued_count",
    "graph_response_time_queued_avg",
    "simulated_time",
    "warm_service_rate",
    "warm_service_mean_time",
    "cold_service_rate",
    "cold_service_mean_time",
    "cold_start_rate",
    "cold_start_mean_time",
    "expiration_rate",
    "expiration_mean_time",
    "total_expected_warm_service_time",
    "total_expected_cold_service_time",
    "total_expected_cold_start_time",
    "total_expected_expiration_time",
    "run_dir",
    "stdout_log",
]

GRAPH_PLOT_SPECS = [
    ("graph_time_avg_cost", "Graph time-average cost", "time_avg_cost"),
    ("graph_event_avg_cost", "Graph event-average cost", "event_avg_cost"),
    ("graph_response_time_avg", "Graph average response time", "response_time_avg"),
    ("graph_response_time_p50", "Graph p50 response time", "response_time_p50"),
    ("graph_response_time_p95", "Graph p95 response time", "response_time_p95"),
    ("graph_response_time_p99", "Graph p99 response time", "response_time_p99"),
    ("graph_response_time_max", "Graph max response time", "response_time_max"),
    ("graph_response_time_warm_avg", "Warm average response time", "response_time_warm_avg"),
    ("graph_response_time_cold_avg", "Cold average response time", "response_time_cold_avg"),
    ("graph_response_time_queued_avg", "Queued average response time", "response_time_queued_avg"),
    ("graph_reqs_total", "Graph total requests", "requests_total"),
    ("graph_reqs_warm", "Warm requests", "requests_warm"),
    ("graph_reqs_cold", "Cold requests", "requests_cold"),
    ("graph_reqs_init_free", "Init-free starts", "requests_init_free"),
    ("graph_reqs_init_reserved", "Init-reserved starts", "requests_init_reserved"),
    ("graph_reqs_queued", "Queued requests", "requests_queued"),
    ("graph_reqs_reject", "Rejected requests", "requests_reject"),
    ("external_arrivals", "External arrivals", "external_arrivals"),
    ("internal_arrivals", "Internal arrivals", "internal_arrivals"),
    ("graph_prob_cold", "Cold-start probability", "prob_cold"),
    ("graph_prob_warm", "Warm-start probability", "prob_warm"),
    ("graph_prob_queued", "Queued probability", "prob_queued"),
    ("graph_prob_reject", "Rejection probability", "prob_reject"),
    ("graph_inst_count_avg", "Average allocated instances", "inst_count_avg"),
    ("graph_inst_running_count_avg", "Average running instances", "inst_running_count_avg"),
    ("graph_inst_idle_count_avg", "Average idle instances", "inst_idle_count_avg"),
    ("graph_inst_init_free_count_avg", "Average init-free instances", "inst_init_free_count_avg"),
    ("graph_inst_init_reserved_count_avg", "Average init-reserved instances", "inst_init_reserved_count_avg"),
    ("graph_inst_queued_jobs_count_avg", "Average queued jobs", "inst_queued_jobs_count_avg"),
    ("warm_service_rate", "Warm service rate", "warm_service_rate"),
    ("warm_service_mean_time", "Warm service mean time", "warm_service_mean_time"),
    ("cold_service_rate", "Cold service rate", "cold_service_rate"),
    ("cold_service_mean_time", "Cold service mean time", "cold_service_mean_time"),
    ("cold_start_rate", "Cold-start rate", "cold_start_rate"),
    ("cold_start_mean_time", "Cold-start mean time", "cold_start_mean_time"),
    ("expiration_rate", "Expiration rate", "expiration_rate"),
    ("expiration_mean_time", "Expiration mean time", "expiration_mean_time"),
    ("total_expected_warm_service_time", "Total expected warm service time", "total_expected_warm_service_time"),
    ("total_expected_cold_service_time", "Total expected cold service time", "total_expected_cold_service_time"),
    ("total_expected_cold_start_time", "Total expected cold-start time", "total_expected_cold_start_time"),
    ("total_expected_expiration_time", "Total expected expiration time", "total_expected_expiration_time"),
]

NODE_HEATMAP_SPECS = [
    ("prob_cold", "Node cold-start probability", "node_prob_cold"),
    ("prob_warm", "Node warm-start probability", "node_prob_warm"),
    ("prob_queued", "Node queued probability", "node_prob_queued"),
    ("prob_reject", "Node rejection probability", "node_prob_reject"),
    ("reqs_total", "Node total requests", "node_reqs_total"),
    ("reqs_warm", "Node warm requests", "node_reqs_warm"),
    ("reqs_cold", "Node cold requests", "node_reqs_cold"),
    ("reqs_init_free", "Node init-free starts", "node_reqs_init_free"),
    ("reqs_init_reserved", "Node init-reserved starts", "node_reqs_init_reserved"),
    ("reqs_queued", "Node queued requests", "node_reqs_queued"),
    ("reqs_reject", "Node rejected requests", "node_reqs_reject"),
    ("inst_count_avg", "Node average allocated instances", "node_inst_count_avg"),
    ("inst_running_count_avg", "Node average running instances", "node_inst_running_count_avg"),
    ("inst_idle_count_avg", "Node average idle instances", "node_inst_idle_count_avg"),
    ("inst_init_free_count_avg", "Node average init-free instances", "node_inst_init_free_count_avg"),
    ("inst_init_reserved_count_avg", "Node average init-reserved instances", "node_inst_init_reserved_count_avg"),
    ("inst_queued_jobs_count_avg", "Node average queued jobs", "node_inst_queued_jobs_count_avg"),
    ("response_time_avg", "Node average response time", "node_response_time_avg"),
    ("response_time_p50", "Node p50 response time", "node_response_time_p50"),
    ("response_time_p95", "Node p95 response time", "node_response_time_p95"),
    ("response_time_p99", "Node p99 response time", "node_response_time_p99"),
    ("response_time_max", "Node max response time", "node_response_time_max"),
    ("response_time_warm_avg", "Node warm average response time", "node_response_time_warm_avg"),
    ("response_time_cold_avg", "Node cold average response time", "node_response_time_cold_avg"),
    ("response_time_queued_avg", "Node queued average response time", "node_response_time_queued_avg"),
    ("warm_service_rate", "Node warm service rate", "node_warm_service_rate"),
    ("warm_service_mean_time", "Node warm service mean time", "node_warm_service_mean_time"),
    ("cold_service_rate", "Node cold service rate", "node_cold_service_rate"),
    ("cold_service_mean_time", "Node cold service mean time", "node_cold_service_mean_time"),
    ("cold_start_rate", "Node cold-start rate", "node_cold_start_rate"),
    ("cold_start_mean_time", "Node cold-start mean time", "node_cold_start_mean_time"),
    ("expiration_rate", "Node expiration rate", "node_expiration_rate"),
    ("expiration_mean_time", "Node expiration mean time", "node_expiration_mean_time"),
]


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


def output_root(output_prefix):
    path = Path(output_prefix)
    return path if path.is_absolute() else ROOT / path


def build_input(scenario_name, scenario, length, log_dir, max_concurrency=MAX_CONCURRENCY):
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
        "max_concurrency": max_concurrency,
        "max_time": scenario["max_time"],
        "log_dir": str(log_dir),
        "K": K,
        "seeds": SEEDS,
        "k_delta": 1,
        "k_gamma": [1,1, 1],
        "prtb": [[-0.5, 0.5], [-0.5, 0.5], [-1, 1]],
        "learn_mask": [True, True, True],
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


def generate_configs(selected_scenarios, chain_lengths, output_dir, max_concurrency=MAX_CONCURRENCY):
    generated = []
    config_dir = output_dir / "configs"
    log_dir = output_dir / "logs"
    for scenario_name in selected_scenarios:
        scenario = SCENARIOS[scenario_name]
        dag_dir = config_dir / scenario_name / "dags"
        input_dir = config_dir / scenario_name / "inputs"
        for length in chain_lengths:
            dag_path = dag_dir / f"DAG_chain_L{length:02d}.json"
            input_path = input_dir / f"input_chain_L{length:02d}.json"
            write_json(dag_path, build_dag(length))
            write_json(input_path, build_input(scenario_name, scenario, length, log_dir, max_concurrency))
            generated.append((scenario_name, length, input_path, dag_path))
    return generated


def find_latest_run_dir(experiment_name, arrival_rate, log_dir):
    pattern = f"{experiment_name}_arr{arrival_rate}_*"
    candidates = sorted(log_dir.glob(pattern), key=lambda p: p.stat().st_mtime)
    if not candidates:
        raise FileNotFoundError(f"No log directory found for {log_dir / pattern}")
    return candidates[-1]


def load_first_result(run_dir):
    with (run_dir / "aggregated_results.json").open() as f:
        aggregated = json.load(f)
    if not aggregated["runs"]:
        raise RuntimeError(f"No completed runs in {run_dir}")
    return aggregated["runs"][0]


def load_first_graph_metrics_row(run_dir):
    metrics_path = run_dir / "all_runs_metrics.csv"
    if not metrics_path.exists():
        return None
    with metrics_path.open(newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row["level"] == "graph":
                return row
    raise RuntimeError(f"No graph-level row found in {metrics_path}")


def load_metrics_rows(run_dir):
    metrics_path = run_dir / "all_runs_metrics.csv"
    if not metrics_path.exists():
        return []
    with metrics_path.open(newline="") as f:
        return list(csv.DictReader(f))


def number_from_csv(row, key, cast=float, default=None):
    if key not in row:
        return default
    value = row[key]
    if value == "":
        return default
    if cast is int:
        return int(float(value))
    return cast(value)


def sum_per_node(result, key):
    return sum(node.get(key, 0) for node in result.get("per_node", {}).values())


def graph_value_from_metrics(metrics_row, result, csv_key, result_key=None, cast=float, default=0):
    if metrics_row is not None:
        return number_from_csv(metrics_row, csv_key, cast, default)
    if result_key is not None:
        return result.get(result_key, default)
    return default


def result_graph_metrics(metrics_row, result):
    graph_reqs_total = graph_value_from_metrics(
        metrics_row, result, "reqs_total", "graph_reqs_total", int, 0
    )
    graph_reqs_cold = graph_value_from_metrics(
        metrics_row, result, "reqs_cold", "graph_reqs_cold", int, 0
    )
    graph_reqs_warm = graph_value_from_metrics(
        metrics_row, result, "reqs_warm", "graph_reqs_warm", int, 0
    )
    graph_reqs_init_free = (
        number_from_csv(metrics_row, "reqs_init_free", int, 0)
        if metrics_row is not None
        else sum_per_node(result, "reqs_init_free")
    )
    graph_reqs_init_reserved = (
        number_from_csv(metrics_row, "reqs_init_reserved", int, 0)
        if metrics_row is not None
        else sum_per_node(result, "reqs_init_reserved")
    )
    graph_reqs_queued = (
        number_from_csv(metrics_row, "reqs_queued", int, 0)
        if metrics_row is not None
        else sum_per_node(result, "reqs_queued")
    )
    graph_reqs_reject = graph_value_from_metrics(
        metrics_row, result, "reqs_reject", "graph_reqs_reject", int, 0
    )

    def probability(numerator):
        return numerator / graph_reqs_total if graph_reqs_total else 0

    values = {
        "graph_time_avg_cost": graph_value_from_metrics(
            metrics_row, result, "graph_time_avg_cost", "graph_time_avg_cost"
        ),
        "graph_event_avg_cost": graph_value_from_metrics(
            metrics_row, result, "graph_event_avg_cost", "graph_cost_avg"
        ),
        "graph_response_time_count": graph_value_from_metrics(
            metrics_row, result, "response_time_count", "graph_response_time_count", int, 0
        ),
        "graph_response_time_avg": graph_value_from_metrics(
            metrics_row, result, "response_time_avg", "graph_response_time_avg"
        ),
        "graph_response_time_p50": graph_value_from_metrics(
            metrics_row, result, "response_time_p50", "graph_response_time_p50"
        ),
        "graph_response_time_p95": graph_value_from_metrics(
            metrics_row, result, "response_time_p95", "graph_response_time_p95"
        ),
        "graph_response_time_p99": graph_value_from_metrics(
            metrics_row, result, "response_time_p99", "graph_response_time_p99"
        ),
        "graph_response_time_max": graph_value_from_metrics(
            metrics_row, result, "response_time_max", "graph_response_time_max"
        ),
        "graph_response_time_warm_count": graph_value_from_metrics(
            metrics_row, result, "response_time_warm_count", None, int, 0
        ),
        "graph_response_time_warm_avg": graph_value_from_metrics(
            metrics_row, result, "response_time_warm_avg"
        ),
        "graph_response_time_cold_count": graph_value_from_metrics(
            metrics_row, result, "response_time_cold_count", None, int, 0
        ),
        "graph_response_time_cold_avg": graph_value_from_metrics(
            metrics_row, result, "response_time_cold_avg"
        ),
        "graph_response_time_queued_count": graph_value_from_metrics(
            metrics_row, result, "response_time_queued_count", None, int, 0
        ),
        "graph_response_time_queued_avg": graph_value_from_metrics(
            metrics_row, result, "response_time_queued_avg"
        ),
        "graph_reqs_total": graph_reqs_total,
        "graph_reqs_cold": graph_reqs_cold,
        "graph_reqs_warm": graph_reqs_warm,
        "graph_reqs_init_free": graph_reqs_init_free,
        "graph_reqs_init_reserved": graph_reqs_init_reserved,
        "graph_reqs_queued": graph_reqs_queued,
        "graph_reqs_reject": graph_reqs_reject,
        "external_arrivals": graph_value_from_metrics(
            metrics_row, result, "external_arrivals", "external_arrivals", int, 0
        ),
        "internal_arrivals": graph_value_from_metrics(
            metrics_row, result, "internal_arrivals", "internal_arrivals", int, 0
        ),
        "graph_prob_cold": graph_value_from_metrics(
            metrics_row, result, "prob_cold", "graph_prob_cold", float, probability(graph_reqs_cold)
        ),
        "graph_prob_warm": graph_value_from_metrics(
            metrics_row, result, "prob_warm", None, float, probability(graph_reqs_warm)
        ),
        "graph_prob_queued": graph_value_from_metrics(
            metrics_row, result, "prob_queued", None, float, probability(graph_reqs_queued)
        ),
        "graph_prob_reject": graph_value_from_metrics(
            metrics_row, result, "prob_reject", "graph_prob_reject", float, probability(graph_reqs_reject)
        ),
        "simulated_time": graph_value_from_metrics(
            metrics_row, result, "simulated_time", "simulated_time"
        ),
    }

    for key in [
        "inst_count_avg",
        "inst_running_count_avg",
        "inst_idle_count_avg",
        "inst_init_free_count_avg",
        "inst_init_reserved_count_avg",
        "inst_queued_jobs_count_avg",
        "warm_service_rate",
        "warm_service_mean_time",
        "cold_service_rate",
        "cold_service_mean_time",
        "cold_start_rate",
        "cold_start_mean_time",
        "expiration_rate",
        "expiration_mean_time",
        "total_expected_warm_service_time",
        "total_expected_cold_service_time",
        "total_expected_cold_start_time",
        "total_expected_expiration_time",
    ]:
        values[f"graph_{key}" if key.startswith("inst_") else key] = (
            number_from_csv(metrics_row, key, float, 0) if metrics_row is not None else 0
        )
    return values


def run_experiment(python, scenario_name, length, input_path, dag_path, output_dir):
    with input_path.open() as f:
        config = json.load(f)

    run_output_dir = output_dir / "logs" / "stdout"
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

    run_dir = find_latest_run_dir(
        config["experiment_name"], config["arrival_rate"], Path(config["log_dir"])
    )
    metrics_row = load_first_graph_metrics_row(run_dir)
    result = load_first_result(run_dir) if metrics_row is None else None
    row = {
        "scenario": scenario_name,
        "chain_length": length,
        "graph": chain_label(length),
        "run_dir": str(run_dir),
        "stdout_log": str(stdout_path),
    }
    row.update(result_graph_metrics(metrics_row, result))
    return row


def save_results_csv(rows, output_prefix, output_dir):
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / f"{output_prefix}_results.csv"
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=GRAPH_RESULT_FIELDNAMES, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    return csv_path


def plot_metric(rows, selected_scenarios, chain_lengths, metric, ylabel, output_name, output_dir):
    output_dir.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(10, 5))
    for scenario_name in selected_scenarios:
        scenario_rows = sorted(
            [row for row in rows if row["scenario"] == scenario_name],
            key=lambda row: row["chain_length"],
        )
        points = [
            (row["chain_length"], row.get(metric))
            for row in scenario_rows
            if row.get(metric) is not None
        ]
        if not points:
            continue
        xs = [point[0] for point in points]
        ys = [point[1] for point in points]
        plt.plot(xs, ys, marker="o", label=scenario_name)
    plt.xlabel("Chain length")
    plt.ylabel(ylabel)
    tick_step = 1 if len(chain_lengths) <= 15 else 5
    ticks = [x for x in chain_lengths if (x - chain_lengths[0]) % tick_step == 0]
    if chain_lengths[-1] not in ticks:
        ticks.append(chain_lengths[-1])
    plt.xticks(ticks)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    output_path = output_dir / output_name
    plt.savefig(output_path, dpi=160)
    plt.close()
    return output_path


def metric_float(row, key):
    if key not in row or row[key] in ("", None):
        return None
    try:
        return float(row[key])
    except (TypeError, ValueError):
        return None


def save_combined_metrics_csv(rows, output_prefix, output_dir):
    metrics_rows = []
    for row in rows:
        run_dir = Path(row["run_dir"])
        for metric_row in load_metrics_rows(run_dir):
            enriched = {
                "scenario": row["scenario"],
                "chain_length": row["chain_length"],
                "graph": row["graph"],
                "run_dir": row["run_dir"],
                "stdout_log": row["stdout_log"],
            }
            enriched.update(metric_row)
            metrics_rows.append(enriched)

    if not metrics_rows:
        return None, []

    fieldnames = [
        "scenario",
        "chain_length",
        "graph",
        "run_dir",
        "stdout_log",
    ]
    for metric_row in metrics_rows:
        for key in metric_row:
            if key not in fieldnames:
                fieldnames.append(key)

    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / f"{output_prefix}_metrics.csv"
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(metrics_rows)
    return csv_path, metrics_rows


def plot_graph_metrics(rows, selected_scenarios, chain_lengths, output_prefix, output_dir):
    plot_dir = output_dir / "plots" / "graph"
    figures = []
    for metric, ylabel, suffix in GRAPH_PLOT_SPECS:
        if not any(row.get(metric) not in ("", None) for row in rows):
            continue
        figures.append(
            plot_metric(
                rows,
                selected_scenarios,
                chain_lengths,
                metric,
                ylabel,
                f"{output_prefix}_{suffix}.png",
                output_dir=plot_dir,
            )
        )
    return figures


def plot_node_heatmap(metric_rows, selected_scenarios, chain_lengths, metric, ylabel, output_name, output_dir):
    output_dir.mkdir(parents=True, exist_ok=True)
    figures = []
    for scenario_name in selected_scenarios:
        scenario_rows = [
            row for row in metric_rows
            if row.get("scenario") == scenario_name and row.get("level") == "node"
        ]
        values = [
            metric_float(row, metric)
            for row in scenario_rows
            if metric_float(row, metric) is not None
        ]
        if not values:
            continue

        max_node = max(int(float(row["node_index"])) for row in scenario_rows if row.get("node_index"))
        chain_index = {length: idx for idx, length in enumerate(chain_lengths)}
        matrix = [[float("nan") for _ in range(max_node)] for _ in chain_lengths]

        for row in scenario_rows:
            length = int(row["chain_length"])
            if length not in chain_index or not row.get("node_index"):
                continue
            value = metric_float(row, metric)
            if value is None:
                continue
            node_index = int(float(row["node_index"])) - 1
            matrix[chain_index[length]][node_index] = value

        plt.figure(figsize=(11, 7))
        cmap = plt.get_cmap("viridis").copy()
        cmap.set_bad("#f2f2f2")
        image = plt.imshow(matrix, aspect="auto", origin="lower", interpolation="nearest", cmap=cmap)
        plt.colorbar(image, label=ylabel)
        plt.xlabel("Node index")
        plt.ylabel("Chain length")

        x_ticks = list(range(max_node))
        x_step = 1 if max_node <= 20 else 5
        plt.xticks(x_ticks[::x_step], [idx + 1 for idx in x_ticks[::x_step]])

        y_ticks = list(range(len(chain_lengths)))
        y_step = 1 if len(chain_lengths) <= 20 else 5
        plt.yticks(y_ticks[::y_step], [chain_lengths[idx] for idx in y_ticks[::y_step]])

        plt.title(f"{scenario_name}: {ylabel}")
        plt.tight_layout()
        output_path = output_dir / f"{output_name}_{scenario_name}.png"
        plt.savefig(output_path, dpi=160)
        plt.close()
        figures.append(output_path)
    return figures


def plot_node_heatmaps(metric_rows, selected_scenarios, chain_lengths, output_prefix, output_dir):
    plot_dir = output_dir / "plots" / "node"
    figures = []
    for metric, ylabel, suffix in NODE_HEATMAP_SPECS:
        figures.extend(
            plot_node_heatmap(
                metric_rows,
                selected_scenarios,
                chain_lengths,
                metric,
                ylabel,
                f"{output_prefix}_{suffix}",
                plot_dir,
            )
        )
    return figures


def relative_path(path):
    if not path.is_absolute():
        return path
    try:
        return path.relative_to(ROOT)
    except ValueError:
        return path


def save_metric_report(rows, output_prefix, output_dir, csv_path, metrics_csv_path, graph_figures, node_figures):
    report_path = output_dir / f"{output_prefix}_metric_report.md"
    final_rows = sorted(rows, key=lambda row: (row["scenario"], row["chain_length"]))
    last_row = final_rows[-1] if final_rows else None

    lines = [
        "# Sanitycheck Metric Report",
        "",
        "This report is generated from the run-level CSV outputs. It includes graph-level line plots and node-level heatmaps for response time, request counts, probabilities, instance counts, and configured distribution parameters.",
        "",
        "## Source Data",
        "",
        f"- Results CSV: `{relative_path(csv_path)}`",
    ]
    if metrics_csv_path:
        lines.append(f"- Combined metrics CSV: `{relative_path(metrics_csv_path)}`")
    lines.extend([
        "",
        "## Final Chain-Length Snapshot",
        "",
        "| scenario | L | time cost | event cost | avg RT | p95 RT | warm reqs | cold reqs | queued reqs | rejected reqs | p cold | p warm | p queued | p reject |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ])
    if last_row:
        lines.append(
            "| {scenario} | {chain_length} | {graph_time_avg_cost:.4f} | "
            "{graph_event_avg_cost:.4f} | {graph_response_time_avg:.4f} | "
            "{graph_response_time_p95:.4f} | {graph_reqs_warm} | {graph_reqs_cold} | "
            "{graph_reqs_queued} | {graph_reqs_reject} | {graph_prob_cold:.4f} | "
            "{graph_prob_warm:.4f} | {graph_prob_queued:.4f} | {graph_prob_reject:.4f} |".format(**last_row)
        )

    lines.extend([
        "",
        "## Interpretation",
        "",
        "- Cost alone is not enough: lower resource cost can coincide with higher rejection.",
        "- Response-time metrics are reported separately from cost, so they describe service quality without changing the optimizer objective.",
        "- Warm, cold, queued, init, and rejection metrics separate service quality from resource usage.",
        "- Node heatmaps show whether graph-level behavior is spread across the chain or concentrated at specific nodes.",
        "- Distribution-rate plots document the configured rates and mean times used to make longer chains comparable.",
        "",
        "## Graph-Level Plots",
        "",
    ])
    for figure in graph_figures:
        lines.append(f"- `{relative_path(figure)}`")

    lines.extend([
        "",
        "## Node-Level Heatmaps",
        "",
    ])
    for figure in node_figures:
        lines.append(f"- `{relative_path(figure)}`")

    report_path.write_text("\n".join(lines) + "\n")
    return report_path


def save_summary(rows, output_prefix, output_dir):
    summary_path = output_dir / f"{output_prefix}_summary.md"
    lines = [
        "# Sanitycheck Results",
        "",
        "| scenario | L | time avg cost | event avg cost | avg RT | p95 RT | external | internal | warm | cold | queued | rejected | p cold | p warm | p queued | p reject |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in sorted(rows, key=lambda item: (item["scenario"], item["chain_length"])):
        lines.append(
            "| {scenario} | {chain_length} | {graph_time_avg_cost:.4f} | "
            "{graph_event_avg_cost:.4f} | {graph_response_time_avg:.4f} | "
            "{graph_response_time_p95:.4f} | {external_arrivals} | {internal_arrivals} | "
            "{graph_reqs_warm} | {graph_reqs_cold} | {graph_reqs_queued} | {graph_reqs_reject} | "
            "{graph_prob_cold:.4f} | {graph_prob_warm:.4f} | {graph_prob_queued:.4f} | "
            "{graph_prob_reject:.4f} |".format(**row)
        )
    summary_path.write_text("\n".join(lines) + "\n")
    return summary_path


def parse_args():
    parser = argparse.ArgumentParser(description="Generate and run graph sanitycheck experiments.")
    parser.add_argument("--max-length", type=int, default=DEFAULT_MAX_LENGTH)
    parser.add_argument("--min-length", type=int, default=1)
    parser.add_argument(
        "--scenario",
        action="append",
        choices=sorted(SCENARIOS.keys()),
        help="Scenario to run. Repeat for multiple. Defaults to all scenarios.",
    )
    parser.add_argument(
        "--output-prefix",
        default="sanitycheck",
        help="Output folder for configs, logs, CSVs, summaries, and figures.",
    )
    parser.add_argument(
        "--max-concurrency",
        type=int,
        default=MAX_CONCURRENCY,
        help="Max concurrency value to write into generated input configs.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    if args.min_length < 1 or args.max_length < args.min_length:
        raise ValueError("--max-length must be >= --min-length >= 1")

    output_dir = output_root(args.output_prefix)
    artifact_prefix = output_dir.name
    python = PYTHON if PYTHON.exists() else Path(sys.executable)
    selected_scenarios = args.scenario or list(SCENARIOS.keys())
    chain_lengths = list(range(args.min_length, args.max_length + 1))
    generated = generate_configs(selected_scenarios, chain_lengths, output_dir, args.max_concurrency)

    rows = []
    for scenario_name, length, input_path, dag_path in generated:
        rows.append(run_experiment(python, scenario_name, length, input_path, dag_path, output_dir))

    csv_path = save_results_csv(rows, artifact_prefix, output_dir)
    metrics_csv_path, metric_rows = save_combined_metrics_csv(rows, artifact_prefix, output_dir)
    cost_fig = plot_metric(
        rows,
        selected_scenarios,
        chain_lengths,
        "graph_time_avg_cost",
        "Graph time-average cost",
        f"{artifact_prefix}_time_avg_cost.png",
        output_dir / "plots" / "summary",
    )
    event_cost_fig = plot_metric(
        rows,
        selected_scenarios,
        chain_lengths,
        "graph_event_avg_cost",
        "Graph event-average cost",
        f"{artifact_prefix}_event_avg_cost.png",
        output_dir / "plots" / "summary",
    )
    external_fig = plot_metric(
        rows,
        selected_scenarios,
        chain_lengths,
        "external_arrivals",
        "External arrivals",
        f"{artifact_prefix}_external_arrivals.png",
        output_dir / "plots" / "summary",
    )
    cold_fig = plot_metric(
        rows,
        selected_scenarios,
        chain_lengths,
        "graph_prob_cold",
        "Cold-start probability",
        f"{artifact_prefix}_cold_probability.png",
        output_dir / "plots" / "summary",
    )
    summary_path = save_summary(rows, artifact_prefix, output_dir)
    graph_figures = plot_graph_metrics(rows, selected_scenarios, chain_lengths, artifact_prefix, output_dir)
    node_figures = plot_node_heatmaps(metric_rows, selected_scenarios, chain_lengths, artifact_prefix, output_dir)
    report_path = save_metric_report(
        rows,
        artifact_prefix,
        output_dir,
        csv_path,
        metrics_csv_path,
        graph_figures,
        node_figures,
    )

    print("\nDone.")
    print(f"Output folder: {output_dir}")
    print(f"Generated configs: {output_dir / 'configs'}")
    print(f"Simulator logs: {output_dir / 'logs'}")
    print(f"CSV: {csv_path}")
    if metrics_csv_path:
        print(f"Metrics CSV: {metrics_csv_path}")
    print(f"Summary: {summary_path}")
    print(f"Metric report: {report_path}")
    print(f"Figures: {cost_fig}, {event_cost_fig}, {external_fig}, {cold_fig}")
    print(f"Generated {len(graph_figures)} graph metric plots and {len(node_figures)} node heatmaps.")


if __name__ == "__main__":
    main()
