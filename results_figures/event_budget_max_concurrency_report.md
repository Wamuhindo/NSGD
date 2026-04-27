# Event-Budget Sanitycheck Report

## Purpose

This report summarizes the `event_budget` sanitycheck runs for chain lengths `n=1..20`, with and without a tight maximum concurrency cap. It also explains why graph-level and node-level metrics are both needed, and why the lower cost under `max_concurrency=50` should not be interpreted as better performance by itself.

## Experiment Setup

Two runs are compared:

| run | max_concurrency | output |
|---|---:|---|
| baseline | 100000 | `results_figures/event_budget_n1_20_results.csv` |
| capped | 50 | `results_figures/event_budget_maxconcurrency50_n1_20_results.csv` |

Both runs use the `event_budget` scenario, so the simulator uses the same update/event budget for every chain length. As the chain grows, each external arrival creates more internal arrivals, so external arrivals decrease while the graph-level event/request count stays around the same order.

## Graph-Level Meaning

Graph-level metrics aggregate over the whole DAG/chain, not over a single node. For a chain like:

```text
A1 -> A2 -> A3 -> A4
```

`graph_prob_cold` is computed from cold-start counts across all nodes divided by total graph requests across all nodes. It answers:

```text
Across the entire workflow, what fraction of function executions saw a cold start?
```

This is useful because a user request experiences the whole chain, not only one node. However, graph-level metrics hide where the behavior comes from. That is why node-level metrics are needed too.

## Why Node-Level Metrics Are Needed

The graph-level cold-start probability, rejection probability, and cost are useful summary numbers, but they can hide uneven behavior across the chain. For example, a graph-level cold probability of `0.04` could mean:

- every node has about `4%` cold starts, or
- one node is much worse and the others are fine.

For this reason, each run now writes CSV data in the log path:

| file | contents |
|---|---|
| `metrics.csv` | per-run graph row plus one row per node |
| `all_runs_metrics.csv` | combined metrics for all runs in that experiment log directory |

The metrics include:

- request counts: total, cold, warm, rejected
- probabilities: cold-start and rejection
- average instance counts: total, running, idle, init-free, init-reserved, queued
- configured rates and mean times for warm service, cold service, cold start, and expiration

This is needed so plots and final summaries can be regenerated from compact CSV data, without re-parsing large JSON logs, and so graph-level conclusions can be checked against node-level behavior.

## Max-Concurrency Implementation Check

The `max_concurrency` cap is implemented as a graph-wide resource pool.

Relevant code paths:

- `AutoscalerFaasScalarVectorial/simulate.py`: graph-wide active concurrency uses `running + init_reserved + init_free_booked`.
- `AutoscalerFaasScalarVectorial/simulate.py`: graph-wide allocated capacity uses total `server_count` across nodes.
- Arrivals reject when active concurrency is full or there are no cold slots left.

A direct validation run for `L=20` with `max_concurrency=50` gave:

```text
config_max_concurrency=50
history_points=50000
max_allocated_servers=50
max_running=17
max_init_reserved=19
max_active_proxy_running_reserved_queued=24
graph_reqs_total=24216
graph_reqs_reject=784
graph_prob_reject=0.032375
```

The important check is:

```text
max_allocated_servers=50
```

So the cap is being enforced correctly. The capped run also produces nonzero rejection, which confirms that the cap is binding.

## Result Comparison

Selected chain lengths:

| L | baseline time cost | capped time cost | delta | baseline event cost | capped event cost | delta | baseline p_cold | capped p_cold | baseline p_reject | capped p_reject |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 118.5565 | 118.5565 | 0.0000 | 122.5062 | 122.5062 | 0.0000 | 0.0078 | 0.0078 | 0.0000 | 0.0000 |
| 4 | 235.5215 | 218.6725 | -16.8490 | 249.8990 | 228.0835 | -21.8155 | 0.0174 | 0.0161 | 0.0000 | 0.0015 |
| 10 | 400.7621 | 367.2920 | -33.4701 | 427.5410 | 375.8069 | -51.7341 | 0.0304 | 0.0275 | 0.0000 | 0.0062 |
| 15 | 553.0113 | 431.3388 | -121.6726 | 605.1927 | 426.9247 | -178.2680 | 0.0414 | 0.0363 | 0.0000 | 0.0191 |
| 20 | 606.6242 | 446.3876 | -160.2366 | 661.4637 | 438.0662 | -223.3975 | 0.0465 | 0.0444 | 0.0000 | 0.0324 |

At `L=20`, the capped run has lower cost:

```text
time-average cost: 606.6242 -> 446.3876
event-average cost: 661.4637 -> 438.0662
```

But it also rejects requests:

```text
rejection probability: 0.0000 -> 0.0324
```

This means about `3.24%` of graph requests are rejected in the capped run.

## Why Capped Cost Looks Better

The current cost function is mostly a resource occupancy cost:

```text
cost = idle_on * 2
     + busy * 1
     + initializing * 5
     + init_reserved * 100
     + rejected_event * 200
```

This comes from `AutoscalerFaasScalarVectorial/Algorithm.py`.

When `max_concurrency=50`, the simulator is forced to keep fewer instances and fewer reserved/initializing slots alive. That reduces resource occupancy cost. The rejection penalty is only applied as an event indicator, not as a persistent penalty over time, so the resource savings can outweigh the rejection penalty.

Therefore:

```text
lower capped cost != better performance
```

It means:

```text
lower resource cost + higher rejection
```

The capped system is cheaper under the current objective, but lower quality from a service perspective.

## Why We Need These Metrics

We need both graph-level and node-level CSV metrics because cost alone can be misleading. In the capped run, cost decreases, but rejection increases. Without `p_reject`, the capped system would look better. Without node-level rows, we cannot tell which parts of the chain are responsible for cold starts, queueing, or rejection.

For future plots and reports, the minimum useful metric set is:

- graph time-average cost
- graph event-average cost
- graph cold-start probability
- graph rejection probability
- graph total requests, external arrivals, internal arrivals
- node-level cold-start probability
- node-level rejection probability
- node-level warm/cold/queued request counts
- configured rates and mean times for warm service, cold service, cold start, and expiration

## Recommendation

Do not rank policies by cost alone. Use one of these approaches:

1. Report cost and rejection probability side by side.
2. Add an SLA-aware score, for example:

```text
effective_cost = graph_time_avg_cost + large_penalty * graph_prob_reject
```

3. Increase `w_rej` substantially if rejection should be treated as unacceptable.
4. Plot node-level cold and reject probabilities to identify where the graph is failing.

The current result is useful because it confirms the cap works, but it also shows why the final analysis must include rejection and node-level metrics.

## Generated Artifacts

Baseline run:

- `results_figures/event_budget_n1_20_results.csv`
- `results_figures/event_budget_n1_20_summary.md`

Capped run:

- `results_figures/event_budget_maxconcurrency50_n1_20_results.csv`
- `results_figures/event_budget_maxconcurrency50_n1_20_summary.md`

Per-run metrics:

- `logs/sanitycheck_event_budget_L*/.../metrics.csv`
- `logs/sanitycheck_event_budget_L*/all_runs_metrics.csv`
