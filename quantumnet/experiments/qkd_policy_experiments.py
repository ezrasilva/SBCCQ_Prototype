from __future__ import annotations

import csv
import hashlib
import os
import random
import statistics
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from typing import Any, Iterable, Sequence

try:
    import numpy as np  # Optional; used if available
except Exception:  # pragma: no cover
    np = None

from quantumnet.topology import Network

DEFAULT_POLICIES: tuple[str, ...] = ("threshold", "on_demand", "hybrid")


def stable_seed(base_seed: int, *parts: Any) -> int:
    """Generate a deterministic 32-bit seed from a base seed + arbitrary parts.

    Avoid using Python's built-in hash() which is process-randomized.
    """
    payload = "|".join(str(p) for p in parts)
    digest = hashlib.sha256(f"{int(base_seed)}|{payload}".encode("utf-8")).digest()
    return int.from_bytes(digest[:4], "big", signed=False)


def seed_global_rng(seed: int) -> None:
    """Seed all RNG sources used by the simulator."""
    random.seed(int(seed))
    if np is not None:
        np.random.seed(int(seed) % (2**32 - 1))


@dataclass(frozen=True)
class LinkPair:
    alice_id: int
    bob_id: int

    def as_tuple(self) -> tuple[int, int]:
        return tuple(sorted((int(self.alice_id), int(self.bob_id))))


def build_network_for_experiment(
    *,
    topology_name: str,
    topology_nodes: int,
    policy: str,
    link_pairs: Sequence[tuple[int, int]],
    min_threshold_bits: int | None,
    buffer_capacity_bits: int | None,
    seed: int,
) -> Network:
    """Create a Network configured consistently for one trial."""
    seed_global_rng(seed)

    net = Network()
    net.set_ready_topology(topology_name, int(topology_nodes))
    net.controller.set_policy(policy)

    # Configure link-level parameters.
    for alice_id, bob_id in link_pairs:
        edge = tuple(sorted((int(alice_id), int(bob_id))))
        edge_data = net.graph.edges[edge]

        if buffer_capacity_bits is not None:
            edge_data["qkd_max_buffer_bits"] = max(0, int(buffer_capacity_bits))
        else:
            edge_data["qkd_max_buffer_bits"] = None

        if min_threshold_bits is not None:
            net.controller.set_minimum_stock(int(alice_id), int(bob_id), int(min_threshold_bits))

    return net


def _sum_link_metric(net: Network, link_pairs: Sequence[tuple[int, int]], metric: str) -> int:
    total = 0
    for alice_id, bob_id in link_pairs:
        state = net.get_qkd_link_state(int(alice_id), int(bob_id))
        total += int(state.get(metric, 0) or 0)
    return total


def _get_capacity_bits(net: Network, link_pairs: Sequence[tuple[int, int]]) -> int | None:
    capacities: list[int] = []
    for alice_id, bob_id in link_pairs:
        state = net.get_qkd_link_state(int(alice_id), int(bob_id))
        cap = state.get("max_buffer_bits", None)
        if cap is None:
            continue
        capacities.append(int(cap))
    if not capacities:
        return None
    # If capacities differ across links, we keep the sum capacity for utilization aggregation.
    return sum(capacities)


def run_trial_from_request_schedule(
    *,
    policy: str,
    trial_id: int,
    base_seed: int,
    topology_name: str,
    topology_nodes: int,
    link_pairs: Sequence[tuple[int, int]],
    request_schedule: Sequence[tuple[int, int, int]],
    min_threshold_bits: int | None,
    buffer_capacity_bits: int | None,
) -> dict[str, Any]:
    """Run a single trial given a deterministic request schedule.

    request_schedule items: (alice_id, bob_id, requested_bits)
    """
    schedule_len = len(request_schedule)
    schedule_requested_bits = sum(int(item[2]) for item in request_schedule) if request_schedule else 0
    seed = stable_seed(
        base_seed,
        "trial",
        trial_id,
        "policy",
        policy,
        "topology",
        topology_name,
        topology_nodes,
        "min_threshold_bits",
        min_threshold_bits,
        "buffer_capacity_bits",
        buffer_capacity_bits,
        "schedule_len",
        schedule_len,
        "schedule_requested_bits",
        schedule_requested_bits,
    )
    net = build_network_for_experiment(
        topology_name=topology_name,
        topology_nodes=topology_nodes,
        policy=policy,
        link_pairs=link_pairs,
        min_threshold_bits=min_threshold_bits,
        buffer_capacity_bits=buffer_capacity_bits,
        seed=seed,
    )

    # Buffer sampling (occupation) before each request, aggregated across links.
    buffer_samples_sum = 0
    buffer_samples_count = 0

    for alice_id, bob_id, requested_bits in request_schedule:
        state_before = net.get_qkd_link_state(int(alice_id), int(bob_id))
        buffer_samples_sum += int(state_before.get("bits_available", 0) or 0)
        buffer_samples_count += 1

        net.controller.handle_key_request(int(alice_id), int(bob_id), int(requested_bits))

    totals = {
        "total_generated_bits": _sum_link_metric(net, link_pairs, "total_generated_bits"),
        "total_dropped_bits": _sum_link_metric(net, link_pairs, "total_dropped_bits"),
        "total_consumed_bits": _sum_link_metric(net, link_pairs, "total_consumed_bits"),
        "total_requested_bits": _sum_link_metric(net, link_pairs, "total_requested_bits"),
        "served_requests": _sum_link_metric(net, link_pairs, "served_requests"),
        "failed_requests": _sum_link_metric(net, link_pairs, "failed_requests"),
        "denied_requests": _sum_link_metric(net, link_pairs, "denied_requests"),
        "replenishment_events": _sum_link_metric(net, link_pairs, "replenishment_events"),
        "bits_available": _sum_link_metric(net, link_pairs, "bits_available"),
    }

    served = int(totals["served_requests"])
    denied = int(totals["denied_requests"])
    failed = int(totals["failed_requests"])
    total_reqs = served + denied + failed

    service_rate = served / total_reqs if total_reqs > 0 else 0.0
    denial_rate = (denied + failed) / total_reqs if total_reqs > 0 else 0.0

    generated = int(totals["total_generated_bits"])
    consumed = int(totals["total_consumed_bits"])
    efficiency = consumed / generated if generated > 0 else 0.0

    buffer_mean_bits = buffer_samples_sum / buffer_samples_count if buffer_samples_count else 0.0

    total_capacity = _get_capacity_bits(net, link_pairs)
    if total_capacity and total_capacity > 0:
        # Average utilization across links (sampled per request) relative to total capacity.
        # Each sample is for a single link; scale by number of links.
        links = max(1, len(link_pairs))
        buffer_util_mean = (buffer_mean_bits * links) / total_capacity
    else:
        buffer_util_mean = None

    return {
        "policy": policy,
        "trial": int(trial_id),
        "seed": int(seed),
        **totals,
        "total_requests": int(total_reqs),
        "service_rate": float(service_rate),
        "denial_rate": float(denial_rate),
        "efficiency": float(efficiency),
        "buffer_mean_bits": float(buffer_mean_bits),
        "buffer_util_mean": (float(buffer_util_mean) if buffer_util_mean is not None else None),
    }


def aggregate_trials(
    rows: Sequence[dict[str, Any]],
    *,
    group_keys: Sequence[str],
    metric_keys: Sequence[str],
) -> list[dict[str, Any]]:
    """Group trial rows and compute mean/std for each metric."""
    buckets: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
    for row in rows:
        key = tuple(row.get(k) for k in group_keys)
        buckets.setdefault(key, []).append(row)

    summary: list[dict[str, Any]] = []
    for key, group_rows in sorted(buckets.items(), key=lambda kv: kv[0]):
        out: dict[str, Any] = {k: v for k, v in zip(group_keys, key)}
        out["n"] = len(group_rows)

        for metric in metric_keys:
            values = [r.get(metric) for r in group_rows]
            values = [float(v) for v in values if v is not None]

            if not values:
                out[f"{metric}_mean"] = None
                out[f"{metric}_std"] = None
                continue

            out[f"{metric}_mean"] = float(statistics.mean(values))
            out[f"{metric}_std"] = float(statistics.stdev(values)) if len(values) > 1 else 0.0

        summary.append(out)

    return summary


def write_csv(path: str, rows: Sequence[dict[str, Any]]) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    if not rows:
        raise ValueError("No rows to write")

    fieldnames: list[str] = []
    for row in rows:
        for key in row.keys():
            if key not in fieldnames:
                fieldnames.append(key)

    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _run_trial_task(task: dict[str, Any]) -> dict[str, Any]:
    """Worker entrypoint for multiprocessing.

    `task` must contain only picklable values.
    """
    base = run_trial_from_request_schedule(
        policy=task["policy"],
        trial_id=task["trial_id"],
        base_seed=task["base_seed"],
        topology_name=task["topology_name"],
        topology_nodes=task["topology_nodes"],
        link_pairs=task["link_pairs"],
        request_schedule=task["request_schedule"],
        min_threshold_bits=task.get("min_threshold_bits", None),
        buffer_capacity_bits=task.get("buffer_capacity_bits", None),
    )
    extra = task.get("extra", {})
    if extra:
        base.update(extra)
    return base


def run_trials(
    tasks: Sequence[dict[str, Any]],
    *,
    n_jobs: int = 1,
    mp_start_method: str = "spawn",
) -> list[dict[str, Any]]:
    """Run many independent trials, optionally in parallel.

    This is intended for speeding up 30x repetitions per setting.
    Use `n_jobs=1` for the most notebook-friendly behavior.

    Args:
        tasks: Sequence of task dicts accepted by `_run_trial_task`.
        n_jobs: Number of worker processes. If <=1, runs serially.
        mp_start_method: 'spawn' (safer in notebooks) or 'fork' (Linux only).

    Returns:
        List of per-trial result dicts.
    """
    if not tasks:
        return []

    workers = int(n_jobs)
    if workers <= 1:
        return [_run_trial_task(t) for t in tasks]

    # In notebooks, 'spawn' tends to be more stable than 'fork'.
    ctx = mp.get_context(mp_start_method)
    with ProcessPoolExecutor(max_workers=workers, mp_context=ctx) as ex:
        return list(ex.map(_run_trial_task, tasks))
