"""Reachability diagnostics for the 3-D vascular swarm benchmark.

This script is a research diagnostic, not a policy benchmark.  In particular,
``topology_oracle`` uses the full centerline graph and clot locations to test
whether the shared actuator can physically reach both branch clots in sequence.
"""
from __future__ import annotations

import argparse
import csv
import heapq
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Callable

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from environments.vascular_3d_swarm_env import Vascular3DSwarmEnv


Controller = Callable[[Vascular3DSwarmEnv], np.ndarray]


def _action_from_direction(direction: np.ndarray, morphology: float) -> np.ndarray:
    norm = float(np.linalg.norm(direction))
    velocity = direction / norm if norm > 1e-8 else np.zeros(3, dtype=np.float32)
    return np.asarray([velocity[0], velocity[1], velocity[2], morphology], dtype=np.float32)


def _nearest_indices(points: np.ndarray, positions: np.ndarray) -> np.ndarray:
    distances = np.linalg.norm(positions[:, None, :] - points[None, :, :], axis=2)
    return np.argmin(distances, axis=1)


def _branch_counts(env: Vascular3DSwarmEnv) -> dict[int, int]:
    assert env.scene is not None
    indices = _nearest_indices(env.scene.points, env.robot_positions)
    counts = Counter(int(env.scene.branch_ids[i]) for i in indices)
    return dict(sorted(counts.items()))


def _encode_counts(counts: dict[int, int]) -> str:
    return ",".join(f"{branch}:{count}" for branch, count in sorted(counts.items()))


def _scene_graph(points: np.ndarray) -> list[list[tuple[int, float]]]:
    if len(points) < 2:
        return [[] for _ in range(len(points))]
    pairwise = np.linalg.norm(points[:, None, :] - points[None, :, :], axis=2)
    nearest = np.partition(pairwise + np.eye(len(points)) * 10.0, 1, axis=1)[:, 1]
    threshold = max(0.025, float(np.percentile(nearest, 90)) * 1.8)
    graph: list[list[tuple[int, float]]] = [[] for _ in range(len(points))]
    for i in range(len(points)):
        for j in range(i + 1, len(points)):
            distance = float(pairwise[i, j])
            if 1e-6 < distance <= threshold:
                graph[i].append((j, distance))
                graph[j].append((i, distance))
    return graph


def _dijkstra_path(graph: list[list[tuple[int, float]]], start: int, goal: int) -> list[int]:
    queue: list[tuple[float, int]] = [(0.0, start)]
    previous = {start: -1}
    best = {start: 0.0}
    while queue:
        distance, node = heapq.heappop(queue)
        if node == goal:
            break
        if distance > best.get(node, math.inf):
            continue
        for neighbor, weight in graph[node]:
            candidate = distance + weight
            if candidate < best.get(neighbor, math.inf):
                best[neighbor] = candidate
                previous[neighbor] = node
                heapq.heappush(queue, (candidate, neighbor))
    if goal not in previous:
        return [start]
    path = [goal]
    while path[-1] != start:
        path.append(previous[path[-1]])
    path.reverse()
    return path


def _junction_indices(graph: list[list[tuple[int, float]]]) -> list[int]:
    indices = [i for i, edges in enumerate(graph) if len(edges) >= 3]
    return indices if indices else [0]


def _constant_forward(morphology: float) -> Controller:
    def act(_env: Vascular3DSwarmEnv) -> np.ndarray:
        return np.asarray([1.0, 0.0, 0.0, morphology], dtype=np.float32)

    return act


def _greedy_euclidean(morphology: float) -> Controller:
    def act(env: Vascular3DSwarmEnv) -> np.ndarray:
        active = np.where(env.clot_masses > 0.0)[0]
        if len(active) == 0:
            return np.asarray([0.0, 0.0, 0.0, morphology], dtype=np.float32)
        centroid = env.robot_positions.mean(axis=0)
        clot_delta = env.clot_positions[active] - centroid[None, :]
        target = active[int(np.argmin(np.linalg.norm(clot_delta, axis=1)))]
        direction = env.clot_positions[target] - centroid
        return _action_from_direction(direction, morphology)

    return act


def _topology_oracle(morphology: float, lookahead: int) -> Controller:
    graph_cache: dict[int, list[list[tuple[int, float]]]] = {}

    def act(env: Vascular3DSwarmEnv) -> np.ndarray:
        assert env.scene is not None
        scene_id = id(env.scene)
        if scene_id not in graph_cache:
            graph_cache[scene_id] = _scene_graph(env.scene.points)
        graph = graph_cache[scene_id]
        active = np.where(env.clot_masses > 0.0)[0]
        if len(active) == 0:
            return np.asarray([0.0, 0.0, 0.0, morphology], dtype=np.float32)
        centroid = env.robot_positions.mean(axis=0)
        start = int(_nearest_indices(env.scene.points, centroid[None, :])[0])
        clot_nodes = _nearest_indices(env.scene.points, env.clot_positions[active])
        paths = [_dijkstra_path(graph, start, int(node)) for node in clot_nodes]
        path_lengths = [
            sum(float(np.linalg.norm(env.scene.points[a] - env.scene.points[b])) for a, b in zip(path[:-1], path[1:]))
            for path in paths
        ]
        path = paths[int(np.argmin(path_lengths))]
        waypoint = env.scene.points[path[min(lookahead, len(path) - 1)]]
        direction = waypoint - centroid
        return _action_from_direction(direction, morphology)

    return act


def run_episode(
    scenario: str,
    controller_name: str,
    controller: Controller,
    seed: int,
    num_robots: int,
    num_clots: int,
    horizon: int,
    lookahead: int,
) -> dict[str, object]:
    env = Vascular3DSwarmEnv(
        scenario=scenario,
        num_robots=num_robots,
        num_clots=num_clots,
        horizon=horizon,
        seed=seed,
    )
    env.reset(seed=seed)
    assert env.scene is not None
    graph = _scene_graph(env.scene.points)
    junctions = _junction_indices(graph)
    clot_nodes = _nearest_indices(env.scene.points, env.clot_positions)
    clot_branches = [int(env.scene.branch_ids[i]) for i in clot_nodes]
    first_contact = [-1] * num_clots
    first_cleared_step = -1
    returned_to_junction_after_clear = False
    min_junction_distance_after_clear = math.inf
    max_branches_occupied = 0
    centroid_branch_visits: set[int] = set()
    previous_masses = env.clot_masses.copy()
    terminal_info: dict[str, object] = {}

    for step in range(horizon):
        action = controller(env)
        _obs, _reward, terminated, truncated, info = env.step(action)
        terminal_info = info
        counts = _branch_counts(env)
        max_branches_occupied = max(max_branches_occupied, len(counts))
        centroid = env.robot_positions.mean(axis=0)
        centroid_node = int(_nearest_indices(env.scene.points, centroid[None, :])[0])
        centroid_branch_visits.add(int(env.scene.branch_ids[centroid_node]))
        for clot_id in range(num_clots):
            if first_contact[clot_id] < 0 and bool(np.any(env._last_contact[:, clot_id])):
                first_contact[clot_id] = step + 1
        newly_cleared = np.where((previous_masses > 0.0) & (env.clot_masses <= 0.0))[0]
        if first_cleared_step < 0 and len(newly_cleared) > 0 and np.any(env.clot_masses > 0.0):
            first_cleared_step = step + 1
        if first_cleared_step >= 0 and np.any(env.clot_masses > 0.0):
            junction_distance = float(
                np.min(np.linalg.norm(env.scene.points[junctions] - centroid[None, :], axis=1))
            )
            min_junction_distance_after_clear = min(min_junction_distance_after_clear, junction_distance)
            if junction_distance <= env.tube_radius * 1.5:
                returned_to_junction_after_clear = True
        previous_masses = env.clot_masses.copy()
        if terminated or truncated:
            break

    if not terminal_info:
        terminal_info = {
            "success": False,
            "remaining_mass": float(env.clot_masses.sum()),
            "collision_rate": 0.0,
            "path_length": env.path_length,
        }
    final_counts = _branch_counts(env)
    removal_rate = 1.0 - float(env.clot_masses.sum()) / float(num_clots)
    contacted = [x for x in first_contact if x >= 0]
    return {
        "scenario": scenario,
        "controller": controller_name,
        "seed": seed,
        "robots": num_robots,
        "clots": num_clots,
        "horizon": horizon,
        "lookahead": lookahead if controller_name == "topology_oracle" else "",
        "success": int(bool(terminal_info.get("success", False))),
        "removal_rate": removal_rate,
        "remaining_mass": float(env.clot_masses.sum()),
        "first_contact_step": min(contacted) if contacted else -1,
        "per_clot_first_contact": ",".join(str(x) for x in first_contact),
        "clot_branches": ",".join(str(x) for x in clot_branches),
        "final_branch_occupancy": _encode_counts(final_counts),
        "max_branches_occupied": max_branches_occupied,
        "centroid_branch_visits": ",".join(str(x) for x in sorted(centroid_branch_visits)),
        "first_cleared_step": first_cleared_step,
        "returned_to_junction_after_first_clear": int(returned_to_junction_after_clear),
        "min_junction_distance_after_first_clear": (
            min_junction_distance_after_clear if math.isfinite(min_junction_distance_after_clear) else -1.0
        ),
        "collision_rate": float(env.collision_count) / float(max(num_robots * max(env.step_count, 1), 1)),
        "path_length": float(env.path_length),
        "steps": env.step_count,
        "final_masses": ",".join(f"{mass:.4f}" for mass in env.clot_masses),
    }


def _aggregate(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    groups: dict[tuple[str, str], list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        groups[(str(row["scenario"]), str(row["controller"]))].append(row)
    summary = []
    for (scenario, controller), group in sorted(groups.items()):
        contact_steps = [float(row["first_contact_step"]) for row in group if float(row["first_contact_step"]) >= 0]
        summary.append(
            {
                "scenario": scenario,
                "controller": controller,
                "episodes": len(group),
                "success_rate": float(np.mean([float(row["success"]) for row in group])),
                "removal_rate": float(np.mean([float(row["removal_rate"]) for row in group])),
                "remaining_mass": float(np.mean([float(row["remaining_mass"]) for row in group])),
                "first_contact_step": float(np.mean(contact_steps)) if contact_steps else -1.0,
                "returned_to_junction_rate": float(
                    np.mean([float(row["returned_to_junction_after_first_clear"]) for row in group])
                ),
                "max_branches_occupied": float(np.mean([float(row["max_branches_occupied"]) for row in group])),
                "collision_rate": float(np.mean([float(row["collision_rate"]) for row in group])),
                "steps": float(np.mean([float(row["steps"]) for row in group])),
            }
        )
    return summary


def _write_tsv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenarios", nargs="+", default=["bifurcation", "anastomosis"])
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    parser.add_argument("--robots", type=int, default=32)
    parser.add_argument("--clots", type=int, default=2)
    parser.add_argument("--horizon", type=int, default=1000)
    parser.add_argument("--lookahead", type=int, default=6)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("autoresearch/vascular-3d-research-260811"),
    )
    args = parser.parse_args()

    controllers: dict[str, Controller] = {
        "forward_split_m-1": _constant_forward(-1.0),
        "forward_cohesive_m1": _constant_forward(1.0),
        "greedy_euclidean_m1": _greedy_euclidean(1.0),
        "topology_oracle": _topology_oracle(1.0, args.lookahead),
    }

    rows = [
        run_episode(
            scenario=scenario,
            controller_name=name,
            controller=controller,
            seed=seed,
            num_robots=args.robots,
            num_clots=args.clots,
            horizon=args.horizon,
            lookahead=args.lookahead,
        )
        for scenario in args.scenarios
        for name, controller in controllers.items()
        for seed in args.seeds
    ]
    summary = _aggregate(rows)
    _write_tsv(args.output_dir / "round2_reachability_results.tsv", rows)
    _write_tsv(args.output_dir / "round2_reachability_summary.tsv", summary)
    with (args.output_dir / "round2_reachability_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
