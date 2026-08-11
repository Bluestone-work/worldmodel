#!/usr/bin/env python3
"""Evaluate transparent baselines on the 3-D vascular benchmark."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from environments.vascular_3d_swarm_env import Vascular3DSwarmEnv


def greedy_action(env: Vascular3DSwarmEnv) -> np.ndarray:
    active = env.clot_masses > 0
    if not np.any(active):
        return np.zeros(4, dtype=np.float32)
    centroid = env.robot_positions.mean(axis=0)
    target = env.clot_positions[active][np.argmin(np.linalg.norm(env.clot_positions[active] - centroid, axis=1))]
    delta = target - centroid
    # Deliberately use only currently sensed clots.  The 3-D oracle is retained
    # as a diagnostic baseline, while this controller models a local policy.
    if np.linalg.norm(delta) > env.sensor_radius:
        nearest_distance, nearest_index, nearest_point = env._nearest(centroid[None, :])
        target = nearest_point[0]
        delta = target - centroid
    return np.r_[delta / max(np.linalg.norm(delta), 1e-8), 0.0].astype(np.float32)


def run(env: Vascular3DSwarmEnv, controller: str, seed: int) -> dict:
    env.reset(seed=seed)
    total_reward = 0.0
    first_contact_step = None
    for step in range(env.horizon):
        if controller == "random":
            action = env.action_space.sample()
        elif controller == "greedy":
            action = greedy_action(env)
        elif controller == "flow_only":
            action = np.zeros(4, dtype=np.float32)
        else:
            raise ValueError(controller)
        _, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        if info["active_contacts"] > 0 and first_contact_step is None:
            first_contact_step = env.step_count
        if terminated or truncated:
            break
    return {
        "controller": controller,
        "seed": seed,
        "steps": env.step_count,
        "success": float(info["success"]),
        "remaining_mass": info["remaining_mass"],
        "removal_rate": 1.0 - info["remaining_mass"] / env.num_clots,
        "total_reward": total_reward,
        "collision_rate": info["collision_rate"],
        "known_fraction": info["known_fraction"],
        "path_length": info["path_length"],
        "first_contact_step": first_contact_step if first_contact_step is not None else env.step_count,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenario", default="bifurcation")
    parser.add_argument("--controllers", default="flow_only,random,greedy")
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--robots", type=int, default=24)
    parser.add_argument("--clots", type=int, default=2)
    parser.add_argument("--distance-reward-scale", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=20260811)
    parser.add_argument("--output", default="autoresearch/vascular-3d-baselines")
    args = parser.parse_args()
    out = ROOT / args.output
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    for controller in args.controllers.split(","):
        env = Vascular3DSwarmEnv(scenario=args.scenario, num_robots=args.robots, num_clots=args.clots,
                                 distance_reward_scale=args.distance_reward_scale)
        for i in range(args.episodes):
            rows.append(run(env, controller.strip(), args.seed + i))
        env.close()
    frame = pd.DataFrame(rows)
    frame.to_csv(out / f"episodes_{args.scenario}.csv", index=False)
    summary = frame.groupby("controller").agg({
        "success": ["mean", "std"],
        "removal_rate": ["mean", "std"],
        "steps": "mean",
        "collision_rate": "mean",
        "known_fraction": "mean",
        "path_length": "mean",
        "first_contact_step": "mean",
    }).round(5)
    summary.to_csv(out / f"summary_{args.scenario}.csv")
    serializable = {str(controller): {str(metric): float(value) if np.isfinite(value) else None
                                      for metric, value in values.items()}
                    for controller, values in summary.to_dict(orient="index").items()}
    with (out / f"summary_{args.scenario}.json").open("w") as handle:
        json.dump({"scenario": args.scenario, "episodes": args.episodes, "summary": serializable}, handle, indent=2)
    print(summary)


if __name__ == "__main__":
    main()
