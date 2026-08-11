"""A lightweight 3-D vascular benchmark for swarm navigation and thrombolysis.

This environment is deliberately a control benchmark, not a CFD or clinical
simulator.  A vascular scene is represented by sampled 3-D tube centerlines;
robots are overdamped particles subject to one shared actuation command.  The
representation is useful for testing partial observability and sim-to-real
ideas from recent microrobot-swarm work without granting each particle an
independent actuator.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from gymnasium import Env, spaces


@dataclass(frozen=True)
class _Scene:
    points: np.ndarray
    radii: np.ndarray
    branch_ids: np.ndarray
    clot_indices: np.ndarray
    junction_indices: np.ndarray


def _unit(vector: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vector))
    return vector / norm if norm > 1e-8 else np.zeros_like(vector)


class Vascular3DSwarmEnv(Env):
    """3-D tube-network navigation with a shared swarm actuator.

    The action is ``[vx, vy, vz, morphology]``.  The first three entries are
    a normalized common transport command.  ``morphology`` changes the
    strength of a weak cohesion term and therefore cannot independently move
    individual robots.  This is the physical authority boundary used by the
    project: a graph policy may propose particle-level intents, but the
    environment executes one shared command.
    """

    metadata = {"render_modes": [None, "rgb_array"]}

    def __init__(
        self,
        scenario: str = "bifurcation",
        num_robots: int = 24,
        num_clots: int = 2,
        horizon: int = 300,
        sensor_radius: float = 0.22,
        grid_size: int = 9,
        seed: int | None = None,
        render_mode: str | None = None,
        distance_reward_scale: float = 0.0,
        junction_reward_scale: float = 0.0,
    ) -> None:
        super().__init__()
        if scenario not in {"straight", "bifurcation", "anastomosis", "stenotic"}:
            raise ValueError(f"unknown vascular scenario: {scenario}")
        if num_robots < 2 or num_clots < 1:
            raise ValueError("num_robots >= 2 and num_clots >= 1 are required")
        self.scenario = scenario
        self.num_robots = int(num_robots)
        self.num_clots = int(num_clots)
        self.horizon = int(horizon)
        self.sensor_radius = float(sensor_radius)
        self.grid_size = int(grid_size)
        self.render_mode = render_mode
        self.distance_reward_scale = float(distance_reward_scale)
        self.junction_reward_scale = float(junction_reward_scale)
        self.max_speed = 0.018
        self.flow_speed = 0.004
        self.tube_radius = 0.055
        self.clot_contact_radius = 0.035
        self.lysis_rate = 0.018
        self.collision_penalty = 0.03
        self.action_space = spaces.Box(-1.0, 1.0, shape=(4,), dtype=np.float32)
        half = self.grid_size // 2
        self.observation_space = spaces.Dict(
            {
                "local_occupancy": spaces.Box(-1.0, 1.0, (3, self.grid_size, self.grid_size, self.grid_size), dtype=np.float32),
                "swarm_state": spaces.Box(-1.0, 1.0, (8,), dtype=np.float32),
                "clot_state": spaces.Box(-1.0, 1.0, (self.num_clots, 5), dtype=np.float32),
                "last_action": spaces.Box(-1.0, 1.0, (4,), dtype=np.float32),
            }
        )
        self._rng = np.random.default_rng(seed)
        self.scene: _Scene | None = None
        self.robot_positions = np.zeros((self.num_robots, 3), dtype=np.float32)
        self.clot_positions = np.zeros((self.num_clots, 3), dtype=np.float32)
        self.clot_masses = np.ones(self.num_clots, dtype=np.float32)
        self.last_action = np.zeros(4, dtype=np.float32)
        self.step_count = 0
        self.path_length = 0.0
        self.collision_count = 0
        self._known_points: set[int] = set()
        self._last_contact = np.zeros((self.num_robots, self.num_clots), dtype=bool)
        self._first_clear_step: int | None = None

    def _make_scene(self) -> _Scene:
        def line(start: list[float], end: list[float], n: int, branch: int) -> tuple[list[np.ndarray], list[float], list[int]]:
            s, e = np.asarray(start, dtype=np.float32), np.asarray(end, dtype=np.float32)
            pts = [s + (e - s) * (i / (n - 1)) for i in range(n)]
            return pts, [self.tube_radius] * n, [branch] * n

        paths: list[tuple[list[float], list[float], int, int]]
        if self.scenario == "straight":
            paths = [([0.08, 0.50, 0.50], [0.92, 0.50, 0.50], 0, 80)]
        elif self.scenario == "stenotic":
            paths = [([0.08, 0.50, 0.50], [0.92, 0.50, 0.50], 0, 100)]
        elif self.scenario == "anastomosis":
            paths = [
                ([0.08, 0.50, 0.50], [0.48, 0.50, 0.50], 0, 40),
                ([0.48, 0.50, 0.50], [0.92, 0.68, 0.64], 1, 50),
                ([0.48, 0.50, 0.50], [0.92, 0.32, 0.36], 2, 50),
                ([0.92, 0.68, 0.64], [0.92, 0.32, 0.36], 3, 35),
            ]
        else:
            paths = [
                ([0.08, 0.50, 0.50], [0.48, 0.50, 0.50], 0, 45),
                ([0.48, 0.50, 0.50], [0.90, 0.76, 0.70], 1, 55),
                ([0.48, 0.50, 0.50], [0.90, 0.24, 0.30], 2, 55),
            ]
        points: list[np.ndarray] = []
        radii: list[float] = []
        branch_ids: list[int] = []
        for start, end, branch, n in paths:
            p, r, b = line(start, end, n, branch)
            had_points = bool(points)
            points.extend(p[1:] if had_points else p)
            radii.extend(r[1:] if had_points else r)
            branch_ids.extend(b[1:] if had_points else b)
        points_array = np.asarray(points, dtype=np.float32)
        radii_array = np.asarray(radii, dtype=np.float32)
        branch_array = np.asarray(branch_ids, dtype=np.int32)
        pairwise = np.linalg.norm(points_array[:, None, :] - points_array[None, :, :], axis=2)
        if len(points_array) > 1:
            nearest = np.partition(pairwise + np.eye(len(points_array)) * 10.0, 1, axis=1)[:, 1]
            threshold = max(0.025, float(np.percentile(nearest, 90)) * 1.8)
            junction_indices = np.array(
                [i for i in range(len(points_array)) if np.count_nonzero((pairwise[i] > 1e-6) & (pairwise[i] <= threshold)) >= 3],
                dtype=np.int32,
            )
        else:
            junction_indices = np.zeros(1, dtype=np.int32)
        if len(junction_indices) == 0:
            junction_indices = np.array([len(points_array) // 2], dtype=np.int32)
        # One clot per selected branch, always away from the injection point.
        branches = sorted(set(int(x) for x in branch_array))
        # Reuse a branch when a simple scene has fewer branches than clots;
        # choose separated points so the task still contains distinct targets.
        clot_indices_list: list[int] = []
        for clot_id in range(self.num_clots):
            branch = branches[-1 - (clot_id % len(branches))]
            branch_points = np.where(branch_array == branch)[0]
            quantile = (clot_id // len(branches) + 1) / (self.num_clots // len(branches) + 1)
            clot_indices_list.append(int(branch_points[int(quantile * (len(branch_points) - 1))]))
        clot_indices = np.asarray(clot_indices_list, dtype=np.int32)
        if self.scenario == "stenotic":
            for i, p in enumerate(points_array):
                if 0.46 < float(p[0]) < 0.56:
                    radii_array[i] *= 0.48
        return _Scene(points_array, radii_array, branch_array, clot_indices, junction_indices)

    def _nearest(self, positions: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        assert self.scene is not None
        delta = positions[:, None, :] - self.scene.points[None, :, :]
        distances = np.linalg.norm(delta, axis=2)
        indices = np.argmin(distances, axis=1)
        return distances[np.arange(len(positions)), indices], indices, self.scene.points[indices]

    def _project_to_tube(self, position: np.ndarray) -> tuple[np.ndarray, bool]:
        distance, index, nearest = self._nearest(position[None, :])
        allowed = float(self.scene.radii[index[0]]) if self.scene is not None else self.tube_radius
        if float(distance[0]) <= allowed:
            return position.astype(np.float32), False
        return nearest[0].astype(np.float32), True

    def _flow_at(self, positions: np.ndarray) -> np.ndarray:
        assert self.scene is not None
        _, indices, _ = self._nearest(positions)
        points = self.scene.points
        tangents = np.zeros_like(positions)
        for row, idx in enumerate(indices):
            lo, hi = max(0, int(idx) - 1), min(len(points) - 1, int(idx) + 1)
            tangents[row] = _unit(points[hi] - points[lo])
        return tangents * self.flow_speed

    def _observe(self) -> dict[str, np.ndarray]:
        assert self.scene is not None
        centroid = self.robot_positions.mean(axis=0)
        # Vectorized local voxelization.  Keeping this independent of a mesh or
        # CFD library makes the benchmark portable while avoiding a Python
        # triple loop in every environment step.
        axis = np.linspace(-self.sensor_radius, self.sensor_radius, self.grid_size, dtype=np.float32)
        gx, gy, gz = np.meshgrid(axis, axis, axis, indexing="ij")
        voxels = centroid[None, :] + np.stack([gx, gy, gz], axis=-1).reshape(-1, 3)
        distance, index, _ = self._nearest(voxels)
        inside_sensor = np.linalg.norm(voxels - centroid[None, :], axis=1) <= self.sensor_radius
        local_flat = np.full((3, len(voxels)), -1.0, dtype=np.float32)
        local_flat[0, inside_sensor] = (
            distance[inside_sensor] <= self.scene.radii[index[inside_sensor]]
        ).astype(np.float32)
        if self.clot_masses.sum() > 0:
            clot_distance = np.min(
                np.linalg.norm(self.clot_positions[None, :, :] - voxels[:, None, :], axis=2), axis=1
            )
            local_flat[1, inside_sensor] = (
                clot_distance[inside_sensor] < self.clot_contact_radius * 1.8
            ).astype(np.float32)
        robot_distance = np.min(
            np.linalg.norm(self.robot_positions[None, :, :] - voxels[:, None, :], axis=2), axis=1
        )
        local_flat[2, inside_sensor] = (robot_distance[inside_sensor] < 0.025).astype(np.float32)
        local = local_flat.reshape(3, self.grid_size, self.grid_size, self.grid_size)
        nearest_dist, _, _ = self._nearest(self.robot_positions)
        swarm = np.array(
            [*centroid, float(self.robot_positions[:, 0].ptp()), float(self.robot_positions[:, 1].ptp()), float(self.robot_positions[:, 2].ptp()), self.step_count / self.horizon, float(np.mean(nearest_dist))],
            dtype=np.float32,
        )
        clot = np.zeros((self.num_clots, 5), dtype=np.float32)
        clot[:, :3] = self.clot_positions
        clot[:, 3] = self.clot_masses
        clot[:, 4] = (self.clot_masses > 0).astype(np.float32)
        scene_distance = np.linalg.norm(self.scene.points - centroid[None, :], axis=1)
        self._known_points.update(int(i) for i in np.where(scene_distance <= self.sensor_radius)[0])
        return {"local_occupancy": local, "swarm_state": np.clip(swarm, -1, 1), "clot_state": clot, "last_action": self.last_action.copy()}

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None):
        super().reset(seed=seed)
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        self.scene = self._make_scene()
        self.robot_positions = self.scene.points[:1] + self._rng.normal(0.0, 0.012, (self.num_robots, 3)).astype(np.float32)
        self.robot_positions = np.asarray([self._project_to_tube(p)[0] for p in self.robot_positions], dtype=np.float32)
        self.clot_positions = self.scene.points[self.scene.clot_indices].copy()
        self.clot_masses = np.ones(self.num_clots, dtype=np.float32)
        self.last_action.fill(0.0)
        self.step_count = 0
        self.path_length = 0.0
        self.collision_count = 0
        self._known_points.clear()
        self._last_contact.fill(False)
        self._first_clear_step = None
        return self._observe(), {"scenario": self.scenario, "known_fraction": 0.0}

    def step(self, action: np.ndarray):
        action = np.clip(np.asarray(action, dtype=np.float32), -1.0, 1.0).reshape(-1)
        if action.shape != (4,):
            raise ValueError(f"expected action shape (4,), got {action.shape}")
        centroid = self.robot_positions.mean(axis=0)
        active_before = self.clot_masses > 0.0
        previous_distance = float(
            np.min(np.linalg.norm(self.clot_positions[active_before] - centroid[None, :], axis=1))
            if np.any(active_before) else 0.0
        )
        transport = _unit(action[:3]) * self.max_speed * min(1.0, float(np.linalg.norm(action[:3])))
        flow = self._flow_at(self.robot_positions)
        cohesion = (centroid[None, :] - self.robot_positions) * (0.35 + 0.25 * float(action[3]))
        noise = self._rng.normal(0.0, 0.0007, self.robot_positions.shape).astype(np.float32)
        proposed = self.robot_positions + transport[None, :] + flow + cohesion + noise
        collisions = 0
        for i in range(self.num_robots):
            projected, collided = self._project_to_tube(proposed[i])
            collisions += int(collided)
            self.path_length += float(np.linalg.norm(projected - self.robot_positions[i]))
            self.robot_positions[i] = projected
        self.collision_count += collisions
        distances = np.linalg.norm(self.robot_positions[:, None, :] - self.clot_positions[None, :, :], axis=2)
        contact = (distances <= self.clot_contact_radius) & (self.clot_masses[None, :] > 0)
        self._last_contact = contact
        removed = self.lysis_rate * np.exp(-np.square(distances / self.clot_contact_radius)) * contact
        removed_mass = float(removed.sum())
        self.clot_masses = np.maximum(0.0, self.clot_masses - removed.sum(axis=0)).astype(np.float32)
        self.clot_masses[self.clot_masses < 0.01] = 0.0
        centroid_after = self.robot_positions.mean(axis=0)
        active_after = self.clot_masses > 0.0
        current_distance = float(
            np.min(np.linalg.norm(self.clot_positions[active_after] - centroid_after[None, :], axis=1))
            if np.any(active_after) else 0.0
        )
        distance_progress = previous_distance - current_distance
        junction_distance = float(
            np.min(np.linalg.norm(self.scene.points[self.scene.junction_indices] - centroid_after[None, :], axis=1))
        )
        if self._first_clear_step is None and np.any(self.clot_masses <= 0.0) and np.any(active_after):
            self._first_clear_step = self.step_count + 1
        self.step_count += 1
        self.last_action = action.copy()
        terminated = bool(np.all(self.clot_masses <= 0.0))
        truncated = bool(self.step_count >= self.horizon and not terminated)
        progress = removed_mass * 8.0
        junction_reward = 0.0
        if self.junction_reward_scale > 0.0 and self._first_clear_step is not None and np.any(active_after):
            junction_reward = self.junction_reward_scale * float(
                np.exp(-junction_distance / max(self.tube_radius, 1e-6))
            )
        reward = (
            progress
            + self.distance_reward_scale * distance_progress
            + junction_reward
            - self.collision_penalty * collisions / max(self.num_robots, 1)
            - 0.002
        )
        if terminated:
            reward += 5.0
        info = {
            "success": terminated,
            "removed_mass": removed_mass,
            "remaining_mass": float(self.clot_masses.sum()),
            "active_contacts": int(np.sum(np.any(contact, axis=0))),
            "collisions": collisions,
            "collision_rate": collisions / max(self.num_robots, 1),
            "path_length": self.path_length,
            "distance_progress": distance_progress,
            "junction_distance": junction_distance,
            "known_fraction": len(self._known_points) / max(len(self.scene.points), 1),
            "branch_ids": self.scene.branch_ids.copy(),
        }
        return self._observe(), float(reward), terminated, truncated, info

    def render(self):
        if self.render_mode != "rgb_array":
            return None
        import matplotlib.pyplot as plt
        fig = plt.figure(figsize=(5, 4))
        ax = fig.add_subplot(111, projection="3d")
        assert self.scene is not None
        ax.scatter(self.scene.points[:, 0], self.scene.points[:, 1], self.scene.points[:, 2], s=3, alpha=0.15)
        ax.scatter(self.robot_positions[:, 0], self.robot_positions[:, 1], self.robot_positions[:, 2], s=12, c="tab:blue")
        active = self.clot_masses > 0
        ax.scatter(self.clot_positions[active, 0], self.clot_positions[active, 1], self.clot_positions[active, 2], s=45, c="tab:red")
        ax.set(xlim=(0, 1), ylim=(0, 1), zlim=(0, 1))
        fig.canvas.draw()
        image = np.asarray(fig.canvas.buffer_rgba())[..., :3].copy()
        plt.close(fig)
        return image
