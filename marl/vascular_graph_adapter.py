"""DGR-style graph adapter for :class:`Vascular3DSwarmEnv`.

The adapter separates policy representation from actuator authority.  A graph
policy can emit one 3-D intent per particle, but execution is reduced to the
single shared four-dimensional action accepted by the vascular environment.
"""
from __future__ import annotations

import numpy as np


def build_vascular_graph_observation(env, sensing_radius: float | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Return node tokens and a proximity graph from local state.

    Node token layout is ``[x, y, z, clot_dx, clot_dy, clot_dz,
    clot_distance, contact, time, local_clearance]``.  The clot terms are
    nearest-active-clot features, not an oracle list of all clots.
    """
    positions = np.asarray(env.robot_positions, dtype=np.float32)
    centroid = positions.mean(axis=0)
    active = np.asarray(env.clot_masses) > 0.0
    if np.any(active):
        clots = np.asarray(env.clot_positions)[active]
        delta = clots[None, :, :] - positions[:, None, :]
        distances = np.linalg.norm(delta, axis=2)
        nearest = np.argmin(distances, axis=1)
        nearest_delta = delta[np.arange(len(positions)), nearest]
        nearest_distance = distances[np.arange(len(positions)), nearest]
    else:
        nearest_delta = np.zeros_like(positions)
        nearest_distance = np.ones(len(positions), dtype=np.float32)
    distance_to_tube, _, _ = env._nearest(positions)
    contact = np.any(np.asarray(env._last_contact), axis=1).astype(np.float32)
    node = np.column_stack(
        [
            positions,
            np.clip(nearest_delta, -1.0, 1.0),
            np.clip(nearest_distance, 0.0, 1.0),
            contact,
            np.full(len(positions), env.step_count / max(env.horizon, 1), dtype=np.float32),
            np.clip(1.0 - distance_to_tube / max(env.tube_radius, 1e-6), 0.0, 1.0),
        ]
    ).astype(np.float32)
    radius = float(sensing_radius if sensing_radius is not None else env.sensor_radius)
    pairwise = np.linalg.norm(positions[:, None, :] - positions[None, :, :], axis=2)
    adjacency = (pairwise <= radius).astype(np.float32)
    np.fill_diagonal(adjacency, 1.0)
    return node, adjacency


def shared_action_from_particle_intents(intents: np.ndarray) -> np.ndarray:
    """Project ``(N, 3)`` particle intents to one executable 4-D action."""
    intents = np.asarray(intents, dtype=np.float32)
    if intents.ndim != 2 or intents.shape[1] != 3:
        raise ValueError(f"expected intents with shape (N, 3), got {intents.shape}")
    mean_intent = np.clip(intents.mean(axis=0), -1.0, 1.0)
    disagreement = float(np.mean(np.linalg.norm(intents - mean_intent[None, :], axis=1)))
    # A high disagreement asks for a mild morphology change while preserving
    # one shared transport vector.  It is a diagnostic projection, not an
    # independent per-particle actuator.
    morphology = float(np.clip(disagreement * 2.0, -1.0, 1.0))
    return np.r_[mean_intent, morphology].astype(np.float32)

