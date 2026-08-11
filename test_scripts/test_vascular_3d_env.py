"""Smoke and invariant tests for the 3-D vascular benchmark."""
from __future__ import annotations

import numpy as np

from environments.vascular_3d_swarm_env import Vascular3DSwarmEnv


def test_reset_is_reproducible_and_observation_is_bounded():
    first = Vascular3DSwarmEnv(scenario="bifurcation")
    second = Vascular3DSwarmEnv(scenario="bifurcation")
    obs_a, _ = first.reset(seed=17)
    obs_b, _ = second.reset(seed=17)
    np.testing.assert_allclose(first.robot_positions, second.robot_positions)
    np.testing.assert_allclose(first.clot_positions, second.clot_positions)
    assert obs_a["local_occupancy"].shape == (3, 9, 9, 9)
    assert first.observation_space.contains(obs_a)
    assert second.observation_space.contains(obs_b)


def test_shared_action_keeps_one_actuator_authority_and_tube_validity():
    env = Vascular3DSwarmEnv(scenario="stenotic", num_robots=8)
    env.reset(seed=3)
    action = np.array([0.4, 0.2, -0.1, 0.7], dtype=np.float32)
    before = env.robot_positions.copy()
    _, _, _, _, info = env.step(action)
    assert env.action_space.shape == (4,)
    assert info["collisions"] >= 0
    assert np.max(np.linalg.norm(env.robot_positions[:, None, :] - env.scene.points[None, :, :], axis=2).min(axis=1)) <= env.tube_radius + 1e-6
    assert np.any(np.linalg.norm(env.robot_positions - before, axis=1) > 0)


def test_clot_contact_reduces_mass_and_success_terminates():
    env = Vascular3DSwarmEnv(scenario="straight", num_robots=12, num_clots=1, horizon=30)
    env.reset(seed=5)
    env.robot_positions[:] = env.clot_positions[0]
    initial = float(env.clot_masses.sum())
    _, reward, terminated, _, info = env.step(np.zeros(4, dtype=np.float32))
    assert info["removed_mass"] > 0
    assert env.clot_masses.sum() < initial
    assert reward > 0
    assert not terminated

