from __future__ import annotations

import numpy as np

from environments.vascular_3d_swarm_env import Vascular3DSwarmEnv
from marl.vascular_graph_adapter import build_vascular_graph_observation, shared_action_from_particle_intents


def test_graph_adapter_and_shared_projection():
    env = Vascular3DSwarmEnv(num_robots=10, scenario="bifurcation")
    env.reset(seed=12)
    node, adjacency = build_vascular_graph_observation(env)
    assert node.shape == (10, 10)
    assert adjacency.shape == (10, 10)
    assert np.allclose(np.diag(adjacency), 1.0)
    action = shared_action_from_particle_intents(np.ones((10, 3), dtype=np.float32))
    assert action.shape == (4,)
    assert np.all(action[:3] == 1.0)
    assert action[3] == 0.0

