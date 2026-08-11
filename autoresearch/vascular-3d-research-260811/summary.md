# 3D vascular bridge: round-0 result

Date: 2026-08-11
Base repository commit: `a2e8838f2f688b3e6d47721f2b8ac9545fd3dc8a`

## Implemented

- `environments/vascular_3d_swarm_env.py`: 3D straight/bifurcation/anastomosis/stenotic tube scenes, local voxel observation, flow, tube projection, shared actuator, clot contact/lysis proxy.
- `marl/vascular_graph_adapter.py`: DGR-style 10-D node tokens and proximity graph; `(N,3)` particle intents are projected to one `(4,)` executable action.
- `scripts/evaluate_vascular_3d_baselines.py`: reproducible fixed-seed baseline evaluation.
- `doc/vascular-3d-dgr-research-protocol.zh.md`: literature/code alignment, assumptions, metrics, and staged experiment protocol.

## Verification

Direct test functions passed:

- deterministic reset and Gymnasium observation bounds;
- tube validity and one-actuator authority;
- clot contact reduces mass;
- graph adapter shape and shared projection;
- existing GAT forward pass with `action_dim=3`.

`pytest` was unavailable in the container (`No module named pytest`), so the same tests were invoked directly and Python bytecode compilation passed.

## Round-0 observation

On a one-episode bifurcation smoke run (300-step horizon), `flow_only`, `random`, and `greedy` all had success 0 and removal rate 0.5, with first contact around step 229--300. This is a smoke result, not a statistical claim. Short runs on straight/anastomosis/stenotic scenes show that some scenes are too easy under the current flow and clot placement; formal comparisons need at least 5 seeds and 100 held-out episodes per scenario.

## Next experiment

Vectorize/benchmark the voxelizer (already vectorized in this round), then train and compare MLP-MAPPO, GAT-MAPPO, and GAT-LSTM-MAPPO with the same local observation and the shared-action projection. Keep independent per-particle action only as an explicitly labelled unrealistic upper bound.

