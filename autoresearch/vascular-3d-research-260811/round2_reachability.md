# Round 2 reachability diagnosis

Date: 2026-08-11

## Purpose

Round 1 showed that short-budget GAT/MLP policies and simple baselines usually clear only one clot in the bifurcation task. This round tests whether the two-clot bifurcation is physically unreachable under the shared magnetic actuator, or whether it mainly requires topology-aware navigation.

## Diagnostic setup

- Script: `scripts/diagnose_vascular_reachability.py`
- Scenarios: `bifurcation`, `anastomosis`
- Seeds: 0, 1, 2, 3, 4
- Robots: 32
- Clots: 2
- Horizon: 1000
- Actuation: one shared action `[vx, vy, vz, morphology]`
- Controllers:
  - `forward_split_m-1`: constant forward transport with weak cohesion.
  - `forward_cohesive_m1`: constant forward transport with strong cohesion.
  - `greedy_euclidean_m1`: global Euclidean clot direction, no topology memory.
  - `topology_oracle`: global centerline graph and clot positions; used only as a reachability upper bound.

Important correction: the first draft of `topology_oracle` passed short waypoint vectors directly to the environment, which reduced transport speed below the flow field. The final script normalizes controller directions before execution.

## Results

| Scenario | Controller | Success | Removal | First contact | Return-to-junction | Steps |
|---|---:|---:|---:|---:|---:|---:|
| bifurcation | forward_split_m-1 | 0.00 | 0.50 | 56.6 | 0.60 | 1000.0 |
| bifurcation | forward_cohesive_m1 | 0.00 | 0.40 | 47.25 | 0.80 | 1000.0 |
| bifurcation | greedy_euclidean_m1 | 0.00 | 0.50 | 42.0 | 1.00 | 1000.0 |
| bifurcation | topology_oracle | 1.00 | 1.00 | 41.0 | 1.00 | 101.8 |
| anastomosis | forward_split_m-1 | 1.00 | 1.00 | 95.6 | 0.00 | 102.8 |
| anastomosis | forward_cohesive_m1 | 1.00 | 1.00 | 105.8 | 0.00 | 110.2 |
| anastomosis | greedy_euclidean_m1 | 1.00 | 1.00 | 39.8 | 0.00 | 41.6 |
| anastomosis | topology_oracle | 1.00 | 1.00 | 39.0 | 0.00 | 41.0 |

Full per-episode records:

- `autoresearch/vascular-3d-research-260811/round2_reachability_results.tsv`
- `autoresearch/vascular-3d-research-260811/round2_reachability_summary.tsv`
- `autoresearch/vascular-3d-research-260811/round2_reachability_summary.json`

## Interpretation

The bifurcation task is reachable under the shared actuator: the topology oracle clears both clots in all five seeds. Therefore the previous failures should not be framed as evidence that the environment is impossible. They identify a navigation gap: policies need to infer or remember vascular topology well enough to leave a completed branch, return to the junction, and enter the other branch.

A follow-up 8-robot run, matched to the training regime used in the round-1 learning experiments, gave the same qualitative result: the topology oracle still solved all five seeds, while the non-topological controllers still stalled after clearing only one clot. So the gap is not caused by the robot count in the learning setup.

The anastomosis task is currently too easy for the main benchmark because all simple controllers solved it. It can still be useful as a curriculum or sanity scene, but it should not be the headline task without harder clot placement, adverse flow, partial observability, or randomized topology.

Greedy Euclidean control returns to the junction after the first clear in bifurcation but still repeatedly fails to reach and clear the second branch. This separates "can move back near the junction" from "can select and commit to the unvisited branch." That is the exact opening for DGR-style graph memory, GAT message passing, and recurrent policy state.

## Next research action

Use `topology_oracle` as an upper-bound diagnostic only. The next implementation target should be a learnable topology-memory treatment:

- expose branch/topology progress signals only during training/evaluation diagnostics, not as privileged observations;
- add an optional recurrent policy path or graph-history aggregator to the vascular GAT pipeline;
- train bifurcation two-clot with shaped distance/contact reward and evaluate whether GAT/history closes the gap to the oracle;
- keep anastomosis as curriculum unless it is made harder.
