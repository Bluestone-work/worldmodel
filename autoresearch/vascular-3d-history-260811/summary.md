# Vascular history-augmented GAT feasibility check

Date: 2026-08-11

## Question

Does adding short action history to the 3-D vascular GAT policy make the two-clot bifurcation task materially more feasible?

## Setup

- Task: `bifurcation`
- Robots: 8
- Clots: 2
- Reward shaping: `distance_reward_scale=1.0`
- Training budget: 8 PPO updates, 128 steps/update, 2 epochs, batch size 64
- Evaluation: 5 held-out seeds
- Model: shared-actuator GAT
- Memory variant: concatenate the last 3 executed shared actions to every node token

## Result

| Variant | Success | Removal rate | Remaining mass | First contact | Collision rate | Known fraction |
|---|---:|---:|---:|---:|---:|---:|
| Baseline GAT | 0.00 | 0.50 | 1.00 | 95.0 | 0.20 | 0.809 |
| History GAT (3 actions) | 0.00 | 0.40 | 1.20 | 139.0 | 0.00 | 0.795 |

Full logs:

- [gat_base/evaluation.json](./gat_base/evaluation.json)
- [gat_hist3/evaluation.json](./gat_hist3/evaluation.json)
- [gat_base/training_log.json](./gat_base/training_log.json)
- [gat_hist3/training_log.json](./gat_hist3/training_log.json)

## Interpretation

Short action history did not close the feasibility gap. It slightly delayed first contact and reduced removal rate in this short run. That is not a proof that memory is useless; it is evidence that generic recent-action concatenation is too weak for the branch-switching problem.

This is consistent with the reachability diagnosis: the bifurcation task is solvable with an oracle that reasons over centerline topology, so the bottleneck is not physics. The bottleneck is a structured policy representation that can remember which branch was already serviced and commit to returning to the junction.

## Next step

Move from generic history to explicit topology memory:

- recurrent or graph-history state over branch visits;
- auxiliary branch-switch / return-to-junction signal;
- optional oracle imitation on the branch-selection subproblem.
