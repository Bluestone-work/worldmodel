# Junction-shaped and curriculum transfer check

Date: 2026-08-11

## Question

What should be adjusted next after the reachability check: reward shaping, one-clot pretraining, or explicit topology memory?

## Runs

All runs use `bifurcation`, 8 robots, 8 PPO updates for short checks unless noted.

### 1. Baseline two-clot GAT, 8 updates

- `distance_reward_scale=1.0`
- `junction_reward_scale=0.0`
- `history_actions=0`

Result:

- success `0.0`
- removal rate `0.5`
- first contact `95.0`

### 2. Junction-shaped two-clot GAT, 8 updates

- `distance_reward_scale=1.0`
- `junction_reward_scale=0.5`
- `history_actions=0`

Result:

- success `0.0`
- removal rate `0.5`
- first contact `94.6`

### 3. Baseline two-clot GAT, 32 updates

Result:

- success `0.0`
- removal rate `0.5`
- first contact `46.6`
- collision rate `0.525`

### 4. Junction-shaped two-clot GAT, 32 updates

Result:

- success `0.0`
- removal rate `0.5`
- first contact `44.8`
- collision rate `0.375`

### 5. One-clot pretraining, 8 updates

- `clots=1`
- success `1.0`
- removal rate `1.0`
- first contact `105.6`

### 6. Transfer from one-clot checkpoint to two-clot, 8 updates

- loaded from `autoresearch/vascular-3d-junction-260811/gat_oneclot8/final_policy.pth`
- success `0.0`
- removal rate `0.5`
- first contact `79.6`

### 7. Transfer from one-clot checkpoint to two-clot, 32 updates

- success `0.0`
- removal rate `0.5`
- first contact `52.4`

## Interpretation

The new knobs changed secondary metrics but not the core success gap.

- Junction shaping lowered collisions and moved first contact slightly earlier, but did not improve two-clot success or removal.
- One-clot pretraining is easy, which confirms the policy can learn contact/lysis.
- One-clot pretraining does not transfer the branch-switch behavior to the two-clot task within this budget.

So the next adjustment should not be more generic reward shaping or ordinary curriculum alone. The bottleneck is the branch-switch decision itself.

## Next step

Implement explicit topology memory for branch commitment:

- recurrent state over recent branch occupancy;
- auxiliary branch-switch / return-to-junction head;
- optional imitation target from the topology oracle for the branch-choice subproblem only.
