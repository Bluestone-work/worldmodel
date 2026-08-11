# Round 1 learning comparison

Configuration shared by both models: `bifurcation`, 8 robots, 8 PPO updates, 128 transitions/update, 2 PPO epochs, batch size 64, 5 held-out seeds, shared 4-D action projection.

| Policy | Success | Removal rate | Remaining mass | First contact | Collision rate |
|---|---:|---:|---:|---:|---:|
| GAT-MAPPO | 0.00 | 0.500 | 1.000 | 180.0 | 0.000 |
| MLP-MAPPO | 0.00 | 0.444 | 1.112 | 171.6 | 0.000 |
| Flow-only baseline | 0.00 | 0.500 | 1.000 | 223.2 | 0.000 |
| Greedy baseline | 0.00 | 0.200 | 1.600 | 235.6 | 0.000 |
| Random baseline | 0.00 | 0.461 | 1.078 | 246.0 | 0.025 |

The result is an engineering and hypothesis check, not a final claim: all learned policies were trained for only 1,024 environment transitions and none cleared both clots. GAT has earlier contact than flow-only and MLP in this run, but the confidence interval is not estimated and the task may be partially solved by the deterministic flow. The next controlled study should use one-clot and two-clot tasks separately, add a progress-shaped treatment reward, and scale training to at least 5 seeds before interpreting the architecture gap.

