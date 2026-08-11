"""Per-particle MLP actor with centralized mean-pooled critic."""
from __future__ import annotations

import torch
import torch.nn as nn


class MLPParticlePolicy(nn.Module):
    """MLP ablation sharing the GAT policy's Gaussian/PPO interface."""

    def __init__(self, node_obs_dim: int, action_dim: int = 3, hidden_dim: int = 128):
        super().__init__()
        self.encoder = nn.Sequential(nn.Linear(node_obs_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, hidden_dim), nn.ReLU())
        self.actor_mean = nn.Linear(hidden_dim, action_dim)
        self.actor_logstd = nn.Parameter(torch.zeros(action_dim))
        self.critic = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, 1))

    def forward(self, node_obs: torch.Tensor, adj: torch.Tensor | None = None):
        features = self.encoder(node_obs)
        return self.actor_mean(features), self.critic(features.mean(dim=1))

    def get_action(self, node_obs, adj, deterministic: bool = False):
        mean, value = self.forward(node_obs, adj)
        if deterministic:
            action = mean
        else:
            action = mean + torch.randn_like(mean) * torch.exp(self.actor_logstd)
        return torch.clamp(action, -1.0, 1.0), value

    def evaluate_actions(self, node_obs, adj, actions):
        mean, value = self.forward(node_obs, adj)
        std = torch.exp(self.actor_logstd)
        var = std.square()
        normalizer = torch.log(torch.tensor(2.0 * 3.141592653589793, device=actions.device))
        log_prob = (-0.5 * ((actions - mean).square() / var + 2.0 * torch.log(std) + normalizer)).sum(dim=-1)
        entropy = (0.5 + 0.5 * torch.log(2.0 * 3.141592653589793 * var)).sum(dim=-1)
        return log_prob, value, entropy

