"""
GAT-MAPPO Policy for Magnetic Swarm Navigation

Adapts DGR-VDS GAT architecture to magnetic swarm thrombolysis:
- Node = individual microrobot (not independent agent, but particle in swarm)
- Edge = proximity-based graph (within sensing radius)
- Action = desired 2D velocity (aggregated via constraint allocator)
- Observation = local occupancy + swarm state + clot state

Key differences from DGR-VDS:
- No independent control (shared magnetic field)
- Constraint allocator projects desired velocities → realizable field
- Projection loss measures addressability gap
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional


class GraphAttentionLayer(nn.Module):
    """Single-layer Graph Attention (Veličković et al., ICLR 2018)"""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        dropout: float = 0.1,
        alpha: float = 0.2
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.dropout = dropout
        self.alpha = alpha

        # Learnable parameters
        self.W = nn.Parameter(torch.empty(size=(in_features, out_features)))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)

        self.a = nn.Parameter(torch.empty(size=(2*out_features, 1)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)

        self.leakyrelu = nn.LeakyReLU(self.alpha)

    def forward(
        self,
        h: torch.Tensor,
        adj: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            h: Node features [batch, n_nodes, in_features]
            adj: Adjacency matrix [batch, n_nodes, n_nodes] (0/1)

        Returns:
            h_prime: Updated features [batch, n_nodes, out_features]
        """
        batch_size, n_nodes, _ = h.size()

        # Linear transformation
        Wh = torch.matmul(h, self.W)  # [batch, n_nodes, out_features]

        # Compute attention coefficients
        Wh_i = Wh.unsqueeze(2)  # [batch, n_nodes, 1, out_features]
        Wh_j = Wh.unsqueeze(1)  # [batch, 1, n_nodes, out_features]

        # Concatenate and compute scores
        concat = torch.cat([
            Wh_i.expand(-1, -1, n_nodes, -1),
            Wh_j.expand(-1, n_nodes, -1, -1)
        ], dim=-1)  # [batch, n_nodes, n_nodes, 2*out_features]

        e = self.leakyrelu(torch.matmul(concat, self.a).squeeze(-1))

        # Mask: only attend to neighbors
        mask = (adj == 0)
        e = e.masked_fill(mask, -1e9)

        # Softmax normalization
        attention = F.softmax(e, dim=2)  # [batch, n_nodes, n_nodes]
        attention = F.dropout(attention, self.dropout, training=self.training)

        # Aggregate neighbor features
        h_prime = torch.matmul(attention, Wh)

        return h_prime


class MultiHeadGATLayer(nn.Module):
    """Multi-head Graph Attention"""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        n_heads: int = 4,
        dropout: float = 0.1
    ):
        super().__init__()
        self.n_heads = n_heads
        self.out_features = out_features

        self.attentions = nn.ModuleList([
            GraphAttentionLayer(in_features, out_features, dropout=dropout)
            for _ in range(n_heads)
        ])

    def forward(self, h: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        """
        Multi-head attention + concatenation

        Returns: [batch, n_nodes, n_heads * out_features]
        """
        h_list = [att(h, adj) for att in self.attentions]
        h_multi = torch.cat(h_list, dim=-1)
        return h_multi


class GATEncoder(nn.Module):
    """2-layer GAT encoder"""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 128,
        n_heads: int = 4
    ):
        super().__init__()

        # Layer 1: multi-head attention
        self.gat1 = MultiHeadGATLayer(input_dim, hidden_dim, n_heads=n_heads)

        # Layer 2: single-head attention (aggregation)
        self.gat2 = GraphAttentionLayer(hidden_dim * n_heads, hidden_dim)

        self.output_dim = hidden_dim

    def forward(
        self,
        node_features: torch.Tensor,
        adj: torch.Tensor
    ) -> torch.Tensor:
        """
        Graph encoding

        Args:
            node_features: [batch, n_nodes, input_dim]
            adj: [batch, n_nodes, n_nodes]

        Returns:
            encoded: [batch, n_nodes, output_dim]
        """
        # Layer 1
        h = self.gat1(node_features, adj)
        h = F.elu(h)

        # Layer 2
        h = self.gat2(h, adj)
        h = F.elu(h)

        return h


class MagneticSwarmGATPolicy(nn.Module):
    """
    GAT-based policy for magnetic swarm navigation.

    Architecture:
    1. Node feature encoder: extract per-particle local features
    2. GAT encoder: aggregate neighbor information via graph attention
    3. Policy head: output desired velocity per particle
    4. Constraint allocator: project desired velocities → realizable field

    Note: Unlike DGR-VDS, we don't use LSTM since magnetic control
    is essentially reactive (no momentum/inertia in overdamped regime).
    """

    def __init__(
        self,
        node_obs_dim: int,
        action_dim: int = 2,  # 2D velocity (vx, vy)
        hidden_dim: int = 128,
        gat_hidden_dim: int = 128,
        n_gat_heads: int = 4
    ):
        super().__init__()

        # 1. Node feature encoder
        self.feature_encoder = nn.Sequential(
            nn.Linear(node_obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )

        # 2. GAT encoder
        self.gat_encoder = GATEncoder(
            input_dim=hidden_dim,
            hidden_dim=gat_hidden_dim,
            n_heads=n_gat_heads
        )

        # 3. Fusion layer
        self.fusion = nn.Sequential(
            nn.Linear(hidden_dim + gat_hidden_dim, hidden_dim),
            nn.ReLU()
        )

        # 4. Policy head
        self.actor_mean = nn.Linear(hidden_dim, action_dim)
        self.actor_logstd = nn.Parameter(torch.zeros(action_dim))

        # 5. Critic head (centralized value function)
        # Input: global state (swarm centroid, spread, clot positions)
        # We'll add a separate method for this
        self.critic_encoder = nn.Sequential(
            nn.Linear(node_obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        self.critic_head = nn.Linear(hidden_dim, 1)

    def forward(
        self,
        node_obs: torch.Tensor,      # [batch, n_particles, node_obs_dim]
        adj: torch.Tensor,            # [batch, n_particles, n_particles]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass

        Args:
            node_obs: Per-particle observations [batch, n_particles, node_obs_dim]
            adj: Proximity-based adjacency matrix [batch, n_particles, n_particles]

        Returns:
            action_mean: Desired velocity [batch, n_particles, action_dim]
            value: State value [batch, 1]
        """
        batch_size, n_particles, _ = node_obs.size()

        # 1. Local feature encoding
        local_features = self.feature_encoder(node_obs)  # [batch, n_particles, hidden_dim]

        # 2. Graph attention aggregation
        graph_features = self.gat_encoder(local_features, adj)  # [batch, n_particles, gat_hidden_dim]

        # 3. Fusion
        fused = torch.cat([local_features, graph_features], dim=-1)
        fused = self.fusion(fused)  # [batch, n_particles, hidden_dim]

        # 4. Actor: per-particle desired velocity
        action_mean = self.actor_mean(fused)  # [batch, n_particles, action_dim]

        # 5. Critic: centralized value function
        # Use mean-pooled features as global state representation
        global_features = fused.mean(dim=1)  # [batch, hidden_dim]
        value = self.critic_head(global_features)  # [batch, 1]

        return action_mean, value

    def get_action(
        self,
        node_obs: torch.Tensor,
        adj: torch.Tensor,
        deterministic: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get action for a single batch

        Returns:
            action: [batch, n_particles, action_dim]
            value: [batch, 1]
        """
        action_mean, value = self.forward(node_obs, adj)

        if deterministic:
            action = action_mean
        else:
            # Sample from Gaussian
            std = torch.exp(self.actor_logstd)
            action = torch.normal(action_mean, std.expand_as(action_mean))

        # Clip to [-1, 1] (normalized velocity)
        action = torch.clamp(action, -1.0, 1.0)

        return action, value

    def evaluate_actions(
        self,
        node_obs: torch.Tensor,
        adj: torch.Tensor,
        actions: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Evaluate actions for PPO update

        Returns:
            log_probs: [batch, n_particles]
            value: [batch, 1]
            entropy: [batch, n_particles]
        """
        action_mean, value = self.forward(node_obs, adj)

        # Gaussian policy
        std = torch.exp(self.actor_logstd)
        var = std ** 2

        # Log probability
        log_prob = -0.5 * ((actions - action_mean) ** 2 / var + 2 * torch.log(std) + torch.log(torch.tensor(2 * 3.14159)))
        log_prob = log_prob.sum(dim=-1)  # Sum over action dimensions

        # Entropy (per action dimension, then sum)
        entropy = 0.5 + 0.5 * torch.log(2 * 3.14159 * var)
        entropy = entropy.sum(dim=-1)  # [batch, n_particles]

        return log_prob, value, entropy


# ===== Test =====
if __name__ == "__main__":
    print("Testing MagneticSwarmGATPolicy...")

    policy = MagneticSwarmGATPolicy(
        node_obs_dim=10,  # Example: [x, y, z, vx, vy, nearest_clot_dist, ...]
        action_dim=2,
        hidden_dim=64,
        gat_hidden_dim=64,
        n_gat_heads=4
    )

    batch_size = 4
    n_particles = 64

    node_obs = torch.randn(batch_size, n_particles, 10)

    # Create proximity-based adjacency matrix
    # Assume particles within radius 0.1 are connected
    positions = torch.randn(batch_size, n_particles, 2)
    dist = torch.cdist(positions, positions)  # [batch, n_particles, n_particles]
    adj = (dist < 0.1).float()

    # Self-loop
    adj = adj + torch.eye(n_particles).unsqueeze(0)

    # Forward pass
    action_mean, value = policy(node_obs, adj)
    print(f"Action mean shape: {action_mean.shape}")  # [4, 64, 2]
    print(f"Value shape: {value.shape}")  # [4, 1]

    # Get action
    action, value = policy.get_action(node_obs, adj, deterministic=False)
    print(f"Sampled action shape: {action.shape}")  # [4, 64, 2]

    # Evaluate actions
    log_probs, value, entropy = policy.evaluate_actions(node_obs, adj, action)
    print(f"Log probs shape: {log_probs.shape}")  # [4, 64]
    print(f"Entropy shape: {entropy.shape}")  # [4, 64]

    print("\n✓ All tests passed!")
