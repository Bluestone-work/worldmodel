"""
PPO Trainer for Magnetic Swarm GAT Policy

Simplified MAPPO implementation for magnetic swarm navigation:
- Centralized critic (observes global state)
- Decentralized actor (per-particle desired velocity)
- Constraint allocator (project to realizable field)
- Experience replay buffer
- GAE for advantage estimation
"""
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from typing import Tuple, Dict, List
from collections import deque


class PPOBuffer:
    """Experience replay buffer for PPO"""

    def __init__(self, capacity: int = 100000):
        self.capacity = capacity
        self.buffer = deque(maxlen=capacity)

    def store(
        self,
        node_obs: np.ndarray,
        adj: np.ndarray,
        action: np.ndarray,
        reward: float,
        value: float,
        log_prob: np.ndarray,
        done: bool
    ):
        """Store one transition"""
        self.buffer.append({
            'node_obs': node_obs,
            'adj': adj,
            'action': action,
            'reward': reward,
            'value': value,
            'log_prob': log_prob,
            'done': done
        })

    def get_batch(self, batch_size: int = None) -> Dict[str, torch.Tensor]:
        """Get a batch of transitions"""
        if batch_size is None:
            batch_size = len(self.buffer)

        indices = np.random.choice(len(self.buffer), batch_size, replace=False)
        batch = [self.buffer[i] for i in indices]

        return {
            'node_obs': torch.FloatTensor(np.array([t['node_obs'] for t in batch])),
            'adj': torch.FloatTensor(np.array([t['adj'] for t in batch])),
            'action': torch.FloatTensor(np.array([t['action'] for t in batch])),
            'reward': torch.FloatTensor(np.array([t['reward'] for t in batch])),
            'value': torch.FloatTensor(np.array([t['value'] for t in batch])),
            'log_prob': torch.FloatTensor(np.array([t['log_prob'] for t in batch])),
            'done': torch.FloatTensor(np.array([t['done'] for t in batch])),
        }

    def compute_advantages(self, gamma: float = 0.99, gae_lambda: float = 0.95):
        """Compute GAE advantages in-place"""
        if len(self.buffer) == 0:
            return

        # Convert to numpy arrays
        rewards = np.array([t['reward'] for t in self.buffer])
        values = np.array([t['value'] for t in self.buffer])
        dones = np.array([t['done'] for t in self.buffer])

        # Compute advantages using GAE
        advantages = np.zeros_like(rewards)
        last_advantage = 0
        last_value = 0

        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                next_value = 0
            else:
                next_value = values[t + 1]

            delta = rewards[t] + gamma * next_value * (1 - dones[t]) - values[t]
            advantages[t] = delta + gamma * gae_lambda * (1 - dones[t]) * last_advantage
            last_advantage = advantages[t]

        # Store advantages and returns
        returns = advantages + values

        for i, transition in enumerate(self.buffer):
            transition['advantage'] = advantages[i]
            transition['return'] = returns[i]

    def clear(self):
        """Clear buffer"""
        self.buffer.clear()

    def __len__(self):
        return len(self.buffer)


class PPOTrainer:
    """PPO trainer for GAT policy"""

    def __init__(
        self,
        policy: nn.Module,
        lr: float = 3e-4,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_epsilon: float = 0.2,
        value_coef: float = 0.5,
        entropy_coef: float = 0.01,
        max_grad_norm: float = 0.5,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    ):
        self.policy = policy.to(device)
        self.optimizer = optim.Adam(policy.parameters(), lr=lr)
        self.device = device

        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_epsilon = clip_epsilon
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef
        self.max_grad_norm = max_grad_norm

        self.buffer = PPOBuffer()

    def collect_experience(
        self,
        env,
        n_steps: int = 2048
    ) -> Dict[str, float]:
        """Collect experience using current policy"""
        self.policy.eval()
        episode_rewards = []
        episode_lengths = []
        current_episode_reward = 0
        current_episode_length = 0

        obs, info = env.reset()
        done = False

        for step in range(n_steps):
            # Build node observations and adjacency matrix
            node_obs, adj = self._build_graph_observation(obs, env)

            # Get action
            with torch.no_grad():
                node_obs_tensor = torch.FloatTensor(node_obs).unsqueeze(0).to(self.device)
                adj_tensor = torch.FloatTensor(adj).unsqueeze(0).to(self.device)

                action, value = self.policy.get_action(node_obs_tensor, adj_tensor, deterministic=False)
                action = action.squeeze(0).cpu().numpy()
                value = value.squeeze(0).cpu().numpy()

                # Compute log prob for storage
                log_prob, _, _ = self.policy.evaluate_actions(
                    node_obs_tensor,
                    adj_tensor,
                    torch.FloatTensor(action).unsqueeze(0).to(self.device)
                )
                log_prob = log_prob.squeeze(0).cpu().numpy()

            # Aggregate desired velocities → realizable field action
            field_action = self._aggregate_to_field(action, env)

            # Step environment
            next_obs, reward, terminated, truncated, info = env.step(field_action)
            done = terminated or truncated

            # Store transition
            # Sum log_probs across particles for single transition probability
            total_log_prob = log_prob.sum()

            self.buffer.store(
                node_obs=node_obs,
                adj=adj,
                action=action,
                reward=reward,
                value=value.item() if isinstance(value, np.ndarray) else value,
                log_prob=total_log_prob,
                done=done
            )

            current_episode_reward += reward
            current_episode_length += 1

            if done:
                episode_rewards.append(current_episode_reward)
                episode_lengths.append(current_episode_length)
                current_episode_reward = 0
                current_episode_length = 0
                obs, info = env.reset()
                done = False
            else:
                obs = next_obs

        # Compute advantages
        self.buffer.compute_advantages(self.gamma, self.gae_lambda)

        return {
            'mean_reward': np.mean(episode_rewards) if episode_rewards else 0.0,
            'mean_length': np.mean(episode_lengths) if episode_lengths else 0.0,
            'n_episodes': len(episode_rewards)
        }

    def _build_graph_observation(
        self,
        obs: Dict[str, np.ndarray],
        env
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Build per-particle node observations and proximity graph.

        Node features: [x, y, z, dist_to_nearest_clot, dist_to_centroid, ...]
        Adjacency: particles within sensing_radius are connected
        """
        robot_positions = env.robot_positions  # (N, 3)
        clot_positions = env.clot_positions    # (M, 3)
        clot_masses = env.clot_masses          # (M,)

        n_particles = len(robot_positions)

        # Per-particle features
        node_features = []
        for i in range(n_particles):
            pos = robot_positions[i]

            # Distance to active clots
            active_clots = clot_positions[clot_masses > 0]
            if len(active_clots) > 0:
                dist_to_clots = np.linalg.norm(active_clots - pos, axis=1)
                min_clot_dist = dist_to_clots.min()
            else:
                min_clot_dist = 1.0  # Normalized max distance

            # Distance to swarm centroid
            centroid = robot_positions.mean(axis=0)
            dist_to_centroid = np.linalg.norm(pos - centroid)

            # Normalized position
            norm_pos = pos / max(env.map_size, 1.0)

            # Node feature vector
            feature = np.array([
                norm_pos[0],
                norm_pos[1],
                norm_pos[2] / max(1, env.num_layers - 1),
                min_clot_dist,
                dist_to_centroid,
                float(env.step_count) / max(1, env.horizon),
            ], dtype=np.float32)

            node_features.append(feature)

        node_obs = np.array(node_features)  # (N, feature_dim)

        # Proximity-based adjacency matrix
        dist_matrix = np.linalg.norm(
            robot_positions[:, None, :2] - robot_positions[None, :, :2],
            axis=2
        )
        radius = env.local_sensing_radius if hasattr(env, 'local_sensing_radius') else 0.2
        adj = (dist_matrix < radius * env.map_size).astype(np.float32)

        # Add self-loops
        adj = adj + np.eye(n_particles, dtype=np.float32)

        return node_obs, adj

    def _aggregate_to_field(
        self,
        desired_velocities: np.ndarray,
        env
    ) -> np.ndarray:
        """
        Aggregate per-particle desired velocities to realizable field action.

        Simple approach: mean desired velocity → field [Bx, By, 0, 0, 0]
        (no gradient control for now)

        Returns: [Bx, By, Gxx, Gxy, Gyy] (5D field action)
        """
        # Mean desired velocity (transport field)
        mean_velocity = desired_velocities.mean(axis=0)  # (2,)

        # Map to field action [Bx, By, Gxx, Gxy, Gyy]
        field_action = np.array([
            mean_velocity[0],
            mean_velocity[1],
            0.0,  # No gradient control yet
            0.0,
            0.0
        ], dtype=np.float32)

        # Clip to [-1, 1]
        field_action = np.clip(field_action, -1.0, 1.0)

        return field_action

    def update(
        self,
        n_epochs: int = 10,
        batch_size: int = 256
    ) -> Dict[str, float]:
        """Update policy using collected experience"""
        self.policy.train()

        if len(self.buffer) == 0:
            return {}

        # Compute advantages if not already done
        if 'advantage' not in self.buffer.buffer[0]:
            self.buffer.compute_advantages(self.gamma, self.gae_lambda)

        # Training metrics
        total_policy_loss = 0.0
        total_value_loss = 0.0
        total_entropy = 0.0
        n_updates = 0

        for epoch in range(n_epochs):
            # Sample batches
            n_samples = len(self.buffer)
            indices = np.random.permutation(n_samples)

            for start_idx in range(0, n_samples, batch_size):
                end_idx = min(start_idx + batch_size, n_samples)
                batch_indices = indices[start_idx:end_idx]

                # Get batch
                batch = [self.buffer.buffer[i] for i in batch_indices]

                node_obs = torch.FloatTensor(np.array([t['node_obs'] for t in batch])).to(self.device)
                adj = torch.FloatTensor(np.array([t['adj'] for t in batch])).to(self.device)
                actions = torch.FloatTensor(np.array([t['action'] for t in batch])).to(self.device)
                old_log_probs = torch.FloatTensor(np.array([t['log_prob'] for t in batch])).to(self.device)
                advantages = torch.FloatTensor(np.array([t['advantage'] for t in batch])).to(self.device)
                returns = torch.FloatTensor(np.array([t['return'] for t in batch])).to(self.device)

                # Normalize advantages
                advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

                # Evaluate actions
                log_probs, values, entropy = self.policy.evaluate_actions(node_obs, adj, actions)

                # Sum log_probs across particles (batch dimension remains)
                log_probs_sum = log_probs.sum(dim=1)  # [batch]

                # PPO policy loss
                ratio = torch.exp(log_probs_sum - old_log_probs)
                surr1 = ratio * advantages
                surr2 = torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * advantages
                policy_loss = -torch.min(surr1, surr2).mean()

                # Value loss
                value_loss = ((values.squeeze() - returns) ** 2).mean()

                # Entropy bonus (mean across particles and batch)
                entropy_loss = -entropy.mean()

                # Total loss
                loss = policy_loss + self.value_coef * value_loss + self.entropy_coef * entropy_loss

                # Optimize
                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                self.optimizer.step()

                # Record metrics
                total_policy_loss += policy_loss.item()
                total_value_loss += value_loss.item()
                total_entropy += entropy.mean().item()
                n_updates += 1

        # Clear buffer after update
        self.buffer.clear()

        return {
            'policy_loss': total_policy_loss / max(1, n_updates),
            'value_loss': total_value_loss / max(1, n_updates),
            'entropy': total_entropy / max(1, n_updates),
        }

    def save(self, path: str):
        """Save policy checkpoint"""
        torch.save({
            'policy_state_dict': self.policy.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
        }, path)

    def load(self, path: str):
        """Load policy checkpoint"""
        checkpoint = torch.load(path, map_location=self.device)
        self.policy.load_state_dict(checkpoint['policy_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
