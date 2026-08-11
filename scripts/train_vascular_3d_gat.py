#!/usr/bin/env python3
"""Train/evaluate a shared-actuator GAT policy on the 3-D vascular task."""
from __future__ import annotations

import argparse
import json
import sys
from collections import deque
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from environments.vascular_3d_swarm_env import Vascular3DSwarmEnv
from marl.gat_policy import MagneticSwarmGATPolicy
from marl.mlp_swarm_policy import MLPParticlePolicy
from marl.ppo_trainer import PPOTrainer
from marl.vascular_graph_adapter import build_vascular_graph_observation, shared_action_from_particle_intents


class Vascular3DTrainer(PPOTrainer):
    """Reuse the project's PPO/GAE implementation with a 3-D action bridge."""

    def _build_graph_observation(self, obs, env):
        return build_vascular_graph_observation(env)

    def _aggregate_to_field(self, desired_velocities, env):
        return shared_action_from_particle_intents(desired_velocities)


def _augment_with_action_history(
    node_obs: np.ndarray,
    action_history: deque[np.ndarray],
    history_actions: int,
) -> np.ndarray:
    if history_actions <= 0:
        return node_obs
    history = np.zeros((history_actions, 4), dtype=np.float32)
    recent = list(action_history)[-history_actions:]
    if recent:
        history[-len(recent):] = np.asarray(recent, dtype=np.float32)
    history_flat = history.reshape(-1)
    tiled = np.broadcast_to(history_flat, (node_obs.shape[0], len(history_flat)))
    return np.concatenate([node_obs, tiled], axis=1)


class HistoryVascular3DTrainer(Vascular3DTrainer):
    def __init__(self, *args, history_actions: int = 3, **kwargs):
        super().__init__(*args, **kwargs)
        self.history_actions = int(history_actions)
        self._action_history: deque[np.ndarray] = deque(maxlen=max(1, self.history_actions))

    def _build_graph_observation(self, obs, env):
        node_obs, adj = super()._build_graph_observation(obs, env)
        node_obs = _augment_with_action_history(node_obs, self._action_history, self.history_actions)
        return node_obs, adj

    def collect_experience(self, env, n_steps: int = 2048) -> dict[str, float]:
        self.policy.eval()
        episode_rewards = []
        episode_lengths = []
        current_episode_reward = 0
        current_episode_length = 0

        self._action_history.clear()
        obs, info = env.reset()
        done = False

        for step in range(n_steps):
            node_obs, adj = self._build_graph_observation(obs, env)

            with torch.no_grad():
                node_obs_tensor = torch.FloatTensor(node_obs).unsqueeze(0).to(self.device)
                adj_tensor = torch.FloatTensor(adj).unsqueeze(0).to(self.device)

                action, value = self.policy.get_action(node_obs_tensor, adj_tensor, deterministic=False)
                action = action.squeeze(0).cpu().numpy()
                value = value.squeeze(0).cpu().numpy()

                log_prob, _, _ = self.policy.evaluate_actions(
                    node_obs_tensor,
                    adj_tensor,
                    torch.FloatTensor(action).unsqueeze(0).to(self.device)
                )
                log_prob = log_prob.squeeze(0).cpu().numpy()

            field_action = self._aggregate_to_field(action, env)
            next_obs, reward, terminated, truncated, info = env.step(field_action)
            done = terminated or truncated
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
                self._action_history.clear()
                obs, info = env.reset()
                done = False
            else:
                self._action_history.append(np.asarray(field_action, dtype=np.float32))
                obs = next_obs

        self.buffer.compute_advantages(self.gamma, self.gae_lambda)

        return {
            "mean_reward": np.mean(episode_rewards) if episode_rewards else 0.0,
            "mean_length": np.mean(episode_lengths) if episode_lengths else 0.0,
            "n_episodes": len(episode_rewards),
        }


def evaluate(
    trainer: Vascular3DTrainer,
    env: Vascular3DSwarmEnv,
    episodes: int,
    seed: int,
    history_actions: int = 0,
) -> dict:
    trainer.policy.eval()
    rows = []
    for episode in range(episodes):
        env.reset(seed=seed + episode)
        action_history: deque[np.ndarray] = deque(maxlen=max(1, history_actions))
        total_reward = 0.0
        first_contact = None
        for _ in range(env.horizon):
            node, adj = build_vascular_graph_observation(env)
            node = _augment_with_action_history(node, action_history, history_actions)
            with torch.no_grad():
                action, _ = trainer.policy.get_action(
                    torch.as_tensor(node, dtype=torch.float32, device=trainer.device).unsqueeze(0),
                    torch.as_tensor(adj, dtype=torch.float32, device=trainer.device).unsqueeze(0),
                    deterministic=True,
                )
            shared = shared_action_from_particle_intents(action.squeeze(0).cpu().numpy())
            _, reward, terminated, truncated, info = env.step(shared)
            if history_actions > 0:
                action_history.append(np.asarray(shared, dtype=np.float32))
            total_reward += float(reward)
            if info["active_contacts"] > 0 and first_contact is None:
                first_contact = env.step_count
            if terminated or truncated:
                break
        rows.append({
            "seed": seed + episode,
            "success": float(info["success"]),
            "removal_rate": float(1.0 - info["remaining_mass"] / env.num_clots),
            "remaining_mass": float(info["remaining_mass"]),
            "steps": int(env.step_count),
            "total_reward": total_reward,
            "collision_rate": float(info["collision_rate"]),
            "known_fraction": float(info["known_fraction"]),
            "first_contact_step": int(first_contact if first_contact is not None else env.step_count),
        })
    return {
        "success_rate": float(np.mean([x["success"] for x in rows])),
        "removal_rate": float(np.mean([x["removal_rate"] for x in rows])),
        "remaining_mass": float(np.mean([x["remaining_mass"] for x in rows])),
        "mean_steps": float(np.mean([x["steps"] for x in rows])),
        "mean_reward": float(np.mean([x["total_reward"] for x in rows])),
        "collision_rate": float(np.mean([x["collision_rate"] for x in rows])),
        "known_fraction": float(np.mean([x["known_fraction"] for x in rows])),
        "first_contact_step": float(np.mean([x["first_contact_step"] for x in rows])),
        "episodes": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenario", default="bifurcation")
    parser.add_argument("--model", choices=("gat", "mlp"), default="gat")
    parser.add_argument("--robots", type=int, default=12)
    parser.add_argument("--clots", type=int, default=2)
    parser.add_argument("--distance-reward-scale", type=float, default=0.0)
    parser.add_argument("--junction-reward-scale", type=float, default=0.0)
    parser.add_argument("--history-actions", type=int, default=0)
    parser.add_argument("--updates", type=int, default=10)
    parser.add_argument("--steps-per-update", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--eval-episodes", type=int, default=10)
    parser.add_argument("--seed", type=int, default=20260811)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output", default="autoresearch/vascular-3d-gat")
    parser.add_argument("--load", default="")
    args = parser.parse_args()
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    output = ROOT / args.output
    output.mkdir(parents=True, exist_ok=True)
    env = Vascular3DSwarmEnv(
        scenario=args.scenario,
        num_robots=args.robots,
        num_clots=args.clots,
        distance_reward_scale=args.distance_reward_scale,
        junction_reward_scale=args.junction_reward_scale,
    )
    node_obs_dim = 10 + max(0, args.history_actions) * 4
    if args.model == "gat":
        policy = MagneticSwarmGATPolicy(node_obs_dim=node_obs_dim, action_dim=3, hidden_dim=64, gat_hidden_dim=64, n_gat_heads=2)
    else:
        policy = MLPParticlePolicy(node_obs_dim=node_obs_dim, action_dim=3, hidden_dim=128)
    trainer_cls = HistoryVascular3DTrainer if args.history_actions > 0 else Vascular3DTrainer
    trainer = trainer_cls(
        policy=policy,
        device=args.device,
        entropy_coef=0.005,
        **({"history_actions": args.history_actions} if args.history_actions > 0 else {}),
    )
    if args.load:
        trainer.load(args.load)
    log = []
    for update in range(args.updates):
        collect = trainer.collect_experience(env, n_steps=args.steps_per_update)
        train = trainer.update(n_epochs=args.epochs, batch_size=args.batch_size)
        row = {"update": update + 1, **collect, **train}
        log.append(row)
        print(json.dumps(row, sort_keys=True))
    evaluation = evaluate(trainer, env, args.eval_episodes, args.seed + 100000, history_actions=args.history_actions)
    trainer.save(output / "final_policy.pth")
    (output / "training_log.json").write_text(json.dumps(log, indent=2))
    (output / "evaluation.json").write_text(json.dumps(evaluation, indent=2))
    (output / "config.json").write_text(json.dumps(vars(args), indent=2))
    print(json.dumps({k: v for k, v in evaluation.items() if k != "episodes"}, indent=2))
    env.close()


if __name__ == "__main__":
    main()
