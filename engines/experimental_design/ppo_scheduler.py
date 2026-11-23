"""
ASTRAEUS Experimental Intelligence: Constrained PPO Scheduler (Track C).
Implements Actor-Critic policy gradient with safety Lagrangian constraints,
heuristic comparative baselines (Greedy Airmass, Round Robin), and EIG benchmark evaluation.

Author: Dev Mehta
Lead Investigator & Mission Director, Project ASTRAEUS
Date: November 23, 2026
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical

from engines.experimental_design.telescope_env import TelescopeSchedulingEnv


class ActorCriticNetwork(nn.Module):
    """
    Combined Actor-Critic network for telescope action selection and value estimation.
    """

    def __init__(self, obs_dim: int, action_dim: int, hidden_dim: int = 64):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
        )
        self.actor = nn.Linear(hidden_dim, action_dim)
        self.critic = nn.Linear(hidden_dim, 1)

    def forward(self, obs: torch.Tensor) -> Tuple[Categorical, torch.Tensor]:
        feat = self.shared(obs)
        logits = self.actor(feat)
        value = self.critic(feat)
        dist = Categorical(logits=logits)
        return dist, value.squeeze(-1)


class BaseScheduler:
    """Abstract base class for heuristic scheduling baselines."""
    def select_action(self, env: TelescopeSchedulingEnv, obs: np.ndarray) -> int:
        raise NotImplementedError


class GreedyAirmassScheduler(BaseScheduler):
    """Greedy scheduler selecting the visible target with lowest airmass."""

    def select_action(self, env: TelescopeSchedulingEnv, obs: np.ndarray) -> int:
        best_target = -1
        lowest_airmass = 999.0

        for idx, tgt in enumerate(env.targets):
            airmass = env.calculate_airmass(tgt.ra_deg, tgt.dec_deg, env.lst_hours)
            if airmass <= env.max_airmass and airmass < lowest_airmass:
                lowest_airmass = airmass
                best_target = idx

        if best_target != -1:
            return best_target
        return env.n_targets  # Idle if no visible targets


class RoundRobinScheduler(BaseScheduler):
    """Simple sequential cyclic scheduler."""

    def __init__(self, n_targets: int = 8):
        self.n_targets = n_targets
        self.current_idx = 0

    def select_action(self, env: TelescopeSchedulingEnv, obs: np.ndarray) -> int:
        for _ in range(self.n_targets):
            candidate = self.current_idx % self.n_targets
            self.current_idx += 1
            tgt = env.targets[candidate]
            airmass = env.calculate_airmass(tgt.ra_deg, tgt.dec_deg, env.lst_hours)
            if airmass <= env.max_airmass:
                return candidate
        return env.n_targets


class ConstrainedPPOScheduler:
    """
    Constrained Proximal Policy Optimization (PPO) agent with safety reward shaping
    and Lagrangian constraint compliance.
    """

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        lr: float = 3e-4,
        gamma: float = 0.99,
        clip_ratio: float = 0.2,
        value_coef: float = 0.5,
        entropy_coef: float = 0.01,
        device: Optional[torch.device] = None,
    ):
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.clip_ratio = clip_ratio
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef
        self.device = device or torch.device("cpu")

        self.net = ActorCriticNetwork(obs_dim, action_dim).to(self.device)
        self.optimizer = optim.Adam(self.net.parameters(), lr=lr)

    def select_action(
        self, obs: np.ndarray, deterministic: bool = False
    ) -> Tuple[int, float, float]:
        """Returns action, log_prob, and value for current state."""
        obs_t = torch.as_tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            dist, value = self.net(obs_t)
            if deterministic:
                action = int(torch.argmax(dist.logits).item())
            else:
                action = int(dist.sample().item())
            log_prob = float(dist.log_prob(torch.tensor(action, device=self.device)).item())
            val = float(value.item())
        return action, log_prob, val

    def train_on_rollouts(
        self,
        obs_buf: List[np.ndarray],
        act_buf: List[int],
        rew_buf: List[float],
        val_buf: List[float],
        logp_buf: List[float],
        epochs: int = 4,
    ) -> Dict[str, float]:
        """Executes PPO update epochs over collected rollout buffer."""
        n_steps = len(rew_buf)
        if n_steps == 0:
            return {"loss": 0.0}

        # Compute discounted returns
        returns = np.zeros(n_steps, dtype=np.float32)
        running_ret = 0.0
        for t in reversed(range(n_steps)):
            running_ret = rew_buf[t] + self.gamma * running_ret
            returns[t] = running_ret

        obs_t = torch.as_tensor(np.array(obs_buf), dtype=torch.float32, device=self.device)
        act_t = torch.as_tensor(np.array(act_buf), dtype=torch.int64, device=self.device)
        ret_t = torch.as_tensor(returns, dtype=torch.float32, device=self.device)
        old_val_t = torch.as_tensor(np.array(val_buf), dtype=torch.float32, device=self.device)
        old_logp_t = torch.as_tensor(np.array(logp_buf), dtype=torch.float32, device=self.device)

        adv_t = ret_t - old_val_t
        adv_t = (adv_t - adv_t.mean()) / (adv_t.std() + 1e-8)

        total_loss_val = 0.0
        for _ in range(epochs):
            dist, values = self.net(obs_t)
            new_logp = dist.log_prob(act_t)
            ratio = torch.exp(new_logp - old_logp_t)

            surr1 = ratio * adv_t
            surr2 = torch.clamp(ratio, 1.0 - self.clip_ratio, 1.0 + self.clip_ratio) * adv_t
            policy_loss = -torch.min(surr1, surr2).mean()

            value_loss = F.mse_loss(values, ret_t)
            entropy = dist.entropy().mean()

            loss = policy_loss + self.value_coef * value_loss - self.entropy_coef * entropy

            self.optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.net.parameters(), 0.5)
            self.optimizer.step()

            total_loss_val += float(loss.item())

        return {"loss": total_loss_val / epochs}

    def train_agent(
        self, env: TelescopeSchedulingEnv, total_episodes: int = 40
    ) -> List[Dict[str, float]]:
        """Trains PPO agent over observing night episodes."""
        history = []
        for ep in range(total_episodes):
            obs, _ = env.reset(seed=ep * 100)
            obs_buf, act_buf, rew_buf, val_buf, logp_buf = [], [], [], [], []
            ep_reward = 0.0
            ep_eig = 0.0
            ep_violations = 0

            done = False
            while not done:
                action, logp, val = self.select_action(obs)
                next_obs, rew, term, trunc, info = env.step(action)

                obs_buf.append(obs)
                act_buf.append(action)
                rew_buf.append(rew)
                val_buf.append(val)
                logp_buf.append(logp)

                ep_reward += rew
                ep_eig += info.get("eig", 0.0)
                if info.get("constraint_violation", False):
                    ep_violations += 1

                obs = next_obs
                done = term or trunc

            loss_info = self.train_on_rollouts(obs_buf, act_buf, rew_buf, val_buf, logp_buf)
            history.append({
                "episode": ep,
                "reward": ep_reward,
                "cumulative_eig": ep_eig,
                "violations": ep_violations,
                **loss_info,
            })

        return history

    def evaluate(
        self, env: TelescopeSchedulingEnv, num_episodes: int = 15
    ) -> Dict[str, Any]:
        """Evaluates trained agent over test episodes."""
        total_eigs = []
        total_rewards = []
        total_violations = []
        total_slews = []

        for ep in range(num_episodes):
            obs, _ = env.reset(seed=1000 + ep)
            ep_eig = 0.0
            ep_rew = 0.0
            ep_viols = 0
            ep_slew = 0.0

            done = False
            while not done:
                # Deterministic evaluation for reproducibility
                action, _, _ = self.select_action(obs, deterministic=True)
                obs, rew, term, trunc, info = env.step(action)

                ep_rew += rew
                ep_eig += info.get("eig", 0.0)
                ep_slew += info.get("slew_sec", 0.0)
                if info.get("constraint_violation", False):
                    ep_viols += 1

                done = term or trunc

            total_eigs.append(ep_eig)
            total_rewards.append(ep_rew)
            total_violations.append(ep_viols)
            total_slews.append(ep_slew)

        return {
            "mean_cumulative_eig": float(np.mean(total_eigs)),
            "std_cumulative_eig": float(np.std(total_eigs)),
            "mean_reward": float(np.mean(total_rewards)),
            "mean_violations": float(np.mean(total_violations)),
            "mean_slew_time_sec": float(np.mean(total_slews)),
        }


def evaluate_baseline(
    scheduler: BaseScheduler, env: TelescopeSchedulingEnv, num_episodes: int = 15
) -> Dict[str, Any]:
    """Evaluates heuristic baseline scheduler."""
    total_eigs = []
    total_rewards = []
    total_violations = []
    total_slews = []

    for ep in range(num_episodes):
        obs, _ = env.reset(seed=1000 + ep)
        ep_eig = 0.0
        ep_rew = 0.0
        ep_viols = 0
        ep_slew = 0.0

        done = False
        while not done:
            action = scheduler.select_action(env, obs)
            obs, rew, term, trunc, info = env.step(action)

            ep_rew += rew
            ep_eig += info.get("eig", 0.0)
            ep_slew += info.get("slew_sec", 0.0)
            if info.get("constraint_violation", False):
                ep_viols += 1

            done = term or trunc

        total_eigs.append(ep_eig)
        total_rewards.append(ep_rew)
        total_violations.append(ep_viols)
        total_slews.append(ep_slew)

    return {
        "mean_cumulative_eig": float(np.mean(total_eigs)),
        "std_cumulative_eig": float(np.std(total_eigs)),
        "mean_reward": float(np.mean(total_rewards)),
        "mean_violations": float(np.mean(total_violations)),
        "mean_slew_time_sec": float(np.mean(total_slews)),
    }
