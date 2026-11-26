"""PPO: 基于策略梯度的深度强化学习 (Actor-Critic + Action Masking)。

网络: Actor-Critic 双头 MLP —— 共享躯干 9 -> 64 -> 64, 分叉出
    Actor 头: 9 维动作 logits (Softmax 前将非法动作 logits 置为 -1e9,
              非法动作概率严格为 0);
    Critic 头: 标量状态价值 V(s)。
机制: GAE (λ) 优势估计 + PPO-Clip 目标 + 熵正则, 直接优化动作概率分布。

训练流: select_action 采样时缓存 (obs, action, old_logp);
        observe 将转移追加进 rollout buffer; update() 攒够 min_batch
        条样本后, 用当前网络重算 value 做 GAE, 跑 epochs 轮裁剪更新。
"""
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical

from agents.base import BaseAgent, Transition

MASK_VALUE = -1e9  # 非法动作 logits 屏蔽值 (Softmax 后概率严格为 0)


class ActorCritic(nn.Module):
    """双头 MLP: 输出 (动作 logits, 状态价值)。"""

    def __init__(self, obs_dim: int = 9, hidden: int = 64, n_actions: int = 9):
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Linear(obs_dim, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden), nn.Tanh(),
        )
        self.actor = nn.Linear(hidden, n_actions)
        self.critic = nn.Linear(hidden, 1)

    def forward(self, x: torch.Tensor):
        h = self.trunk(x)
        return self.actor(h), self.critic(h).squeeze(-1)

    def masked_logits(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Softmax 前屏蔽非法动作: logits 置 MASK_VALUE, 概率严格为 0。"""
        logits, _ = self.forward(x)
        return torch.where(mask.bool(), logits, torch.full_like(logits, MASK_VALUE))


class PPOAgent(BaseAgent):
    def __init__(self, name: str = "PPO", lr: float = 3e-4, gamma: float = 0.99,
                 lam: float = 0.95, clip_epsilon: float = 0.2,
                 entropy_coef: float = 0.01, value_coef: float = 0.5,
                 epochs: int = 8, min_batch: int = 64, mini_batch: int = 32,
                 max_grad_norm: float = 0.5,
                 device: str = "cpu", rng: Optional[np.random.Generator] = None):
        super().__init__(name)
        self.gamma = gamma
        self.lam = lam
        self.clip_epsilon = clip_epsilon
        self.entropy_coef = entropy_coef
        self.value_coef = value_coef
        self.epochs = epochs
        self.min_batch = min_batch
        self.mini_batch = mini_batch
        self.max_grad_norm = max_grad_norm
        self.rng = rng if rng is not None else np.random
        self.device = torch.device(device)

        self.net = ActorCritic().to(self.device)
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=lr)

        # rollout buffer: (obs, action, old_logp, reward, done, valid_mask, next_obs)
        self.buffer = []
        self._last = None  # select_action 时缓存的 (obs, action, logp, valid_mask)

    # ------------------------------------------------------------------ #
    def select_action(self, obs: np.ndarray, valid_mask: np.ndarray,
                      training: bool = True) -> int:
        x = self._tensor(obs)
        mask_t = self._mask_tensor(valid_mask)
        with torch.no_grad():
            logits = self.net.masked_logits(x, mask_t)
            dist = Categorical(logits=logits)
            if training:
                action = int(dist.sample())
            else:
                action = int(torch.argmax(logits))
            logp = dist.log_prob(torch.tensor(action, device=self.device))
        self._last = (obs, action, float(logp), valid_mask)
        return action

    def observe(self, transition: Transition) -> None:
        obs, action, logp, valid_mask = self._last
        self.buffer.append((obs, action, logp, transition.reward, transition.done,
                            valid_mask, transition.next_obs))
        self._last = None

    # ------------------------------------------------------------------ #
    def update(self) -> dict:
        if len(self.buffer) < self.min_batch:
            return {}
        buffer, self.buffer = self.buffer, []
        return self._ppo_update(buffer)

    def _ppo_update(self, buffer) -> dict:
        obs = torch.stack([self._tensor(e[0]).squeeze(0) for e in buffer])
        actions = torch.tensor([e[1] for e in buffer], dtype=torch.long, device=self.device)
        old_logps = torch.tensor([e[2] for e in buffer], dtype=torch.float32, device=self.device)
        rewards = torch.tensor([e[3] for e in buffer], dtype=torch.float32, device=self.device)
        dones = torch.tensor([e[4] for e in buffer], dtype=torch.float32, device=self.device)
        masks = torch.stack([self._mask_tensor(e[5]) for e in buffer])
        next_obs = torch.stack([self._tensor(e[6]).squeeze(0) for e in buffer])

        with torch.no_grad():
            values = self.net(obs)[1]
            # GAE 引导值: 最后一条样本的后继状态价值 (若终局则为 0)
            last_next_value = (1.0 - dones[-1]) * self.net(next_obs[-1:])[1][0]
            adv = torch.zeros_like(rewards)
            next_adv = 0.0
            next_val = last_next_value
            for i in range(len(buffer) - 1, -1, -1):
                delta = rewards[i] + self.gamma * next_val * (1.0 - dones[i]) - values[i]
                adv_i = delta + self.gamma * self.lam * next_adv * (1.0 - dones[i])
                adv[i] = adv_i
                next_adv = adv_i
                next_val = values[i]
            returns = adv + values
            adv = (adv - adv.mean()) / (adv.std() + 1e-8)

        n = len(buffer)
        metrics = {"ppo_policy_loss": 0.0, "ppo_value_loss": 0.0, "ppo_entropy": 0.0}
        for _ in range(self.epochs):
            perm = torch.randperm(n)
            for start in range(0, n, self.mini_batch):
                idx = perm[start:start + self.mini_batch]
                logits = self.net.masked_logits(obs[idx], masks[idx])
                dist = Categorical(logits=logits)
                new_logps = dist.log_prob(actions[idx])
                entropy = dist.entropy().mean()

                ratio = torch.exp(new_logps - old_logps[idx])
                surr1 = ratio * adv[idx]
                surr2 = torch.clamp(ratio, 1.0 - self.clip_epsilon,
                                    1.0 + self.clip_epsilon) * adv[idx]
                policy_loss = -torch.min(surr1, surr2).mean()

                value_pred = self.net(obs[idx])[1]
                value_loss = F.mse_loss(value_pred, returns[idx])

                loss = (policy_loss + self.value_coef * value_loss
                        - self.entropy_coef * entropy)
                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.net.parameters(), self.max_grad_norm)
                self.optimizer.step()

                metrics["ppo_policy_loss"] += float(policy_loss)
                metrics["ppo_value_loss"] += float(value_loss)
                metrics["ppo_entropy"] += float(entropy)

        n_steps = max(1, self.epochs * ((n + self.mini_batch - 1) // self.mini_batch))
        return {k: v / n_steps for k, v in metrics.items()}

    # ------------------------------------------------------------------ #
    def _tensor(self, obs: np.ndarray) -> torch.Tensor:
        return torch.tensor(np.asarray(obs, dtype=np.float32),
                            device=self.device).unsqueeze(0)

    def _mask_tensor(self, valid_mask: np.ndarray) -> torch.Tensor:
        return torch.tensor(np.asarray(valid_mask, dtype=np.float32),
                            device=self.device).unsqueeze(0)

    def save(self, path: str) -> None:
        torch.save({"net": self.net.state_dict()}, path)

    def load(self, path: str) -> None:
        self.net.load_state_dict(
            torch.load(path, map_location=self.device, weights_only=True)["net"])
