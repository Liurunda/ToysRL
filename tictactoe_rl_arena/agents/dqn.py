"""DQN: 基于价值的深度强化学习。

网络: MLP 9 -> 64 -> 64 -> 9 (Relu), 输入为扁平化 canonical 9 维观测。
机制:
  - Replay Buffer + Target Net: 缓解函数近似下的训练不稳定;
  - Action Masking: argmax 前将非法动作的 Q 值置为 -inf;
  - ε-greedy 探索 (仅在合法动作中采样), 每回合乘法衰减。

演示重点: 从"查表"到"函数近似"的跨越, 观察网络稳定性与样本效率差异
(神经网络前期收敛慢, 但无需枚举状态即具备泛化能力)。
"""
from collections import deque
from typing import List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from agents.base import BaseAgent, Transition


class QNet(nn.Module):
    """MLP: 9 -> 64 -> 64 -> 9。"""

    def __init__(self, obs_dim: int = 9, hidden: int = 64, n_actions: int = 9):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, n_actions),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ReplayBuffer:
    def __init__(self, capacity: int = 10000):
        self.buffer = deque(maxlen=capacity)

    def push(self, transition: Transition) -> None:
        self.buffer.append(transition)

    def sample(self, batch_size: int, rng: np.random.Generator) -> List[Transition]:
        idx = rng.choice(len(self.buffer), size=batch_size, replace=False)
        return [self.buffer[i] for i in idx]

    def __len__(self) -> int:
        return len(self.buffer)


class DQNAgent(BaseAgent):
    def __init__(self, name: str = "DQN", lr: float = 1e-3, gamma: float = 0.95,
                 buffer_size: int = 10000, batch_size: int = 64,
                 target_update_every: int = 100,
                 epsilon_start: float = 1.0, epsilon_end: float = 0.05,
                 epsilon_decay: float = 0.999,
                 device: str = "cpu", rng: Optional[np.random.Generator] = None):
        super().__init__(name)
        self.gamma = gamma
        self.batch_size = batch_size
        self.target_update_every = target_update_every
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.rng = rng if rng is not None else np.random
        self.device = torch.device(device)

        self.qnet = QNet().to(self.device)
        self.target_net = QNet().to(self.device)
        self.target_net.load_state_dict(self.qnet.state_dict())
        self.optimizer = torch.optim.Adam(self.qnet.parameters(), lr=lr)

        self.buffer = ReplayBuffer(buffer_size)
        self.n_updates = 0

    # ------------------------------------------------------------------ #
    def select_action(self, obs: np.ndarray, valid_mask: np.ndarray,
                      training: bool = True) -> int:
        eps = self.epsilon if training else 0.0
        valid = np.flatnonzero(valid_mask)
        if self.rng.random() < eps:
            return int(self.rng.choice(valid))
        with torch.no_grad():
            q = self.qnet(self._tensor(obs)).cpu().numpy()
        q = np.where(valid_mask, q, -np.inf)  # Action Masking
        return int(np.argmax(q))

    def observe(self, transition: Transition) -> None:
        self.buffer.push(transition)

    def end_episode(self) -> None:
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)

    # ------------------------------------------------------------------ #
    def update(self) -> dict:
        if len(self.buffer) < self.batch_size:
            return {}
        batch = self.buffer.sample(self.batch_size, self.rng)

        obs = torch.stack([self._tensor(t.obs).squeeze(0) for t in batch])
        actions = torch.tensor([t.action for t in batch], dtype=torch.long, device=self.device)
        rewards = torch.tensor([t.reward for t in batch], dtype=torch.float32, device=self.device)
        dones = torch.tensor([t.done for t in batch], dtype=torch.float32, device=self.device)
        next_obs = torch.stack([self._tensor(t.next_obs).squeeze(0) for t in batch])
        next_masks = torch.tensor(np.stack([t.next_valid_mask for t in batch]),
                                  dtype=torch.float32, device=self.device)

        # 目标值仅在下一状态的合法动作上取 max (Action Masking)
        # 注意: 终局转移的 next_mask 全 0 (max 为 -inf), 不能写成
        # (1-done)*max, 否则 0 * -inf = NaN 污染整个批次。
        with torch.no_grad():
            next_q = self.target_net(next_obs)
            next_q = next_q.masked_fill(next_masks == 0, float("-inf"))
            max_next = next_q.max(dim=1).values
            targets = torch.where(dones.bool(), rewards,
                                  rewards + self.gamma * max_next)

        q_values = self.qnet(obs).gather(1, actions.unsqueeze(1)).squeeze(1)
        loss = F.mse_loss(q_values, targets)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.n_updates += 1
        if self.n_updates % self.target_update_every == 0:
            self.target_net.load_state_dict(self.qnet.state_dict())

        return {"dqn_loss": float(loss), "q_mean": float(q_values.mean())}

    # ------------------------------------------------------------------ #
    def _tensor(self, obs: np.ndarray) -> torch.Tensor:
        return torch.tensor(np.asarray(obs, dtype=np.float32),
                            device=self.device).unsqueeze(0)

    def save(self, path: str) -> None:
        torch.save({
            "qnet": self.qnet.state_dict(),
            "target_net": self.target_net.state_dict(),
            "epsilon": self.epsilon,
            "gamma": self.gamma,
        }, path)

    def load(self, path: str) -> None:
        ckpt = torch.load(path, map_location=self.device, weights_only=True)
        self.qnet.load_state_dict(ckpt["qnet"])
        self.target_net.load_state_dict(ckpt["target_net"])
        self.epsilon = ckpt["epsilon"]
