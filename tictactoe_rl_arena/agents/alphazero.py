"""AlphaZero-Lite: MCTS + Policy-Value 网络的策略迭代闭环。

Policy-Value 网络输出 (P, v):
    P 为 9 维动作先验概率, v ∈ [-1, 1] 为状态价值预测 (Tanh)。
树搜索用 PUCT 公式:
    a* = argmax_a [ Q(s,a) + c_puct * P(a) * sqrt(N(s)) / (1 + N(s,a)) ]
    Q(s,a) 由自博弈结果回传估计, 替代纯 MCTS 的随机 Rollout;
叶节点不再模拟, 直接用网络价值 v 估值。

训练闭环: MCTS 搜索后的访问分布 π 反过来监督训练网络:
    loss = (z - v)^2 - πᵀ log p + c‖θ‖²
(可选棋盘对称性增强, 见 utils/symmetry.py)
"""
import math
from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from agents.base import BaseAgent
from envs.tictactoe import TicTacToeEnv, state_result
from utils.symmetry import transform_pi, transform_state


class PolicyValueNet(nn.Module):
    """双头 MLP: 策略头输出动作 logits, 价值头输出 Tanh 标量。"""

    def __init__(self, obs_dim: int = 9, hidden: int = 64, n_actions: int = 9):
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Linear(obs_dim, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden), nn.Tanh(),
        )
        self.policy = nn.Linear(hidden, n_actions)
        self.value = nn.Linear(hidden, 1)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        h = self.trunk(x)
        return self.policy(h), torch.tanh(self.value(h)).squeeze(-1)

    @torch.no_grad()
    def predict(self, state: tuple, device) -> Tuple[np.ndarray, float]:
        """单个 canonical 状态的 (P, v), 返回 numpy 结果。"""
        x = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
        logits, value = self.forward(x)
        p = F.softmax(logits, dim=-1)
        return p.cpu().numpy().reshape(-1), float(value)


class AZNode:
    """树节点: state 为 canonical 元组 (轮到走的一方视角)。"""

    __slots__ = ("state", "valid", "action", "P", "N", "W", "N_total",
                 "children", "expanded")

    def __init__(self, state: tuple, valid: tuple, action: Optional[int] = None):
        self.state = state
        self.valid = valid
        self.action = action        # 从父节点到达本节点的动作
        self.P = None               # 网络先验 (展开时填充, 长度 9)
        self.N = np.zeros(9, dtype=float)   # 每条边的访问次数
        self.W = 0.0                # 累计价值 (本节点玩家视角)
        self.N_total = 0            # 本节点访问次数
        self.children = {}          # action -> AZNode
        self.expanded = False


class AlphaZeroAgent(BaseAgent):
    def __init__(self, name: str = "AlphaZero", n_simulations: int = 200,
                 c_puct: float = 1.4, lr: float = 1e-3, weight_decay: float = 1e-4,
                 epochs: int = 3, batch_size: int = 256, use_symmetry: bool = True,
                 device: str = "cpu", rng: Optional[np.random.Generator] = None):
        super().__init__(name)
        self.n_simulations = n_simulations
        self.c_puct = c_puct
        self.epochs = epochs
        self.batch_size = batch_size
        self.use_symmetry = use_symmetry
        self.rng = rng if rng is not None else np.random
        self.device = torch.device(device)

        self.net = PolicyValueNet().to(self.device)
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=lr,
                                          weight_decay=weight_decay)
        self.train_buffer: List[Tuple[tuple, np.ndarray, float]] = []  # (s, π, z)

    # ------------------------------------------------------------------ #
    # 决策: PUCT 树搜索
    # ------------------------------------------------------------------ #
    def select_action(self, obs: np.ndarray, valid_mask: np.ndarray,
                      training: bool = True) -> int:
        state = tuple(int(x) for x in obs)
        valid = tuple(int(a) for a in np.flatnonzero(valid_mask))
        root = self._search(state, valid)
        if training:
            # 温度 τ=1: 按访问分布采样 (训练时鼓励探索)
            probs = root.N[list(valid)]
            probs = probs / probs.sum()
            return int(self.rng.choice(list(valid), p=probs))
        # 评估/对弈: 取访问次数最多的动作
        return int(max(valid, key=lambda a: root.N[a]))

    def _search(self, state: tuple, valid: tuple) -> AZNode:
        root = AZNode(state, valid)
        for _ in range(self.n_simulations):
            self._simulate(root)
        return root

    def _simulate(self, root: AZNode) -> None:
        node = root
        path = [root]

        # 1) 沿 PUCT 下降至叶子
        while True:
            result = state_result(node.state)
            if result is not None:
                value = float(result)
                break
            if node.expanded:
                node = self._puct_child(node)
                path.append(node)
                continue
            # 2) 叶子: 网络估值并展开 (叶节点估值替代随机 Rollout)
            p, v = self.net.predict(node.state, self.device)
            mask = np.zeros(9)
            mask[list(node.valid)] = 1.0
            node.P = p * mask
            node.P /= node.P.sum() + 1e-12   # 合法动作上重归一化
            node.expanded = True
            for a in node.valid:
                child_state, child_valid = self._apply(node.state, a)
                node.children[a] = AZNode(child_state, child_valid, action=a)
            value = float(v)
            break

        # 3) 回传: 自底向上, 每上一层翻转一次视角
        v = value
        for n in reversed(path):
            n.N_total += 1
            n.W += v
            v = -v
        # 边访问计数
        for parent, child in zip(path, path[1:]):
            parent.N[child.action] += 1

    def _puct_child(self, node: AZNode) -> AZNode:
        best_child, best_value = None, -math.inf
        sqrt_n = math.sqrt(node.N_total + 1)
        for a, child in node.children.items():
            # 子节点价值为对手视角, 取负翻回本节点玩家视角
            q = -child.W / (child.N_total + 1e-8)
            u = self.c_puct * node.P[a] * sqrt_n / (1.0 + child.N_total)
            if q + u > best_value:
                best_value, best_child = q + u, child
        return best_child

    # ------------------------------------------------------------------ #
    # 自博弈: 收集 (s, π, z) 训练样本
    # ------------------------------------------------------------------ #
    def self_play_episode(self) -> List[Tuple[tuple, np.ndarray, float]]:
        """一局完整自博弈, 返回 [(state, π, z)] —— z 为该状态玩家视角的终局值。"""
        env = TicTacToeEnv()
        states, pis = [], []
        obs = env.reset()
        while True:
            mask = env.get_valid_mask()
            state = tuple(int(x) for x in obs)
            valid = tuple(int(a) for a in np.flatnonzero(mask))
            root = self._search(state, valid)
            pi = root.N / root.N.sum()
            action = int(self.rng.choice(list(valid), p=pi[list(valid)] / pi[list(valid)].sum()))
            states.append(state)
            pis.append(pi)
            obs, r, done, _ = env.step(action)
            if done:
                break
        z = float(r)  # 最后落子方视角: 胜 +1 / 平 0 / 负 -1
        m = len(states) - 1
        return [(s, pis[i], z if (m - i) % 2 == 0 else -z)
                for i, s in enumerate(states)]

    # ------------------------------------------------------------------ #
    # 网络训练: loss = (z - v)^2 - πᵀ log p (+ L2 正则)
    # ------------------------------------------------------------------ #
    def update(self) -> dict:
        if len(self.train_buffer) < self.batch_size:
            return {}
        idx = self.rng.choice(len(self.train_buffer), size=self.batch_size, replace=False)
        states, pis, zs = [], [], []
        for i in idx:
            s, pi, z = self.train_buffer[i]
            if self.use_symmetry:  # 随机选一个对称变换做数据增强
                k = int(self.rng.integers(8))
                s, pi = transform_state(s, k), transform_pi(pi, k)
            states.append(s)
            pis.append(pi)
            zs.append(z)

        x = torch.tensor(np.array(states), dtype=torch.float32, device=self.device)
        pi_t = torch.tensor(np.array(pis), dtype=torch.float32, device=self.device)
        z_t = torch.tensor(zs, dtype=torch.float32, device=self.device)

        metrics = {"az_loss": 0.0, "az_value_loss": 0.0, "az_policy_loss": 0.0}
        for _ in range(self.epochs):
            logits, v = self.net(x)
            log_p = F.log_softmax(logits, dim=-1)
            policy_loss = -(pi_t * log_p).sum(dim=-1).mean()
            value_loss = F.mse_loss(v, z_t)
            loss = policy_loss + value_loss
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            metrics["az_loss"] += float(loss)
            metrics["az_value_loss"] += float(value_loss)
            metrics["az_policy_loss"] += float(policy_loss)
        # 周期性清空训练缓冲 (FIFO 概念: 保留最近的样本)
        if len(self.train_buffer) > self.batch_size * 50:
            del self.train_buffer[:-self.batch_size * 50]
        return {k: v / self.epochs for k, v in metrics.items()}

    # ------------------------------------------------------------------ #
    @staticmethod
    def _apply(state: tuple, action: int):
        """在 canonical 状态下落子 (+1), 返回对手视角的新状态与合法动作。"""
        b = list(state)
        b[action] = 1
        new_state = tuple(-x for x in b)
        valid = tuple(i for i, c in enumerate(new_state) if c == 0)
        return new_state, valid

    def save(self, path: str) -> None:
        torch.save({"net": self.net.state_dict()}, path)

    def load(self, path: str) -> None:
        self.net.load_state_dict(
            torch.load(path, map_location=self.device, weights_only=True)["net"])
