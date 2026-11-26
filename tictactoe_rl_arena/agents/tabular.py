"""经典表格型强化学习算法: Monte Carlo / Q-Learning / SARSA。

三者共享一张 Q 表: dict[state_tuple, np.array(9)], 状态键为 canonical
观测的不可变元组 (如 (1,0,-1,...)), 探索统一为仅在 valid_mask==1 的
动作中采样的 ε-greedy。

算法差异 (演示重点):
  - Monte Carlo: 回合结束后用真实回报 G_t 更新 (First-visit), 高方差、无偏差;
  - Q-Learning:  Off-policy 单步 TD, 目标值仅在下一状态合法动作上取 max, 收敛最快;
  - SARSA:       On-policy, 目标值使用下一状态实际选择的动作 Q(s', a'),
                 训练中期 ε 较大时策略偏保守。
"""
from collections import defaultdict
from typing import Optional

import numpy as np

from agents.base import BaseAgent, Transition


class TabularAgent(BaseAgent):
    """表格型算法基类: 共享 Q 表、ε-greedy 探索与状态键编码。"""

    def __init__(self, name: str, lr: float = 0.2, gamma: float = 0.9,
                 epsilon_start: float = 1.0, epsilon_end: float = 0.05,
                 epsilon_decay: float = 0.999, rng: Optional[np.random.Generator] = None):
        super().__init__(name)
        self.lr = lr
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay  # 每回合乘法衰减
        self.rng = rng if rng is not None else np.random
        self.Q = defaultdict(lambda: np.zeros(9, dtype=float))  # Q 表

    # ------------------------------------------------------------------ #
    @staticmethod
    def _key(obs: np.ndarray) -> tuple:
        """长度为 9 的观测向量 → 不可变元组 (字典键)。"""
        return tuple(int(x) for x in obs)

    def select_action(self, obs: np.ndarray, valid_mask: np.ndarray,
                      training: bool = True) -> int:
        eps = self.epsilon if training else 0.0
        valid = np.flatnonzero(valid_mask)
        if self.rng.random() < eps:
            return int(self.rng.choice(valid))
        q = self.Q[self._key(obs)]
        masked = np.where(valid_mask, q, -np.inf)  # Action Masking
        return int(np.argmax(masked))

    def _decay_epsilon(self) -> None:
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)

    # ------------------------------------------------------------------ #
    def save(self, path: str) -> None:
        import pickle
        with open(path, "wb") as f:
            pickle.dump({"Q": dict(self.Q), "epsilon": self.epsilon}, f)

    def load(self, path: str) -> None:
        import pickle
        with open(path, "rb") as f:
            data = pickle.load(f)
        self.Q.update(data["Q"])
        self.epsilon = data["epsilon"]


class MonteCarloAgent(TabularAgent):
    """First-visit Monte Carlo: 回合结束后用真实回报 G_t 更新 Q。

    演示重点: 高方差、无偏差 —— 只有在整个回合结束后才能学习。
    """

    def __init__(self, **kwargs):
        super().__init__(name=kwargs.pop("name", "MC"), **kwargs)
        self._trajectory = []

    def observe(self, transition: Transition) -> None:
        self._trajectory.append(transition)

    def end_episode(self) -> None:
        G = 0.0
        visited = set()
        for t in reversed(self._trajectory):  # 从后往前累加真实回报
            G = t.reward + self.gamma * G
            key = self._key(t.obs)
            if key not in visited:  # First-visit
                visited.add(key)
                self.Q[key][t.action] += self.lr * (G - self.Q[key][t.action])
        self._trajectory.clear()
        self._decay_epsilon()


class QLearningAgent(TabularAgent):
    """Off-policy 单步 TD: 目标值仅在下一状态的合法动作上取 max。

    演示重点: 收敛速度快于 MC, 是表格型方法的"标配"。
    """

    def __init__(self, **kwargs):
        super().__init__(name=kwargs.pop("name", "QLearning"), **kwargs)

    def observe(self, transition: Transition) -> None:
        key = self._key(transition.obs)
        if transition.done:
            target = transition.reward
        else:
            next_q = self.Q[self._key(transition.next_obs)]
            next_q_masked = np.where(transition.next_valid_mask, next_q, -np.inf)
            target = transition.reward + self.gamma * float(np.max(next_q_masked))
        self.Q[key][transition.action] += self.lr * (target - self.Q[key][transition.action])

    def end_episode(self) -> None:
        self._decay_epsilon()


class SarsaAgent(TabularAgent):
    """On-policy TD: 目标值使用下一状态实际选择的动作 Q(s', a')。

    实现说明: 单步 TD 需要知道 a', 而 a' 要等下一个转移到来时才知道,
    因此更新滞后一步, end_episode 时补上终局转移的最后一拍更新。
    """

    def __init__(self, **kwargs):
        super().__init__(name=kwargs.pop("name", "SARSA"), **kwargs)
        self._pending = None  # 等待配对 a' 的上一步转移

    def observe(self, transition: Transition) -> None:
        if self._pending is not None:
            p = self._pending
            if p.done:
                target = p.reward
            else:
                # transition.action 正是 p 的下一状态实际选择的动作 a'
                target = p.reward + self.gamma * self.Q[self._key(p.next_obs)][transition.action]
            key = self._key(p.obs)
            self.Q[key][p.action] += self.lr * (target - self.Q[key][p.action])
        self._pending = transition

    def end_episode(self) -> None:
        if self._pending is not None:
            p = self._pending
            target = p.reward  # 终局转移: 无后续状态
            key = self._key(p.obs)
            self.Q[key][p.action] += self.lr * (target - self.Q[key][p.action])
            self._pending = None
        self._decay_epsilon()
