"""三档难度固定的基准对手, 用于客观衡量各 RL 算法的棋力。

Level 1 - RandomAgent:     在合法动作中均匀随机采样。
Level 2 - HeuristicAgent:  单步启发式规则 (赢 > 堵 > 中心 > 角 > 随机)。
Level 3 - MinimaxAgent:    带记忆化的 Negamax 完美策略, 保证绝对不败。

Minimax 的核心评估意义: 井字棋先手最优与后手最优的结果都是平局,
因此任何 RL 算法面对 Minimax 的"100% 不败率 (胜+平)"即代表达到理论最优。
"""
from typing import Optional

import numpy as np

from agents.base import BaseAgent
from envs.tictactoe import check_winner, state_result


class RandomAgent(BaseAgent):
    """Level 1: 在合法动作中均匀随机采样。用于检验算法是否具备基础学习能力。"""

    def __init__(self, name: str = "Random", rng: Optional[np.random.Generator] = None):
        super().__init__(name)
        self.rng = rng if rng is not None else np.random

    def select_action(self, obs, valid_mask, training=True) -> int:
        valid = np.flatnonzero(valid_mask)
        return int(self.rng.choice(valid))


class HeuristicAgent(BaseAgent):
    """Level 2: 单步启发式规则。

    优先级: 有连三直接赢 → 对手下步连三则堵截 → 占中心 → 占角 → 随机。
    用于区分"只会抓随机对手漏洞"与"真正学会基础防守"的算法。
    """

    def __init__(self, name: str = "Heuristic", rng: Optional[np.random.Generator] = None):
        super().__init__(name)
        self.rng = rng if rng is not None else np.random

    def select_action(self, obs, valid_mask, training=True) -> int:
        board = np.asarray(obs, dtype=int)  # canonical: +1 己方, -1 对手
        valid = np.flatnonzero(valid_mask).tolist()

        # 1. 有连三直接赢
        for a in valid:
            b = board.copy()
            b[a] = 1
            if check_winner(b) == 1:
                return a
        # 2. 对手下步能连三则堵截 (模拟对手落子)
        for a in valid:
            b = board.copy()
            b[a] = -1
            if check_winner(b) == -1:
                return a
        # 3. 占中心
        if 4 in valid:
            return 4
        # 4. 占角
        corners = [c for c in (0, 2, 6, 8) if c in valid]
        if corners:
            return int(self.rng.choice(corners))
        # 5. 其余随机
        return int(self.rng.choice(valid))


class MinimaxAgent(BaseAgent):
    """Level 3: 带记忆化的 Negamax 完美策略, 保证绝对不败 (最优对最优必平)。

    因为 canonical 状态本身已编码"轮到谁走", 记忆化键仅需状态元组本身。
    终局值的判断统一复用 envs.tictactoe.state_result (与 MCTS/AZ 一致)。
    平局分支 (价值相同的动作) 按 rng 随机选择, 使对局具备一定多样性。
    """

    def __init__(self, name: str = "Minimax",
                 rng: Optional[np.random.Generator] = None):
        super().__init__(name)
        self.rng = rng if rng is not None else np.random
        self._memo = {}  # state_tuple -> 轮到走的一方的最优值

    # ------------------------------------------------------------------ #
    def select_action(self, obs, valid_mask, training=True) -> int:
        state = tuple(int(x) for x in obs)
        best_value = -1.0
        best_actions = []
        for a in np.flatnonzero(valid_mask):
            child = self._next_state(state, int(a))
            v = -self._negamax(child)  # 对手最优视角翻转为己方视角
            if v > best_value:
                best_value = v
                best_actions = [int(a)]
            elif v == best_value:
                best_actions.append(int(a))
        return int(self.rng.choice(best_actions))

    def _next_state(self, state, action: int) -> tuple:
        """己方在 action 落子 (+1), 返回对手视角的 canonical 状态。"""
        b = list(state)
        b[action] = 1
        return tuple(-x for x in b)

    def _negamax(self, state: tuple) -> int:
        """state 为轮到走的一方的 canonical 视角; 返回该方的最优终局值 (+1/0/-1)。"""
        if state in self._memo:
            return self._memo[state]
        result = state_result(state)
        if result is not None:
            return result
        best = -1
        for a, cell in enumerate(state):
            if cell == 0:
                v = -self._negamax(self._next_state(state, a))
                if v > best:
                    best = v
                    if best == 1:  # 已找到必胜着法, 剪枝
                        break
        self._memo[state] = best
        return best
