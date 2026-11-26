"""MCTS: 纯 UCT 蒙特卡洛树搜索 (基于搜索与规划的算法)。

树节点记录 (N, W, Q): 访问次数、累计价值、平均价值 (均为该节点"轮到走
的一方"的视角)。选择用 UCT 公式:
    a* = argmax_a [ -Q(child) + c * sqrt( ln N(parent) / N(child) ) ]
(Q 前的负号来自视角翻转: 子节点价值是对手视角。)

默认策略: 随机 Rollout 模拟至终局, 回溯真实对局结果。

演示重点: 零训练 —— 不更新任何参数, 仅靠在线推演即可达到接近完美棋力,
用于对比"搜索"与"学习"的算力权衡。
"""
import math
from typing import Optional

import numpy as np

from agents.base import BaseAgent
from envs.tictactoe import state_result


class MCTSNode:
    """树节点: state 为 canonical 元组 (轮到走的一方视角)。"""

    __slots__ = ("state", "valid", "N", "W", "Q", "children")

    def __init__(self, state: tuple, valid: tuple):
        self.state = state
        self.valid = valid          # 合法动作列表
        self.N = 0                  # 访问次数
        self.W = 0.0                # 累计价值 (本节点玩家视角)
        self.Q = 0.0                # 平均价值 W / N
        self.children = {}          # action -> MCTSNode


class MCTSAgent(BaseAgent):
    def __init__(self, name: str = "MCTS", n_simulations: int = 300,
                 c_uct: float = 1.414, rng: Optional[np.random.Generator] = None):
        super().__init__(name)
        self.n_simulations = n_simulations
        self.c_uct = c_uct
        self.rng = rng if rng is not None else np.random

    # ------------------------------------------------------------------ #
    def select_action(self, obs: np.ndarray, valid_mask: np.ndarray,
                      training: bool = True) -> int:
        state = tuple(int(x) for x in obs)
        valid = tuple(int(a) for a in np.flatnonzero(valid_mask))
        root = MCTSNode(state, valid)

        for _ in range(self.n_simulations):
            self._simulate(root)

        # 选择访问次数最多的动作 (稳健子节点)
        counts = {a: root.children[a].N for a in valid}
        return int(max(counts, key=counts.get))

    # ------------------------------------------------------------------ #
    def _simulate(self, root: MCTSNode) -> None:
        node = root
        path = [root]

        # 1) Selection: 沿 UCT 下降直到未完全展开的节点
        while node.children and len(node.children) == len(node.valid):
            node = self._uct_child(node)
            path.append(node)

        # 2) Expansion: 若未终局, 展开一个随机未扩展动作
        result = state_result(node.state)
        if result is not None:
            value = result
        else:
            unexpanded = [a for a in node.valid if a not in node.children]
            action = int(self.rng.choice(unexpanded))
            child_state, child_valid = self._apply(node.state, action)
            child = MCTSNode(child_state, child_valid)
            node.children[action] = child
            path.append(child)
            # 3) Rollout: 从新节点随机模拟至终局
            value = self._rollout(child_state)

        # 4) Backprop: 自底向上回传, 每上一层翻转一次视角
        v = float(value)
        for n in reversed(path):
            n.N += 1
            n.W += v
            n.Q = n.W / n.N
            v = -v

    def _uct_child(self, node: MCTSNode) -> MCTSNode:
        """UCT: 子节点价值为对手视角, 取负翻转回本节点视角。"""
        best_child, best_value = None, -math.inf
        log_n = math.log(node.N + 1)
        for child in node.children.values():
            v = -child.Q + self.c_uct * math.sqrt(log_n / (child.N + 1e-8))
            if v > best_value:
                best_value, best_child = v, child
        return best_child

    def _rollout(self, state: tuple) -> int:
        """随机 Rollout 至终局, 返回该状态"轮到走的一方"的终局值。

        注意视角跟踪: 每落一子视角翻转一次, 终局检测时的 state 可能已是
        对方视角, 返回值需要按翻转次数还原到 rollout 起始玩家的视角。
        """
        player = 1  # +1: 当前 state 仍是起始玩家视角; -1: 已翻转为对手视角
        while True:
            result = state_result(state)
            if result is not None:
                return result * player
            empty = [i for i, c in enumerate(state) if c == 0]
            action = int(self.rng.choice(empty))
            state, _ = self._apply(state, action)
            player = -player

    @staticmethod
    def _apply(state: tuple, action: int):
        """在 canonical 状态下落子 (+1), 返回对手视角的新状态。"""
        b = list(state)
        b[action] = 1
        new_state = tuple(-x for x in b)
        valid = tuple(i for i, c in enumerate(new_state) if c == 0)
        return new_state, valid
