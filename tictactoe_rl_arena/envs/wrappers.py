"""单智能体包装器: 把双人轮流环境封装为标准单智能体 MDP。

每次 step(action) 包含一次完整转移:
    己方落子 → 若未结束, 对手自动落子
返回 (obs', reward, done, info), 其中:
    obs'   为下一决策点的 canonical 观测 (己方视角);
    reward 始终从己方视角给出 (己胜 +1 / 平局 0 / 己负 -1 / 己方非法 -10)。

支持每局随机分配己方执先手 (X) 或执后手 (O);
若执后手, reset() 时由对手先走一步再返回初始观测 s0。
"""
from typing import Optional, Tuple

import numpy as np

from envs.tictactoe import TicTacToeEnv


class SingleAgentWrapper:
    def __init__(self, env: TicTacToeEnv, opponent_agent,
                 randomize_side: bool = True, fixed_side: Optional[int] = None,
                 rng: Optional[np.random.Generator] = None):
        self.env = env
        self.opponent = opponent_agent
        self.randomize_side = randomize_side
        self.fixed_side = fixed_side  # 1=执先手, -1=执后手; 仅 randomize_side=False 时生效
        self.rng = rng if rng is not None else np.random
        self.agent_player = 1  # 1=执先手 X, -1=执后手 O

    # ------------------------------------------------------------------ #
    # Gym 风格接口
    # ------------------------------------------------------------------ #
    def reset(self) -> Tuple:
        if self.randomize_side:
            self.agent_player = 1 if self.rng.random() < 0.5 else -1
        else:
            self.agent_player = self.fixed_side if self.fixed_side is not None else 1
        self.env.reset()
        if self.agent_player == -1:
            # 己方执后手: 对手 (先手) 先走一步
            self._opponent_step()
        return self.get_obs(), self.get_valid_mask()

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, dict]:
        # 1) 己方落子
        obs, r, done, info = self.env.step(action)
        if not done:
            # 2) 对手自动落子, 其奖励需从对手视角翻转为己方视角
            obs, r_opp, done, info = self._opponent_step()
            reward = -r_opp
        else:
            reward = r  # 己方获胜 +1 / 平局 0 / 己方非法 -10
        info = dict(info)
        info["agent_player"] = self.agent_player
        info["agent_reward"] = reward
        return obs, reward, done, info

    # ------------------------------------------------------------------ #
    # 便捷访问
    # ------------------------------------------------------------------ #
    def _opponent_step(self):
        action = self.opponent.select_action(
            self.env.get_canonical_board(), self.env.get_valid_mask(), training=False)
        return self.env.step(action)

    def get_obs(self) -> np.ndarray:
        """己方视角的 canonical 观测。"""
        return self.env.get_canonical_board()

    def get_valid_mask(self) -> np.ndarray:
        """当前决策点的合法动作掩码 (视角无关, 掩码只取决于占用情况)。"""
        return self.env.get_valid_mask()
