"""智能体统一抽象接口。

所有算法 (表格型 / 深度 / 搜索型) 以及基准对手都实现 BaseAgent 接口,
从而可以在相同的训练循环、评估协议与天梯赛中被无差别调用。

接口约定
--------
select_action(obs, valid_mask, training):
    obs        为"轮到该智能体走"的 canonical 观测 (+1 己方 / -1 对手 / 0 空);
    valid_mask 为长度 9 的 0/1 数组, 1 表示合法空位。实现方必须保证
    返回的动作在 valid_mask 之内 (即 Action Masking);
    training=True 时允许探索 (ε-greedy / 采样), False 时贪婪 / 确定。

observe(transition):  存入一条经验 (DQN 进 Replay Buffer, 表格型就地更新)。
update():             执行参数更新, 返回 loss 等指标 dict (无可更新时返回 {})。
end_episode():        回合结束钩子 —— SARSA 补最后一拍 TD、MC 计算整回合
                      回报、ε 按回合衰减等。
"""
import pickle
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass
class Transition:
    """一次完整单步转移 (全部字段均为智能体自身视角)。"""
    obs: np.ndarray            # 决策点观测 s
    action: int                # 己方选择的动作 a
    reward: float              # 己方视角即时奖励 r
    next_obs: np.ndarray       # 下一决策点观测 s' (若 done 则无意义)
    done: bool                 # 对局是否已结束
    valid_mask: np.ndarray     # s 处的合法动作掩码
    next_valid_mask: np.ndarray  # s' 处的合法动作掩码 (若 done 则无意义)


class BaseAgent(ABC):
    def __init__(self, name: Optional[str] = None):
        self.name = name or type(self).__name__

    # ------------------------------------------------------------------ #
    # 必须实现的接口
    # ------------------------------------------------------------------ #
    @abstractmethod
    def select_action(self, obs: np.ndarray, valid_mask: np.ndarray,
                      training: bool = True) -> int:
        """在合法动作中选择一个动作返回。"""

    # ------------------------------------------------------------------ #
    # 可选钩子 (默认空实现)
    # ------------------------------------------------------------------ #
    def observe(self, transition: Transition) -> None:
        """存入经验或轨迹。"""

    def update(self) -> dict:
        """执行参数 / Q 表更新, 返回 loss 等指标。"""
        return {}

    def end_episode(self) -> None:
        """回合结束钩子 (SARSA 滞后更新 / MC 回报计算 / ε 衰减)。"""

    # ------------------------------------------------------------------ #
    # 保存 / 加载 (默认 pickle, 深度算法各自覆写)
    # ------------------------------------------------------------------ #
    def save(self, path: str) -> None:
        with open(path, "wb") as f:
            pickle.dump(self.__dict__, f)

    def load(self, path: str) -> None:
        with open(path, "rb") as f:
            self.__dict__.update(pickle.load(f))

    def __repr__(self) -> str:
        return f"<{self.name}>"
