"""3×3 井字棋核心双人轮流环境。

棋盘编码
--------
底层棋盘为长度 9 的整型数组 (行优先, 格子索引 0~8):
    X (先手) = 1, O (后手) = -1, 空位 = 0

视角归一化 (Canonical Observation)
---------------------------------
每次向"当前轮到的玩家"返回状态时, 统一乘以 current_player:
    智能体眼中 +1 永远代表己方棋子, -1 永远代表对手棋子, 0 代表空位。
    同一套策略网络 / Q 表即可无缝执先手或执后手。

奖励机制 (稀疏终局奖励)
-----------------------
己方落子获胜 +1.0 (对手视角记为 -1.0), 平局 0.0, 中间步 0.0。
兜底: 若未开启掩码防护且强行落子于已占用格, 直接判负 -10.0 并结束对局
(正常训练路径由 Action Mask 杜绝非法动作)。
"""
from typing import Tuple

import numpy as np

# 所有三连组合: 3 行 + 3 列 + 2 对角线
WIN_LINES = (
    (0, 1, 2), (3, 4, 5), (6, 7, 8),   # 行
    (0, 3, 6), (1, 4, 7), (2, 5, 8),   # 列
    (0, 4, 8), (2, 4, 6),              # 对角线
)


def check_winner(board: np.ndarray) -> int:
    """检查棋盘上是否有三连。返回 1 (X 胜) / -1 (O 胜) / 0 (未分胜负)。"""
    for a, b, c in WIN_LINES:
        s = int(board[a]) + int(board[b]) + int(board[c])
        if s == 3:
            return 1
        if s == -3:
            return -1
    return 0


def state_result(state) -> int:
    """判断一个 canonical 状态 (tuple, 轮到走的一方视角) 的终局结果。

    返回 1 (轮到走的一方胜) / 0 (平局) / -1 (轮到走的一方负) / None (未终局)。
    注意: 正常对局中终局永远出现在"对手刚连三"之后, 因此终局 canonical 状态
    里三连的棋子是 -1 (对方), 轮到走的一方已输。
    该函数被 Minimax / MCTS / AlphaZero 共用, 保证语义一致。
    """
    winner = check_winner(np.asarray(state, dtype=int))
    if winner == 1:
        return 1
    if winner == -1:
        return -1
    if 0 not in state:
        return 0
    return None


class TicTacToeEnv:
    """双人轮流 3×3 井字棋环境。

    约定: step(action) 的返回值中,
      - obs:    下一轮应落子玩家的 canonical 观测 (棋盘 × 该玩家标记);
      - reward: 刚刚落子一方视角的奖励 (获胜 +1, 平局 0, 非法 -10)。
    要获得"己方视角"的奖励, 由上层包装器根据落子方身份做符号翻转。
    """

    def __init__(self):
        self.board = None
        self.current_player = None
        self.reset()

    # ------------------------------------------------------------------ #
    # 核心接口
    # ------------------------------------------------------------------ #
    def reset(self) -> np.ndarray:
        self.board = np.zeros(9, dtype=np.int8)
        self.current_player = 1  # X 先手
        return self.get_canonical_board()

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, dict]:
        action = int(action)  # 兼容 np 标量 / one-hot 索引
        mover = self.current_player

        if self.board[action] != 0:
            # 非法动作兜底: 直接判负并结束对局
            self.current_player = -mover
            info = {"winner": -mover, "illegal": True}
            return self.get_canonical_board(), -10.0, True, info

        self.board[action] = mover
        winner = check_winner(self.board)
        self.current_player = -mover

        if winner != 0:
            return self.get_canonical_board(), 1.0, True, {"winner": winner, "illegal": False}
        if np.all(self.board != 0):
            return self.get_canonical_board(), 0.0, True, {"winner": 0, "illegal": False}
        return self.get_canonical_board(), 0.0, False, {"winner": 0, "illegal": False}

    # ------------------------------------------------------------------ #
    # 观测 / 掩码 / 克隆 / 渲染
    # ------------------------------------------------------------------ #
    def get_canonical_board(self) -> np.ndarray:
        """当前轮到玩家的 canonical 观测: 己方=+1, 对手=-1, 空=0。"""
        return (self.board * self.current_player).astype(np.int8)

    def get_valid_mask(self) -> np.ndarray:
        """合法动作掩码: 1 表示可落子的空位, 0 表示已占用。"""
        return (self.board == 0).astype(np.int8)

    def clone(self) -> "TicTacToeEnv":
        """深拷贝环境状态 (供 MCTS 模拟推演使用, 不共享任何可变状态)。"""
        env = TicTacToeEnv.__new__(TicTacToeEnv)
        env.board = self.board.copy()
        env.current_player = self.current_player
        return env

    def render(self) -> None:
        """以文本形式打印棋盘。"""
        symbols = {1: "X", -1: "O", 0: "."}
        rows = []
        for r in range(3):
            rows.append(" " + " | ".join(symbols[int(self.board[r * 3 + c])] for c in range(3)))
        print("\n".join(rows))
        print(f"轮到: {'X' if self.current_player == 1 else 'O'} (canonical 视角 +1 为当前玩家)")
