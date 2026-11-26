"""标准化评估协议 (Evaluator)。

每训练一定步数 (默认 500 局), 冻结智能体 (training=False, 关闭探索),
执行固定评估:
    对阵 Random 100 局 (50 先手 + 50 后手)
    对阵 Heuristic 100 局 (50 先手 + 50 后手)
    对阵 Minimax 50 局 (25 先手 + 25 后手; Minimax 确定性较强,
                       平局分支随机选择保证对局多样性)

关键指标: 对阵 Minimax 的不败率 (win + draw) ≥ 95% 即认为达到理论最优。
"""
from typing import Dict, Optional

import numpy as np

from agents.base import BaseAgent
from agents.baselines import HeuristicAgent, MinimaxAgent, RandomAgent
from envs.tictactoe import TicTacToeEnv


def play_game(agent_a: BaseAgent, agent_b: BaseAgent,
              side_a: int = 1, rng: Optional[np.random.Generator] = None) -> int:
    """两个智能体在裸环境下对弈一局。

    side_a: agent_a 执 X(1) 或 O(-1)。
    返回 agent_a 视角的结果: 1 胜 / 0 平 / -1 负。
    双方均以 training=False (贪婪/确定策略) 行动。
    """
    env = TicTacToeEnv()
    while True:
        player = env.current_player
        agent = agent_a if player == side_a else agent_b
        action = agent.select_action(
            env.get_canonical_board(), env.get_valid_mask(), training=False)
        _, _, done, info = env.step(action)
        if done:
            winner = info["winner"]
            if winner == 0:
                return 0
            return 1 if winner == side_a else -1


class Evaluator:
    def __init__(self, games_vs_random: int = 100, games_vs_heuristic: int = 100,
                 games_vs_minimax: int = 50, rng: Optional[np.random.Generator] = None):
        self.rng = rng if rng is not None else np.random
        self.games = {
            "random": games_vs_random,
            "heuristic": games_vs_heuristic,
            "minimax": games_vs_minimax,
        }
        # 每个待评估智能体使用独立的一组基准对手实例 (避免共享 Minimax 记忆化缓存
        # 跨智能体串扰 —— 实际为共享加速也无妨, 此处保持语义清晰)
        self.baselines = {
            "random": RandomAgent(rng=self.rng),
            "heuristic": HeuristicAgent(rng=self.rng),
            "minimax": MinimaxAgent(rng=self.rng),
        }

    # ------------------------------------------------------------------ #
    def evaluate(self, agent: BaseAgent, episode: int) -> Dict[str, float]:
        """运行一轮标准化评估, 返回指标 dict (episode + 各对手胜/平/负率)。"""
        out = {"episode": episode}
        for key, opponent in self.baselines.items():
            n = self.games[key]
            wins = draws = 0
            for g in range(n):
                side = 1 if g % 2 == 0 else -1  # 先手 / 后手各半
                res = play_game(agent, opponent, side_a=side, rng=self.rng)
                wins += (res == 1)
                draws += (res == 0)
            out[f"win_vs_{key}"] = wins / n
            out[f"draw_vs_{key}"] = draws / n
            if key == "minimax":
                out["notlose_vs_minimax"] = (wins + draws) / n
        return out

    def format(self, metrics: Dict[str, float]) -> str:
        return (
            f"ep {metrics['episode']:>7} | "
            f"vsRandom win {metrics['win_vs_random']:5.2f} | "
            f"vsHeuristic win {metrics['win_vs_heuristic']:5.2f} | "
            f"vsMinimax win {metrics['win_vs_minimax']:5.2f} "
            f"notlose {metrics['notlose_vs_minimax']:5.2f}"
        )
