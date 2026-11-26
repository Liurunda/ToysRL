"""多算法混战与 Elo 天梯榜 (Arena)。

所有算法训练完成后, 互相进行全循环对弈 (Round-Robin), 以 Elo 评分
(初始 1500, K=32) 输出最终排行榜。
"""
from typing import Dict, List, Optional, Tuple

import numpy as np

from agents.base import BaseAgent
from evaluation.evaluator import play_game


class Arena:
    def __init__(self, K: float = 32.0, rng: Optional[np.random.Generator] = None):
        self.K = K
        self.rng = rng if rng is not None else np.random

    def round_robin(self, agents: Dict[str, BaseAgent], games_per_pair: int = 40,
                    verbose: bool = True) -> Tuple[List[Tuple[str, float]], Dict]:
        """全循环对弈并计算 Elo 排名。

        返回 (排名列表 [(名称, Elo)], 对战结果 {(a, b): (a胜, 平, a负)})。
        """
        names = list(agents)
        elo = {n: 1500.0 for n in names}
        results = {}
        for i in range(len(names)):
            for j in range(i + 1, len(names)):
                a, b = names[i], names[j]
                wins = draws = 0
                for g in range(games_per_pair):
                    side = 1 if g % 2 == 0 else -1  # 先手 / 后手各半
                    res = play_game(agents[a], agents[b], side_a=side, rng=self.rng)
                    if res == 1:
                        wins += 1
                        score_a, score_b = 1.0, 0.0
                    elif res == -1:
                        score_a, score_b = 0.0, 1.0
                    else:
                        draws += 1
                        score_a = score_b = 0.5
                    # 标准 Elo 更新
                    expected_a = 1.0 / (1.0 + 10 ** ((elo[b] - elo[a]) / 400.0))
                    elo[a] += self.K * (score_a - expected_a)
                    elo[b] += self.K * (score_b - (1.0 - expected_a))
                results[(a, b)] = (wins, draws, games_per_pair - wins - draws)
                if verbose:
                    print(f"  {a:>12} vs {b:<12}: {wins}胜 {draws}平 "
                          f"{games_per_pair - wins - draws}负")

        ranking = sorted(elo.items(), key=lambda kv: -kv[1])
        return ranking, results

    @staticmethod
    def print_ranking(ranking: List[Tuple[str, float]]) -> None:
        print("\n========== Elo 天梯榜 ==========")
        for rank, (name, rating) in enumerate(ranking, start=1):
            print(f"  {rank:>2}. {name:<14} Elo {rating:7.1f}")
        print("================================")
