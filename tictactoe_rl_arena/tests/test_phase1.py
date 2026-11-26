"""Phase 1 验收测试: 核心环境 + 基准对手。

验证目标 (对应设计文档 Phase 1):
  1. Minimax vs Minimax 100% 平局;
  2. Heuristic vs Random 胜率 > 90%;
  3. 环境规则 (胜负判断 / canonical 视角 / 非法动作兜底 / clone) 零 Bug;
  4. SingleAgentWrapper 的先/后手与奖励视角正确。
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from agents.baselines import HeuristicAgent, MinimaxAgent, RandomAgent  # noqa: E402
from envs.tictactoe import TicTacToeEnv, check_winner, state_result  # noqa: E402
from envs.wrappers import SingleAgentWrapper  # noqa: E402


def play_game(agent_a, agent_b, side_a=1, rng=None):
    """agent_a 执 side_a (1=先手X, -1=后手O), 返回 agent_a 视角结果 (1胜/0平/-1负)。"""
    env = TicTacToeEnv()
    env.reset()
    while True:
        player = env.current_player
        agent = agent_a if player == side_a else agent_b
        action = agent.select_action(env.get_canonical_board(), env.get_valid_mask(), training=False)
        _, _, done, info = env.step(action)
        if done:
            winner = info["winner"]
            if winner == 0:
                return 0
            return 1 if winner == side_a else -1


def test_minimax_vs_minimax_all_draws(rng):
    a, b = MinimaxAgent(rng=rng), MinimaxAgent(rng=rng)
    draws = 0
    for g in range(100):
        if play_game(a, b, side_a=1 if g % 2 == 0 else -1, rng=rng) == 0:
            draws += 1
    print(f"[1] Minimax vs Minimax: 100 局中平局 {draws} 局")
    assert draws == 100, "完美策略对弈必须 100% 平局!"


def test_heuristic_beats_random(rng):
    h, r = HeuristicAgent(rng=rng), RandomAgent(rng=rng)
    wins = sum(1 for g in range(200) if play_game(h, r, side_a=1 if g % 2 == 0 else -1, rng=rng) == 1)
    rate = wins / 200
    print(f"[2] Heuristic vs Random: 胜率 {rate:.2%} (200 局)")
    assert rate > 0.9, "Heuristic 对 Random 的胜率应 > 90%!"


def test_env_rules():
    env = TicTacToeEnv()
    # canonical 视角: X 视角 obs == 原始棋盘
    obs = env.reset()
    assert np.array_equal(obs, np.zeros(9)), "初始 X 视角应为全 0"

    # X 下 4 (中心) 后, O 的 canonical 视角中中心应为 -1
    _, r, done, _ = env.step(4)
    assert r == 0.0 and not done
    assert env.current_player == -1
    assert env.get_canonical_board()[4] == -1, "对手视角应看到己方棋子为 -1"
    assert np.array_equal(env.get_valid_mask(), np.array([1, 1, 1, 1, 0, 1, 1, 1, 1]))

    # clone 独立性与对称对局不变量: clone 上模拟落子不影响原环境
    clone = env.clone()
    clone.step(0)
    assert clone.board[0] == -1 and env.board[0] == 0, "clone 不应共享棋盘内存"

    # 非法动作兜底: 直接判负 -10 并结束
    _, r, done, info = env.step(4)
    assert r == -10.0 and done and info["illegal"]

    # X 胜: 0,1,2 连线
    env = TicTacToeEnv()
    for a in (0, 3, 1, 4, 2):  # X: 0,1,2 / O: 3,4
        _, r, done, info = env.step(a)
    assert done and info["winner"] == 1 and r == 1.0
    assert check_winner(env.board) == 1
    # 终局 canonical 视角: 败方视角看到胜方三连为 -1 → state_result 应为 -1
    assert state_result(tuple(env.get_canonical_board())) == -1

    # 平局局面: 仅剩一格空且无人连三
    board = np.array([1, -1, 1, -1, -1, 1, 1, 1, 0])
    env.board, env.current_player = board, -1
    _, r, done, info = env.step(8)
    assert done and info["winner"] == 0 and r == 0.0
    print("[3] 环境规则 / canonical 视角 / clone / 非法动作兜底: 全部通过")


def test_wrapper_sides_and_rewards(rng):
    random = RandomAgent(rng=rng)

    # 执后手: reset 后应由对手 (X) 先走一步
    wrapper = SingleAgentWrapper(TicTacToeEnv(), random, randomize_side=False, fixed_side=-1, rng=rng)
    obs, mask = wrapper.reset()
    assert np.count_nonzero(obs) == 1, "己方执后手时, reset 后对手应先落一子"

    # 己方视角奖励: 手动构造一个己方下一步获胜的局面
    # 原始棋盘: X=1 在 (0,1), O=-1 在 (3,4), 轮到 X
    env = TicTacToeEnv()
    env.board = np.array([1, 1, 0, -1, -1, 0, 0, 0, 0], dtype=np.int8)
    env.current_player = 1
    wrapper = SingleAgentWrapper(env, random, randomize_side=False, rng=rng)
    wrapper.agent_player = 1
    obs, reward, done, info = wrapper.step(2)  # X 连三获胜
    assert done and reward == 1.0 and info["agent_reward"] == 1.0

    # 己方输棋视角: X 在 (3,4) 已两连, 己方 O 乱走一步后 X 落 5 获胜 → 己方 -1
    env = TicTacToeEnv()
    env.board = np.array([1, 1, 0, -1, -1, 0, 0, 0, 0], dtype=np.int8)
    env.current_player = -1
    wrapper = SingleAgentWrapper(env, random, randomize_side=False, rng=rng)
    wrapper.agent_player = -1
    obs, reward, done, info = wrapper.step(8)  # O 走角, X 必走 2 获胜
    assert done and reward == -1.0 and info["agent_reward"] == -1.0
    print("[4] SingleAgentWrapper 先/后手轮换与奖励视角: 全部通过")


if __name__ == "__main__":
    rng = np.random.default_rng(42)
    test_minimax_vs_minimax_all_draws(rng)
    test_heuristic_beats_random(rng)
    test_env_rules()
    test_wrapper_sides_and_rewards(rng)
    print("\nPhase 1 验收全部通过 ✔")
