"""人机交互脚本: 亲自下场测试训练好的 AI。

用法:
  python play_human.py --agent qlearning --checkpoint results/qlearning_fixed
  python play_human.py --agent mcts                                # 零训练直接对弈
  python play_human.py --agent minimax                             # 挑战完美对手
  python play_human.py --agent alphazero --checkpoint results/alphazero_selfplay
  python play_human.py --agent qlearning --checkpoint ... --side o # 人类执后手

输入 1-9 对应棋盘位置:
  1 | 2 | 3
  --+---+--
  4 | 5 | 6
  --+---+--
  7 | 8 | 9
"""
import argparse
import os
import sys

import numpy as np
import yaml

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from agents.base import BaseAgent  # noqa: E402
from envs.tictactoe import TicTacToeEnv  # noqa: E402
from run_experiment import build_agent, load_checkpoint  # noqa: E402


def render_board(env: TicTacToeEnv, human_player: int) -> None:
    """打印棋盘: 已落子显示 X/O, 空位显示输入编号 (人类视角)。"""
    symbols = {1: "X", -1: "O", 0: None}
    print("\n  当前棋盘 (人类执 %s):" % ("X" if human_player == 1 else "O"))
    for r in range(3):
        cells = []
        for c in range(3):
            idx = r * 3 + c
            cell = env.board[idx]
            cells.append(symbols[int(cell)] if cell != 0 else str(idx + 1))
        print("   " + " | ".join(cells))
        if r < 2:
            print("  ---+---+---")
    print()


def play(agent: BaseAgent, human_side: int, rng) -> None:
    env = TicTacToeEnv()
    env.reset()
    while True:
        render_board(env, human_side)
        if env.current_player == human_side:
            while True:
                try:
                    move = int(input("你的回合, 输入格子编号 (1-9): ")) - 1
                except (ValueError, EOFError):
                    print("再见!")
                    return
                if not 0 <= move < 9:
                    print("编号应在 1-9 之间, 重新输入。")
                    continue
                if env.get_valid_mask()[move] == 0:
                    print("该格已有棋子, 重新输入。")
                    continue
                break
        else:
            move = agent.select_action(env.get_canonical_board(),
                                       env.get_valid_mask(), training=False)
            print(f"{agent.name} 落子: {move + 1}\n")
        _, _, done, info = env.step(move)
        if done:
            render_board(env, human_side)
            winner = info["winner"]
            if winner == 0:
                print("平局! 🤝")
            else:
                you_win = (winner == human_side)
                print("你赢了! 🎉" if you_win else f"你输了, {agent.name} 获胜。")
            return


def main():
    parser = argparse.ArgumentParser(description="人机对弈")
    parser.add_argument("--agent", type=str, required=True,
                        help="算法名: qlearning/sarsa/mc/dqn/ppo/mcts/alphazero/random/heuristic/minimax")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="已训练 run 目录 (results/<name>_<mode>), 内部含 meta.json 与 checkpoint")
    parser.add_argument("--side", type=str, default="x", choices=["x", "o"],
                        help="人类执 X(先手) 或 O(后手)")
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()

    with open(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           "configs", "default.yaml")) as f:
        cfg = yaml.safe_load(f)
    rng = np.random.default_rng(args.seed if args.seed is not None else cfg["seed"])

    if args.checkpoint:
        agent = load_checkpoint(args.checkpoint, rng)
        print(f"已加载 checkpoint: {args.checkpoint} ({agent.name})")
    else:
        agent = build_agent(args.agent, cfg, rng)

    human_side = 1 if args.side == "x" else -1
    play(agent, human_side, rng)


if __name__ == "__main__":
    main()
