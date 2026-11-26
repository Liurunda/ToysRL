"""主入口: 一键运行指定算法或全量对比。

用法示例:
  python run_experiment.py --agent qlearning --mode fixed --episodes 10000
  python run_experiment.py --agent dqn --mode selfplay --episodes 20000
  python run_experiment.py --agent mcts                        # 零训练, 直接评估
  python run_experiment.py --all                               # 全算法训练 + Elo 天梯赛
  python run_experiment.py --arena                             # 用已保存 checkpoint 打天梯
  python run_experiment.py --agent qlearning --viz             # 策略可解释性可视化

输出目录: results/<算法>_<模式>/
    metrics.csv         训练/评估全量指标
    learning_curves.png 学习曲线图
    summary.json        样本效率对比 (对 Minimax 不败率 ≥95% 所需对局数与耗时)
    agent.pkl / agent.pt  模型 checkpoint
    meta.json           重建智能体所需的超参数元信息
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import yaml

from agents.alphazero import AlphaZeroAgent
from agents.base import BaseAgent
from agents.baselines import HeuristicAgent, MinimaxAgent, RandomAgent
from agents.dqn import DQNAgent
from agents.mcts import MCTSAgent
from agents.ppo import PPOAgent
from agents.tabular import MonteCarloAgent, QLearningAgent, SarsaAgent
from evaluation.arena import Arena
from evaluation.evaluator import Evaluator
from trainers.fixed_opponent import train as train_fixed
from trainers.self_play import train_alpha_zero, train_self_play
from utils.logger import Logger

ROOT = os.path.dirname(os.path.abspath(__file__))
TRAINABLE = ["mc", "sarsa", "qlearning", "dqn", "ppo", "alphazero"]
BASELINES = ["random", "heuristic", "minimax"]


# ---------------------------------------------------------------------- #
# 智能体构建 / 加载
# ---------------------------------------------------------------------- #
def build_agent(name: str, cfg: dict, rng) -> BaseAgent:
    if name == "mc":
        return MonteCarloAgent(**cfg["tabular"], rng=rng)
    if name == "sarsa":
        return SarsaAgent(**cfg["tabular"], rng=rng)
    if name == "qlearning":
        return QLearningAgent(**cfg["tabular"], rng=rng)
    if name == "dqn":
        return DQNAgent(**cfg["dqn"], rng=rng)
    if name == "ppo":
        return PPOAgent(**cfg["ppo"], rng=rng)
    if name == "mcts":
        return MCTSAgent(**cfg["mcts"], rng=rng)
    if name == "alphazero":
        return AlphaZeroAgent(**cfg["alphazero"], rng=rng)
    if name == "random":
        return RandomAgent(rng=rng)
    if name == "heuristic":
        return HeuristicAgent(rng=rng)
    if name == "minimax":
        return MinimaxAgent(rng=rng)
    raise ValueError(f"未知智能体: {name} (可选: {TRAINABLE + BASELINES})")


def save_path(run_dir: str, agent: BaseAgent) -> str:
    """深度算法存 .pt (torch), 其余存 .pkl (pickle)。"""
    ext = "pt" if isinstance(agent, (DQNAgent, PPOAgent, AlphaZeroAgent)) else "pkl"
    return os.path.join(run_dir, f"agent.{ext}")


def load_checkpoint(run_dir: str, rng) -> BaseAgent:
    with open(os.path.join(run_dir, "meta.json")) as f:
        meta = json.load(f)
    agent = build_agent(meta["name"], meta["cfg"], rng)
    path = os.path.join(run_dir, "agent.pt")
    if not os.path.exists(path):
        path = os.path.join(run_dir, "agent.pkl")
    agent.load(path)
    return agent


# ---------------------------------------------------------------------- #
# 训练与保存
# ---------------------------------------------------------------------- #
def train_and_save(name: str, cfg: dict, args, rng) -> str:
    mode = args.mode
    if mode == "auto":
        # 文档默认: 表格型与深度算法用固定对手, AZ 必须自博弈
        mode = "selfplay" if name == "alphazero" else "fixed"
    run_dir = os.path.join(ROOT, "results", f"{name}_{mode}")
    os.makedirs(run_dir, exist_ok=True)

    agent = build_agent(name, cfg, rng)
    logger = Logger(run_dir)
    evaluator = Evaluator(**cfg["eval"], rng=rng)
    tr = cfg["training"]
    t0 = time.time()

    print(f"\n=== 训练 {name} (mode={mode}, episodes={args.episodes}) ===")
    if name == "mcts":
        # 零训练: 仅靠在线推演, 直接跑一轮评估
        results = evaluator.evaluate(agent, 0)
        results["wall_time"] = time.time() - t0
        logger.log(**results)
        print("[" + agent.name + "] " + evaluator.format(results))
    elif mode == "selfplay":
        if name == "alphazero":
            train_alpha_zero(agent, args.episodes, args.eval_every, evaluator,
                             logger, rng, tr["az_games_per_update"])
        else:
            train_self_play(agent, args.episodes, args.eval_every, evaluator,
                            logger, rng, tr["save_every"], tr["pool_size"],
                            tr["pool_prob"])
    else:
        opponent_name = args.opponent
        opponent = {"random": RandomAgent, "heuristic": HeuristicAgent}[opponent_name](rng=rng)
        train_fixed(agent, opponent, args.episodes, args.eval_every,
                    evaluator, logger, rng)

    # -------- 保存 checkpoint / 指标 / 图表 --------
    agent.save(save_path(run_dir, agent))
    logger.save_csv(os.path.join(run_dir, "metrics.csv"))
    # 图表标题用英文 (无显示环境通常缺少中文字体, 避免乱码)
    logger.plot(["win_vs_random", "win_vs_heuristic", "notlose_vs_minimax"],
                title=f"{agent.name} learning curves ({mode})",
                path=os.path.join(run_dir, "learning_curves.png"))

    # 样本效率: 达到"对 Minimax 不败率 ≥ 95%"所需的对局数与耗时
    ep95 = logger.time_to_threshold("notlose_vs_minimax", 0.95)
    wall95 = None
    if ep95 is not None:
        for e in logger.history:
            if e.get("episode") == ep95 and "wall_time" in e:
                wall95 = e["wall_time"]
                break
    summary = {
        "name": name, "mode": mode, "episodes": args.episodes,
        "sample_efficiency": {
            "episodes_to_notlose95": ep95,
            "wall_time_to_notlose95_s": round(wall95, 1) if wall95 is not None else None,
        },
        "final": {k: logger.latest(k) for k in
                  ("win_vs_random", "win_vs_heuristic", "win_vs_minimax",
                   "notlose_vs_minimax", "wall_time")},
    }
    with open(os.path.join(run_dir, "summary.json"), "w") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    with open(os.path.join(run_dir, "meta.json"), "w") as f:
        json.dump({"name": name, "mode": mode, "cfg": {k: cfg[k] for k in
                   ("tabular", "dqn", "ppo", "mcts", "alphazero")}}, f,
                  ensure_ascii=False, indent=2)
    print(f"结果已保存至 {run_dir} (总耗时 {time.time() - t0:.1f}s)")
    return run_dir


# ---------------------------------------------------------------------- #
# 策略可解释性可视化 (文本热力图)
# ---------------------------------------------------------------------- #
def viz_policy(agent: BaseAgent, rng) -> None:
    """打印各算法在经典棋局下的策略视图:
       (a) 开局空棋盘 (先手视角); (b) 对手占中心 (先手视角)。"""
    import torch
    import torch.nn.functional as F

    positions = {
        "开局空棋盘 (我方先手)": (np.zeros(9, dtype=int),
                                  np.ones(9, dtype=np.int8)),
        "对手占中心 (我方先手)": (np.array([0, 0, 0, 0, -1, 0, 0, 0, 0], dtype=int),
                                  np.array([1, 1, 1, 1, 0, 1, 1, 1, 1], dtype=np.int8)),
    }

    def policy_grid(obs, mask):
        """提取当前智能体的动作评分向量 (Q 值 / 概率 / 访问次数)。"""
        if isinstance(agent, (MonteCarloAgent, QLearningAgent, SarsaAgent)):
            q = agent.Q[tuple(int(x) for x in obs)]
            return np.where(mask, q, np.nan), "Q 值"
        if isinstance(agent, DQNAgent):
            with torch.no_grad():
                q = agent.qnet(torch.tensor(obs, dtype=torch.float32)).numpy()
            return np.where(mask, q, np.nan), "Q 值"
        if isinstance(agent, PPOAgent):
            with torch.no_grad():
                logits = agent.net.masked_logits(
                    torch.tensor(obs, dtype=torch.float32).unsqueeze(0),
                    torch.tensor(mask, dtype=torch.float32).unsqueeze(0))
                probs = F.softmax(logits, dim=-1).numpy().reshape(-1)
            return np.where(mask, probs, np.nan), "动作概率"
        if isinstance(agent, MCTSAgent):
            from agents.mcts import MCTSNode
            root = MCTSNode(tuple(int(x) for x in obs),
                            tuple(int(a) for a in np.flatnonzero(mask)))
            for _ in range(agent.n_simulations):
                agent._simulate(root)
            counts = np.array([root.children[a].N if a in root.children else 0
                               for a in range(9)], dtype=float)
            return np.where(mask, counts, np.nan), "访问次数"
        if isinstance(agent, AlphaZeroAgent):
            root = agent._search(tuple(int(x) for x in obs),
                                 tuple(int(a) for a in np.flatnonzero(mask)))
            return np.where(mask, root.N, np.nan), "访问次数"
        return None, ""

    print(f"\n=== {agent.name} 的策略视图 ===")
    for title, (obs, mask) in positions.items():
        values, label = policy_grid(obs, mask)
        if values is None:
            print(f"[{title}] 该算法无可视化内部状态")
            continue
        print(f"\n{title} —— {label}:")
        for r in range(3):
            row = values[r * 3:(r + 1) * 3]
            print("   " + " | ".join(
                f"{v:>7.3f}" if not np.isnan(v) else "   ✗   " for v in row))


# ---------------------------------------------------------------------- #
# 天梯赛
# ---------------------------------------------------------------------- #
def run_arena(cfg: dict, rng, games_per_pair: int = None) -> None:
    arena_cfg = cfg["arena"]
    games_per_pair = games_per_pair or arena_cfg["games_per_pair"]
    agents = {}
    for name in BASELINES + ["mcts"]:
        agents[name.capitalize() if name != "mcts" else "MCTS"] = build_agent(name, cfg, rng)
    for name in TRAINABLE:
        for mode in ("fixed", "selfplay"):
            run_dir = os.path.join(ROOT, "results", f"{name}_{mode}")
            if os.path.exists(os.path.join(run_dir, "meta.json")):
                label = name.upper() + ("(sp)" if mode == "selfplay" else "")
                agents[label] = load_checkpoint(run_dir, rng)
                break  # 每个算法取一个已训练版本 (优先固定对手版)
    if len(agents) <= len(BASELINES) + 1:
        print("未找到已训练的 checkpoint, 请先运行训练 (如 --all)。")
        return
    print(f"\n=== 全循环对弈 (每对 {games_per_pair} 局) ===")
    arena = Arena(K=arena_cfg["K"], rng=rng)
    ranking, _ = arena.round_robin(agents, games_per_pair=games_per_pair)
    arena.print_ranking(ranking)


# ---------------------------------------------------------------------- #
def main():
    parser = argparse.ArgumentParser(description="多范式强化学习对比实验平台")
    parser.add_argument("--agent", type=str, default=None,
                        help="算法名: " + "/".join(TRAINABLE + BASELINES + ["mcts"]))
    parser.add_argument("--all", action="store_true", help="训练全部算法并打天梯")
    parser.add_argument("--arena", action="store_true", help="用已保存 checkpoint 打 Elo 天梯")
    parser.add_argument("--viz", action="store_true", help="打印策略可解释性视图")
    parser.add_argument("--mode", type=str, default="auto",
                        choices=["auto", "fixed", "selfplay"])
    parser.add_argument("--opponent", type=str, default="random",
                        choices=["random", "heuristic"], help="固定对手训练的对手")
    parser.add_argument("--episodes", type=int, default=None)
    parser.add_argument("--eval-every", type=int, default=None)
    parser.add_argument("--games-per-pair", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--config", type=str, default=os.path.join(ROOT, "configs", "default.yaml"))
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    if args.seed is not None:
        cfg["seed"] = args.seed
    rng = np.random.default_rng(cfg["seed"])
    args.episodes = args.episodes or cfg["training"]["episodes"]
    args.eval_every = args.eval_every or cfg["training"]["eval_every"]

    if args.all:
        # 各算法按规模分配训练量 (深度算法样本效率低, 多训; AZ 树搜索慢, 少训)
        per_agent_episodes = {
            "mc": min(args.episodes, 5000), "sarsa": min(args.episodes, 5000),
            "qlearning": min(args.episodes, 5000), "dqn": max(args.episodes, 10000),
            "ppo": max(args.episodes, 10000), "alphazero": min(args.episodes, 3000),
        }
        for name in TRAINABLE:
            saved = args.episodes
            args.episodes = per_agent_episodes[name]
            train_and_save(name, cfg, args, rng)
            args.episodes = saved
        run_arena(cfg, rng, args.games_per_pair)
        return

    if args.arena:
        run_arena(cfg, rng, args.games_per_pair)
        return

    if args.agent is None:
        parser.print_help()
        return

    if args.viz:
        # 加载 checkpoint (若有) 或直接构建 (MCTS/基准无需训练)
        mode = "selfplay" if (args.mode == "auto" and args.agent == "alphazero") \
            else ("fixed" if args.mode == "auto" else args.mode)
        run_dir = os.path.join(ROOT, "results", f"{args.agent}_{mode}")
        if not os.path.exists(os.path.join(run_dir, "meta.json")):
            # 回退: 该算法任意已训练 run 目录均可
            candidates = [os.path.join(ROOT, "results", d) for d in
                          os.listdir(os.path.join(ROOT, "results"))
                          if d.startswith(f"{args.agent}_")]
            for cand in candidates:
                if os.path.exists(os.path.join(cand, "meta.json")):
                    run_dir = cand
                    break
        if os.path.exists(os.path.join(run_dir, "meta.json")):
            agent = load_checkpoint(run_dir, rng)
            print(f"已加载 checkpoint: {run_dir}")
        else:
            agent = build_agent(args.agent, cfg, rng)
        viz_policy(agent, rng)
        return

    train_and_save(args.agent, cfg, args, rng)


if __name__ == "__main__":
    main()
