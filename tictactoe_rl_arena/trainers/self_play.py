"""自博弈训练调度流 (Self-Play 模式)。

适用阶段: 进阶对比演示 (Q-Learning / DQN / PPO / AlphaZero-Lite)。

防崩塌机制 (历史对手池 Opponent Pool):
    若只让当前网络和最新的自己对弈, 容易产生"策略遗忘"和循环克制。
    每训练 K 轮 (默认 500 局), 将当前智能体权重深拷贝存入历史池
    (最多保留最近 10 个版本); 每局训练以 80% 概率匹配最新自己,
    20% 概率从历史池随机抽取老对手, 确保对弱点策略的持续压制。

AlphaZero-Lite 走独立的训练流: MCTS 自博弈收集 (s, π, z) → 监督训练
Policy-Value 网络, 形成策略迭代闭环。
"""
import copy
import time
from typing import Optional

import numpy as np

from agents.base import BaseAgent, Transition
from envs.tictactoe import TicTacToeEnv
from envs.wrappers import SingleAgentWrapper
from evaluation.evaluator import Evaluator
from utils.logger import Logger


def train_self_play(agent: BaseAgent, episodes: int, eval_every: int,
                    evaluator: Evaluator, logger: Logger,
                    rng: np.random.Generator,
                    save_every: int = 500, pool_size: int = 10,
                    pool_prob: float = 0.2, verbose: bool = True) -> BaseAgent:
    """自博弈 + 历史对手池训练主循环。"""
    pool = []                                # 历史对手池 (权重深拷贝)
    latest = copy.deepcopy(agent)            # 最新的自己 (冻结快照)
    start = time.time()
    step = 0
    for ep in range(1, episodes + 1):
        # 80% 匹配最新自己, 20% 从历史池随机抽取老对手
        if pool and rng.random() < pool_prob:
            opponent = pool[int(rng.integers(len(pool)))]
        else:
            opponent = latest
        wrapper = SingleAgentWrapper(TicTacToeEnv(), opponent,
                                     randomize_side=True, rng=rng)
        obs, mask = wrapper.reset()
        done = False
        ep_reward = 0.0
        while not done:
            action = agent.select_action(obs, mask, training=True)
            next_obs, reward, done, info = wrapper.step(action)
            next_mask = wrapper.get_valid_mask()
            agent.observe(Transition(obs, action, reward, next_obs, done,
                                     mask, next_mask))
            metrics = agent.update()
            if metrics:
                logger.log(step=step, **metrics)
            ep_reward += reward
            obs, mask = next_obs, next_mask
            step += 1
        agent.end_episode()

        winner = info.get("winner", 0)
        agent_result = 0 if winner == 0 else (1 if winner == wrapper.agent_player else -1)
        logger.log(episode=ep, ep_reward=ep_reward, ep_result=agent_result,
                   pool_size=len(pool))

        # 定期把当前智能体权重深拷贝存入历史池 (最多保留最近 pool_size 个版本)
        if ep % save_every == 0:
            pool.append(latest)
            latest = copy.deepcopy(agent)
            if len(pool) > pool_size:
                pool.pop(0)

        if ep % eval_every == 0:
            results = evaluator.evaluate(agent, ep)
            results["wall_time"] = time.time() - start
            logger.log(**results)
            if verbose:
                print(f"[{agent.name} self-play] " + evaluator.format(results))
    return agent


def train_alpha_zero(agent: BaseAgent, episodes: int, eval_every: int,
                     evaluator: Evaluator, logger: Logger,
                     rng: np.random.Generator,
                     games_per_update: int = 10, verbose: bool = True) -> BaseAgent:
    """AlphaZero-Lite 训练流: 自博弈收集数据 → 监督训练网络 (策略迭代闭环)。

    agent 必须是 AlphaZeroAgent: 其 select_action 内部使用 PUCT 树搜索,
    访问分布 π 反过来监督 Policy-Value 网络。
    """
    start = time.time()
    for ep in range(1, episodes + 1):
        # 一局 MCTS 自博弈, 得到 (state, π, z) 训练样本
        data = agent.self_play_episode()
        agent.train_buffer.extend(data)
        logger.log(episode=ep, buffer_size=len(agent.train_buffer))

        if ep % games_per_update == 0:
            metrics = agent.update()
            logger.log(episode=ep, **metrics)

        if ep % eval_every == 0:
            results = evaluator.evaluate(agent, ep)
            results["wall_time"] = time.time() - start
            logger.log(**results)
            if verbose:
                print(f"[{agent.name} self-play] " + evaluator.format(results))
    return agent
