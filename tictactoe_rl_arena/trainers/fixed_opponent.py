"""固定对手训练流 (Stationary MDP 模式)。

适用阶段: 初级对比演示 (MC / Q-Learning / DQN / PPO)。
流程: 固定对手为 RandomAgent 或 HeuristicAgent, 每局随机分配 RL 智能体
执先手或执后手, 跑通标准单智能体训练循环。

预期现象: 针对 RandomAgent 训练出的模型, 面对 HeuristicAgent / Minimax
时往往暴露盲区 (过拟合弱对手)。
"""
import time
from typing import Optional

import numpy as np

from agents.base import BaseAgent, Transition
from envs.tictactoe import TicTacToeEnv
from envs.wrappers import SingleAgentWrapper
from evaluation.evaluator import Evaluator
from utils.logger import Logger


def train(agent: BaseAgent, opponent: BaseAgent, episodes: int,
          eval_every: int, evaluator: Evaluator, logger: Logger,
          rng: np.random.Generator, verbose: bool = True) -> BaseAgent:
    """固定对手训练主循环。

    注意: agent.update() 在每一步后都被调用, 由各算法自行决定是否执行
    参数更新 (Q-Learning/SARSA 在 observe 中就地更新, MC 在回合末更新,
    DQN 按 buffer 就绪程度更新, PPO 攒够 batch 才更新)。
    """
    wrapper = SingleAgentWrapper(TicTacToeEnv(), opponent,
                                 randomize_side=True, rng=rng)
    start = time.time()
    step = 0
    for ep in range(1, episodes + 1):
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

        # 记录本回合从智能体视角的结果 (1 胜 / 0 平 / -1 负)
        winner = info.get("winner", 0)
        agent_result = 0 if winner == 0 else (1 if winner == wrapper.agent_player else -1)
        logger.log(episode=ep, ep_reward=ep_reward, ep_result=agent_result,
                   opponent=opponent.name)

        if ep % eval_every == 0:
            results = evaluator.evaluate(agent, ep)
            results["wall_time"] = time.time() - start
            logger.log(**results)
            if verbose:
                print(f"[{agent.name} vs {opponent.name}] " + evaluator.format(results))
    return agent
