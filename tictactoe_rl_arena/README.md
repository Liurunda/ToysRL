# 基于 3×3 井字棋的多范式强化学习演示平台

依据《基于 3×3 井字棋的多范式强化学习演示平台设计文档》实现的模块化、可扩展
RL 对比实验框架。在规则极简但包含双人零和博弈核心特性的井字棋上, 演示并对比
**表格型、深度价值、策略梯度、搜索与规划**四大类算法的原理、收敛速度与棋力。

## 算法矩阵

| 类别 | 算法 | 状态/价值表达 | 核心机制 |
| :--- | :--- | :--- | :--- |
| 表格型 | Monte Carlo / SARSA / Q-Learning | `dict[state, np.array(9)]` Q 表 | ε-greedy + Action Masking |
| 深度价值 | DQN | MLP 9→64→64→9 | Replay Buffer + Target Net, 非法动作 Q 置 -inf |
| 深度策略 | PPO | Actor-Critic 双头 MLP | 非法动作 logits 置 -1e9, GAE + PPO-Clip |
| 纯搜索 | MCTS | 树节点 (N, W, Q) | UCT + 随机 Rollout, **零训练** |
| 混合范式 | AlphaZero-Lite | Policy-Value 网络 (P, v) | PUCT, 搜索访问分布 π 监督训练网络 |

三档基准对手: `Random` / `Heuristic` (赢→堵→中心→角→随机) / `Minimax`
(带记忆化 Negamax, 保证不败 —— 对 Minimax 不败率 100% 即达到理论最优)。

## 快速开始

```bash
pip install numpy torch matplotlib pyyaml   # 依赖

# 单个算法 (固定对手训练 / 自博弈训练)
python run_experiment.py --agent qlearning --mode fixed --episodes 10000
python run_experiment.py --agent dqn --mode selfplay --episodes 20000
python run_experiment.py --agent ppo --mode fixed --episodes 10000
python run_experiment.py --agent alphazero --mode selfplay --episodes 3000
python run_experiment.py --agent mcts                   # 零训练, 直接评估

# 全量对比 (所有算法训练后自动打 Elo 天梯)
python run_experiment.py --all

# Elo 天梯赛 (使用已保存 checkpoint)
python run_experiment.py --arena

# 策略可解释性视图 (Q 值 / 动作概率 / MCTS 访问次数热力图)
python run_experiment.py --agent qlearning --viz

# 人机对弈
python play_human.py --agent mcts
python play_human.py --agent dqn --checkpoint results/dqn_fixed

# Phase 1 验收测试 (环境规则 / 基准对手零 Bug 验证)
python tests/test_phase1.py
```

## 关键设计

* **视角归一化 (Canonical Observation)**: 观测永远乘以当前玩家标记,
  智能体眼中 +1 永远是自己 —— 同一套模型无缝执先手/后手;
* **非法动作屏蔽 (Action Masking)**: 所有算法在合法动作内选择/采样;
* **环境接口解耦**: 底层 `TicTacToeEnv` 单步轮流 + `clone()` 供推演,
  上层 `SingleAgentWrapper` 封装为"对手托管"单智能体 MDP;
* **两种训练范式**: 固定对手 (Stationary MDP) 与自博弈
  (历史对手池防策略遗忘: 80% 最新自己 + 20% 老版本);
* **标准化评估协议**: 每 500 局冻结评估 vs Random 100 局 / Heuristic 100 局 /
  Minimax 50 局 (先手后手各半), 输出学习曲线与样本效率对比。

## 典型现象 (全部可在本平台复现)

1. 针对 Random 训练的模型面对 Heuristic/Minimax 暴露盲区 (过拟合弱对手);
2. 神经网络前期收敛慢于 Q 表, 但无需枚举状态;
3. MCTS 零训练仅靠在线推演即可接近完美棋力 (算力换样本);
4. AlphaZero-Lite 数百局自博弈即可逼近理论最优 (样本效率最高)。

## 目录结构

```
tictactoe_rl_arena/
├── configs/default.yaml      # 超参数
├── envs/                     # TicTacToeEnv + SingleAgentWrapper
├── agents/                   # base / baselines / tabular / dqn / ppo / mcts / alphazero
├── trainers/                 # fixed_opponent / self_play (含历史对手池与 AZ 闭环)
├── evaluation/               # evaluator (标准化评估) / arena (循环赛 + Elo)
├── utils/                    # logger (曲线/图表) / symmetry (旋转镜像增强)
├── tests/test_phase1.py      # Phase 1 验收测试
├── run_experiment.py         # 主入口
├── play_human.py             # 人机对弈
└── results/                  # 训练产物 (checkpoint / 曲线 / 样本效率 summary)
```
