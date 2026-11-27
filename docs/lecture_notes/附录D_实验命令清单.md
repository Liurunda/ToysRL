# 附录 D　实验命令清单

所有命令在 `tictactoe_rl_arena/` 目录下执行（Python 3.8+）。

## D.1 环境准备

```bash
pip install numpy torch matplotlib pyyaml
python tests/test_phase1.py        # Phase 1 验收: 环境规则 / 基准对手零 Bug 验证
```

## D.2 各章实验索引

| 实验 | 内容 | 命令 |
| :--- | :--- | :--- |
| 1-1 | 打印一局轨迹（脚本见正文） | 见[第 1 章](01_什么是强化学习.md) |
| 2-1 | 验证视角归一化（脚本见正文） | 见[第 2 章](02_马尔可夫决策过程.md) |
| 3-1 | Q 值热力图 | `python run_experiment.py --agent qlearning --viz` |
| 4-1 | ε 参数扫描（改 `tabular` 段） | `python run_experiment.py --agent qlearning --episodes 5000` |
| 5-1 | 三算法学习曲线对比 | 见下 D.3 |
| 6-1 | DQN 曲线 + `target_update_every` 扫描 | 见下 D.3 |
| 7-1 | PPO 训练与指标观察 | `python run_experiment.py --agent ppo --episodes 10000` |
| 8-1 | MCTS 算力-棋力权衡（改 `mcts.n_simulations`） | `python run_experiment.py --agent mcts` |
| 9-1 | AlphaZero 闭环 + 对称增强对照 | 见下 D.3 |
| 10-1 | 全量对比、天梯、人机 | 见下 D.5 |

## D.3 单算法训练命令

```bash
# 表格型 (固定对手, 5000 局约 3 秒)
python run_experiment.py --agent mc        --mode fixed --episodes 5000
python run_experiment.py --agent sarsa     --mode fixed --episodes 5000
python run_experiment.py --agent qlearning --mode fixed --episodes 5000

# 深度价值 / 深度策略 (固定对手)
python run_experiment.py --agent dqn --mode fixed --episodes 10000
python run_experiment.py --agent ppo --mode fixed --episodes 10000

# 自博弈 (价值方法在非平稳环境挣扎 — 现象 10.2)
python run_experiment.py --agent dqn --mode selfplay --episodes 2000
python run_experiment.py --agent ppo --mode selfplay --episodes 10000

# 纯搜索 (零训练, 直接评估)
python run_experiment.py --agent mcts

# AlphaZero (自博弈闭环)
python run_experiment.py --agent alphazero --mode selfplay --episodes 3000

# 固定对手换 Heuristic (观察训练信号变强/变难)
python run_experiment.py --agent qlearning --mode fixed --opponent heuristic --episodes 10000
```

> `--episodes` 缺省时用 `configs/default.yaml` 的 `training.episodes`；
> `--mode auto` 时只有 `alphazero` 自动走 selfplay，其余走 fixed。

## D.4 输出文件说明（`results/<算法>_<模式>/`）

| 文件 | 内容 |
| :--- | :--- |
| `metrics.csv` | 全量指标：每 500 局一轮的评估胜率/不败率 + 每步的 loss 等 |
| `learning_curves.png` | `win_vs_random / win_vs_heuristic / notlose_vs_minimax` 三条曲线 |
| `summary.json` | 最终胜率/不败率、样本效率（不败率 ≥95% 首次达标的局数与耗时） |
| `agent.pkl` / `agent.pt` | checkpoint（表格/MCTS 为 pickle，深度算法为 torch） |
| `meta.json` | 重建智能体所需的超参数元信息 |

快速读结果：

```bash
cat results/qlearning_fixed/summary.json
python -c "import json;d=json.load(open('results/alphazero_selfplay/summary.json'));print(d['sample_efficiency'], d['final'])"
```

## D.5 全量对比 / 天梯 / 可视化 / 人机

```bash
python run_experiment.py --all     # 全算法训练 (约 10 分钟) + 自动 Elo 天梯
python run_experiment.py --arena   # 只用已有 checkpoint 重打天梯 (约 2-3 分钟)

# 策略可解释性视图 (Q 值 / 动作概率 / 访问次数热力图)
python run_experiment.py --agent qlearning --viz
python run_experiment.py --agent dqn       --viz
python run_experiment.py --agent ppo       --viz
python run_experiment.py --agent mcts      --viz
python run_experiment.py --agent alphazero --viz

# 人机对弈 (输入 1-9 对应格子; --side o 人类执后手)
python play_human.py --agent mcts
python play_human.py --agent qlearning --checkpoint results/qlearning_fixed
python play_human.py --agent alphazero --checkpoint results/alphazero_selfplay
python play_human.py --agent minimax      # 挑战完美对手 (你最多逼平)
```

## D.6 调参速查（改 `configs/default.yaml` 后重跑对应命令）

| 想观察的现象 | 改哪里 | 怎么改 |
| :--- | :--- | :--- |
| 探索过多/过少 | `tabular.epsilon_end / epsilon_decay` | 0.3 或 0.0 / 0.99 |
| 学习率过快/过慢 | `tabular.lr`、`dqn.lr`、`ppo.lr` | 0.05 与 0.8 对比 |
| 价值传播速度 | `tabular.gamma`、`dqn.gamma` | 0.8 与 0.99 对比 |
| 目标网络同步节奏 | `dqn.target_update_every` | 10 与 1000 对比 |
| PPO 探索强度 | `ppo.entropy_coef` | 0.01 与 0.1 对比 |
| GAE 看多远 | `ppo.lam` | 0.0 与 1.0 对比 |
| 搜索深度 | `mcts.n_simulations` | 50 / 300 / 2000 |
| UCT/PUCT 探索宽度 | `mcts.c_uct`、`alphazero.c_puct` | 0.5 与 3.0 对比 |
| 对称增强威力 | `alphazero.use_symmetry` | true / false |
| 历史对手池作用 | `training.pool_prob`、`pool_size` | 0.0（关池）与 0.5 对比 |
| 评估统计置信度 | `eval.games_vs_*` | 局数减半/加倍观察曲线噪声 |

## D.7 完整复现一条龙（约 15 分钟）

```bash
pip install numpy torch matplotlib pyyaml
python tests/test_phase1.py
python run_experiment.py --agent mc --episodes 5000
python run_experiment.py --agent sarsa --episodes 5000
python run_experiment.py --agent qlearning --episodes 5000
python run_experiment.py --agent dqn --episodes 10000
python run_experiment.py --agent ppo --episodes 10000
python run_experiment.py --agent mcts
python run_experiment.py --agent alphazero --mode selfplay --episodes 3000
python run_experiment.py --arena
python play_human.py --agent alphazero --checkpoint results/alphazero_selfplay
```

跑完后对照[第 10 章 10.5 节](10_训练范式与评估.md)的全景表：趋势应一致，具体数值
因随机种子与机器性能会有浮动。
