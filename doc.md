以下是一份完整的 **《基于 3×3 井字棋的多范式强化学习演示平台设计文档》**。该文档旨在建立一个模块化、可扩展的实验框架，使表格型算法、深度强化学习以及树搜索算法能够在完全公平的基准下进行训练、对弈和对比。

---

# 基于 3×3 井字棋的多范式强化学习演示平台设计文档

## 1. 项目概述与目标

### 1.1 设计目标
构建一个轻量级、易读且标准化的强化学习对比实验平台，通过 **3×3 井字棋（Tic-Tac-Toe）** 这一规则极简但包含双人零和博弈核心特性的环境，演示并对比四大类强化学习算法的原理、收敛速度与最终棋力：
1. **经典表格型算法**：Monte Carlo, Q-Learning, SARSA
2. **基于价值的深度强化学习**：DQN
3. **基于策略梯度的深度强化学习**：REINFORCE / PPO
4. **基于搜索与规划的算法**：MCTS（蒙特卡洛树搜索）、AlphaZero-Lite

### 1.2 核心挑战与设计原则
* **视角归一化（Canonical Representation）**：让同一个智能体既能执先手（X）也能执后手（O）。
* **非法动作屏蔽（Action Masking）**：避免智能体浪费样本去学习“不能下在已有棋子的地方”，聚焦策略博弈本身。
* **环境接口解耦**：底层提供“单步轮流”的纯棋盘环境，上层按需封装为“单智能体对手托管环境”（用于固定对手训练）或“自博弈调度器”。

---

## 2. 系统总体架构

```text
+-------------------------------------------------------------------+
|                        实验调度与可视化层                           |
|   Training Pipeline (固定对手 / 自博弈)   |   Arena (天梯/Elo评估)  |
+-------------------------------------------------------------------+
                 |                                   |
                 v                                   v
+---------------------------------+   +-----------------------------+
|        智能体抽象层 (Agents)      |   |      基准对手 (Baselines)    |
|  - BaseAgent (统一接口)          |   |  - RandomAgent              |
|  - Tabular: MC, SARSA, Q-Learn  |   |  - HeuristicAgent (规则)     |
|  - Deep RL: DQN, PPO            |   |  - MinimaxAgent (完美解)     |
|  - Search:  MCTS, AlphaZeroLite |   +-----------------------------+
+---------------------------------+                 |
                 |                                  |
                 +----------------+-----------------+
                                  |
                                  v
+-------------------------------------------------------------------+
|                      核心环境层 (Environment)                       |
|  TicTacToeEnv (基础双人轮流环境)  <-- 封装 --> SinglePlayerWrapper   |
+-------------------------------------------------------------------+
```

---

## 3. 核心模块详细设计

### 3.1 游戏环境模块 (`TicTacToeEnv`)

#### A. 状态与视角设计
* **底层棋盘**：长度为 9（或 $3 \times 3$）的整型数组，先手（X）记为 `1`，后手（O）记为 `-1`，空位记为 `0`。
* **视角归一化观测（Canonical Observation）**：
  每次向当前轮到的玩家返回状态时，统一乘以 `current_player`（`1` 或 `-1`）。
  * 结果：在智能体眼中，`+1` **永远代表己方棋子**，`-1` **永远代表对手棋子**，`0` 代表空位。
  * 优势：无需为先手和后手分别训练两套模型，同一套网络/Q表可无缝切换先后手。
* **算法输入适配**：
  * 表格型算法：将长度为 9 的向量转为不可变元组或字符串（如 `"1,0,-1,..."`）作为字典的 Key。
  * 神经网络算法：支持扁平化 9 维向量（MLP 输入），或 3 通道 $3 \times 3$ 张量（`[己方落子层, 对手落子层, 空位层]`，适合 CNN 或更清晰的特征表达）。

#### B. 动作空间与合法掩码
* **动作空间**：`Discrete(9)`，对应格子索引 `0` 到 `8`（行优先）。
* **Action Mask**：提供 `get_valid_actions_mask()` 方法，返回长度为 9 的布尔数组或 `0/1` 数组（`1` 表示合法空位，`0` 表示已占用）。

#### C. 奖励机制（稀疏终局奖励）
* 己方落子直接获胜：`+1.0`（且对手视角记为 `-1.0`）
* 棋盘填满平局：`0.0`
* 普通中间步：`0.0`
* *(兜底机制)* 若未开启掩码防护且智能体强行选择非法动作：直接判负 `-10.0` 并结束对局（或在演示中严格通过 Action Mask 杜绝非法动作）。

#### D. 环境核心接口定义
```python
class TicTacToeEnv:
    def reset(self) -> np.ndarray: ...
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, dict]: ...
    def get_canonical_board(self) -> np.ndarray: ...
    def get_valid_mask(self) -> np.ndarray: ...
    def clone(self) -> "TicTacToeEnv": ...  # 供 MCTS 模拟推演使用
    def render(self) -> None: ...
```

#### E. 单智能体包装器 (`SingleAgentWrapper`)
对于标准单智能体 RL 算法（如 DQN、PPO、Q-Learning），环境每次 `step(action)` 需包含“己方落子 $\rightarrow$ 若未结束则对手自动落子”的完整转移，才能直接套用标准 MDP 公式 $(s_t, a_t, r_t, s_{t+1})$。
* **设计方案**：包装器内持有一个 `opponent_agent`，并支持配置我方随机执先或执后（若执后，在 `reset()` 时由对手先走一步再返回初始状态 $s_0$）。

---

### 3.2 基准对手模块（Baselines）

为了客观衡量各 RL 算法的棋力上限与收敛速度，必须内置三档难度固定的对手：

1. **Level 1 - `RandomAgent`（随机策略）**
   * 在合法动作中均匀随机采样。用于检验算法是否具备基础的“学习能力”。
2. **Level 2 - `HeuristicAgent`（单步启发式规则）**
   * 规则优先级：有连三直接赢 $\rightarrow$ 对手下步连三则堵截 $\rightarrow$ 优先占中心/角落 $\rightarrow$ 其余随机。用于区分“只会抓随机对手漏洞”与“真正学会基础防守”的算法。
3. **Level 3 - `MinimaxAgent`（带记忆化/Alpha-Beta 的完美策略）**
   * 遍历博弈树，保证**绝对不败**（最优对最优必为平局）。
   * **核心评估意义**：因为井字棋先手最优和后手最优的结果都是**平局**，所以任何 RL 算法在面对 Minimax 时，**100% 不败率（胜率+平局率）** 即代表达到了理论最优水平。

---

### 3.3 智能体统一接口与算法矩阵

#### 统一基类 (`BaseAgent`)
```python
class BaseAgent(ABC):
    @abstractmethod
    def select_action(self, obs: np.ndarray, valid_mask: np.ndarray, training: bool = True) -> int:
        pass

    def observe(self, transition: tuple) -> None:  # 存入经验或轨迹
        pass

    def update(self) -> dict:  # 执行参数/Q表更新，返回 loss 等指标
        pass
```

#### 各算法实现规格

| 类别 | 算法 | 状态/价值表达 | 探索机制 & 动作屏蔽 | 学习特点（演示重点） |
| :--- | :--- | :--- | :--- | :--- |
| **表格型** | **Monte Carlo** | `dict[state_tuple, np.array(9)]` | $\epsilon$-greedy（仅在 `valid_mask==1` 中采样） | 回合结束后用真实回报 $G_t$ 更新，高方差、无偏差。 |
| **表格型** | **Q-Learning** | `dict[state_tuple, np.array(9)]` | 探索同上；目标值仅在下一状态的合法动作上取 $\max$ | Off-policy 单步 TD 更新，收敛快于 MC。 |
| **表格型** | **SARSA** | `dict[state_tuple, np.array(9)]` | 目标值使用下一状态实际选择的动作 $Q(s', a')$ | On-policy，训练中期如果 $\epsilon$ 较大，策略会偏保守。 |
| **深度价值** | **DQN** | MLP (`9 -> 64 -> 64 -> 9`) | Replay Buffer + Target Net；`argmax` 前将非法动作 Q 值置为 $-\infty$ | 演示从“查表”到“函数近似”的跨越，观察网络稳定性。 |
| **深度策略** | **PPO** | Actor-Critic 双头 MLP | Softmax 前将非法动作 logits 置为 `-1e9`（概率严格为 0） | 演示策略梯度直接优化动作概率分布与 GAE 优势估计。 |
| **纯搜索** | **MCTS** | 树节点记录 $(N, W, Q)$ | UCT 公式选子节点，默认策略随机模拟（Rollout） | **零训练**，仅靠在线推演即可达到接近完美棋力，用于对比搜索与学习的算力权衡。 |
| **混合范式** | **AlphaZero-Lite** | Policy-Value 网络输出 $(P, v)$ | PUCT 公式用 $P$ 引导树搜索，用 $v$ 替代随机 Rollout | 用 MCTS 搜索后的访问分布 $\pi$ 反过来监督训练神经网络（策略迭代闭环）。 |

---

## 4. 训练范式设计

平台支持两种核心训练模式，用于对比不同环境平稳性下的表现：

### 4.1 固定对手训练（Stationary MDP 模式）
* **适用阶段**：初级对比演示（MC / Q-Learning / DQN / PPO）。
* **流程**：固定对手为 `RandomAgent` 或 `HeuristicAgent`，每局随机分配 RL 智能体执先手或执后手，跑通标准单智能体训练循环。
* **预期现象展示**：针对 `RandomAgent` 训练出来的模型，在面对 `HeuristicAgent` 或 `Minimax` 时往往会暴露盲区（过拟合弱对手）。

### 4.2 自博弈训练（Self-Play 模式）
* **适用阶段**：进阶对比演示（Q-Learning / DQN / PPO / AlphaZero-Lite）。
* **防崩塌机制（历史对手池 Opponent Pool）**：
  * 若只让当前网络和最新的自己对弈，容易产生“策略遗忘”和循环克制。
  * **设计方案**：每训练 $K$ 轮（如 500 局），将当前智能体权重深拷贝存入历史池（最多保留最近 10 个版本）。每局训练时，以 80% 概率匹配最新自己，20% 概率从历史池随机抽取老对手，确保对弱点策略的持续压制。

---

## 5. 评估与可视化体系

### 5.1 标准化评估协议 (`Evaluator`)
每训练一定步数（例如每 500 局），冻结智能体参数（`training=False`，关闭随机探索），执行固定评估：
* 对阵 **Random** 100 局（50 先手 + 50 后手）
* 对阵 **Heuristic** 100 局（50 先手 + 50 后手）
* 对阵 **Minimax** 50 局（25 先手 + 25 后手，因 Minimax 确定性较强，可辅以轻微随机平局分支）

### 5.2 核心对比指标与图表
1. **学习曲线（Learning Curves）**：
   * X 轴：训练对局数（Episodes）或 实际挂钟时间（Wall-clock Time）。
   * Y 轴：对阵 Heuristic 的胜率、对阵 Minimax 的**不败率（Win + Draw）**。
2. **样本效率对比表**：达到“对 Minimax 不败率 $\ge 95\%$”所需的对局数与耗时。
3. **循环赛与 Elo 天梯榜（Arena）**：
   * 所有算法训练完成后，互相进行全循环对弈（Round-Robin），计算并展示最终 Elo 评分排行榜。
4. **策略可解释性可视化（选做但极佳）**：
   * 给定一个经典棋局（如开局空棋盘，或对手占中心），打印出 Q-Learning 的 Q 值热力图、PPO 的动作概率分布、MCTS 的节点访问次数分布，直观展示不同算法“眼中”的好棋。

---

## 6. 建议代码目录结构

```text
tictactoe_rl_arena/
├── configs/                  # 超参数配置文件 (yaml/json)
│   └── default.yaml
├── envs/
│   ├── tictactoe.py          # 核心双人轮流环境 (含 clone, render)
│   └── wrappers.py           # SingleAgentWrapper (托管对手与先/后手轮换)
├── agents/
│   ├── base.py               # BaseAgent 抽象类
│   ├── baselines.py          # Random, Heuristic, Minimax (含记忆化)
│   ├── tabular.py            # MonteCarlo, SARSA, QLearning
│   ├── dqn.py                # DQN + ReplayBuffer
│   ├── ppo.py                # PPO (带 Action Masking 的 Categorical)
│   ├── mcts.py               # 纯 UCT MCTS
│   └── alphazero.py          # AlphaZero-Lite (MCTS + PolicyValueNet)
├── trainers/
│   ├── fixed_opponent.py     # 固定对手训练流
│   └── self_play.py          # 自博弈与历史池调度流
├── evaluation/
│   ├── evaluator.py          # 定期基准测试
│   └── arena.py              # 多算法混战与 Elo 计算
├── utils/
│   ├── logger.py             # 训练曲线记录与 Matplotlib 绘图
│   └── symmetry.py           # (可选) 棋盘旋转/镜像对称性数据增强
├── run_experiment.py         # 主入口：一键运行指定算法或全量对比
└── play_human.py             # 人机交互脚本：亲自下场测试训练好的 AI
```

---

## 7. 分阶段实施路线图（Roadmap）

* **Phase 1：地基搭建（核心环境 + 基准测试）**
  * 实现 `TicTacToeEnv`、`SingleAgentWrapper`、3 个 Baseline 智能体。
  * 验证 `Minimax vs Minimax` 100% 平局、`Heuristic vs Random` 胜率 > 90%，确保底层规则与视角切换零 Bug。
* **Phase 2：表格型算法闭环与评估流水线**
  * 实现 Q-Learning、SARSA、Monte Carlo 及 `Evaluator` 绘图脚本。
  * 跑通“固定对手”与“自博弈”对比，输出第一组学习曲线。
* **Phase 3：深度强化学习接入**
  * 基于 PyTorch 实现 DQN 和 PPO（重点验证 Action Masking 逻辑正确性）。
  * 观察相比表格型方法在 3×3 棋盘上的样本效率差异（如神经网络前期收敛慢但具备泛化能力）。
* **Phase 4：树搜索与 AlphaZero-Lite**
  * 实现纯 MCTS 及 AlphaZero-Lite。
  * 跑通全算法 Elo 天梯赛与交互式人机对弈脚本（`play_human.py`）。

---

如果你认可这份设计文档，我们可以立刻进入 **Phase 1**：我可以直接为你写出 `envs/tictactoe.py`、`envs/wrappers.py` 以及 `agents/baselines.py`（含 Minimax）的完整可运行代码，或者你希望先针对文档中的某一部分（例如 AlphaZero-Lite 的设计细节或对称性增强）展开讨论？