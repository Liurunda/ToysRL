# 附录 B　代码导读索引

所有路径相对仓库根目录，行号以 commit `fe96caa` 为准。
**用法**：从"概念/公式"查实现位置，或从"文件"查每个文件里有什么。

## B.1 按概念查：公式 → 代码

| 概念 / 公式 | 实现位置 |
| :--- | :--- |
| 交互循环 `reset/step` | `envs/tictactoe.py:78-101` |
| 视角归一化 canonical 观测 | `envs/tictactoe.py:106-108` |
| 合法动作掩码 | `envs/tictactoe.py:110-112` |
| 稀疏终局奖励 + 非法兜底 −10 | `envs/tictactoe.py:83-101` |
| 环境克隆（供 MCTS 推演） | `envs/tictactoe.py:114-119` |
| 终局判定（Minimax/MCTS/AZ 共用） | `envs/tictactoe.py:43-58` |
| 对手托管折叠：双人 → 单智能体 MDP | `envs/wrappers.py:19-56` |
| 奖励视角翻转（对手赢 = 我输） | `envs/wrappers.py:49-50` |
| 统一智能体接口 `select_action/observe/update/end_episode` | `agents/base.py:39-62` |
| 转移数据结构 `Transition` | `agents/base.py:27-36` |
| ε-greedy（表格型，掩码 + 合法集随机） | `agents/tabular.py:43-51` |
| ε 逐回合乘法衰减 | `agents/tabular.py:53-54` |
| Q 表（defaultdict 全 0 初始化） | `agents/tabular.py:35` |
| MC first-visit：$Q \leftarrow Q + \alpha(G_t - Q)$ | `agents/tabular.py:83-93` |
| Q-Learning：$r + \gamma \max_{a'\in\text{合法}} Q(s',a')$ | `agents/tabular.py:105-113` |
| SARSA 滞后一拍：$r + \gamma Q(s',a')$ | `agents/tabular.py:130-149` |
| QNet MLP `9→64→64→9` | `agents/dqn.py:23-35` |
| ReplayBuffer（环形 10000） | `agents/dqn.py:38-50` |
| DQN ε-greedy（网络前向 + 掩码） | `agents/dqn.py:79-88` |
| DQN 目标值：target net + 掩码 + **NaN 陷阱规避** | `agents/dqn.py:110-118` |
| Target Network 同步（每 100 次更新） | `agents/dqn.py:127-129` |
| Actor-Critic 双头网络（Tanh 躯干） | `agents/ppo.py:26-45` |
| 非法动作 logits 屏蔽（−1e9） | `agents/ppo.py:42-45` |
| PPO 采样 + 缓存 `(obs, action, logp)` | `agents/ppo.py:76-89` |
| Rollout buffer 攒批（min_batch=64） | `agents/ppo.py:98-102` |
| GAE 倒序循环：$\delta_t$ 与 $A_t^{GAE}$ | `agents/ppo.py:113-127` |
| PPO-Clip：$L^{CLIP} = \min(rA, \mathrm{clip}(r)A)$ | `agents/ppo.py:140-144` |
| 熵正则与总损失 | `agents/ppo.py:138,149-154` |
| MCTS 节点统计量 $(N, W, Q)$ | `agents/mcts.py:22-33` |
| MCTS 四步循环（Selection/Expansion/Rollout/Backprop） | `agents/mcts.py:59-88` |
| UCT：$-Q(child) + c\sqrt{\ln N/n}$ | `agents/mcts.py:90-98` |
| Rollout 视角奇偶还原 | `agents/mcts.py:100-114` |
| 按访问次数选最终动作（稳健子节点） | `agents/mcts.py:54-56` |
| Policy-Value 网络 $(P, v)$（Tanh 价值头） | `agents/alphazero.py:27-49` |
| PUCT：$Q + c\,P\sqrt{N}/(1+n)$ | `agents/alphazero.py:148-157` |
| 叶节点网络估值替代 Rollout | `agents/alphazero.py:125-136` |
| 自博弈收集 $(s,\pi,z)$ 与 z 视角修正 | `agents/alphazero.py:162-182` |
| AZ 损失：$(z-v)^2 - \pi^\top\log p$ + L2 | `agents/alphazero.py:187-221` |
| 棋盘 D4 对称变换（8 种） | `utils/symmetry.py:21-43` |
| 固定对手训练主循环 | `trainers/fixed_opponent.py:22-65` |
| 自博弈 + 历史对手池（80/20） | `trainers/self_play.py:27-80` |
| AlphaZero 训练流（每 10 局更新一次网络） | `trainers/self_play.py:83-109` |
| 标准化评估（三档对手、先/后手各半、冻结探索） | `evaluation/evaluator.py:21-76` |
| Elo 更新公式（K=32，初始 1500） | `evaluation/arena.py:43-46` |
| 样本效率阈值（不败率 ≥95% 首次达标的局数/耗时） | `run_experiment.py:135-151` |
| `--viz` 策略视图（Q 值/概率/访问次数热力图） | `run_experiment.py:166-220` |

## B.2 按文件查：每个文件的地图

- **`envs/tictactoe.py`**（129 行）：`WIN_LINES`(25) → `check_winner`(32) →
  `state_result`(43) → `TicTacToeEnv`(61)：`reset`(78) / `step`(83) /
  `get_canonical_board`(106) / `get_valid_mask`(110) / `clone`(114) / `render`(121)。
- **`envs/wrappers.py`**（72 行）：`SingleAgentWrapper`(19)：`reset` 含执后手时对手先走(33) /
  `step` 含对手托管与奖励翻转(44) / `_opponent_step`(61)。
- **`agents/base.py`**（76 行）：`Transition`(27) / `BaseAgent`(39)：抽象 `select_action`(46)、
  可选钩子 `observe/update/end_episode`(54-62)、默认 pickle 存取(67-73)。
- **`agents/tabular.py`**（149 行）：`TabularAgent`(21) / `MonteCarloAgent`(70) /
  `QLearningAgent`(96) / `SarsaAgent`(119)。
- **`agents/dqn.py`**（151 行）：`QNet`(23) / `ReplayBuffer`(38) / `DQNAgent`(53)。
- **`agents/ppo.py`**（177 行）：`ActorCritic`(26) / `PPOAgent`(48)。
- **`agents/mcts.py`**（123 行）：`MCTSNode`(22) / `MCTSAgent`(36)。
- **`agents/alphazero.py`**（238 行）：`PolicyValueNet`(27) / `AZNode`(52) / `AlphaZeroAgent`(70)。
- **`agents/baselines.py`**（119 行）：`RandomAgent`(18) / `HeuristicAgent`(30) /
  `MinimaxAgent`(68，记忆化 Negamax)。
- **`trainers/fixed_opponent.py`**（65 行）：`train`(22)。
- **`trainers/self_play.py`**（109 行）：`train_self_play`(27) / `train_alpha_zero`(83)。
- **`evaluation/evaluator.py`**（85 行）：`play_game`(21) / `Evaluator`(43)。
- **`evaluation/arena.py`**（60 行）：`Arena.round_robin`(19) 与 `print_ranking`(55)。
- **`utils/symmetry.py`**（48 行）：`TRANSFORMS`(21) / `transform_state`(34) /
  `transform_pi`(40) / `all_symmetries_state_action`(46)。
- **`utils/logger.py`**（82 行）：`Logger.log`(25) / `series`(35) / `time_to_threshold`(46) /
  `plot`(54) / `save_csv`(75)。
- **`run_experiment.py`**（324 行）：`build_agent`(48) / `train_and_save`(92) /
  `viz_policy`(166) / `run_arena`(226) / `main`(249)。

## B.3 超参数位置（`configs/default.yaml`）

`tabular`（lr/gamma/epsilon 三项衰减）、`dqn`（buffer/batch/target_update_every）、
`ppo`（lam/clip/entropy/epochs/mini_batch）、`mcts`（n_simulations/c_uct）、
`alphazero`（n_simulations/c_puct/use_symmetry）、`training`（eval_every/
save_every/pool_size/pool_prob）、`eval`（三档局数）、`arena`（K/games_per_pair）。
