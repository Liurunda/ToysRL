# 第 6 章　价值函数近似与 DQN

> **本章目标**：理解为什么需要函数近似、它带来什么新风险（死亡三角），
> 掌握 DQN 的两大稳定化技术（Replay Buffer、Target Network）与实现细节，
> 并读懂 `agents/dqn.py` 全文（含一个真实的 NaN 陷阱）。
> **对应代码**：`agents/dqn.py`（全文）。
> **前置知识**：[第 5 章](05_表格型方法.md)的 Q-Learning。

---

## 6.1 表格的极限：当状态数爆炸

井字棋只有 5478 个合法状态，表格法 2 秒训完。但把棋盘从 $3\times3$ 换成 $19\times19$
的围棋：状态数约 $3^{361} \approx 10^{172}$——宇宙中的原子总数（$\sim10^{80}$）在它
面前都是零头。任何"每状态存一行"的方法直接出局。

更根本的是：**表格法无法泛化**。它把"两个相邻状态很相似"这个事实完全丢弃——
井字棋里"我差一格就赢"与"我差两格就赢"在 Q 表里是互不相关的两行，每个都要从头攒经验。
而神经网络天然把相似的输入映射到相似的输出：**见一隅而知全局**。

## 6.2 函数近似：$Q(s,a;\theta)$ 与半梯度更新

把 Q 表换成带参数 $\theta$ 的网络 $Q(s,a;\theta)$。本平台的结构是 MLP
`9 → 64 → 64 → 9`（`agents/dqn.py:23-35`）：输入扁平 9 维 canonical 观测，
输出 9 个动作的 Q 值。

更新目标与 Q-Learning 相同，只是"查表"换成"梯度下降"：

$$L(\theta) = \mathbb{E}\Big[ \big( \underbrace{r + \gamma \max_{a'} Q(s',a';\theta^-)}_{\text{目标}} - Q(s,a;\theta) \big)^2 \Big]$$

注意目标里的 $\theta^-$（一会儿讲）；且**梯度不回传进目标**（"半梯度"方法）——
目标被当作常数。代码（`agents/dqn.py:120-125`）：

```python
q_values = self.qnet(obs).gather(1, actions.unsqueeze(1)).squeeze(1)
loss = F.mse_loss(q_values, targets)
self.optimizer.zero_grad(); loss.backward(); self.optimizer.step()
```

`gather` 只取"实际执行过的动作"的 Q 值来算损失——网络输出 9 个值，但我们只对
走过的那个动作有（TD 目标的）监督。

## 6.3 死亡三角：为什么"查表换成网络"会翻车

函数近似 + 自举（bootstrap）+ off-policy 三者的组合，在理论上是**可能发散**的——
被戏称为"**死亡三角（deadly triad）**"。直觉：

1. **函数近似**：更新一个 $(s,a)$ 会**波及邻居状态**的估值（泛化的副作用）；
2. **自举**：目标值本身由网络给出，误差会被循环放大；
3. **off-policy**：学的策略与行为的策略不同，高估（max 操作）与过时数据叠加。

三者共同作用下，一个局部的估计错误可能正反馈式地滚雪球，让 Q 值发散到天文数字。
DQN 的两大稳定化技术正是冲着这个来的。

## 6.4 DQN 两大稳定化技术

### ① Experience Replay（经验回放）

把每条转移 $(s,a,r,s',\text{done},\text{mask})$ 存进固定容量（10000）的环形缓冲
`ReplayBuffer`（`agents/dqn.py:38-50`），更新时**随机采样一个小批量**（batch_size=64）：

```python
class ReplayBuffer:
    def __init__(self, capacity=10000):
        self.buffer = deque(maxlen=capacity)   # 满了自动丢弃最旧经验
    def push(self, transition): self.buffer.append(transition)
    def sample(self, batch_size, rng):
        idx = rng.choice(len(self.buffer), size=batch_size, replace=False)
        return [self.buffer[i] for i in idx]
```

它同时解决两个问题：

- **打破样本相关性**：连续步的样本高度相关（同一局、相邻棋盘），SGD 假设样本独立，
  随机采样使梯度估计更接近 i.i.d.；
- **样本复用**：每条经验被反复学习多次，缓解"交互昂贵"问题。

### ② Target Network（目标网络）

维护两份网络：`qnet`（在线更新）与 `target_net`（只读，定期拷贝权重）。
计算 TD 目标用 `target_net`（`agents/dqn.py:113-118`），每 100 次更新同步一次
（`agents/dqn.py:127-129`）：

```python
self.n_updates += 1
if self.n_updates % self.target_update_every == 0:
    self.target_net.load_state_dict(self.qnet.state_dict())
```

为什么？如果目标也用 `qnet` 自己算，那么"追逐目标"的过程就是**追着自己的尾巴跑**：
Q 值一升高，目标立刻跟着升高，形成无休止的正反馈。固定目标网络等于把目标"钉住"
100 步，让回归问题暂时静止，训练稳定得多。

## 6.5 逐行精读：一个真实的 NaN 陷阱

`agents/dqn.py:110-118` 计算目标值的部分值得逐行看：

```python
with torch.no_grad():
    next_q = self.target_net(next_obs)
    next_q = next_q.masked_fill(next_masks == 0, float("-inf"))  # 非法动作 -inf
    max_next = next_q.max(dim=1).values
    targets = torch.where(dones.bool(), rewards,
                          rewards + self.gamma * max_next)
```

**陷阱**：终局转移的 `next_mask` 全为 0，`masked_fill` 之后整行都是 $-\infty$，
`max_next = -\infty$。初学者的写法常常是：

```python
targets = rewards + self.gamma * (1 - dones) * max_next   # ❌ 错!
```

当 `done=1` 时，$(1-\text{dones}) \times (-\infty) = 0 \times -\infty = \text{NaN}$——
**一个 NaN 就污染整个批次**，损失变 NaN、梯度变 NaN、网络权重变 NaN，且极难排查
（偶发、只在终局样本上触发）。正确做法是上面的 `torch.where`：`done` 时直接取
`rewards`，**根本不触碰** $-\infty$。

> 这是"公式对、代码错"的典型案例：数学上 $r + \gamma(1-\text{done})\max Q$ 没毛病，
> 数值计算里 $0\times(-\infty)$ 却是 NaN。读懂这类细节，正是读代码的意义。

完整的 `update` 流程（`agents/dqn.py:97-131`）：buffer 不满 64 就返回空 → 采样 →
组装张量 → 目标网络算 `max_next` → MSE 损失 → 反向传播 → 每 100 步同步目标网络。

## 6.6 真实实验结果：从查表到函数近似的跨越

vs Random 训练 10000 局（`results/dqn_fixed/summary.json`，对比表格型 5000 局）：

| 算法 | 局数 | vs Random 胜率 | vs Minimax 不败率 | 训练耗时 |
| :--- | :--- | :--- | :--- | :--- |
| Q-Learning（表格） | 5000 | 0.82 | 0.52 | 2.8s |
| DQN（网络） | 10000 | **0.93** | **0.68** | 146s |

两个教学点：

1. **样本效率差异**：DQN 用了两倍的局数和 50 倍的耗时，才达到略好的水平——
   神经网络前期收敛慢是函数近似的固有代价。但它**从不枚举状态**：
   给 DQN 的输入只是一个 9 维向量，换成 $19\times19$ 围棋它照样能跑
   （表格法则直接内存爆炸）——这是"慢"买来的"可扩展性"。
2. **不败率 0.68 而非 1.0**：即便 10000 局，DQN 仍未完全求解井字棋。
   深层原因包括死亡三角的不稳定性、稀疏奖励下 TD 信号稀薄、以及对手只是 Random
   （学不到防守绝活）。自博弈版更惨（`results/dqn_selfplay`：2000 局，不败率仅 0.06）
   ——非平稳环境对价值方法尤其苛刻，[第 10 章](10_训练范式与评估.md)细讲。

## 🧪 实验 6-1：DQN 曲线与调参

```bash
cd tictactoe_rl_arena
python run_experiment.py --agent dqn --mode fixed --episodes 10000
```

1. 打开 `results/dqn_fixed/learning_curves.png`，对比[第 5 章实验](05_表格型方法.md)的
   Q-Learning 曲线：DQN 的 `notlose_vs_minimax` 是否更晚起步、爬得更抖？
2. 调参练习：把 `configs/default.yaml` 里 `dqn.target_update_every` 改为 10 与 1000
   各跑一次。观察：同步太快（追尾巴）曲线震荡，同步太慢（目标过时）收敛停滞。
3. 进阶验证 NaN 陷阱：把 `agents/dqn.py:117` 的 `torch.where` 改成
   `rewards + self.gamma * (1 - dones) * max_next`，训练几局观察 loss 变 NaN。

## 6.7 小结

- 状态爆炸 → 函数近似；泛化是神经网络的本质优势；
- 函数近似 + 自举 + off-policy = 死亡三角，DQN 用 Replay Buffer 与 Target Network 镇压；
- 数值细节（$0 \times -\infty$ = NaN）提醒我们：公式正确 ≠ 代码正确；
- 表格 → 网络是"用样本效率换可扩展性"的第一笔交易。

**下一章**：[第 7 章　策略梯度与 PPO](07_策略梯度与PPO.md)——不估 Q 值，直接优化策略。
**练习**：见[附录 C](附录C_练习与思考题.md) 基础题 8、进阶题 4。
