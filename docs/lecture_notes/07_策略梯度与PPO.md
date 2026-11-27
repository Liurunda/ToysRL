# 第 7 章　策略梯度与 PPO

> **本章目标**：理解"直接优化策略"与"先学价值再贪心"两条路线的差别，
> 掌握策略梯度定理（直观版）、Actor-Critic、GAE(λ) 与 PPO-Clip，
> 并逐行读懂 `agents/ppo.py`（含倒序 GAE 循环与裁剪目标的实现）。
> **对应代码**：`agents/ppo.py`（全文）。
> **前置知识**：[第 5](05_表格型方法.md)、[6 章](06_价值函数近似与DQN.md)的价值方法。

---

## 7.1 为什么直接优化策略

价值方法（第 5、6 章）的路线是"**先估准 Q，再对 Q 贪心**"。这条路有两个软肋：

1. **贪心动作的脆弱性**：Q 估计只要有一点误差，$\arg\max$ 就可能选出离谱的动作；
2. **表达不了随机策略**：有些场景（石头剪刀布、牌类博弈）最优策略本身就是随机的，
   对 Q 贪心永远学不出来。

策略梯度走另一条路：**把策略 $\pi_\theta(a|s)$ 参数化，沿"提高总回报"的方向直接
做梯度上升**，连价值函数都可以不要。目标：

$$J(\theta) = \mathbb{E}_{\tau \sim \pi_\theta}\big[ G_0 \big] \qquad \text{（从初始状态起的期望回报）}$$

## 7.2 策略梯度定理（直观版）

对 $J(\theta)$ 求梯度，关键技巧是"log 导数恒等式"：$\nabla_\theta \pi_\theta = \pi_\theta\, \nabla_\theta \log \pi_\theta$。
把对轨迹的期望换进来，整理后得到**策略梯度定理**：

$$\nabla_\theta J(\theta) = \mathbb{E}_{\pi_\theta}\Big[ \nabla_\theta \log \pi_\theta(a_t|s_t) \cdot G_t \Big]$$

**直觉**：把每个动作的概率"朝着回报高的方向"推——$G_t$ 大（这局赢了）就提高这条
轨迹上动作的概率，$G_t$ 小（输了）就压低。最简版本（REINFORCE）直接按上式更新。

两个几乎必加的改进：

- **Baseline（减基准）**：把 $G_t$ 换成 $G_t - b(s_t)$，期望不变但方差大降
  （赢一局全轨迹 +1、输一局全轨迹 −1，全正全负会无谓抖动）；
- **用价值网络当 baseline**：$b(s_t) = V_\phi(s_t)$，得到 **Actor-Critic**：
  Actor（策略）负责选动作，Critic（价值）负责打分。

## 7.3 GAE：优势函数的 λ 加权估计

Actor-Critic 里真正驱动策略的是**优势函数** $A_t = Q(s_t,a_t) - V(s_t)$——
"这一步比平均好多少"。GAE（Generalized Advantage Estimation）用 TD 误差的
指数加权和来估计它。先定义一步 TD 误差：

$$\delta_t = r_t + \gamma V(s_{t+1}) - V(s_t)$$

（终局时 $V(s_{t+1})$ 取 0。）然后：

$$A_t^{\text{GAE}} = \delta_t + (\gamma\lambda)\,\delta_{t+1} + (\gamma\lambda)^2\,\delta_{t+2} + \cdots$$

$\lambda \in [0,1]$ 是"看多远"的旋钮：$\lambda=0$ 时 $A_t = \delta_t$（一步 TD，低方差有偏），
$\lambda=1$ 时退化为 $G_t - V(s_t)$（整段回报，无偏高方差）。平台取 $\lambda=0.95$。

**代码对照**（`agents/ppo.py:113-127`，注意是**从后往前**的循环，正好复用滚动量）：

```python
values = self.net(obs)[1]                    # 当前网络重算 V(s)
last_next_value = (1.0 - dones[-1]) * self.net(next_obs[-1:])[1][0]
adv = torch.zeros_like(rewards)
next_adv, next_val = 0.0, last_next_value
for i in range(len(buffer) - 1, -1, -1):
    delta = rewards[i] + self.gamma * next_val * (1.0 - dones[i]) - values[i]
    adv_i  = delta + self.gamma * self.lam * next_adv * (1.0 - dones[i])
    adv[i] = adv_i
    next_adv, next_val = adv_i, values[i]
returns = adv + values                        # GAE 的 V 目标: A + V = Q 估计
adv = (adv - adv.mean()) / (adv.std() + 1e-8) # 标准化: 稳定后续梯度
```

倒序循环时 `next_adv` 恰好携带 $\delta_{t+1} + (\gamma\lambda)\delta_{t+2} + \cdots$，
乘上 $(\gamma\lambda)$ 再接上 $\delta_t$，一行代码实现整个 GAE 求和。
（实现注记：教材版 GAE 用**采集时**记录的 $V$ 计算，本平台用当前网络重算——属于简化实现，
在 buffer 很小、更新频繁时差异可忽略。）

## 7.4 PPO-Clip：让策略"小步快跑"

策略梯度最大的坑是**步长**：一步迈太大，新策略与旧策略差太远，旧数据立刻作废，
甚至崩掉（策略"跳崖"）。PPO 的核心是给更新加上"信任区域"，用**概率比裁剪**实现：

$$r_t(\theta) = \frac{\pi_\theta(a_t|s_t)}{\pi_{\theta_{old}}(a_t|s_t)}, \qquad
L^{CLIP}(\theta) = \mathbb{E}_t\Big[ \min\big( r_t(\theta) A_t,\; \operatorname{clip}(r_t(\theta),\, 1-\epsilon,\, 1+\epsilon)\, A_t \big) \Big]$$

- $r_t = 1$ 表示新旧策略对动作 $a_t$ 的打分没变；
- `clip` 把比值钳在 $[0.8, 1.2]$（$\epsilon=0.2$）内；
- 取 `min` 的语义：当优势为正（好动作）时，**禁止**比值超过 $1+\epsilon$（涨多了不给
  额外收益）；优势为负时，**禁止**比值低于 $1-\epsilon$（跌多了不给额外惩罚）。
  两端都被钳住，更新就不会过冲。

**代码对照**（`agents/ppo.py:135-154`）：

```python
logits = self.net.masked_logits(obs[idx], masks[idx])   # 屏蔽非法动作 (第 4 章)
dist = Categorical(logits=logits)
new_logps = dist.log_prob(actions[idx])
entropy = dist.entropy().mean()

ratio = torch.exp(new_logps - old_logps[idx])           # r_t(θ)
surr1 = ratio * adv[idx]
surr2 = torch.clamp(ratio, 1.0 - self.clip_epsilon, 1.0 + self.clip_epsilon) * adv[idx]
policy_loss = -torch.min(surr1, surr2).mean()           # -L^CLIP (梯度上升→损失取负)

value_pred = self.net(obs[idx])[1]
value_loss = F.mse_loss(value_pred, returns[idx])

loss = policy_loss + self.value_coef * value_loss - self.entropy_coef * entropy
self.optimizer.zero_grad(); loss.backward()
nn.utils.clip_grad_norm_(self.net.parameters(), self.max_grad_norm)  # 梯度裁剪
self.optimizer.step()
```

- `old_logps` 是**采集时**缓存的对数概率（下面讲流程），`new_logps` 用当前网络重算——
  比值就是新旧策略之比；
- 最后一项 $-\,c_e \cdot H(\pi)$ 是**熵正则**：鼓励动作分布不要过早"一根筋"，
  维持探索（[第 4 章](04_探索与利用.md) 4.2 节"更好的探索"）；
- 额外还有一层梯度范数裁剪（0.5），双保险。

## 7.5 完整数据流：rollout buffer 三拍循环

PPO 是 on-policy 方法，数据"过期即弃"，流程比 DQN 更讲究时序（`agents/ppo.py:76-102`）：

1. **`select_action`**（76-89）：算 masked logits → `Categorical` 采样动作（训练）
   / argmax（评估）→ 把 `(obs, action, logp, valid_mask)` 缓存进 `self._last`；
2. **`observe`**（91-95）：转移到达后，把缓存与 `(reward, done, next_obs)` 拼成一条
   `(obs, action, old_logp, reward, done, mask, next_obs)` 追加进 buffer；
3. **`update`**（98-102）：buffer 攒够 `min_batch=64` 条才触发 `_ppo_update`，
   并把 buffer **清空**（on-policy！旧数据不回收）。

`_ppo_update` 内部把 64 条样本按 `epochs=8` 轮 × `mini_batch=32` 的小批反复学习——
这是 PPO 独有的"有限复用"：数据只新鲜一次，但趁新鲜多嚼几口。

## 7.6 真实实验结果：小棋盘上的"水土不服"

vs Random 训练 10000 局（`results/ppo_fixed/summary.json`）：

| 算法 | 局数 | vs Random 胜率 | vs Minimax 不败率 | 耗时 |
| :--- | :--- | :--- | :--- | :--- |
| Q-Learning（表格） | 5000 | 0.82 | 0.52 | 2.8s |
| DQN | 10000 | 0.93 | 0.68 | 146s |
| PPO | 10000 | 0.80 | **0.16** | 51s |

PPO 对 Minimax 的不败率只有 0.16——明显弱于同期训练的价值方法。原因值得认真想：

1. **on-policy 样本利用率低**：buffer 用 8 遍即弃，10000 局的样本量对策略梯度而言偏少；
2. **探索弱**：PPO 的探索只是"按自己的 softmax 分布采样"，没有 ε-greedy 那种
   强制随机，早期分布没成形时几乎在盲走；
3. **对手太弱**：vs Random 训练，Critic 学到的是"对手随机时的价值"，GAE 的优势
   信号被对局噪声淹没。

**这本身是重要的一课**：算法没有绝对优劣，只有合适与否。PPO 的舞台是**状态/动作空间
巨大或连续、需要随机策略**的问题（机器人控制、大语言模型对齐 RLHF），在那里价值方法
的贪心路线根本走不通。在 5478 个状态、9 个离散动作的井字棋上，它是"杀鸡用牛刀"。

## 🧪 实验 7-1：PPO 训练与指标观察

```bash
cd tictactoe_rl_arena
python run_experiment.py --agent ppo --mode fixed --episodes 10000
```

1. 打开 `results/ppo_fixed/metrics.csv`，观察三列指标：`ppo_policy_loss`（策略损失）、
   `ppo_value_loss`（价值损失）、`ppo_entropy`（熵）。熵是否随训练持续下降
   （策略越来越确定）？前期熵是否明显更高（在探索）？
2. 调参练习：把 `configs/default.yaml` 的 `ppo.entropy_coef` 从 0.01 改到 0.1，
   观察探索增强后对 Minimax 不败率是否改善；
3. 对照[实验 6-1](06_价值函数近似与DQN.md) 的 DQN 曲线，体会"样本效率 vs 适用面"的取舍。

## 7.7 小结

- 策略梯度直接优化 $\pi_\theta$，公式核心是 $\nabla \log \pi \cdot A$；
- Actor-Critic 用 $V$ 当 baseline 降方差；GAE 用 $\lambda$ 加权 TD 误差估计优势，
  代码里一个倒序循环搞定；
- PPO-Clip 用概率比裁剪把更新钳在信任区域内，配合熵正则维持探索；
- on-policy 的代价是样本利用率低——小棋盘上 PPO 不如价值方法，大世界才是它的主场。

**下一章**：[第 8 章　搜索与规划：MCTS](08_搜索与规划MCTS.md)——不学习，纯"算"。
**练习**：见[附录 C](附录C_练习与思考题.md) 基础题 9、进阶题 5。
