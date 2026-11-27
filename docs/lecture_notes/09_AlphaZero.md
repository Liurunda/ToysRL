# 第 9 章　AlphaZero：搜索与学习的结合

> **本章目标**：理解 Policy-Value 网络与 PUCT 如何升级纯 MCTS；
> 掌握"搜索产生监督信号 → 训练网络 → 更强的搜索"的策略迭代闭环，
> 并逐行读懂 `agents/alphazero.py`（含自博弈样本 $(s,\pi,z)$ 的视角修正）。
> **对应代码**：`agents/alphazero.py`、`utils/symmetry.py`、`trainers/self_play.py:83-109`。
> **前置知识**：[第 7 章](07_策略梯度与PPO.md)的 Actor-Critic、[第 8 章](08_搜索与规划MCTS.md)的 MCTS。

---

## 9.1 纯 MCTS 的两个浪费

第 8 章的 MCTS 每步模拟 2000 次才能接近完美，但这两千次里有大量浪费：

1. **随机 Rollout 没有智能**：模拟里双方乱下，得到的终局结果噪声巨大
   （在 3×3 棋盘上还勉强有用，棋盘一大基本是纯噪声）；
2. **知识不积累**：上一步搜了 2000 次，下一步从零再来 2000 次——
   同一局棋里，几乎相同的局面被反复重新搜索，什么都不记得。

AlphaZero 的思路直击这两点：**用神经网络给搜索"先验"和"直觉"**——
Policy 头告诉搜索"先看哪些分支"，Value 头替代随机 Rollout 直接给局面估值；
反过来，**搜索的结果又反过来训练这个网络**。搜索与学习互相喂养，形成闭环。

## 9.2 Policy-Value 网络：$(P, v)$ 双头

网络结构（`agents/alphazero.py:27-49`）与 PPO 的 Actor-Critic 神似，但语义不同：

```python
class PolicyValueNet(nn.Module):
    def __init__(self, obs_dim=9, hidden=64, n_actions=9):
        self.trunk = nn.Sequential(nn.Linear(obs_dim, hidden), nn.Tanh(),
                                   nn.Linear(hidden, hidden), nn.Tanh())
        self.policy = nn.Linear(hidden, n_actions)   # 策略头: 动作 logits
        self.value  = nn.Linear(hidden, 1)           # 价值头: Tanh 标量
    def forward(self, x):
        h = self.trunk(x)
        return self.policy(h), torch.tanh(self.value(h)).squeeze(-1)
```

- **Policy 头**输出动作先验 $P(a|s)$（Softmax 后是概率分布）——"直觉：这步棋通常
  值得先看看"；
- **Value 头**输出状态价值 $v \in [-1, 1]$（Tanh 压到终局奖励的范围内）——
  "直觉：这个局面大概谁优"。

两者都从**同一个棋盘向量**读出来——共享躯干意味着"会下棋"和"会判断局面"被绑在一起
学，互相借力。

## 9.3 PUCT：先验引导的树搜索

第 8 章的 UCT 探索项对所有子节点一视同仁（只按访问次数区分）。PUCT 把网络先验
$P(a|s)$ 乘进探索项（`agents/alphazero.py:148-157`）：

$$a^* = \arg\max_a \Big[ Q(s,a) + c_{puct}\, P(a|s)\, \frac{\sqrt{N(s)}}{1 + N(s,a)} \Big]$$

代码里 $Q(s,a)$ 用子节点统计量表示（子节点的 $W$ 是对手视角，取负翻回）：

```python
q = -child.W / (child.N_total + 1e-8)          # 该边平均价值 (本节点视角)
u = self.c_puct * node.P[a] * sqrt_n / (1.0 + child.N_total)
```

含义的变化是根本性的：

- $P(a|s)$ 大 → 探索加成大 → **搜索把算力集中在"直觉看好"的分支**上；
- 网络越准，搜索越高效；搜索越高效，产出的数据越好——这就是闭环的正反馈。

而叶节点的处理也变了（`agents/alphazero.py:125-136`）：不再随机 Rollout 到底，
而是**直接调用网络估值** $v$ 并展开：

```python
p, v = self.net.predict(node.state, self.device)   # 网络替代 Rollout
mask = np.zeros(9); mask[list(node.valid)] = 1.0
node.P = p * mask
node.P /= node.P.sum() + 1e-12                     # 合法动作上重归一化
node.expanded = True
for a in node.valid:
    child_state, child_valid = self._apply(node.state, a)
    node.children[a] = AZNode(child_state, child_valid, action=a)
value = float(v)
```

## 9.4 自博弈闭环：$(s, \pi, z)$ 从哪来

AlphaZero 的训练数据**不需要任何外部对手**——自己跟自己下（`trainers/self_play.py:83-109`）。
一局自博弈里，每个决策点 $s_i$ 产出一条训练样本 $(s_i, \pi_i, z_i)$：

- $\pi_i$：MCTS 在 $s_i$ 跑完 $n_{sim}$ 次模拟后的**访问分布**（`alphazero.py:171-172`：

  ```python
  root = self._search(state, valid)
  pi = root.N / root.N.sum()       # 访问分布 = 搜索的"结论"
  ```

  这是**比硬选一个动作强得多的监督信号**——整张概率分布刻画了各分支的相对优劣；
- $z_i$：本局**真实终局结果**（胜 +1 / 平 0），但要做**视角修正**（`alphazero.py:179-182`）：

  ```python
  z = float(r)                       # 最后落子方视角的结果
  m = len(states) - 1
  return [(s, pis[i], z if (m - i) % 2 == 0 else -z) for i, s in enumerate(states)]
  ```

  $s_i$ 的玩家与最后落子方相隔 $m-i$ 步，每步视角翻转一次，所以相隔偶数步同视角、
  奇数步取反——又是[第 8 章](08_搜索与规划MCTS.md)那个视角翻转，换个马甲再来一次。

训练时从 buffer 采样一个 batch，做 **8 重对称增强**（`utils/symmetry.py`：D4 群旋转/
镜像共 8 个等价变换，棋盘与 $\pi$ 用同一个置换变换），然后最小化
（`agents/alphazero.py:187-221`）：

$$\mathcal{L} = \underbrace{(z - v)^2}_{\text{价值回归}} + \underbrace{-\,\pi^\top \log p}_{\text{策略交叉熵}}$$

外加 L2 正则（实现上由 Adam 的 `weight_decay=1e-4` 承担）。注意这是**纯监督学习**：
目标 $z$ 是真实对局结果（无偏！）、目标 $\pi$ 是搜索算出的强标签（高质量！）。
没有 bootstrap、没有 off-policy 校正——第 6 章"死亡三角"的三大病因一个都不沾。
这就是 AlphaZero 训练稳定、样本效率惊人的根本原因。

> 至此可以看清整个闭环：**MCTS + 网络 → 更强对局 → 更强标签 $(z, \pi)$ → 网络变强 →
> 更强的 MCTS**。这正是经典**策略迭代**（评估-改进循环）的神经网络版。

## 9.5 真实实验结果：样本效率之王

自博弈 3000 局（`results/alphazero_selfplay/summary.json`）：

| 算法 | 训练量 | vs Random 胜率 | vs Minimax 不败率 | 达不败率≥95% 所需 |
| :--- | :--- | :--- | :--- | :--- |
| DQN（vs Random） | 10000 局 | 0.93 | 0.68 | 未达标 |
| PPO（vs Random） | 10000 局 | 0.80 | 0.16 | 未达标 |
| MCTS | **0 局** | 0.92 | 1.00 | 0 局（零训练） |
| AlphaZero-Lite | **3000 局** | 0.91 | **1.00** | **2000 局 / 392s** |

AlphaZero 用 2000 局自博弈（392 秒）达到理论最优，而 DQN/PPO 训练 10000 局都摸不到边。
同时它对 Heuristic/Minimax 的胜率是 0.0——全是平局：不犯错，也不给对手机会，
正是完美棋力的样子（[第 10 章](10_训练范式与评估.md)展开）。

对照 MCTS 看更耐人寻味：**棋力上两者都是 1.00，但成本结构完全不同**。
MCTS 每步现场烧 2000 次模拟（评估 250 局花了 79s）；AlphaZero 把搜索蒸馏进网络后，
**对弈时只需 200 次模拟**（`alphazero.n_simulations=200`，是纯 MCTS 的 1/10）——
训练期的 392s 换来了部署期 10 倍的算力折扣。**学习，就是把搜索的算力"预付"掉**。

（本平台是 AlphaZero-Lite：没有 Dirichlet 探索噪声、每步重建搜索树、网络只有 2 层——
完整版 AlphaZero 在围棋上的那套还多些工程细节，但本章讲的闭环已一字不差。）

## 🧪 实验 9-1：亲眼看策略迭代闭环

```bash
cd tictactoe_rl_arena
python run_experiment.py --agent alphazero --mode selfplay --episodes 3000
```

1. 打开 `results/alphazero_selfplay/learning_curves.png`：`notlose_vs_minimax` 是否
   在几百局内就冲到高位？对照 DQN/PPO 的曲线体会样本效率差距；
2. 看 `metrics.csv` 的 `az_policy_loss` 与 `az_value_loss`：它们随训练下降
   （网络越来越会模仿搜索、越来越会判断局面）；
3. 运行 `python run_experiment.py --agent alphazero --viz`，与[实验 8-1](08_搜索与规划MCTS.md)
   的 MCTS 热力图对比：两者访问分布是否都集中在中心/角落？
4. 变体实验：把 `configs/default.yaml` 里 `alphazero.use_symmetry` 改为 `false`
   重训，观察样本效率下降多少——8 倍数据增强的威力。

## 9.6 小结

- Policy-Value 网络给搜索提供先验 $P$ 与叶节点估值 $v$，替代随机 Rollout；
- PUCT 用 $P$ 引导探索，把算力聚焦在高潜分支；
- 自博弈产生 $(s,\pi,z)$：$\pi$ 是搜索的访问分布（强标签），$z$ 是真实终局结果
  （无偏标签）——训练是稳定的监督学习，绕开了死亡三角；
- 闭环本质是策略迭代：搜索改进网络，网络加速搜索；
- "学习 = 把搜索算力预付"：AZ 对弈时仅需纯 MCTS 十分之一的模拟次数。

**下一章**：[第 10 章　训练范式与评估](10_训练范式与评估.md)——怎样公平地比出谁强谁弱。
**练习**：见[附录 C](附录C_练习与思考题.md) 基础题 11、进阶题 7。
