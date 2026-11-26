"""训练曲线记录与 Matplotlib 绘图工具。

Logger 以 append 方式记录两类指标:
  - 按"训练对局数"对齐的指标 (episode=...): 评估胜率、回合结果等;
  - 按"环境步数"对齐的指标 (step=...): DQN loss 等。
绘图时以 series() 提取同 x 轴的序列, 二者互不干扰。
"""
import csv
import os
from typing import List, Optional

import matplotlib
matplotlib.use("Agg")  # 无显示环境 (WSL/服务器) 下使用非交互后端
import matplotlib.pyplot as plt


class Logger:
    def __init__(self, run_dir: Optional[str] = None):
        self.run_dir = run_dir
        self.history: List[dict] = []
        if run_dir:
            os.makedirs(run_dir, exist_ok=True)

    # ------------------------------------------------------------------ #
    def log(self, episode: Optional[int] = None, step: Optional[int] = None,
            **metrics) -> None:
        entry = dict(metrics)
        if episode is not None:
            entry["episode"] = episode
        if step is not None:
            entry["step"] = step
        self.history.append(entry)

    # ------------------------------------------------------------------ #
    def series(self, key: str, x: str = "episode"):
        """提取 (x, y) 序列, 例如 series('win_vs_heuristic')。"""
        return [(e[x], e[key]) for e in self.history if key in e and x in e]

    def latest(self, key: str):
        """最近一次记录的指标值。"""
        for e in reversed(self.history):
            if key in e:
                return e[key]
        return None

    def time_to_threshold(self, key: str, threshold: float, x: str = "episode"):
        """样本效率指标: 首次达到 threshold 所需的 x (对局数/步数), 未达到返回 None。"""
        for e in self.history:
            if key in e and e[key] >= threshold:
                return e[x]
        return None

    # ------------------------------------------------------------------ #
    def plot(self, keys, title: str, path: str, x: str = "episode",
             ylabel: str = "rate", ylim=(0, 1.05), threshold_lines=None) -> None:
        """绘制学习曲线: 每个 key 一条曲线。threshold_lines: {key: (thr, label)}。"""
        plt.figure(figsize=(8, 5))
        for key in keys:
            pts = self.series(key, x)
            if pts:
                xs, ys = zip(*pts)
                plt.plot(xs, ys, marker="o", markersize=2.5, label=key)
        for thr, label in (threshold_lines or {}).values():
            plt.axhline(thr, ls="--", color="gray", alpha=0.6)
        plt.xlabel(x)
        plt.ylabel(ylabel)
        plt.ylim(*ylim)
        plt.title(title)
        plt.legend()
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(path, dpi=120)
        plt.close()

    def save_csv(self, path: str) -> None:
        if not self.history:
            return
        keys = sorted({k for e in self.history for k in e})
        with open(path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=keys)
            writer.writeheader()
            writer.writerows(self.history)
