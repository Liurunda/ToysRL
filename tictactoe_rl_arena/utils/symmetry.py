"""棋盘对称性数据增强 (旋转 / 镜像, 共 8 个等价变换)。

3×3 棋盘 (行优先索引 0~8) 的二面体群 D4:
    恒等、旋转 90°/180°/270°、水平/垂直镜像、主/副对角线镜像。
TRANSFORMS[k] 为长度 9 的置换, 含义: 旧位置 old 的棋子/概率映射到
新位置 TRANSFORMS[k][old]。
"""
import numpy as np


def _perm(fn):
    """把坐标映射 (i, j) -> (i', j') 转为行优先索引置换 old -> new。"""
    p = [0] * 9
    for idx in range(9):
        i, j = divmod(idx, 3)
        ni, nj = fn(i, j)
        p[idx] = ni * 3 + nj
    return np.array(p, dtype=int)


TRANSFORMS = [
    _perm(lambda i, j: (i, j)),          # 0: 恒等
    _perm(lambda i, j: (j, 2 - i)),      # 1: 旋转 90°
    _perm(lambda i, j: (2 - i, 2 - j)),  # 2: 旋转 180°
    _perm(lambda i, j: (2 - j, i)),      # 3: 旋转 270°
    _perm(lambda i, j: (i, 2 - j)),      # 4: 水平镜像
    _perm(lambda i, j: (2 - i, j)),      # 5: 垂直镜像
    _perm(lambda i, j: (j, i)),          # 6: 主对角线镜像
    _perm(lambda i, j: (2 - j, 2 - i)),  # 7: 副对角线镜像
]
_INV = [np.argsort(p) for p in TRANSFORMS]  # new -> old 的逆置换


def transform_state(state, k: int) -> tuple:
    """对 canonical 状态 (tuple) 应用第 k 个对称变换。"""
    state = np.asarray(state)
    return tuple(state[_INV[k]])


def transform_pi(pi, k: int) -> np.ndarray:
    """对动作概率分布 (长度 9) 应用第 k 个对称变换。"""
    pi = np.asarray(pi)
    return pi[_INV[k]]


def all_symmetries_state_action(state, action: int):
    """返回 (state, action) 的全部 8 个等价对, 可用于 Q 表 / 经验增强。"""
    return [(transform_state(state, k), int(TRANSFORMS[k][action])) for k in range(8)]
