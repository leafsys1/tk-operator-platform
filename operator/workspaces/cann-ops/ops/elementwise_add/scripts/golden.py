"""elementwise_add 的**独立**参考实现（§4.5「与独立参考实现对比」）。

独立性说明（这条必须写清楚，否则「独立」两个字没有约束力）：

- 候选实现走的是 Ascend C kernel（NPU 上的 fp32 加法，硬件做一次增益舍入）；
- 本标杆**不碰 NPU**：在 CPU 上用 fp64 求和，再按输出 dtype 舍入一次 —— 即「正确舍入」的结果。

为什么不直接 `a + b`（numpy fp32）：那与候选实现是同一精度路径，等于拿 A 验 A。
fp64 求和 + 一次舍入给出的是这一档精度标准想逼近的标杆值，
混合容差（atol/rtol）就是用来吸收二者差异的。
"""

from __future__ import annotations

import numpy as np


def compute_golden(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """返回与 a 同 dtype 的标杆输出。要求 a 与 b 同 shape（规格 v1：不广播）。"""
    if a.shape != b.shape:
        raise ValueError(f"标杆要求同 shape：{a.shape} vs {b.shape}")
    with np.errstate(over="ignore", invalid="ignore"):
        return (a.astype(np.float64) + b.astype(np.float64)).astype(a.dtype)


if __name__ == "__main__":  # 自检
    x = np.array([1.5, np.inf, np.nan, np.finfo(np.float32).max], dtype=np.float32)
    print(compute_golden(x, np.full(x.shape, 1.0, dtype=np.float32)))
