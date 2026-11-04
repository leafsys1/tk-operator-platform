"""elementwise_sub 的**独立**参考实现（规格 oracle：CPU fp64 计算后一次舍入到 fp32）。

独立性说明：

- 候选实现走 Ascend C kernel（NPU fp32）；
- 本标杆**不碰 NPU、也不用 torch 计算**（torch 只在 tests/test_elementwise_sub.py 里做
  「接口语义对齐」的第二参照）。它在 CPU 上用 fp64 算 ``y = a - alpha * b``，再一次性舍入到 fp32。

alpha 语义：与 torch 一致 —— ``alpha * b`` 在 fp32 域先算并舍入（torch 的 kernel 实际是
``a - alpha*b``，其中 alpha*b 是 fp32 乘），之后再减。但为了给「按位精确」留出标杆口径，
本函数**两种都给**：

- ``compute_golden(a, b, alpha=1.0)``：默认口径 = fp64 全程计算 + 单次舍入（用于判据比对，
  也是 alpha=1 时唯一正确的结果）；
- ``compute_golden_rounding_chain(a, b, alpha)``：fp64 → 先舍入 alpha*b 再做减法，
  用来解释 alpha≠1 时 torch 与「单次舍入」可能存在 1 ulp 差异的情形。
"""

from __future__ import annotations

import numpy as np


def compute_golden(a: np.ndarray, b: np.ndarray, alpha: float = 1.0) -> np.ndarray:
    """返回与 a 同 dtype 的标杆输出：fp64 计算 ``a - alpha*b`` 后一次舍入。

    要求 a 与 b 同 shape（规格 v1：不广播）。NaN/Inf/±0 走 IEEE 754 语义；
    ``x - x = +0.0``，``0 - 0 = +0.0``，``-0 - 0 = -0.0``。
    """
    if a.shape != b.shape:
        raise ValueError(f"标杆要求同 shape：{a.shape} vs {b.shape}")
    if a.dtype != np.float32 or b.dtype != np.float32:
        raise ValueError(f"标杆只支持 float32：{a.dtype} / {b.dtype}")
    with np.errstate(over="ignore", invalid="ignore"):
        out = a.astype(np.float64) - float(alpha) * b.astype(np.float64)
        return out.astype(np.float32)


def compute_golden_rounding_chain(a: np.ndarray, b: np.ndarray, alpha: float) -> np.ndarray:
    """torch 风格的两次舍入：先 fp32 舍入 ``alpha*b``，再做 fp32 减法（同样用 fp64 承载中间值）。"""
    if a.shape != b.shape:
        raise ValueError(f"标杆要求同 shape：{a.shape} vs {b.shape}")
    with np.errstate(over="ignore", invalid="ignore"):
        m = (float(alpha) * b.astype(np.float64)).astype(np.float32)
        return (a.astype(np.float64) - m.astype(np.float64)).astype(np.float32)


if __name__ == "__main__":  # 自检
    x = np.array([1.5, np.inf, -np.inf, np.nan, np.finfo(np.float32).max], dtype=np.float32)
    y = np.array([1.0, np.inf, np.inf, 0.0, -np.finfo(np.float32).max], dtype=np.float32)
    print(compute_golden(x, y))
    print("−0 符号：", np.signbit(compute_golden(np.zeros(1, np.float32), np.zeros(1, np.float32))))
    print("alpha=2：", compute_golden(np.array([10.0], np.float32), np.array([3.0], np.float32), alpha=2.0))
