"""bid_elementwise_sub_20261101 的独立参考实现（Golden）。

权威参考 = CPU ``torch.sub(x1.cpu(), x2.cpu())``（规格 v3.0 oracle，逐字一致）。
numpy fp64 单次舍入（``compute_golden_fp64``）仅作辅助交叉核对，**不列为判据、不作权威**。

- 候选实现走 Ascend C kernel（NPU fp32），由算子实现节点交付；
- 本 golden 不碰 NPU、不经候选 kernel，只在 CPU 上用 torch 算 ``y = x1 - x2``；
- 四动作 test 门禁（``tools/run_cases.py``）与 pytest 交付件共同复用同一入口
  ``compute_golden(a: np.ndarray, b: np.ndarray) -> np.ndarray``（numpy float32 入/出）。

注意：``tools/run_cases.py`` 以 ``golden_mod.compute_golden(a, b)`` 两参调用本函数，
入参为 ``tools/gen_inputs.py`` 生成的 numpy float32 数组，返回值必须是 numpy float32 数组。
"""

from __future__ import annotations

import numpy as np
import torch


def compute_golden(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """权威 golden：CPU ``torch.sub`` 逐元素减法，返回与输入同 dtype 的 numpy float32 数组。

    要求 ``a`` 与 ``b`` 同 shape（规格：不广播）且均为 float32。NaN/Inf/±0 全部遵循
    CPU torch 的 IEEE 754 语义（``inf-inf=nan``、``x-nan=nan``、``±0`` 相减按符号）。
    """
    if a.shape != b.shape:
        raise ValueError(f"标杆要求同 shape：{a.shape} vs {b.shape}")
    if a.dtype != np.float32 or b.dtype != np.float32:
        raise ValueError(f"标杆只支持 float32：{a.dtype} / {b.dtype}")
    # ascontiguousarray + 明确 float32 保证可写的 C 连续数组，torch.from_numpy 直接共享内存
    ta = torch.from_numpy(np.ascontiguousarray(a, dtype=np.float32))
    tb = torch.from_numpy(np.ascontiguousarray(b, dtype=np.float32))
    return torch.sub(ta, tb).numpy()


def compute_golden_fp64(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """辅助交叉核对（**非权威**）：numpy fp64 全程计算后单次舍入回 float32。

    仅用于观察与 ``compute_golden`` 的一致性，两者不一致时一律以 ``compute_golden``（torch.sub）为准，
    本函数**不得**作为判据权威。
    """
    if a.shape != b.shape:
        raise ValueError(f"辅助核对要求同 shape：{a.shape} vs {b.shape}")
    if a.dtype != np.float32 or b.dtype != np.float32:
        raise ValueError(f"辅助核对只支持 float32：{a.dtype} / {b.dtype}")
    with np.errstate(over="ignore", invalid="ignore"):
        return (a.astype(np.float64) - b.astype(np.float64)).astype(np.float32)


if __name__ == "__main__":  # 自检（需 torch；本节点静态范围不执行，仅作人工/真机辅助）
    x = np.array([1.5, np.inf, -np.inf, np.nan, np.finfo(np.float32).max], dtype=np.float32)
    y = np.array([1.0, np.inf, np.inf, 0.0, -np.finfo(np.float32).max], dtype=np.float32)
    print("torch.sub golden:", compute_golden(x, y))
    print("fp64 aux:        ", compute_golden_fp64(x, y))