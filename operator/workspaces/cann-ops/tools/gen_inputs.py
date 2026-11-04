"""输入数据生成：把用例矩阵里的 `gen_a` / `gen_b` 变成确定性数组。

与用例里的 `seed` 绑定 —— 同一个 (dist, 参数, seed, n) 永远得到同一份输入，
这是「可复现」的最低要求（§4.5：每用例记输入生成方式与随机种子）。
"""

from __future__ import annotations

import numpy as np

#: 特殊值组的头部固定顺序（用例里用 `leading` 声明想要哪些）
SPECIAL_VALUES = {
    "nan": np.nan,
    "inf": np.inf,
    "-inf": -np.inf,
    "float32_max": np.finfo(np.float32).max,
    "float32_min_subnormal": np.finfo(np.float32).tiny * np.finfo(np.float32).eps,
}


def generate(spec: dict, n: int, seed: int, dtype: str = "float32") -> np.ndarray:
    """按一条用例的 gen 声明生成 n 个元素。

    spec 为 None 用于拒绝类用例（gen_a/gen_b 写成 null）：它们只关心拒绝码，
    输入内容不影响判定，这里给确定性占位数据，避免 AttributeError 让整条用例 ERROR。
    """
    if spec is None:
        return np.random.default_rng(seed).uniform(-1.0, 1.0, n).astype(dtype)
    rng = np.random.default_rng(seed)
    dist = spec.get("dist", "uniform")

    if dist == "uniform":
        return rng.uniform(spec.get("low", -1.0), spec.get("high", 1.0), n).astype(dtype)
    if dist == "normal":
        return rng.normal(spec.get("mean", 0.0), spec.get("std", 1.0), n).astype(dtype)
    if dist == "const":
        return np.full(n, spec.get("value", 0.0), dtype=dtype)
    if dist == "zeros":
        return np.zeros(n, dtype=dtype)
    if dist == "special":
        values = [SPECIAL_VALUES[k] for k in spec.get("leading", [])]
        if len(values) > n:
            values = values[:n]
        head = np.array(values, dtype=dtype)
        tail = rng.normal(spec.get("mean", 0.0), spec.get("std", 1.0), n - len(values)).astype(dtype)
        return np.concatenate([head, tail])
    raise ValueError(f"不支持的 gen.dist：{dist}")


def describe(spec) -> str:
    """给人看的输入生成方式（进报告与证据）。

    拒绝类用例（形状/参数/未知键等）的 gen_a/gen_b 常写成 null —— 它们不生成输入，
    这里返回占位符而不是抛异常（旧实现在 spec 为 None 时 AttributeError，整轮 test 中断）。
    """
    if spec is None:
        return "(none)"
    dist = spec.get("dist", "uniform")
    if dist == "uniform":
        return f"uniform({spec.get('low')},{spec.get('high')})"
    if dist == "normal":
        return f"normal({spec.get('mean')},{spec.get('std')})"
    if dist == "const":
        return f"const({spec.get('value')})"
    if dist == "special":
        return "special(" + ",".join(spec.get("leading", [])) + ")"
    return dist
