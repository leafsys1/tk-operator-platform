"""判据档机判入口（`test.sh` / `tools/run_cases.py` 用它判每条用例）。

口径 = 官方 `ops-precision-standard` 混合容差，**外加**非有限值的处理规则：

- 官方实现只对有限浮点做 atol/rtol 统计，遇到 NaN/Inf 会算成不通过
  （`abs(inf - inf) = nan` → 元素判不过）。规格 v1 的「数值边界」组包含 NaN/Inf 传播，
  所以这里**先把非有限位单独逐元素精确比对**（同号 inf / 同为 nan），
  再对**有限子集**跑官方混合容差。
- `max_abs_error` / `matched_ratio` 的统计范围 = 有限元素；非有限位的一致性写在
  `nonfinite` 段里，两者都进 `correctness.json`，不混在一起说「过了」。

这不是把标准放宽：非有限位是**精确判定**（比容差更严），只是不参与容差通过率统计。
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np

PROFILE = "ops-precision-standard"
PROFILE_VERSION = 1
PROFILE_REF = f"{PROFILE}@{PROFILE_VERSION}"

_HERE = Path(__file__).resolve().parent
_PROFILE_DIR = _HERE / f"{PROFILE}@{PROFILE_VERSION}"


def _load_official():
    """加载官方档实现（**逐字节副本**，见该目录 PROVENANCE.md）。"""
    path = _PROFILE_DIR / "mixed_tolerance_check.py"
    spec = importlib.util.spec_from_file_location("opx_criteria_mixed_tolerance", path)
    assert spec and spec.loader, f"判据档实现缺失：{path}"
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_official = _load_official()


def profile_ref() -> str:
    return PROFILE_REF


def check(actual: np.ndarray, golden: np.ndarray) -> dict:
    """判一条用例。actual/golden 必须是同 shape、同 dtype 的数组。

    返回 dict：`is_pass` + 官方指标 + `nonfinite` 段 + 使用的档值。
    """
    if actual.shape != golden.shape:
        raise ValueError(f"shape 不一致：actual {actual.shape} vs golden {golden.shape}")
    if actual.dtype != golden.dtype:
        raise ValueError(f"dtype 不一致：actual {actual.dtype} vs golden {golden.dtype}")

    a = actual.astype(np.float64, copy=False)
    g = golden.astype(np.float64, copy=False)
    a_finite = np.isfinite(a)
    g_finite = np.isfinite(g)
    # ① 非有限位：逐元素精确比对（NaN 对 NaN，inf 对同号 inf）
    both_nonfinite = (~a_finite) & (~g_finite)
    nan_match = np.isnan(a) & np.isnan(g)
    inf_match = np.isinf(a) & np.isinf(g) & (np.sign(a) == np.sign(g))
    nonfinite_ok = bool(np.array_equal(np.isnan(a), np.isnan(g))
                        and np.array_equal(np.isinf(a) & (a > 0), np.isinf(g) & (g > 0))
                        and np.array_equal(np.isinf(a) & (a < 0), np.isinf(g) & (g < 0)))
    nonfinite_count = int((~a_finite | ~g_finite).sum())
    # 有限性类别不一致（一边 inf 一边有限）就是错，单独记，不靠上面的整体比较遮住
    kind_mismatch = int((a_finite != g_finite).sum())

    # ② 有限子集：走官方混合容差。
    #    **必须喂原始 dtype 的数组**：官方实现是按 `npu_output.dtype` 查容差表的
    #    （喂 float64 会 ValueError: Unsupported dtype）——它内部自己会升到 float64 做误差计算。
    finite_mask = a_finite & g_finite
    official = _official.check_mixed_tolerance(actual[finite_mask], golden[finite_mask])

    is_pass = bool(official["is_pass"] and nonfinite_ok and kind_mismatch == 0)
    out = {
        "is_pass": is_pass,
        "profile": PROFILE_REF,
        "rtol": official["rtol"],
        "atol": official["atol"],
        "required_matched_ratio": official["required_matched_ratio"],
        "matched_ratio": official["matched_ratio"],
        "max_abs_error": official["max_abs_error"],
        "max_abs_error_limit": official["max_abs_error_limit"],
        "finite_elements": int(finite_mask.sum()),
        "nonfinite": {
            "count": nonfinite_count,
            "match": nonfinite_ok,
            "finite_kind_mismatch": kind_mismatch,
            "nan_pairs": int(nan_match.sum()),
            "inf_pairs": int(inf_match.sum()),
        },
    }
    reasons = []
    if not official["is_pass"]:
        reasons.extend(official.get("failure_reasons", ["官方混合容差未通过"]))
    if not nonfinite_ok:
        reasons.append("非有限位不一致（NaN/Inf 位置或符号不匹配）")
    if kind_mismatch:
        reasons.append(f"有限性类别不一致：{kind_mismatch} 个元素（一侧 inf 一侧有限）")
    out["failure_reasons"] = reasons
    return out


if __name__ == "__main__":  # 自检：档值能不能读出来
    a = np.array([1.0, np.inf, np.nan], dtype=np.float32)
    r = check(a, a.copy())
    print({"profile": r["profile"], "is_pass": r["is_pass"], "rtol": r["rtol"],
           "atol": r["atol"], "max_abs_error_limit": r["max_abs_error_limit"],
           "nonfinite": r["nonfinite"]})
