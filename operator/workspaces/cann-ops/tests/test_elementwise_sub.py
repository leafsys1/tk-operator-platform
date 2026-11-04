"""elementwise_sub · pytest 交付件（真机运行，读 cases/<op>/cases.json）。

候选走 Python 接口 ``ops/elementwise_sub/python/elementwise_sub.py``（ctypes + 候选 kernel），
参考走 ``ops/elementwise_sub/scripts/golden.py``（CPU fp64，不碰 NPU），判定口径与
``criteria/check.py``（``ops-precision-standard@1``）一致。门禁权威仍是共享 runner（test 动作）。

判据补充（规格要求「按位精确」，本脚本额外报告）：
- ``bit_exact``：与 golden 逐位一致的元素数（NaN 按 payload 一致才算逐位相同）；
- ``torch_diff``：与同卡 ``torch.sub`` 的 fp32 结果做对照（torch 是**第二参照**，不是标杆）。
alpha≠1 时同时给出两种 golden 口径的差异数：
  (A) fp64 全程计算后一次舍入 —— 判据口径；
  (B) 先舍入 ``alpha*b`` 再减 —— torch 的舍入点。
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

WS = Path(__file__).resolve().parents[1]
if str(WS) not in sys.path:
    sys.path.insert(0, str(WS))

np = pytest.importorskip("numpy")
torch = pytest.importorskip("torch")
pytest.importorskip("torch_npu")

from criteria.check import check as criteria_check  # noqa: E402
from ops.elementwise_sub.python.elementwise_sub import sub  # noqa: E402
from ops.elementwise_sub.scripts.golden import (  # noqa: E402
    compute_golden,
    compute_golden_rounding_chain,
)
from tools import gen_inputs  # noqa: E402

CASES = json.loads((WS / "cases/elementwise_sub/cases.json").read_text(encoding="utf-8"))
MATCH = [c for c in CASES["cases"] if c.get("required", True) and c.get("expected") == "match"]
REJECT = [c for c in CASES["cases"] if c.get("required", True) and c.get("expected") == "reject"]

# 规格「附加正向参数」用例的 alpha 值（cases.json 的 note 里写明；判据参数不额外引入新用例）
# argv 级拒绝用例（参数缺失 / 未知参数）：Python 接口不接收 argv，这两条只能由可执行文件驱动
EXE = WS / "ops/elementwise_sub/build/elementwise_sub"
EXE_ARGV_CASES = {"sub_rej_006", "sub_rej_007"}

ALPHA_BY_CASE = {
    "sub_alpha_001": 2.0,
    "sub_alpha_002": 0.5,
    "sub_alpha_003": -1.0,
    "sub_alpha_004": 2.0,
}


def _gen(case: dict, key: str, seed_offset: int) -> "np.ndarray":
    n = int(case["n_a"]) if key == "a" else int(case["n_b"])
    return gen_inputs.generate(case.get(f"gen_{key}", {}), n, int(case["seed"]) + seed_offset,
                               case.get("dtype", "float32"))


def _bit_equal(got: "np.ndarray", want: "np.ndarray") -> int:
    """逐位相等的元素数（NaN 也按位比较，不把 NaN==NaN 当相等）。"""
    return int((got.view(np.uint32) == want.view(np.uint32)).sum())


def _run_case(case: dict) -> "np.ndarray":
    a = _gen(case, "a", 0)
    b = _gen(case, "b", 1)
    alpha = ALPHA_BY_CASE.get(case["case_id"], 1)
    ta = torch.from_numpy(a.copy()).npu()
    tb = torch.from_numpy(b.copy()).npu()

    if case["case_id"] == "sub_out_001":
        # out= 预填充非零的独立缓冲，结果须完全覆盖
        out = torch.full((a.size,), 7.5, dtype=torch.float32).npu()
        y = sub(ta, tb, out=out)
        assert y.data_ptr() == out.data_ptr(), "out= 必须复用调用方缓冲"
    elif case["case_id"] == "sub_out_002":
        # out=a：原地 a-b
        y = sub(ta, tb, out=ta)
        assert y.data_ptr() == ta.data_ptr(), "out=a 必须是原地"
    else:
        y = sub(ta, tb, alpha=alpha)
    return y.cpu().numpy()


@pytest.mark.parametrize("case", MATCH, ids=[c["case_id"] for c in MATCH])
def test_match(case: dict) -> None:
    a = _gen(case, "a", 0)
    b = _gen(case, "b", 1)
    alpha = ALPHA_BY_CASE.get(case["case_id"], 1)
    want = compute_golden(a, b, alpha=alpha)
    got = _run_case(case)

    verdict = criteria_check(got, want)
    bit = _bit_equal(got, want)
    with np.errstate(over="ignore", invalid="ignore"):
        torch_ref = torch.sub(torch.from_numpy(a.copy()), torch.from_numpy(b.copy()),
                              alpha=alpha).numpy()
    torch_diff = int((got.view(np.uint32) != torch_ref.view(np.uint32)).sum())
    report = (f"{case['case_id']}: bit_exact {bit}/{got.size}, "
              f"torch_diff {torch_diff}, matched_ratio {verdict.get('matched_ratio'):.6f}")
    if case["case_id"] in ALPHA_BY_CASE:
        chain = compute_golden_rounding_chain(a, b, alpha)
        report += f", rounding_chain_A_vs_B {_bit_equal(want, chain)}"
    print(report)
    assert verdict["is_pass"], f"{case['case_id']}: {verdict.get('failure_reasons')}；{report}"


def _check_exe_argv(case: dict) -> None:
    """用可执行文件验证 argv 级拒绝（与四动作同一入口；期望 stdout 含拒绝码且 rc=2）。"""
    assert EXE.is_file(), f"缺少可执行文件：{EXE}（先跑 build）"
    base = ["--n-a", str(case["n_a"]), "--n-b", str(case["n_b"]),
            "--dtype", case.get("dtype", "float32"),
            "--in-a", "a.bin", "--in-b", "b.bin", "--out", "y.bin"]
    if case["case_id"] == "sub_rej_006":       # 缺 --in-b
        argv = ["--n-a", str(case["n_a"]), "--n-b", str(case["n_b"]),
                "--dtype", case.get("dtype", "float32"), "--in-a", "a.bin", "--out", "y.bin"]
    else:                                      # 未知参数 --foo
        argv = base + ["--foo", "1"]
    proc = subprocess.run([str(EXE)] + argv, capture_output=True, text=True)
    out = (proc.stdout or "") + (proc.stderr or "")
    code = case.get("expect_reject_code")
    assert code in out and proc.returncode == 2, \
        f"{case['case_id']} 期望 {code}（rc=2），实际 rc={proc.returncode}：{out.strip()[:300]}"
    print(f"{case['case_id']}: exe rc=2 {code}")


@pytest.mark.parametrize("case", REJECT, ids=[c["case_id"] for c in REJECT])
def test_reject(case: dict) -> None:
    if case["case_id"] in EXE_ARGV_CASES:
        _check_exe_argv(case)
        return
    code = case.get("expect_reject_code")
    dtype = {"float16": torch.float16, "int32": torch.int32}.get(case.get("dtype"), torch.float32)
    na, nb = int(case["n_a"]), int(case["n_b"])
    with pytest.raises((ValueError, TypeError, RuntimeError)) as ei:
        sub(torch.zeros(na, dtype=dtype).npu(), torch.zeros(nb, dtype=dtype).npu())
    assert code in str(ei.value), f"{case['case_id']} 期望 {code}，实际：{ei.value}"
