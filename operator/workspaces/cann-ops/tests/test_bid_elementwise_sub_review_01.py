"""bid_elementwise_sub_review_01 · pytest 交付件（真机运行，读 cases/<op>/cases.json）。

候选走 Python 接口 ``ops.bid_elementwise_sub_review_01.python.bid_elementwise_sub_review_01``
（ctypes 加载 ``libbid_elementwise_sub_review_01_host.so`` + 候选 Ascend C kernel），
参考走 ``ops.bid_elementwise_sub_review_01.scripts.golden``（权威 = CPU ``torch.sub``，不碰 NPU），
判定口径与 ``criteria/check.py``（``ops-precision-standard@1``）一致，自建参数化补充用例复用同一 checker。
门禁权威仍是共享 runner（四动作 test；``tools/run_cases.py``），本 pytest 不替代门禁。

- 依赖缺失 = 收集失败（不是跳过）：numpy/torch/torch_npu/候选接口全部硬 import，无跳过钩子；
- 不吞异常：拒绝用例按缺陷定义**精确**断言异常类型，无裸兜底、无宽泛
  ``pytest.raises((ValueError, TypeError, RuntimeError))``：
  REJECT 冻结用例（REJECT_N_OUT_OF_RANGE / REJECT_SHAPE_MISMATCH / REJECT_UNSUPPORTED_DTYPE）
  → ``ValueError``；非 Tensor / CPU 输入 / CPU-NPU 混用 → ``TypeError`` + ``REJECT_BAD_ARGS``；
  rank≠1 / 非连续 → ``ValueError`` + ``REJECT_BAD_ARGS``。
  所有测试输入一律在 ``pytest.raises()`` **之外**构造，不把输入构造错误伪装成候选拒绝；
- 已确认冻结用例（cases.json，只读）：15 条（10 match + 5 reject），本文件只读不复写、不新增；
- 补充用例（接口层 pytest 表达，不写 cases.json）：storage_offset 视图 ×4（sub_sup_001..004）、
  float32 极值（sub_sup_005）、CPU 输入（sub_sup_006）、CPU/NPU 双向混用 + 非 Tensor（sub_sup_007）、
  rank≠1（sub_sup_009）、非连续（sub_sup_010）、非 Tensor None/list × 双参数位（sub_sup_011）、
  NPU 上产生输入后立即调用（sub_sup_012，缺陷 9）。跨 NPU（sub_sup_008）依赖双卡，默认不执行 →
  **未验证**：为保证「不静默跳过」，本文件不编入该用例测试函数，由 0.1 黑盒设计与测试工程说明显式记录为未验证。
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
import torch_npu  # noqa: F401  # 注册 npu 设备；缺失 = 收集失败（不是 skip）

WS = Path(__file__).resolve().parents[1]
if str(WS) not in sys.path:
    sys.path.insert(0, str(WS))

from criteria.check import check as criteria_check  # noqa: E402
from ops.bid_elementwise_sub_review_01.python.bid_elementwise_sub_review_01 import subtract  # noqa: E402
from ops.bid_elementwise_sub_review_01.scripts.golden import compute_golden  # noqa: E402
from tools import gen_inputs  # noqa: E402

CASES = json.loads((WS / "cases/bid_elementwise_sub_review_01/cases.json").read_text(encoding="utf-8"))
MATCH = [c for c in CASES["cases"] if c.get("required", True) and c.get("expected") == "match"]
REJECT = [c for c in CASES["cases"] if c.get("required", True) and c.get("expected") == "reject"]

# storage_offset 视图补充用例（0.1 黑盒设计：sub_sup_001..004；N=8/17、offset=1/4）
VIEW_CASES = [
    ("sub_sup_001", 8, 1),
    ("sub_sup_002", 8, 4),
    ("sub_sup_003", 17, 1),
    ("sub_sup_004", 17, 4),
]

# CPU/NPU 双向混用 + 非 Tensor（sub_sup_007 双向；sub_sup_011 None/list × 首/次参数位）：
# 全部 → TypeError + REJECT_BAD_ARGS。kind 在 _mk_arg 里构造（pytest.raises 之外）。
BAD_ARGS_MIX_CASES = [
    ("sub_sup_007_cpu_npu", "cpu", "npu"),
    ("sub_sup_007_npu_cpu", "npu", "cpu"),
    ("sub_sup_011_none_first", "none", "npu"),
    ("sub_sup_011_none_second", "npu", "none"),
    ("sub_sup_011_list_first", "list", "npu"),
    ("sub_sup_011_list_second", "npu", "list"),
]


def _gen(case: dict, key: str, seed_offset: int) -> "np.ndarray":
    n = int(case["n_a"]) if key == "a" else int(case["n_b"])
    return gen_inputs.generate(case.get(f"gen_{key}", {}), n, int(case["seed"]) + seed_offset,
                               case.get("dtype", "float32"))


def _bit_equal(got: "np.ndarray", want: "np.ndarray") -> int:
    """逐位相等的元素数（NaN 也按位比较，不把 NaN==NaN 当相等）。"""
    return int((got.view(np.uint32) == want.view(np.uint32)).sum())


def _check_output(y: "torch.Tensor", ta: "torch.Tensor", tb: "torch.Tensor", n: int) -> None:
    """输出契约显式断言：新张量、同 shape/dtype/设备、连续、非原地、不共享 storage。"""
    assert isinstance(y, torch.Tensor), "接口必须返回 torch.Tensor"
    assert tuple(y.shape) == (int(n),), f"输出 shape 应为 [N]：{tuple(y.shape)}"
    assert y.dtype == torch.float32, f"输出 dtype 应为 float32：{y.dtype}"
    assert y.device == ta.device, f"输出 device 应与输入 NPU 一致：{y.device} vs {ta.device}"
    assert y.is_contiguous(), "输出必须连续"
    assert y.data_ptr() != ta.data_ptr() and y.data_ptr() != tb.data_ptr(), "输出不得原地写回输入"
    assert y.storage().data_ptr() not in (ta.storage().data_ptr(), tb.storage().data_ptr()), \
        "输出不得与输入共享 storage"


def _mk_arg(kind: str):
    """构造 CPU/NPU/None/list 输入（在 pytest.raises 之外构造，见调用方）。"""
    if kind == "cpu":
        return torch.zeros(8, dtype=torch.float32)
    if kind == "npu":
        return torch.zeros(8, dtype=torch.float32).npu()
    if kind == "none":
        return None
    if kind == "list":
        return [0.0] * 8
    raise AssertionError(f"未知输入类型 kind={kind}")


@pytest.mark.parametrize("case", MATCH, ids=[c["case_id"] for c in MATCH])
def test_match(case: dict) -> None:
    a = _gen(case, "a", 0)
    b = _gen(case, "b", 1)
    want = compute_golden(a, b)
    ta = torch.from_numpy(a.copy()).npu()
    tb = torch.from_numpy(b.copy()).npu()
    y = subtract(ta, tb)
    _check_output(y, ta, tb, int(case["n_a"]))
    got = y.cpu().numpy()

    verdict = criteria_check(got, want)
    bit = _bit_equal(got, want)
    with np.errstate(over="ignore", invalid="ignore"):
        torch_ref = torch.sub(torch.from_numpy(a.copy()), torch.from_numpy(b.copy())).numpy()
    torch_diff = int((got.view(np.uint32) != torch_ref.view(np.uint32)).sum())
    report = (f"{case['case_id']}: bit_exact {bit}/{got.size}, "
              f"torch_diff {torch_diff}, matched_ratio {verdict.get('matched_ratio'):.6f}")
    print(report)
    assert verdict["is_pass"], f"{case['case_id']}: {verdict.get('failure_reasons')}；{report}"


@pytest.mark.parametrize("case", REJECT, ids=[c["case_id"] for c in REJECT])
def test_reject(case: dict) -> None:
    """冻结 REJECT 用例：精确 ``ValueError`` + 对应 REJECT_* 码；输入在 raises 之外构造。"""
    code = case.get("expect_reject_code")
    dtype = {"float16": torch.float16, "int32": torch.int32}.get(case.get("dtype"), torch.float32)
    na, nb = int(case["n_a"]), int(case["n_b"])
    ra = torch.zeros(na, dtype=dtype).npu()  # 输入构造在 pytest.raises 之外
    rb = torch.zeros(nb, dtype=dtype).npu()
    with pytest.raises(ValueError) as ei:
        subtract(ra, rb)
    assert code in str(ei.value), f"{case['case_id']} 期望 {code}，实际：{ei.value}"
    print(f"{case['case_id']}: reject {code}（ValueError）")


@pytest.mark.parametrize("case_id,n,offset", VIEW_CASES, ids=[c[0] for c in VIEW_CASES])
def test_sup_storage_offset_view(case_id: str, n: int, offset: int) -> None:
    """连续但 storage_offset≠0 的视图：整 storage 先放 NPU 再切片，按逻辑元素正确计算。"""
    rng = np.random.default_rng(7_700_000 + n)  # 确定性
    src_a = rng.uniform(-1.0, 1.0, n + offset).astype(np.float32)
    src_b = rng.uniform(-1.0, 1.0, n + offset).astype(np.float32)
    buf_a = torch.from_numpy(src_a).npu()
    buf_b = torch.from_numpy(src_b).npu()
    va = buf_a.narrow(0, offset, n)
    vb = buf_b.narrow(0, offset, n)
    assert va.storage_offset() == offset and vb.storage_offset() == offset, "视图 storage_offset 不符合预期"
    assert va.is_contiguous() and vb.is_contiguous(), "narrow 视图应为连续"

    want = compute_golden(va.cpu().numpy(), vb.cpu().numpy())
    y = subtract(va, vb)
    _check_output(y, va, vb, n)
    got = y.cpu().numpy()
    verdict = criteria_check(got, want)
    print(f"{case_id}: bit_exact {_bit_equal(got, want)}/{got.size}, "
          f"matched_ratio {verdict.get('matched_ratio'):.6f}")
    assert verdict["is_pass"], f"{case_id}: {verdict.get('failure_reasons')}"


def test_sup_float32_extrema() -> None:
    """数值边界补充（sub_sup_005）：float32 max/min 正规/次正规、±0、溢出致 inf。

    先保存 ta/tb，随后调用 ``_check_output`` 显式校验 shape/dtype/device/连续/新张量，
    不只比对数值（缺陷 3）；数值对照权威 CPU ``torch.sub`` oracle 与非有限值规则。
    """
    fm = np.finfo(np.float32)
    tiny = fm.tiny                 # 最小正规正数 2^-126
    sub = tiny * fm.eps            # 最小次正规正数 2^-149
    a = np.array(
        [fm.max, -fm.max, fm.max, -fm.max, fm.max, tiny, -tiny, sub, -sub,
         1.0, 1.0, 0.0, -0.0, 3.5, -2.5, tiny, sub],
        dtype=np.float32)
    b = np.array(
        [-fm.max, fm.max, 1.0, 1.0, fm.max, tiny, tiny, sub, sub,
         -1.0, 1e-38, 0.0, 0.0, 1.5, 0.5, tiny, sub],
        dtype=np.float32)
    assert a.size == 17 and b.size == 17
    want = compute_golden(a, b)
    ta = torch.from_numpy(a.copy()).npu()
    tb = torch.from_numpy(b.copy()).npu()
    y = subtract(ta, tb)
    _check_output(y, ta, tb, a.size)  # 缺陷 3：先校验输出契约，再比对数值
    got = y.cpu().numpy()
    verdict = criteria_check(got, want)
    print(f"sub_sup_005: bit_exact {_bit_equal(got, want)}/{got.size}, "
          f"matched_ratio {verdict.get('matched_ratio'):.6f}")
    assert verdict["is_pass"], f"sub_sup_005: {verdict.get('failure_reasons')}"


def test_sup_cpu_input_reject() -> None:
    """补充（sub_sup_006）：x1、x2 均为 CPU 张量 → TypeError + REJECT_BAD_ARGS。"""
    x1 = torch.zeros(8, dtype=torch.float32)  # 输入在 raises 之外构造
    x2 = torch.zeros(8, dtype=torch.float32)
    with pytest.raises(TypeError) as ei:
        subtract(x1, x2)
    assert "REJECT_BAD_ARGS" in str(ei.value), f"期望 REJECT_BAD_ARGS，实际：{ei.value}"
    print("sub_sup_006: reject REJECT_BAD_ARGS（CPU 双输入，TypeError）")


@pytest.mark.parametrize(
    "case_id,ka,kb", BAD_ARGS_MIX_CASES, ids=[c[0] for c in BAD_ARGS_MIX_CASES])
def test_sup_cpu_npu_mix_reject(case_id: str, ka: str, kb: str) -> None:
    """补充（sub_sup_007 双向 / sub_sup_011 非 Tensor）：→ TypeError + REJECT_BAD_ARGS。

    双向齐备（subtract(CPU,NPU) 与 subtract(NPU,CPU)）；并覆盖 None 与 list 非 Tensor 在
    第一个/第二个参数位。输入一律在 pytest.raises 之外构造。
    """
    x1 = _mk_arg(ka)
    x2 = _mk_arg(kb)
    with pytest.raises(TypeError) as ei:
        subtract(x1, x2)
    assert "REJECT_BAD_ARGS" in str(ei.value), f"期望 REJECT_BAD_ARGS，实际：{ei.value}"
    print(f"{case_id}: reject REJECT_BAD_ARGS（TypeError，{ka}/{kb}）")


def test_sup_rank2_reject() -> None:
    """补充（sub_sup_009）：rank≠1（2-D）→ ValueError + REJECT_BAD_ARGS。"""
    a = torch.zeros((2, 3), dtype=torch.float32).npu()  # 输入在 raises 之外构造
    b = torch.zeros((2, 3), dtype=torch.float32).npu()
    with pytest.raises(ValueError) as ei:
        subtract(a, b)
    assert "REJECT_BAD_ARGS" in str(ei.value), f"期望 REJECT_BAD_ARGS，实际：{ei.value}"
    print("sub_sup_009: reject REJECT_BAD_ARGS（rank≠1，ValueError）")


def test_sup_non_contiguous_reject() -> None:
    """补充（sub_sup_010）：1-D 非连续视图（step=2）→ ValueError + REJECT_BAD_ARGS。"""
    full_a = torch.arange(16, dtype=torch.float32).npu()
    full_b = torch.arange(16, dtype=torch.float32).npu()
    a = full_a[::2]
    b = full_b[::2]
    assert a.ndim == 1 and not a.is_contiguous(), "构造的 1-D 视图应为非连续"
    with pytest.raises(ValueError) as ei:
        subtract(a, b)
    assert "REJECT_BAD_ARGS" in str(ei.value), f"期望 REJECT_BAD_ARGS，实际：{ei.value}"
    print("sub_sup_010: reject REJECT_BAD_ARGS（非连续视图，ValueError）")


def test_sup_npu_produced_input_immediate_call() -> None:
    """缺陷 9（测试侧）：在当前 NPU 产生输入后立即调用候选；测试侧不做任何预同步。

    生产者用 torch_npu 在**当前设备**上直接生成输入（enqueue 到 torch_npu 流）；随后
    **不**调用 .cpu() / torch_npu.synchronize() / .item() / .to() 等任何会强制同步的入口，
    立即把 data_ptr 交给候选。候选（实现侧缺陷由节点 2 修复）必须自行建立生产者↔候选流
    依赖（或保守使用目标设备 torch_npu 同步入口），并在返回前同步 + 传播真实失败。

    断言：
    - 输出契约 ``_check_output``：新张量 / shape / dtype / device / 连续 / 不共享 storage；
    - 数值正确（对照 CPU torch.sub oracle）——若候选未同步已产生的输入，这里读到未定义
      数据并判 FAIL，从而暴露缺陷 9；
    - 返回前同步 + 错误传播：候选返回后读取结果必须与 CPU oracle 一致；任何底层失败由
      ``subtract`` 以 RuntimeError 抛出（接口 ``_raise``，rc≠0 不静默），不会返回错误数值冒充成功。
    仍单卡，不涉及第二张卡。
    """
    n = 64
    x1 = torch.randn(n, device="npu", dtype=torch.float32)  # 当前设备产生输入（无任何预同步）
    x2 = torch.randn(n, device="npu", dtype=torch.float32)
    y = subtract(x1, x2)             # 立即调用候选，无预同步
    _check_output(y, x1, x2, n)
    got = y.cpu().numpy()            # 返回后读取：候选返回前应已同步（正确性在此暴露）
    want = compute_golden(x1.cpu().numpy(), x2.cpu().numpy())
    verdict = criteria_check(got, want)
    print(f"sub_sup_012: bit_exact {_bit_equal(got, want)}/{got.size}, "
          f"matched_ratio {verdict.get('matched_ratio'):.6f}")
    assert verdict["is_pass"], f"sub_sup_012（缺陷 9 同步）：{verdict.get('failure_reasons')}"