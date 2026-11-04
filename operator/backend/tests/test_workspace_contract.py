"""工作区契约自测（M1 起）：用例矩阵 / 判据档 / 四动作入口。

都是「控制台信得过」的前提，而且只看**工作区里的文件**（逐条可重建的运行时数据），
不连远端、不跑容器 —— 跑一次 pytest 就能知道「契约层还是不是完整的」。

对应方案 v1.1：§4.5（用例矩阵与报告）、§5（门禁的判据档）、§6（四动作入口）。
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
WS = REPO / "workspaces" / "cann-ops"
OP = "elementwise_add"
PROFILE_REF = "ops-precision-standard@1"

CASE_GROUPS = {"常规", "边界", "数值边界", "异常", "回归"}
CASE_FIELDS = {"case_id", "group", "required", "n_a", "n_b", "dtype", "gen_a", "gen_b",
               "seed", "expected", "note"}
EXPECTED = {"match", "reject"}

pytestmark = pytest.mark.skipif(not WS.is_dir(), reason="任务工作区不在这台机上（运行时数据）")

# 任务生命周期 v2（2026-10-23）：存量算子产物已按用户要求清空（备份在 archives/）。
# 用例矩阵类断言只在「工作区里有算子产物」时有意义；新建任务产出后再自动恢复执行。
_CASES = WS / "cases" / OP / "cases.json"
_skip_no_operator = pytest.mark.skipif(
    not _CASES.is_file(),
    reason="工作区已清空存量算子产物（task-lifecycle-v2 迁移）；等新任务产出用例后再验",
)


def _cases() -> dict:
    return json.loads((WS / "cases" / OP / "cases.json").read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


@_skip_no_operator
def test_cases_matrix_exists() -> None:
    """plan 的前置检查要求工作区有 cases/；这里再确认矩阵文件本身在。"""
    assert (WS / "cases" / OP / "cases.json").is_file(), "缺用例矩阵：cases/<op>/cases.json"


@_skip_no_operator
def test_case_fields_and_groups() -> None:
    suite = _cases()
    cases = suite["cases"]
    assert cases, "用例矩阵为空"
    for case in cases:
        missing = CASE_FIELDS - set(case)
        assert not missing, f"{case.get('case_id')} 缺字段 {sorted(missing)}"
        assert case["group"] in CASE_GROUPS, f"{case['case_id']} 分组非法：{case['group']}"
        assert case["expected"] in EXPECTED, f"{case['case_id']} expected 非法：{case['expected']}"
        assert isinstance(case["seed"], int), f"{case['case_id']} 没有随机种子（§4.5 要求可复现）"
    assert suite["case_groups"] == ["常规", "边界", "数值边界", "异常", "回归"]


@_skip_no_operator
def test_case_ids_unique_and_named() -> None:
    cases = _cases()["cases"]
    ids = [c["case_id"] for c in cases]
    assert len(ids) == len(set(ids)), "case_id 有重复"
    for cid, case in zip(ids, cases):
        assert cid.startswith("add_"), f"{cid} 命名前缀不对（回归组以后也用同一个前缀）"


@_skip_no_operator
def test_cases_respect_spec_v1_constraints() -> None:
    """用例必须落在规格 v1 声明的范围内（同 shape、N 上下界、支持的 dtype）。"""
    suite = _cases()
    cons = suite["constraints"]
    supported = set(cons["supported_dtypes"])
    for case in suite["cases"]:
        if case["expected"] == "match":
            assert case["dtype"] in supported, f"{case['case_id']} 用了规格外的 dtype"
            assert cons["n_min"] <= case["n_a"] <= cons["n_max"], f"{case['case_id']} N 越出规格范围"
            assert case["n_a"] == case["n_b"], f"{case['case_id']} match 用例必须同 shape（v1 不广播）"


@_skip_no_operator
def test_invalid_cases_declare_reject_code() -> None:
    """异常用例必须写明期望的拒绝码 —— 否则「因为别的原因失败」会被当成拒绝通过。"""
    for case in _cases()["cases"]:
        if case["expected"] == "reject":
            code = case.get("expect_reject_code", "")
            assert code.startswith("REJECT_"), f"{case['case_id']} 没有期望拒绝码"


def test_criteria_profile_is_vendored_and_fingerprinted() -> None:
    """判据档要有实体（容器里能机判），且与 skills 安装指纹对得上。"""
    impl = WS / "criteria" / PROFILE_REF / "mixed_tolerance_check.py"
    assert impl.is_file(), f"判据档实现缺失：{impl}"
    assert (WS / "criteria" / PROFILE_REF / "PROVENANCE.md").is_file(), "判据档缺出处说明"
    lock = REPO / "ops" / "dsh" / "skills.lock"
    if not lock.is_file():
        pytest.skip("skills.lock 不在（未装 skills 的机器）")
    entries = json.loads(lock.read_text(encoding="utf-8")).get("files", {})
    want = entries.get(".dsh/skills/ops-precision-standard/scripts/mixed_tolerance_check.py")
    if want:
        assert _sha256(impl) == want, "判据档副本与 skills 安装指纹不一致（被改过？）"


def test_four_action_entry_scripts_present() -> None:
    """plan 会按契约找 <动作>.sh；缺哪个动作的控制台就会拦下来。"""
    from app.services.exec_actions import ACTION_SPECS

    for action, spec in ACTION_SPECS.items():
        entry = WS / spec["entry"]
        assert entry.is_file(), f"工作区缺入口脚本 {entry.name}（动作 {action}）"
    assert (WS / "cases").is_dir(), "工作区缺用例集目录 cases/"


def test_benchmark_and_profile_entry_scripts_are_wired() -> None:
    """M1-b 起这两个动作必须**真跑**：不得再有「NOT IMPLEMENTED + 故意非 0 退出」的存根。"""
    for name, tool in (("benchmark.sh", "run_benchmark.py"), ("profile.sh", "run_profile.py")):
        text = (WS / name).read_text(encoding="utf-8")
        assert "NOT IMPLEMENTED" not in text, f"{name} 仍是存根（M1-b 应已实装）"
        assert tool in text, f"{name} 没接上执行器 {tool}"
