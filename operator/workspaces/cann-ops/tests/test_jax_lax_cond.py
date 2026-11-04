# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""jax_lax_cond · pytest 交付件（真机；候选 = Python 接口，参考 = 同版本 JAX CPU 子进程）。

结构（设计 §2.2.2 / §5 / §6.4）：

* 冻结 ``cases/jax_lax_cond/cases.json``（16 条 / 49 次）
  + 基准复用 ``benchmark.json``（2 条 / 4 次，仅判正确性）
  + 补充 ``tests/jax_lax_cond_supplement_cases.json``（5 条 / 18 次）
  = **23 条黑盒身份 / 71 个 ``(case_id, variant, mode)`` 参数化项**；
  + 白盒 ``tests/jax_lax_cond_whitebox_cases.json``（节点 3 创建，6 条 / 12 次，走同一 loader）
  = **29 条身份 / 83 个参数化项**。
* 候选：``ops/jax_lax_cond/python/jax_lax_cond.py`` 的 ``cond`` / ``cond_compiled``；
  numpy family 走 ``so``（``abi_calls=1``），jax family（S1/grad）走 ``trace``
  （``abi_calls=0``、backend 非 CPU）；两分支 trace 与单分支 exec 证据分开记录。
  S1 的 ``jit`` 固定为 workflow2 #202 已验证的 **0-d 闭包常量 + 设备标量 pred**
  形态：``operand`` 与同步后的 concrete ``pred`` 都是零参数 ``jax.jit`` 程序的
  闭包常量，绝不作为 ``jax.jit`` 动态实参（镜像 #145 的 AFIR 不能编译 0-d jit
  参数，outer-jit tracer pred 也没有 conditional/select lowering）。
* 参考：``golden_cli.py`` 独立 CPU 子进程（``JAX_PLATFORMS=cpu``）；pytest 主进程只跑候选。
* 判定：逐叶子先设备守卫，再 dtype/shape/结构精确，浮点走 ``criteria/check.py``；
  jax family 的 ``weak_type`` 逐叶精确；非有限位逐元素精确；拒绝按四码 + 上游类型/摘要。
  c13 的 ``int32(0/1)`` 是唯一例外：JAX 0.8.1 CPU 不抛错，拒绝依据是已确认 spec 的
  scalar-bool 规则；golden 记 ``cpu_rejects=false / spec_rule_rejects=true``，候选必须
  以 ``type=None + spec_rule=scalar_bool_pred + cpu_probe`` 明示，不能伪造上游异常。
* 本任务没有合法 ``xfail``/``unsupported``：任何非 PASS 都 fail，不计入通过数。

workflow3 迁移（设计 §2.2.2 / §5.6 / §7.4，A2/A3）：
* **fresh worker**：每个 ``(case_id, variant, mode)`` 的候选执行在独立子进程
  （``python -s <本文件> --worker <payload>``）中完成；父进程只调度/聚合，记录
  子进程 PID、程序哈希、backend/device。每个 worker 在调用候选前先初始化
  PJRT/xla-npu（保持 ``2-探针-r4-pytest-mix2.log`` 已验证的 JAX→ACL 顺序）。
* **D1–D4 迁移专项诊断**：D1 数据派生 predicate + kernel token 独立性、
  D2 真实 cotangent / 扁平 VJP、D3 独立进程隔离、D4 后端能力边界。
  D1–D4 是 workflow3 的迁移验收项，**不计入 pytest 用例计数**；CPU 上只验证
  协议并记 ``BLOCKED``，不冒充 PASS，也不把后端限制写成支持。

报告：``${OPX_OUT:-/out}/cond_correctness.json`` + ``.md``；每条
``(case_id, variant, mode)`` 另打印一行 ``COND_CASE {...}``；诊断另打印
``COND_DIAG {...}``。
"""

from __future__ import annotations

import hashlib
import importlib.util
import inspect
import itertools
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import pytest

WS = Path(__file__).resolve().parents[1]
OP = "jax_lax_cond"
for extra in (WS, WS / "ops" / OP / "scripts"):
    if str(extra) not in sys.path:
        sys.path.insert(0, str(extra))

# P6：numpy 是交付件运行必需依赖。缺失时必须让收集阶段显式失败，
# 不能用 importorskip 把整个模块 skip 成“通过”（skip 不计 PASS）。
import numpy as np

import fixtures  # noqa: E402
import cond_engine as engine  # noqa: E402

CANDIDATE_PATH = WS / "ops" / OP / "python" / f"{OP}.py"
GOLDEN_DIR = WS / "ops" / OP / "scripts"
GOLDEN_CLI = GOLDEN_DIR / "golden_cli.py"
FROZEN_CASES_PATH = WS / "cases" / OP / "cases.json"
BENCH_PATH = WS / "cases" / OP / "benchmark.json"
SUPPLEMENT_PATH = WS / "tests" / f"{OP}_supplement_cases.json"
WHITEBOX_PATH = WS / "tests" / f"{OP}_whitebox_cases.json"

EXPECTED_GATE_CASES = 16
EXPECTED_GATE_MODES = 49
EXPECTED_SUPPLEMENT_CASES = 5
#: 节点 3 白盒用例：6 条身份 / 12 个 (variant, mode) 项（见 3-白盒测试设计.md）。
EXPECTED_WHITEBOX_CASES = 6
EXPECTED_WHITEBOX_MODES = 12
#: 黑盒（冻结 + 基准复用 + 补充）身份/参数化项；白盒另计。
EXPECTED_PYTEST_CASES = 23
EXPECTED_PYTEST_MODES = 71
EXPECTED_TOTAL_CASES = EXPECTED_PYTEST_CASES + EXPECTED_WHITEBOX_CASES
EXPECTED_TOTAL_MODES = EXPECTED_PYTEST_MODES + EXPECTED_WHITEBOX_MODES

CRITERIA_MOD = engine._load_module(WS / "criteria" / "check.py", "opx_cond_pytest_criteria")

SELFCHECK_NAMES = (
    "test_selfcheck_discovery_counts",
    "test_selfcheck_smoke_selection",
    "test_selfcheck_data_generation_frozen_rules",
    "test_selfcheck_criteria_thresholds",
    "test_selfcheck_device_guard_negative",
    "test_selfcheck_golden_env_guard_negative",
    "test_selfcheck_golden_reject_rules",
    "test_selfcheck_golden_signature_guard",
    "test_selfcheck_golden_numeric_binding",
    "test_selfcheck_report_writer",
    "test_selfcheck_whitebox_hook",
    "test_selfcheck_whitebox_cases",
    "test_selfcheck_engine_dispatch_negative",
    "test_selfcheck_s1_jit_closure_route",
    "test_selfcheck_worker_protocol",
)

#: workflow3 迁移专项诊断（设计 §5.6 / §7.4）。不进入 pytest 用例计数。
DIAGNOSTIC_NAMES = (
    "test_diag_d1_data_derived_predicate",
    "test_diag_d2_real_cotangent",
    "test_diag_d3_fresh_worker_isolation",
    "test_diag_d4_backend_capability",
)

WORKER_RESULT_PREFIX = "COND_WORKER_RESULT "
WORKER_TIMEOUT_S = int(os.environ.get("OPX_COND_WORKER_TIMEOUT", "900"))
UNSUPPORTED_HLO_OPS = ("stablehlo.conditional", "stablehlo.select",
                       "stablehlo.convert")

WORKDIR = Path(tempfile.mkdtemp(prefix="opx-cond-pytest-"))

#: 记录收集期是否已有 JAX（例如 pytest 插件预导入）；用于区分“本模块泄漏”与外部预导入。
_JAX_PREIMPORTED = "jax" in sys.modules
_JAXLIB_PREIMPORTED = "jaxlib" in sys.modules

#: 每个用例一个 fresh worker；父进程只聚合结果。
_SPAWN_COUNTER = itertools.count(1)
_DIAGNOSTICS: list[dict] = []


# ---------------------------------------------------------------------------
# 用例加载与发现计数
# ---------------------------------------------------------------------------


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _normalise(case: dict, origin: str) -> dict:
    item = dict(case)
    item["mode"] = list(item.get("mode") or ["eager"])
    item["_origin"] = origin
    return item


def _load_cases(*, frozen_path: Path = FROZEN_CASES_PATH, bench_path: Path = BENCH_PATH,
                supplement_path: Path = SUPPLEMENT_PATH,
                whitebox_path: Path = WHITEBOX_PATH) -> tuple[list[dict], dict]:
    """冻结 + 基准 + 补充（+ 可选白盒）。白盒文件缺失是正常状态，不 skip 成通过。"""
    suites = {
        "frozen": _load_json(frozen_path),
        "benchmark": _load_json(bench_path),
        "supplement": _load_json(supplement_path),
    }
    # 设计 §6.4：基准 2 条以 jit_true/jit_false 复用为 L1 正确性（2 条 / 4 次）。
    for case in suites["benchmark"].get("cases") or []:
        case.setdefault("mode", ["jit_true", "jit_false"])
    cases: list[dict] = []
    for origin in ("frozen", "benchmark", "supplement"):
        cases += [_normalise(case, origin) for case in suites[origin].get("cases") or []]
    if whitebox_path.is_file():
        suites["whitebox"] = _load_json(whitebox_path)
        cases += [_normalise(case, "whitebox") for case in suites["whitebox"].get("cases") or []]
    else:
        suites["whitebox"] = {"cases": []}
    return cases, suites


CASES, SUITES = _load_cases()
FROZEN_SUITE = SUITES["frozen"]
BENCH_SUITE = SUITES["benchmark"]
SUPPLEMENT_SUITE = SUITES["supplement"]
WHITEBOX_SUITE = SUITES["whitebox"]
ITEMS = fixtures.expand_items(CASES)
MODE_COUNT = len(ITEMS)
GATE_MODE_COUNT = len(fixtures.expand_items(FROZEN_SUITE.get("cases") or []))
BASE_CASES = [case for case in CASES if case.get("_origin") != "whitebox"]
BASE_MODE_COUNT = len(fixtures.expand_items(BASE_CASES))

_CANDIDATE = None
_RESULTS: list[dict] = []
_CASE_MODE_RESULTS: dict[tuple[str, str, str], dict] = {}
_SELFCHECK_PASSED: set[str] = set()


def _mark_selfcheck(name: str) -> None:
    assert name in SELFCHECK_NAMES, f"未登记的自检名：{name}"
    _SELFCHECK_PASSED.add(name)


def _candidate():
    """候选 Python 接口（节点 2 交付）。缺失时直接失败，不能 skip 成“通过”。"""
    global _CANDIDATE
    if _CANDIDATE is not None:
        return _CANDIDATE
    if not CANDIDATE_PATH.is_file():
        pytest.fail(f"候选 Python 接口不存在：{CANDIDATE_PATH}（节点 2 必须先交付）")
    spec = importlib.util.spec_from_file_location(f"{OP}_candidate", CANDIDATE_PATH)
    if spec is None or spec.loader is None:
        pytest.fail(f"候选 Python 接口无法加载：{CANDIDATE_PATH}")
    module = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(module)
    except Exception as exc:  # noqa: BLE001
        pytest.fail(f"候选 Python 接口 import 失败：{type(exc).__name__}: {exc}")
    _CANDIDATE = module
    return module


# ---------------------------------------------------------------------------
# 设备守卫（设计 §1.3.5）
# ---------------------------------------------------------------------------


def _leaf_platforms(leaf):
    """返回 ``(platforms: set[str], kind: str)``；无法判定时 platforms 为空。"""
    if callable(getattr(leaf, "devices", None)):
        devices = list(leaf.devices())
        return {str(getattr(device, "platform", "?")) for device in devices}, "jax"
    return set(), "numpy-host-copy"


def assert_outputs_on_accelerator(outputs, case_id: str, dispatch: dict | None) -> list[dict]:
    """numpy family 必须有 so/npu/abi>=1/device0；jax family 每叶 platform 必须含 npu。"""
    recorded: list[dict] = []
    dispatch = dispatch or {}
    for path, leaf in fixtures.flatten_with_paths(outputs, "out"):
        platforms, kind = _leaf_platforms(leaf)
        if kind == "numpy-host-copy":
            assert str(dispatch.get("path")) == "so", (
                f"{case_id}/{path}: numpy 输出但 dispatch.path={dispatch.get('path')!r}，"
                "无法证明设备执行")
            assert str(dispatch.get("backend")) == "npu", (
                f"{case_id}/{path}: dispatch.backend={dispatch.get('backend')!r}，CPU 回退不算 PASS")
            assert int(dispatch.get("abi_calls") or 0) >= 1, (
                f"{case_id}/{path}: dispatch.abi_calls={dispatch.get('abi_calls')!r}")
            assert int(dispatch.get("device_id") or 0) == 0, (
                f"{case_id}/{path}: dispatch.device_id={dispatch.get('device_id')!r}")
        else:
            assert platforms and all("npu" in item.lower() for item in platforms), (
                f"{case_id}/{path}: jax 输出 device platform={sorted(platforms)}，必须含 npu")
        recorded.append({"path": path, "kind": kind, "platforms": sorted(platforms)})
    return recorded


# ---------------------------------------------------------------------------
# 候选调用与判定
# ---------------------------------------------------------------------------


def _build_fixture(case: dict, variant: str, mode: str) -> dict:
    """构造 fixture；白盒 out_tree 用例在此补齐 2 叶输出定义。

    既有 ``fixtures.output_spec`` 只按 case_id 识别 ``out_tree``；为不改动
    operator-v1 快照内的 ``ops/jax_lax_cond/scripts/fixtures.py``，白盒用例在
    pytest 侧对同一 fixture 覆盖 ``output_shapes``（golden/候选/判据仍读同一
    case.json 的 ``_resolved.output_shapes``）。
    """
    fixture = fixtures.make_fixture(case, variant, mode)
    branches = {str(item).replace(" ", "") for item in (case.get("branches") or [])}
    if branches == {"out_tree"}:
        shape = list(fixture["leaves"][0]["shape"])
        fixture["output_shapes"] = {"out/0": shape, "out/1/aux": [shape[0]]}
        fixture["output_paths"] = list(fixture["output_shapes"])
    return fixture


def _prepare(case: dict, variant: str, mode: str,
             workdir: Path | None = None) -> tuple[dict, Path]:
    fixture = _build_fixture(case, variant, mode)
    root = workdir or WORKDIR
    cdir = root / str(case["case_id"]) / (variant or "default") / str(mode)
    cdir.mkdir(parents=True, exist_ok=True)
    fixtures.write_case_json(case, fixture, cdir / "case.json")
    fixtures.write_inputs_npz(fixture, cdir / "inputs.npz")
    return fixture, cdir


def _to_numpy(leaf):
    if callable(getattr(leaf, "block_until_ready", None)):
        leaf = leaf.block_until_ready()
    return np.asarray(leaf)


def _make_numpy_tree(fixture: dict):
    return fixtures.build_operands(fixture, convert=lambda value: np.asarray(value))


def _make_jax_tree(fixture: dict):
    import jax.numpy as jnp  # noqa: PLC0415

    variant = str(fixture["variant"])

    def convert(value):
        if variant == "weak":
            return jnp.asarray(float(np.asarray(value)))
        return jnp.asarray(value, dtype=jnp.float32)

    return fixtures.build_operands(fixture, convert=convert)


def _norm_dispatch(module) -> dict:
    dispatch = dict(getattr(module, "_last_dispatch", None) or {})
    evidence = dispatch.pop("evidence", None)
    if isinstance(evidence, dict):
        for key in ("trace_hits", "exec_hits", "runtime_token", "evidence_level"):
            dispatch.setdefault(key, evidence.get(key))
    if "trace_hits" not in dispatch:
        dispatch["trace_hits"] = {
            "true": dispatch.pop("trace_true", None),
            "false": dispatch.pop("trace_false", None),
        }
    if "exec_hits" not in dispatch:
        dispatch["exec_hits"] = {
            "true": dispatch.pop("exec_true", None),
            "false": dispatch.pop("exec_false", None),
        }
    return dispatch


def _family_of(fixture: dict) -> str:
    if fixture["family"] == "jax" or fixtures.base_mode_of(fixture["mode"]) in engine.GRAD_MODES:
        return "jax"
    return "numpy"


def _dispatch_problems(dispatch: dict, fixture: dict) -> list[str]:
    family = _family_of(fixture)
    problems: list[str] = []
    selected = "true" if fixture["selected"] else "false"
    if str(dispatch.get("selected")) != selected:
        problems.append(f"dispatch.selected={dispatch.get('selected')!r} != {selected!r}")
    trace = dispatch.get("trace_hits") or {}
    for side in ("true", "false"):
        value = engine._as_int(trace.get(side)) if isinstance(trace, dict) else None
        if value is None or value < 1:
            problems.append(f"trace_hits.{side}={value!r}（必须 >=1）")
    token = engine._as_int(dispatch.get("runtime_token"))
    want_token = 1 if fixture["selected"] else 2
    if token != want_token:
        problems.append(f"runtime_token={dispatch.get('runtime_token')!r} != {want_token}")
    if family == "numpy":
        if str(dispatch.get("path")) != "so":
            problems.append(f"numpy family dispatch.path={dispatch.get('path')!r}（必须是 so）")
        if str(dispatch.get("exec_mode")) != "selected_only":
            problems.append(f"numpy family exec_mode={dispatch.get('exec_mode')!r}")
        if engine._as_int(dispatch.get("abi_calls")) != 1:
            problems.append(f"numpy family abi_calls={dispatch.get('abi_calls')!r}（必须是 1）")
        if str(dispatch.get("backend")) != "npu":
            problems.append(f"numpy family backend={dispatch.get('backend')!r}（必须是 npu）")
        if engine._as_int(dispatch.get("device_id")) != 0:
            problems.append(f"numpy family device_id={dispatch.get('device_id')!r}（必须是 0）")
        hits = dispatch.get("exec_hits") or {}
        want = {"true": 1, "false": 0} if fixture["selected"] else {"true": 0, "false": 1}
        got = {side: engine._as_int(hits.get(side)) if isinstance(hits, dict) else None
               for side in ("true", "false")}
        if got != want:
            problems.append(f"exec_hits={got!r} != {want!r}（运行期只执行选中分支）")
        if fixtures.base_mode_of(fixture["mode"]) in engine.JIT_MODES:
            if engine._as_int(dispatch.get("compile_count")) != 1:
                problems.append(f"jit compile_count={dispatch.get('compile_count')!r}（必须是 1）")
    else:
        if str(dispatch.get("path")) != "trace":
            problems.append(f"jax family dispatch.path={dispatch.get('path')!r}（必须是 trace）")
        if engine._as_int(dispatch.get("abi_calls")) != 0:
            problems.append(f"jax family abi_calls={dispatch.get('abi_calls')!r}（必须是 0）")
        if str(dispatch.get("backend")) in ("", "None", "cpu"):
            problems.append(f"jax family backend={dispatch.get('backend')!r}（必须非 CPU）")
    return problems


def _judge_outputs(outputs, golden_arrays: dict, golden_meta: dict, fixture: dict,
                   dispatch: dict) -> tuple[list[dict], list[str]]:
    leaves: list[dict] = []
    problems: list[str] = []
    expected_keys = list(fixture["output_shapes"])
    if isinstance(outputs, dict) and set(outputs) == set(expected_keys):
        actual = dict(outputs)          # grad: {"grad": ...} 已按输出路径命名
    else:
        actual = dict(fixtures.flatten_with_paths(outputs, "out"))
    if set(actual) != set(expected_keys):
        return [], [f"输出叶子集合不一致：候选 {sorted(actual)} != 期望 {sorted(expected_keys)}"]
    golden_leaves = engine._meta_leaf_map(golden_meta)
    family = _family_of(fixture)
    platforms = {item["path"]: item for item in
                 assert_outputs_on_accelerator(outputs, fixture["case_id"], dispatch)}
    # grad 的候选输出是按输出路径命名的 dict（{"grad": ...}），而
    # assert_outputs_on_accelerator 按 pytree 展开会加 "out/" 前缀；这里补一份
    # 去前缀别名，避免逐叶判定时平台证据 KeyError（判据与阈值不变）。
    for _path, _item in list(platforms.items()):
        if _path.startswith("out/"):
            platforms.setdefault(_path[len("out/"):], _item)
    for key in expected_keys:
        got = _to_numpy(actual[key])
        want = np.asarray(golden_arrays[key])
        info = fixtures.leaf_bytes(list(want.shape), str(want.dtype))
        leaf = {"path": key, "dtype": str(want.dtype), "shape": list(want.shape),
                **info, "is_pass": False, "weak_type": None,
                "platforms": platforms[key]["platforms"], "kind": platforms[key]["kind"]}
        if got.shape != want.shape:
            problems.append(f"{key}: shape {got.shape} != {want.shape}")
            leaves.append(leaf)
            continue
        if got.dtype != want.dtype:
            problems.append(f"{key}: dtype {got.dtype} != {want.dtype}")
            leaves.append(leaf)
            continue
        if family == "jax":
            leaf["weak_type"] = bool(getattr(actual[key], "weak_type", False))
            want_weak = (golden_leaves.get(key) or {}).get("weak_type")
            leaf["golden_weak_type"] = want_weak
            if want_weak is None:
                problems.append(f"{key}: golden 未记录 weak_type")
            elif leaf["weak_type"] != bool(want_weak):
                problems.append(f"{key}: weak_type={leaf['weak_type']} != golden {bool(want_weak)}")
        verdict = CRITERIA_MOD.check(np.ascontiguousarray(got, dtype=np.float32),
                                     np.ascontiguousarray(want, dtype=np.float32))
        leaf.update({
            "is_pass": bool(verdict["is_pass"]),
            "matched_ratio": verdict["matched_ratio"],
            "max_abs_error": verdict["max_abs_error"],
            "nonfinite": verdict["nonfinite"],
            "failure_reasons": verdict.get("failure_reasons") or [],
        })
        if not verdict["is_pass"]:
            problems.append(f"{key}: " + "；".join(verdict.get("failure_reasons") or ["数值未通过"]))
        leaves.append(leaf)
    return leaves, problems


# ---------------------------------------------------------------------------
# S1 / jax family jit：0-d 闭包常量 + 设备标量 pred（workflow2 #202 已验证路线）
# ---------------------------------------------------------------------------


def _platforms_of(value) -> list[str]:
    devices = getattr(value, "devices", None)
    if not callable(devices):
        return []
    return sorted({str(getattr(device, "platform", "?")) for device in devices()})


def _jax_device_scalar(operand) -> dict:
    """在设备上算 ``sum(operand * operand)`` 并同步回 host，作为 S1 pred 的来源。"""
    import jax.numpy as jnp  # noqa: PLC0415

    raw = jnp.sum(operand * operand)
    ready = raw.block_until_ready()
    array = np.ascontiguousarray(np.asarray(ready))
    return {
        "value": float(array.reshape(())),
        "platforms": _platforms_of(raw),
        "sha256": hashlib.sha256(array.tobytes()).hexdigest(),
    }


def _s1_jit_program(module, pair: dict, operand, pred_value: bool):
    """构造**零参数** jax.jit 程序：0-d operand 与 concrete pred 都是闭包常量。

    镜像 #145 的 AFIR 后端不能编译 0-d ``jax.jit`` 动态参数（``Failed to parse
    AFIR string``），outer-jit tracer pred 也没有 conditional/select lowering。
    workflow2 run #202 已验证的唯一可执行路线是把二者都放在闭包里；本函数拒绝
    任何 Tracer，并返回一个 ``inspect.signature`` 为空的 program。
    """
    import jax  # noqa: PLC0415
    import jax.numpy as jnp  # noqa: PLC0415

    if isinstance(pred_value, jax.core.Tracer):
        raise TypeError("S1 jit 的 pred 必须是同步后的具体值，不能是 Tracer")
    pred_arr = jnp.asarray(bool(pred_value))
    if not jax.core.is_concrete(pred_arr):
        raise TypeError("S1 jit 的 pred 闭包常量必须是 concrete jax.Array")
    # workflow2 #202 路线：pred 必须先同步成 host 具体值，再进入 trace；
    # 不能把异步设备数组或 tracer 留在闭包里让 lowering 自己解析。
    pred_synced = bool(np.asarray(pred_arr.block_until_ready()))
    if pred_synced != bool(pred_value):
        raise TypeError(
            f"S1 jit 的 pred 同步值 {pred_synced} 与派生值 {bool(pred_value)} 不一致")
    if not jax.core.is_concrete(operand):
        raise TypeError("S1 jit 的 0-d operand 闭包常量必须是 concrete jax.Array")

    def program():
        return module.cond(pred_arr, pair["true_fn"], pair["false_fn"], operand)

    return program, pred_arr


def _s1_jit_execute(module, pair: dict, fixture: dict, operand, cdir: Path):
    """S1 的 ``jit`` 正式执行形态。

    * pred：设备标量 ``sum(operand*operand)`` 同步为 Python float 后再比较，
      得到一个 concrete 0-d bool 设备数组；
    * operand/pred：闭包常量，``jax.jit`` 程序零参数，``lower()`` 与调用都不传
      动态实参；
    * 记录 HLO 摘要：0-d closure 程序在镜像 #145 上是常量图（``stablehlo.constant``），
      即分支在 closure 外以 eager 设备执行；这是已记录的 R2 后端限制，不写成
      “分支在 jit 图内运行”。
    """
    import jax  # noqa: PLC0415

    scalar = _jax_device_scalar(operand)
    pred_value = bool(scalar["value"] >= 0.0)
    expected = bool(fixture["pred"])
    if pred_value != expected:
        raise AssertionError(
            f"S1 jit 的设备标量派生 pred={pred_value} 与 fixture pred={expected} 不一致")
    program, pred_arr = _s1_jit_program(module, pair, operand, pred_value)
    pred_synced = bool(np.asarray(pred_arr))
    if pred_synced != pred_value:
        raise AssertionError(
            f"S1 jit 的 pred 同步值 {pred_synced} 与派生值 {pred_value} 不一致")
    program_params = list(inspect.signature(program).parameters)
    if program_params:
        raise AssertionError(
            f"S1 jit 程序必须是零参数闭包，实际参数={program_params}；"
            "禁止把 0-d operand/pred 作为 jax.jit 动态实参")
    lowered = jax.jit(program).lower()
    hlo_text = lowered.as_text()
    hlo_path = cdir / "hlo_s1_jit.mlir"
    hlo_path.write_text(hlo_text, encoding="utf-8")
    hlo_flags = {op: bool(op in hlo_text) for op in UNSUPPORTED_HLO_OPS}
    compiled = lowered.compile()
    outputs = compiled()
    evidence = {
        "exec_form": "closure_jit_zero_arg",
        "jit_dynamic_args": 0,
        "jit_lower_args": 0,
        "jit_call_args": 0,
        "program_params": program_params,
        "closure_constants": ["operand", "pred"],
        "pred_source": "jax_device_scalar",
        "pred_derivation": "sum(operand*operand) >= 0.0",
        "device_scalar": scalar["value"],
        "device_scalar_platforms": scalar["platforms"],
        "device_scalar_sha256": scalar["sha256"],
        "pred_value": pred_value,
        "pred_synced_value": pred_synced,
        "pred_is_concrete": bool(jax.core.is_concrete(pred_arr)),
        "pred_weak_type": bool(getattr(pred_arr, "weak_type", False)),
        "operand_is_concrete": bool(jax.core.is_concrete(operand)),
        "operand_weak_type": bool(getattr(operand, "weak_type", False)),
        "hlo_path": str(hlo_path),
        "hlo_sha256": hashlib.sha256(hlo_text.encode("utf-8")).hexdigest(),
        "hlo_flags": hlo_flags,
        "hlo_constant_graph": "stablehlo.constant" in hlo_text,
    }
    return outputs, evidence


def _run_numeric(module, case: dict, fixture: dict, cdir: Path) -> dict:
    pair = fixtures.branch_pair(case)
    pred = fixture["pred"]
    base_mode = fixtures.base_mode_of(fixture["mode"])
    family = _family_of(fixture)
    if hasattr(module, "_reset_evidence"):
        module._reset_evidence()
    s1_jit = None
    if family == "jax":
        operands = (_make_jax_tree(fixture),)
        if base_mode == "jit":
            outputs, s1_jit = _s1_jit_execute(
                module, pair, fixture, operands[0], cdir)
        else:
            outputs = module.cond(pred, pair["true_fn"], pair["false_fn"], *operands)
    else:
        operands = (_make_numpy_tree(fixture),)
        entry = module.cond_compiled if base_mode == "jit" else module.cond
        outputs = entry(pred, pair["true_fn"], pair["false_fn"], *operands)

    dispatch = _norm_dispatch(module)
    rec = {"dispatch": dispatch, "evidence": {
        "trace_hits": dispatch.get("trace_hits"),
        "exec_hits": dispatch.get("exec_hits"),
        "runtime_token": dispatch.get("runtime_token"),
        "evidence_level": dispatch.get("evidence_level"),
    }}
    if s1_jit is not None:
        rec["s1_jit"] = s1_jit
        rec["evidence"]["s1_jit"] = s1_jit
    problems = _dispatch_problems(dispatch, fixture)

    # c03：同一 jit 逻辑执行内重复 3 次，证明 plan 只编译一次、后续命中缓存。
    if case["case_id"] == "cond_jit_static_pred" and base_mode == "jit" and not problems:
        repeats = []
        for _ in range(2):
            module.cond_compiled(pred, pair["true_fn"], pair["false_fn"], *operands)
            rep = _norm_dispatch(module)
            repeats.append({"compile_count": rep.get("compile_count"),
                            "plan_cache": rep.get("plan_cache"),
                            "abi_calls": rep.get("abi_calls"),
                            "plan_id": rep.get("plan_id")})
        rec["repeats"] = repeats
        if any(engine._as_int(item["compile_count"]) != 1 for item in repeats):
            problems.append(f"c03 重复调用后 compile_count 不再是 1：{repeats}")
        if any(item["plan_cache"] != "hit" for item in repeats):
            problems.append(f"c03 重复调用后 plan_cache 不是 hit：{repeats}")
        if any(engine._as_int(item["abi_calls"]) != 1 for item in repeats):
            problems.append(f"c03 每次调用 abi_calls 必须为 1：{repeats}")

    golden = engine._run_golden(WS, cdir / "case.json", cdir / "inputs.npz",
                                base_mode, fixture["variant"], cdir)
    rec["golden"] = {"ok": golden.get("ok"), "rc": golden.get("rc"),
                     "backend": (golden.get("meta") or {}).get("backend"),
                     "jax": (golden.get("meta") or {}).get("jax")}
    if not golden["ok"]:
        rec.update(status="ERROR", detail=f"golden 失败：{golden.get('detail')}")
        return rec
    leaves, judge_problems = _judge_outputs(outputs, golden["arrays"],
                                            golden["meta"], fixture, dispatch)
    problems.extend(judge_problems)
    rec.update({
        "leaves": leaves,
        "matched_ratio": min((leaf.get("matched_ratio", 1.0) or 0.0)
                             for leaf in leaves) if leaves else 0.0,
        "max_abs_error": max((leaf.get("max_abs_error") or 0.0) for leaf in leaves)
        if leaves else None,
    })
    if problems:
        rec.update(status="FAIL", detail="；".join(problems))
    else:
        rec.update(status="PASS", detail=f"{fixture['mode']} 通过（{len(leaves)} 个叶子）")
    return rec


def _run_grad(module, case: dict, fixture: dict, cdir: Path) -> dict:
    import jax  # noqa: PLC0415
    import jax.numpy as jnp  # noqa: PLC0415

    pair = fixtures.branch_pair(case)
    pred = bool(fixture["pred"])
    x_np = np.ascontiguousarray(
        np.asarray(fixture["arrays"][fixture["leaves"][0]["path"]], dtype=np.float32))
    orig_shape = tuple(x_np.shape)
    if hasattr(module, "_reset_evidence"):
        module._reset_evidence()

    # 镜像 #145 的 AFIR 对 2-D auto-fuse 输出只写首列，且不支持 20→5x4 的
    # reshape（workflow2 #202 / workflow3 `2-探针-grad-r1.log`：2-D 仅 5/20 正确，
    # 扁平 [N] 20/20）。冻结 grad 分支均为逐元素，故 jit grad 以扁平 operand 编译，
    # 梯度数值不变；比较前在 **host** 恢复原 shape（不引入设备 reshape）。
    # eager 仍以原 2-D operand 调用（候选走真实 jax.vjp，2-D 写出完整）。
    grad_flat = fixture["mode"] == "grad_jit"
    if grad_flat:
        x_np = np.ascontiguousarray(x_np.reshape(-1))
    x = jnp.asarray(x_np)

    def loss(value):
        return jnp.sum(module.cond(pred, pair["true_fn"], pair["false_fn"], value))

    if grad_flat:
        grad = jax.jit(jax.grad(loss))(x)
        if tuple(grad.shape) != orig_shape:
            grad = jnp.asarray(
                np.asarray(grad.block_until_ready(), dtype=np.float32).reshape(orig_shape))
    else:
        grad = jax.grad(loss)(x)
    outputs = {"grad": grad}
    dispatch = _norm_dispatch(module)
    rec = {"dispatch": dispatch, "evidence": {
        "trace_hits": dispatch.get("trace_hits"),
        "exec_hits": dispatch.get("exec_hits"),
        "runtime_token": dispatch.get("runtime_token"),
        "evidence_level": dispatch.get("evidence_level"),
        "cotangent_route": dispatch.get("cotangent_route"),
        "grad_input_layout": "flat[N]" if grad_flat else "as_is",
        "original_shape": list(orig_shape),
    }}
    problems = _dispatch_problems(dispatch, fixture)
    golden = engine._run_golden(WS, cdir / "case.json", cdir / "inputs.npz",
                                fixture["mode"], fixture["variant"], cdir)
    rec["golden"] = {"ok": golden.get("ok"), "rc": golden.get("rc"),
                     "backend": (golden.get("meta") or {}).get("backend"),
                     "jax": (golden.get("meta") or {}).get("jax")}
    if not golden["ok"]:
        rec.update(status="ERROR", detail=f"golden 失败：{golden.get('detail')}")
        return rec
    leaves, judge_problems = _judge_outputs(outputs, golden["arrays"],
                                            golden["meta"], fixture, dispatch)
    problems.extend(judge_problems)
    analytic = 2.0 if pred else 3.0
    rec["analytic_expected"] = analytic
    rec["analytic_ok"] = bool(np.allclose(np.asarray(grad), analytic, rtol=0, atol=0))
    rec.update({
        "leaves": leaves,
        "matched_ratio": min((leaf.get("matched_ratio", 1.0) or 0.0)
                             for leaf in leaves) if leaves else 0.0,
        "max_abs_error": max((leaf.get("max_abs_error") or 0.0) for leaf in leaves)
        if leaves else None,
    })
    if problems:
        rec.update(status="FAIL", detail="；".join(problems))
    else:
        rec.update(status="PASS",
                   detail=f"{fixture['mode']} 通过；解析诊断 2x/3x={analytic}（不作通过依据）")
    return rec


def _reject_dispatch_problems(dispatch: dict) -> list[str]:
    """pytest 接口层的拒绝 dispatch 契约（CLI ``meta`` 由 cond_engine 门禁校验）。

    ``meta.status`` / ``meta.leaves`` / ``meta.upstream`` 是候选可执行文件的落盘证据，
    pytest 走 Python 接口时拿不到 ``meta.json``；因此这里只断言接口层可观测的
    ``path`` / ``exec_mode`` / ``abi_calls``，CLI ``meta`` 契约由
    ``engine.reject_meta_problems`` 在门禁引擎里校验，并有自检覆盖。
    """
    problems: list[str] = []
    if str(dispatch.get("path")) != "reject":
        problems.append(f"拒绝时 dispatch.path={dispatch.get('path')!r}（必须是 reject）")
    if str(dispatch.get("exec_mode")) != "reject":
        problems.append(f"拒绝时 dispatch.exec_mode={dispatch.get('exec_mode')!r}"
                        "（必须是 reject）")
    if engine._as_int(dispatch.get("abi_calls")) != 0:
        problems.append(f"拒绝时 abi_calls={dispatch.get('abi_calls')!r}（必须是 0）")
    return problems


def _run_reject(module, case: dict, fixture: dict, cdir: Path) -> dict:
    pair = fixtures.branch_pair(case)
    pred = fixture["pred"]
    base_mode = fixtures.base_mode_of(fixture["mode"])
    entry = module.cond_compiled if base_mode == "jit" else module.cond
    if hasattr(module, "_reset_evidence"):
        module._reset_evidence()
    rec: dict = {"dispatch": {}, "leaves": []}
    try:
        entry(pred, pair["true_fn"], pair["false_fn"], _make_numpy_tree(fixture))
    except Exception as exc:  # noqa: BLE001 —— 预期就是 CondError
        code = getattr(exc, "code", None)
        upstream_type = getattr(exc, "upstream_exception_type", None)
        upstream_message = getattr(exc, "upstream_message", "")
        upstream_sha = getattr(exc, "upstream_message_sha256", "")
        upstream_evidence = getattr(exc, "upstream_evidence", None)
        rec.update({
            "raised": type(exc).__name__,
            "code": code,
            "upstream": {
                "type": upstream_type,
                "message": upstream_message,
                "sha256": upstream_sha,
                "evidence": upstream_evidence if isinstance(upstream_evidence, dict) else {},
            },
        })
        if code != case.get("expected_error"):
            rec.update(status="FAIL",
                       detail=f"拒绝码 {code!r} != {case.get('expected_error')!r}")
            return rec
        if not str(upstream_message).strip():
            rec.update(status="FAIL", detail="upstream_message 为空")
            return rec
        if upstream_sha != engine._sha256_text(str(upstream_message)):
            rec.update(status="FAIL", detail="upstream_message_sha256 不可复算")
            return rec
    else:
        rec.update(status="FAIL",
                   detail=f"该拒绝却成功返回；期望 {case.get('expected_error')!r}")
        return rec

    dispatch = _norm_dispatch(module)
    rec["dispatch"] = dispatch
    dispatch_problems = _reject_dispatch_problems(dispatch)
    if dispatch_problems:
        rec.update(status="FAIL", detail="；".join(dispatch_problems))
        return rec

    golden = engine._run_golden(WS, cdir / "case.json", cdir / "inputs.npz",
                                "reject", fixture["variant"], cdir)
    rec["golden"] = {"ok": golden.get("ok"), "rc": golden.get("rc"),
                     "backend": (golden.get("meta") or {}).get("backend"),
                     "jax": (golden.get("meta") or {}).get("jax"),
                     "stderr": golden.get("stderr")}
    if golden.get("rc") == 5:
        rec.update(status="ERROR",
                   detail="CPU 未抛错，且不属于已确认的规格规则拒绝；"
                          "不修改期望、不把 JAX 接受伪造成拒绝")
        return rec
    if not golden["ok"]:
        rec.update(status="ERROR", detail=f"golden 失败：{golden.get('detail')}")
        return rec
    golden_meta = golden.get("meta") or {}
    reject_evidence = golden_meta.get("reject_evidence") or {}
    cpu_rejects = bool(reject_evidence.get("cpu_rejects"))
    spec_rule_rejects = bool(reject_evidence.get("spec_rule_rejects"))
    int_spec_rule = (str(fixture.get("pred_kind", "")).startswith("int32")
                     and case.get("expected_error") == "INVALID_PREDICATE")
    if not cpu_rejects and not (spec_rule_rejects and int_spec_rule):
        rec.update(status="ERROR",
                   detail="CPU 未抛错，且该变体不是 c13 int32(0/1) 的规格规则拒绝")
        return rec
    cpu = golden_meta.get("upstream") or {}
    if cpu_rejects:
        if str(rec["upstream"]["type"]) != str(cpu.get("type")):
            rec.update(status="FAIL",
                       detail=f"upstream_exception_type={rec['upstream']['type']!r} "
                              f"!= CPU {cpu.get('type')!r}")
            return rec
    else:
        # 规格规则拒绝：CPU 无异常，接口层必须明确 type=None；Python 接口可选提供
        # upstream_evidence（spec_rule/cpu_probe），提供时按同一契约校验。
        if rec["upstream"]["type"] not in (None, "", "None", "none", "null"):
            rec.update(status="FAIL",
                       detail=f"规格规则拒绝的 upstream_exception_type 必须是 None，"
                              f"收到 {rec['upstream']['type']!r}")
            return rec
        evidence = rec["upstream"].get("evidence") or {}
        if evidence:
            if str(evidence.get("spec_rule") or "") != "scalar_bool_pred":
                rec.update(status="FAIL",
                           detail="规格规则拒绝的 upstream_evidence.spec_rule "
                                  "必须是 scalar_bool_pred")
                return rec
            probe = evidence.get("cpu_probe")
            if not isinstance(probe, dict) or probe.get("rejects") is not False or \
                    probe.get("type") is not None:
                rec.update(status="FAIL",
                           detail="规格规则拒绝的 upstream_evidence.cpu_probe "
                                  "必须记录 rejects=false/type=null")
                return rec
    rec["upstream"]["cpu_type"] = cpu.get("type")
    rec["upstream"]["cpu_sha256"] = cpu.get("sha256")
    rec["upstream"]["keyword_hit"] = reject_evidence.get("upstream_keyword_hit")
    rec["upstream"]["cpu_rejects"] = cpu_rejects
    rec["upstream"]["spec_rule_rejects"] = spec_rule_rejects
    rec["upstream"]["reject_basis"] = reject_evidence.get("reject_basis")
    if cpu_rejects:
        detail = (f"按预期拒绝：{case.get('expected_error')}；abi_calls=0；上游 "
                  f"{cpu.get('type')}（sha256 可复算）")
    else:
        detail = (f"按已确认规格规则拒绝：{case.get('expected_error')}；abi_calls=0；"
                  f"CPU 0.8.1 无异常（cpu_rejects=false），拒绝依据 scalar_bool_pred")
    rec.update(status="PASS", detail=detail)
    return rec


def _run_mode_local(case: dict, variant: str, mode: str,
                    workdir: Path | None = None) -> dict:
    """在**当前进程**执行一条 `(case, variant, mode)`（只由 fresh worker 调用）。"""
    fixture = None
    try:
        module = _candidate()
        fixture, cdir = _prepare(case, variant, mode, workdir)
        if case.get("expected_error"):
            rec = _run_reject(module, case, fixture, cdir)
        elif fixtures.base_mode_of(mode) in engine.GRAD_MODES:
            rec = _run_grad(module, case, fixture, cdir)
        else:
            rec = _run_numeric(module, case, fixture, cdir)
    except Exception as exc:  # noqa: BLE001 —— 执行器异常也必须进报告，不能丢条目
        rec = {"status": "ERROR", "dispatch": {}, "leaves": [], "evidence": {},
               "detail": f"执行器异常：{type(exc).__name__}: {exc}"}
    rec.update({
        "case_id": case["case_id"],
        "variant": variant,
        "mode": mode,
        "origin": case.get("_origin"),
        "group": case.get("group"),
        "required": bool(case.get("required", True)),
        "expected": case.get("expected", "match_cpu"),
        "expected_error": case.get("expected_error"),
        "dtype": case.get("dtype", (fixture or {}).get("dtype", "float32")),
        "seed": (fixture or {}).get("seed"),
        "note": case.get("note"),
    })
    return rec


# ---------------------------------------------------------------------------
# fresh worker 协议（workflow3 W3-4 / D3；设计 §2.2.2）
# ---------------------------------------------------------------------------


def _json_default(value):
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    return str(value)


def _worker_program_sha256(case: dict, variant: str, mode: str) -> str:
    """程序哈希 = 用例身份 + 候选源码哈希 + 测试入口哈希。"""
    digest = hashlib.sha256()
    digest.update(b"opx-cond-worker-program/1\0")
    identity = {"case_id": case.get("case_id"), "variant": variant, "mode": mode,
                "origin": case.get("_origin")}
    digest.update(json.dumps(identity, sort_keys=True).encode("utf-8"))
    digest.update(b"\0")
    for path in (CANDIDATE_PATH, Path(__file__).resolve()):
        digest.update(_sha256_file(path).encode("ascii") if path.is_file() else b"-")
        digest.update(b"\0")
    return digest.hexdigest()


def _spawn_worker(payload: dict, *, timeout: int = WORKER_TIMEOUT_S) -> dict:
    """在 fresh 子进程中执行 worker，返回结构化结果（父进程不做候选计算）。"""
    spawn_index = next(_SPAWN_COUNTER)
    body = dict(payload)
    body["spawn_index"] = spawn_index
    wdir = WORKDIR / "workers"
    wdir.mkdir(parents=True, exist_ok=True)
    payload_path = wdir / f"payload-{spawn_index:04d}.json"
    payload_path.write_text(
        json.dumps(body, ensure_ascii=False, default=_json_default), encoding="utf-8")
    # 注意：不能用 ``-I``。镜像 #145 的 AFIR/AscGen 编译器依赖 PYTHONPATH 里的
    # /usr/local/Ascend/**/python/site-packages 与 Ascend-MLIR；``-I`` 隐含 ``-E``
    # 会把它们从 sys.path 去掉，grad/AD 图在设备编译时报
    # "Failed to create AscGen instance"（workflow3 `2-探针-worker-r3.log`：
    # 同代码 ``-I`` FAIL / 无 ``-I`` PASS）。``-s`` 只忽略用户 site，保留必需路径。
    cmd = [sys.executable, "-s", str(Path(__file__).resolve()),
           "--worker", str(payload_path)]
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                            text=True, env=os.environ.copy(), cwd=str(WS))
    try:
        stdout, stderr = proc.communicate(timeout=timeout)
        timed_out = False
    except subprocess.TimeoutExpired:
        proc.kill()
        stdout, stderr = proc.communicate()
        timed_out = True
    result = None
    for line in (stdout or "").splitlines():
        if line.startswith(WORKER_RESULT_PREFIX):
            result = json.loads(line[len(WORKER_RESULT_PREFIX):])
    box = {
        "ok": result is not None and not timed_out,
        "spawn_index": spawn_index,
        "spawn_pid": proc.pid,
        "returncode": proc.returncode,
        "payload_path": str(payload_path),
        "stdout_tail": (stdout or "")[-2000:],
        "stderr_tail": (stderr or "")[-2000:],
        "result": result,
    }
    if timed_out:
        box["error"] = f"worker 超时（{timeout}s）"
    elif result is None:
        box["error"] = f"worker rc={proc.returncode} 未输出 {WORKER_RESULT_PREFIX.strip()}"
    return box


def _worker_init_backend() -> dict:
    """worker 内先初始化 PJRT/xla-npu，再让直调 .so 使用 ACL（顺序已验证）。"""
    info: dict = {"backend": None, "devices": [], "jax": None, "jaxlib": None,
                  "error": None}
    try:
        import jax  # noqa: PLC0415
        import jaxlib  # noqa: PLC0415

        info["jax"] = getattr(jax, "__version__", None)
        info["jaxlib"] = getattr(jaxlib, "__version__", None)
        info["devices"] = [str(device) for device in jax.devices()]
        info["backend"] = jax.default_backend()
    except Exception as exc:  # noqa: BLE001 —— 由用例自己按判据报 ERROR
        info["error"] = f"{type(exc).__name__}: {exc}"
    return info


class _RecordingCondModule:
    """自检用假候选：只记录 ``cond`` 实参是否 concrete，并返回选中分支结果。

    该模块**不实现**候选语义，只用于在 CPU fresh worker 内验证 S1 jit 的调用形态：
    ``jax.jit`` 程序零参数，``pred``/``operand`` 在 ``cond`` 内必须是 concrete
    ``jax.Array``，不能是 ``jax.core.Tracer``。
    """

    def __init__(self):
        # 只记录调用形态；不构造任何 dispatch/backend/token 证据，
        # 避免把 CPU 自检伪装成 NPU 路由或设备 PASS。
        self.calls: list[dict] = []

    def cond(self, pred, true_fun, false_fun, *operands):
        import jax  # noqa: PLC0415

        self.calls.append({
            "pred_is_tracer": isinstance(pred, jax.core.Tracer),
            "pred_is_concrete": bool(jax.core.is_concrete(pred)),
            "operand_is_tracer": any(
                isinstance(operand, jax.core.Tracer) for operand in operands),
            "operand_is_concrete": all(
                bool(jax.core.is_concrete(operand)) for operand in operands),
        })
        branch = true_fun if bool(np.asarray(pred)) else false_fun
        return branch(*operands)


def _selfcheck_s1_jit_worker(case: dict, variant: str, workdir: Path) -> dict:
    """fresh worker：验证 S1 jit 走零参数闭包，不把 0-d operand/pred 当动态实参。"""
    import jax  # noqa: PLC0415

    pair = fixtures.branch_pair(case)
    fixture = fixtures.make_fixture(case, variant, "jit")
    module = _RecordingCondModule()
    try:
        cdir = workdir / "selfcheck-s1-jit" / variant
        cdir.mkdir(parents=True, exist_ok=True)
        operand = _make_jax_tree(fixture)
        outputs, evidence = _s1_jit_execute(module, pair, fixture, operand, cdir)
        run_numeric_source = inspect.getsource(_run_numeric)
        ok = (
            evidence["jit_dynamic_args"] == 0
            and evidence["program_params"] == []
            and evidence["closure_constants"] == ["operand", "pred"]
            and evidence["pred_source"] == "jax_device_scalar"
            and evidence["pred_is_concrete"] is True
            and evidence["operand_is_concrete"] is True
            and len(module.calls) == 1
            and module.calls[0]["pred_is_tracer"] is False
            and module.calls[0]["pred_is_concrete"] is True
            and module.calls[0]["operand_is_tracer"] is False
            and module.calls[0]["operand_is_concrete"] is True
            and bool(getattr(outputs, "weak_type", False)) == (variant == "weak")
            and "_s1_jit_execute(" in run_numeric_source
            and "jax.jit(lambda" not in run_numeric_source
        )
        return {
            "status": "PASS" if ok else "FAIL",
            "case_id": case["case_id"], "variant": variant,
            "backend": jax.default_backend(),
            "evidence": evidence,
            "cond_calls": module.calls,
            "output_weak_type": bool(getattr(outputs, "weak_type", False)),
            "expected_weak_type": variant == "weak",
            "run_numeric_delegates_to_helper": "_s1_jit_execute(" in run_numeric_source,
            "run_numeric_has_outer_jit_lambda": "jax.jit(lambda" in run_numeric_source,
        }
    except Exception as exc:  # noqa: BLE001 —— 自检失败也必须落盘
        return {"status": "ERROR", "case_id": case.get("case_id"),
                "variant": variant,
                "detail": f"{type(exc).__name__}: {exc}"}


def _worker_main(payload_path: str) -> int:
    payload = json.loads(Path(payload_path).read_text(encoding="utf-8"))
    kind = payload.get("kind")
    spawn_index = payload.get("spawn_index")
    if kind == "selfcheck_echo":
        record = {"kind": kind, "echo": payload.get("echo"), "pid": os.getpid(),
                  "spawn_index": spawn_index}
    elif kind == "selfcheck_crash":
        return 9
    else:
        backend_info = _worker_init_backend()
        if kind == "case":
            record = _run_mode_local(payload["case"], payload["variant"],
                                     payload["mode"], Path(payload["workdir"]))
        elif kind == "d1":
            record = _diag_d1_worker(payload["case"], Path(payload["workdir"]))
        elif kind == "d2":
            record = _diag_d2_worker(payload["case"], bool(payload["pred"]),
                                     payload.get("extra"), Path(payload["workdir"]))
        elif kind == "d3":
            record = _diag_d3_worker(payload["tag"])
        elif kind == "d4":
            record = _diag_d4_worker(payload["probe"], payload["case"],
                                     Path(payload["workdir"]))
        elif kind == "selfcheck_s1_jit":
            record = _selfcheck_s1_jit_worker(
                payload["case"], payload["variant"], Path(payload["workdir"]))
        else:
            record = {"status": "ERROR", "detail": f"未知 worker kind={kind!r}"}
        record["worker_pid"] = os.getpid()
        record["pid"] = os.getpid()
        record["spawn_index"] = spawn_index
        record.update({f"worker_{key}": value for key, value in backend_info.items()
                       if key in ("backend", "devices", "jax", "jaxlib", "error")})
    print(WORKER_RESULT_PREFIX + json.dumps(
        record, ensure_ascii=False, default=_json_default), flush=True)
    return 0


def _run_mode(case: dict, variant: str, mode: str) -> dict:
    """父进程调度：每条用例在 fresh worker 中执行，父子进程不共享 JAX 编译缓存。"""
    program_sha = _worker_program_sha256(case, variant, mode)
    payload = {"kind": "case", "case": case, "variant": variant, "mode": mode,
               "workdir": str(WORKDIR), "program_sha256": program_sha}
    box = _spawn_worker(payload)
    if box.get("ok"):
        rec = dict(box["result"])
        worker_pid = rec.pop("worker_pid", None)
        rec["worker"] = {
            "mode": "fresh_subprocess",
            "pid": worker_pid,
            "spawn_pid": box["spawn_pid"],
            "spawn_index": box["spawn_index"],
            "pid_matches_spawn": bool(worker_pid) and int(worker_pid) == int(box["spawn_pid"]),
            "returncode": box["returncode"],
            "program_sha256": program_sha,
            "payload": box["payload_path"],
        }
    else:
        rec = {"status": "ERROR", "dispatch": {}, "leaves": [], "evidence": {},
               "detail": f"fresh worker 失败：{box.get('error')}",
               "worker": {
                   "mode": "fresh_subprocess",
                   "pid": None,
                   "spawn_pid": box["spawn_pid"],
                   "spawn_index": box["spawn_index"],
                   "pid_matches_spawn": False,
                   "returncode": box["returncode"],
                   "program_sha256": program_sha,
                   "payload": box["payload_path"],
                   "stderr_tail": box["stderr_tail"],
               }}
    rec.setdefault("status", "ERROR")
    rec.update({
        "case_id": case["case_id"],
        "variant": variant,
        "mode": mode,
        "origin": case.get("_origin"),
        "group": case.get("group"),
        "required": bool(case.get("required", True)),
        "expected": case.get("expected", "match_cpu"),
        "expected_error": case.get("expected_error"),
        "dtype": rec.get("dtype", case.get("dtype", "float32")),
        "seed": rec.get("seed", case.get("seed")),
        "note": case.get("note"),
    })
    # 动态谓词两种取值必须共享同一 plan_id（两个 fresh worker 各自首调；
    # plan_id 由 (branch_ids, shapes, dtypes, pred_kind) 决定，pred 值不进入 key）。
    if rec["status"] == "PASS" and mode.endswith("_false"):
        true_mode = mode[:-len("_false")] + "_true"
        prior = _CASE_MODE_RESULTS.get((case["case_id"], variant, true_mode))
        if prior and prior.get("status") == "PASS":
            id_true = (prior.get("dispatch") or {}).get("plan_id")
            id_false = (rec.get("dispatch") or {}).get("plan_id")
            if id_true and id_false and str(id_true) != str(id_false):
                rec.update(status="FAIL",
                           detail=f"jit_true plan_id={id_true!r} != jit_false plan_id={id_false!r}")
    _CASE_MODE_RESULTS[(case["case_id"], variant, mode)] = dict(rec)
    return rec


# ---------------------------------------------------------------------------
# 报告
# ---------------------------------------------------------------------------


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def build_report_payload(results: list[dict],
                         diagnostics: list[dict] | None = None) -> dict:
    diagnostics = list(diagnostics or [])
    summary = {status: sum(1 for item in results if item["status"] == status)
               for status in engine.CASE_STATUSES}
    summary.update(
        TOTAL=len(results),
        identity_count=len({item["case_id"] for item in results}),
        mode_executions=len(results),
        expected_cases=EXPECTED_TOTAL_CASES,
        expected_mode_executions=EXPECTED_TOTAL_MODES,
        expected_gate_cases=EXPECTED_GATE_CASES,
        expected_gate_mode_executions=EXPECTED_GATE_MODES,
        selfcheck_count=len(SELFCHECK_NAMES),
        selfcheck_pass=len(_SELFCHECK_PASSED & set(SELFCHECK_NAMES)),
        diagnostic_types=len(DIAGNOSTIC_NAMES),
        diagnostic_records=len(diagnostics),
        diagnostic_pass_records=sum(1 for item in diagnostics
                                    if item.get("status") == "PASS"),
        diagnostic_block_records=sum(1 for item in diagnostics
                                     if item.get("status") in
                                     ("BLOCKED", "BACKEND_LIMITATION", "RECORDED")),
        PASS_modes=sum(1 for item in results if item["status"] == "PASS"),
    )
    by_mode: dict[str, dict] = {}
    for item in results:
        bucket = by_mode.setdefault(item["mode"], {"PASS": 0, "FAIL": 0, "ERROR": 0,
                                                   "INVALID": 0, "NOT_RUN": 0})
        bucket[item["status"]] = bucket.get(item["status"], 0) + 1
    golden = next((item.get("golden") for item in results
                   if (item.get("golden") or {}).get("backend")), {})
    workers = [item.get("worker") or {} for item in results]
    worker_isolation = {
        "mode": "one fresh subprocess per (case_id, variant, mode)",
        "case_workers": len(workers),
        "pid_matches_spawn": sum(1 for item in workers
                                 if item.get("pid_matches_spawn")),
        "program_hashes": len({item.get("program_sha256") for item in workers
                               if item.get("program_sha256")}),
        "spawn_indices": len({item.get("spawn_index") for item in workers
                              if item.get("spawn_index") is not None}),
    }
    return {
        "schema": "opx-jax-cond-correctness/1",
        "op": OP,
        "cases_sha256": _sha256_file(FROZEN_CASES_PATH),
        "benchmark_sha256": _sha256_file(BENCH_PATH),
        "supplement_sha256": _sha256_file(SUPPLEMENT_PATH),
        "whitebox_sha256": _sha256_file(WHITEBOX_PATH) if WHITEBOX_PATH.is_file() else None,
        "suite_ref": os.environ.get("OPX_SUITE_REF") or None,
        "candidate": {
            "interface": f"ops/{OP}/python/{OP}.py::cond",
            "compiled_interface": f"ops/{OP}/python/{OP}.py::cond_compiled",
            "tensor_family": {
                "executed": sorted({str((item.get("dispatch") or {}).get("family"))
                                    for item in results
                                    if (item.get("dispatch") or {}).get("family")}),
                "description": "由本次真实 dispatch 生成，不对未执行的族做声明",
            },
        },
        "golden": {
            "backend": golden.get("backend", "cpu"),
            "jax": golden.get("jax"),
            "jaxlib": golden.get("jaxlib"),
        },
        "execution": {
            "model": "fresh worker subprocess per case; parent only orchestrates",
            "worker_result_prefix": WORKER_RESULT_PREFIX.strip(),
            "isolation": worker_isolation,
        },
        "cases": results,
        "diagnostics": diagnostics,
        "summary": summary,
        "by_mode": by_mode,
    }


def _render_md(payload: dict) -> str:
    lines = [
        f"# {OP} · pytest 正确性报告",
        "",
        f"- 用例集：`cases/{OP}/cases.json`（sha256 `{payload['cases_sha256'][:16]}…`）"
        f" + benchmark + supplement",
        f"- suite_ref：`{payload['suite_ref']}` ｜ golden：CPU JAX {payload['golden'].get('jax')}",
        f"- 汇总：{payload['summary']}",
        f"- 按 mode：{payload['by_mode']}",
        f"- 进程隔离：{payload['execution']['isolation']}",
        "",
        "| 用例 | 来源 | variant | mode | 期望 | 状态 | matched_ratio | max_abs_error | 说明 |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for item in payload["cases"]:
        lines.append(
            f"| {item['case_id']} | {item.get('origin', '')} | {item.get('variant') or '—'} | "
            f"{item.get('mode', '')} | {item.get('expected')} | **{item['status']}** | "
            f"{item.get('matched_ratio')} | {item.get('max_abs_error')} | "
            f"{(item.get('detail') or '').replace('|', '/')[:140]} |")
    lines += [
        "",
        "> 本任务没有合法 xfail/unsupported：任何非 PASS 都不计入通过数。",
        "> 判定口径与门禁同为 criteria/check.py；结构/shape/dtype 精确；",
        "> jax family weak_type 逐叶精确；非有限位逐元素精确；候选必须在非 CPU 后端执行。",
        "> 每条用例在 fresh worker 子进程中执行；父进程只调度/聚合，不共享 AFIR 编译缓存。",
    ]
    diagnostics = payload.get("diagnostics") or []
    if diagnostics:
        lines += [
            "",
            "## workflow3 迁移专项诊断（D1–D4；不计入 pytest 用例计数）",
            "",
            "| 诊断 | 状态 | 说明 |",
            "|---|---|---|",
        ]
        for item in diagnostics:
            lines.append(
                f"| {item.get('diagnostic') or item.get('case_id')} | "
                f"**{item.get('status')}** | "
                f"{(item.get('detail') or '').replace('|', '/')[:220]} |")
        lines.append("")
        lines.append("> D1–D4 不进入 pytest 用例 PASS 计数；"
                     "BLOCKED/BACKEND_LIMITATION 不是 PASS。")
    return "\n".join(lines) + "\n"


def write_report(results: list[dict], out_dir: Path | None = None,
                 diagnostics: list[dict] | None = None) -> tuple[Path, Path]:
    out = out_dir or Path(os.environ.get("OPX_OUT", "/out"))
    out.mkdir(parents=True, exist_ok=True)
    payload = build_report_payload(results, diagnostics)
    json_path = out / "cond_correctness.json"
    md_path = out / "cond_correctness.md"
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2,
                                    default=_json_default), encoding="utf-8")
    md_path.write_text(_render_md(payload), encoding="utf-8")
    return json_path, md_path


# 进程隔离说明（workflow3）：候选不再在 pytest 主进程执行。每个用例由
# `_spawn_worker` 拉起 fresh 子进程，worker 内先 `jax.devices()` 初始化
# PJRT/xla-npu，再调用候选，保持 `2-探针-r4-pytest-mix2.log` 验证过的
# JAX→ACL 顺序；父进程只聚合 JSON/日志，不持有 JAX/AFIR 编译缓存。


@pytest.fixture(scope="session", autouse=True)
def _session_report_writer():
    yield
    if _RESULTS or _DIAGNOSTICS:
        paths = write_report(_RESULTS, diagnostics=_DIAGNOSTICS)
        print(f"\nCOND_REPORT {paths[0]} {paths[1]}")


# ---------------------------------------------------------------------------
# 正确性用例（23 条黑盒 + 6 条白盒 = 29 个身份 / 83 个 (case_id, variant, mode) 参数化项）
# ---------------------------------------------------------------------------


@pytest.mark.opx_case
@pytest.mark.parametrize(
    "case,variant,mode", ITEMS,
    ids=[f"{case['case_id']}[{variant or 'default'}][{mode}]"
         for case, variant, mode in ITEMS])
def test_jax_lax_cond_case(case, variant, mode):
    rec = _run_mode(case, variant, mode)
    _RESULTS.append(rec)
    print("COND_CASE " + json.dumps(rec, ensure_ascii=False, default=_json_default))
    if rec["status"] == "PASS":
        return
    pytest.fail(f"{case['case_id']}/{variant or 'default'}/{mode}: "
                f"{rec['status']} - {rec.get('detail')}")


# ---------------------------------------------------------------------------
# 自检（CPU 可跑；不需要候选/设备；不计入 23/71）
# ---------------------------------------------------------------------------


def test_selfcheck_discovery_counts():
    assert len(FROZEN_SUITE["cases"]) == EXPECTED_GATE_CASES, (
        f"门禁冻结用例数变成 {len(FROZEN_SUITE['cases'])}，期望 {EXPECTED_GATE_CASES}")
    assert GATE_MODE_COUNT == EXPECTED_GATE_MODES, (
        f"门禁 mode 次数变成 {GATE_MODE_COUNT}，期望 {EXPECTED_GATE_MODES}")
    assert len(SUPPLEMENT_SUITE["cases"]) == EXPECTED_SUPPLEMENT_CASES, (
        f"补充用例数变成 {len(SUPPLEMENT_SUITE['cases'])}，期望 {EXPECTED_SUPPLEMENT_CASES}")
    assert len(BASE_CASES) == EXPECTED_PYTEST_CASES, (
        f"pytest 身份数（不含白盒）变成 {len(BASE_CASES)}，期望 {EXPECTED_PYTEST_CASES}")
    assert BASE_MODE_COUNT == EXPECTED_PYTEST_MODES, (
        f"pytest mode 参数化项（不含白盒）变成 {BASE_MODE_COUNT}，期望 {EXPECTED_PYTEST_MODES}")
    assert len({case["case_id"] for case in BASE_CASES}) == EXPECTED_PYTEST_CASES
    assert len(WHITEBOX_SUITE.get("cases") or []) == EXPECTED_WHITEBOX_CASES
    assert len(CASES) == EXPECTED_TOTAL_CASES
    assert MODE_COUNT == EXPECTED_TOTAL_MODES, (
        f"pytest (case, variant, mode) 项变成 {MODE_COUNT}，期望 {EXPECTED_TOTAL_MODES}")
    assert len(fixtures.expand_items(WHITEBOX_SUITE.get("cases") or [])) == \
        EXPECTED_WHITEBOX_MODES
    origins: dict[str, int] = {}
    for case in CASES:
        origins[case["_origin"]] = origins.get(case["_origin"], 0) + 1
    expected_origins = {"frozen": 16, "benchmark": 2, "supplement": 5}
    if EXPECTED_WHITEBOX_CASES:
        expected_origins["whitebox"] = EXPECTED_WHITEBOX_CASES
    assert origins == expected_origins, f"来源分布变化：{origins}"
    assert len(SELFCHECK_NAMES) == len(set(SELFCHECK_NAMES)) == 15
    assert len(DIAGNOSTIC_NAMES) == len(set(DIAGNOSTIC_NAMES)) == 4
    # 冻结 49 / 基准 4 / 补充 18 的分解必须与设计 §6.4 一致
    expanded = fixtures.expand_items(FROZEN_SUITE["cases"])
    assert len(expanded) == 49
    assert len(fixtures.expand_items(BENCH_SUITE["cases"])) == 4
    assert len(fixtures.expand_items(SUPPLEMENT_SUITE["cases"])) == 18
    assert len(expanded) + 4 + 18 == EXPECTED_PYTEST_MODES
    _mark_selfcheck("test_selfcheck_discovery_counts")


def test_selfcheck_smoke_selection():
    selected = engine.pick_smoke_case(FROZEN_SUITE["cases"])
    assert selected["case_id"] == "cond_true_eager"
    assert selected.get("expected") == "match_cpu" and "eager" in selected.get("mode")
    old_rule = [case for case in FROZEN_SUITE["cases"]
                if case.get("expected", "match") == "match"
                and isinstance(case.get("n_a"), int) and case.get("n_a") == case.get("n_b")]
    assert old_rule == [], "旧 elementwise 规则不应命中 cond 用例"
    fixture = fixtures.make_fixture(selected, "", "eager")
    assert fixture["numel"] == 128 and fixture["output_shapes"] == {"out": [8, 16]}
    with pytest.raises(SystemExit):
        engine.pick_smoke_case([case for case in FROZEN_SUITE["cases"]
                                if case["case_id"] != "cond_true_eager"])
    _mark_selfcheck("test_selfcheck_smoke_selection")


def test_selfcheck_data_generation_frozen_rules():
    case = next(item for item in FROZEN_SUITE["cases"]
                if item["case_id"] == "cond_true_eager")
    f1 = fixtures.make_fixture(case, "", "eager")
    f2 = fixtures.make_fixture(case, "", "eager")
    assert f1["inputs_sha256"] == f2["inputs_sha256"]
    expected = np.random.default_rng(20261027).uniform(
        -1.0, 1.0, size=(8, 16)).astype(np.float32)
    assert np.array_equal(f1["arrays"]["operands/x"], expected)
    assert f1["arrays"]["operands/x"].dtype == np.float32
    assert f1["pred_kind"] == "python_bool" and f1["pred"] is True

    # c05：dict 按 key 字典序、同一个 rng 连续取样
    pytree = next(item for item in FROZEN_SUITE["cases"]
                  if item["case_id"] == "cond_input_pytree")
    fp = fixtures.make_fixture(pytree, "", "eager_true")
    assert fp["leaf_order"] == ["operands/a", "operands/b"]
    rng = np.random.default_rng(20261027)
    a = rng.uniform(-1.0, 1.0, size=(6, 4)).astype(np.float32)
    b = rng.uniform(-1.0, 1.0, size=(6, 2)).astype(np.float32)
    assert np.array_equal(fp["arrays"]["operands/a"], a)
    assert np.array_equal(fp["arrays"]["operands/b"], b)
    assert fp["pred_kind"] == "scalar_bool_array"

    # c11：17/33 元素，68/132 B，bytes%32=4（实算）
    assert fixtures.leaf_bytes([17]) == {"elements": 17, "dtype": "float32",
                                        "dtype_size": 4, "bytes": 68, "bytes_mod_32": 4}
    assert fixtures.leaf_bytes([33]) == {"elements": 33, "dtype": "float32",
                                        "dtype_size": 4, "bytes": 132, "bytes_mod_32": 4}
    c11 = next(item for item in FROZEN_SUITE["cases"]
               if item["case_id"] == "cond_unaligned_tail_widths")
    assert fixtures.variants_for(c11) == ["w17", "w33"]
    assert list(fixtures.make_fixture(c11, "w17", "eager_false")["arrays"].values())[0].shape == (17,)
    assert list(fixtures.make_fixture(c11, "w33", "eager_false")["arrays"].values())[0].shape == (33,)

    # c13：四个非法 pred 变体
    c13 = next(item for item in FROZEN_SUITE["cases"]
               if item["case_id"] == "cond_invalid_predicate")
    assert fixtures.variants_for(c13) == ["bool1", "bool2", "int0", "int1"]
    kinds = {v: fixtures.pred_value(c13, v, "eager")[1] for v in ("bool1", "bool2", "int0", "int1")}
    assert kinds == {"bool1": "bool_array_1", "bool2": "bool_array_2",
                     "int0": "int32_scalar", "int1": "int32_scalar"}

    # S5：特殊值冻结
    s5 = next(item for item in SUPPLEMENT_SUITE["cases"]
              if item["case_id"] == "cond_sup_nonfinite")
    fs5 = fixtures.make_fixture(s5, "", "eager_true")
    assert np.array_equal(fs5["arrays"]["operands/x"], fixtures.NONFINITE_VALUES,
                          equal_nan=True)
    _mark_selfcheck("test_selfcheck_data_generation_frozen_rules")


def test_selfcheck_criteria_thresholds():
    good = CRITERIA_MOD.check(np.ones(8, np.float32), np.ones(8, np.float32))
    assert good["is_pass"] is True
    assert good["profile"] == "ops-precision-standard@1"
    assert good["rtol"] == 2 ** -10 and good["atol"] == 2 ** -16
    assert good["required_matched_ratio"] == 0.99
    assert good["max_abs_error_limit"] == 1e-2
    bad = CRITERIA_MOD.check(np.ones(8, np.float32), np.zeros(8, np.float32))
    assert bad["is_pass"] is False
    nan_ok = CRITERIA_MOD.check(np.array([np.nan, 1.0], np.float32),
                                np.array([np.nan, 1.0], np.float32))
    assert nan_ok["is_pass"] is True and nan_ok["nonfinite"]["count"] == 1
    inf_bad = CRITERIA_MOD.check(np.array([np.inf, 1.0], np.float32),
                                 np.array([-np.inf, 1.0], np.float32))
    assert inf_bad["is_pass"] is False
    _mark_selfcheck("test_selfcheck_criteria_thresholds")


class _FakeDevice:
    def __init__(self, platform: str):
        self.platform = platform

    def __str__(self):
        return self.platform


class _FakeJaxLeaf:
    def __init__(self, platform: str):
        self._device = _FakeDevice(platform)

    def devices(self):
        return [self._device]


def test_selfcheck_device_guard_negative():
    with pytest.raises(AssertionError):
        assert_outputs_on_accelerator(np.zeros(4, np.float32), "neg_no_dispatch", {})
    with pytest.raises(AssertionError):
        assert_outputs_on_accelerator(np.zeros(4, np.float32), "neg_trace",
                                      {"path": "trace", "backend": "npu", "abi_calls": 0})
    with pytest.raises(AssertionError):
        assert_outputs_on_accelerator(np.zeros(4, np.float32), "neg_cpu",
                                      {"path": "so", "backend": "cpu", "abi_calls": 2})
    with pytest.raises(AssertionError):
        assert_outputs_on_accelerator(_FakeJaxLeaf("cpu"), "neg_jax_cpu", {})
    with pytest.raises(AssertionError):
        assert_outputs_on_accelerator((_FakeJaxLeaf("npu"), _FakeJaxLeaf("cpu")),
                                      "neg_jax_mixed", {})
    recorded = assert_outputs_on_accelerator(
        np.zeros(4, np.float32), "pos_numpy",
        {"path": "so", "backend": "npu", "abi_calls": 2, "device_id": 0})
    assert recorded[0]["kind"] == "numpy-host-copy"
    _mark_selfcheck("test_selfcheck_device_guard_negative")


def test_selfcheck_golden_env_guard_negative(tmp_path):
    case = next(item for item in FROZEN_SUITE["cases"]
                if item["case_id"] == "cond_true_eager")
    fixture = fixtures.make_fixture(case, "", "eager")
    case_json = tmp_path / "case.json"
    inputs = tmp_path / "inputs.npz"
    fixtures.write_case_json(case, fixture, case_json)
    fixtures.write_inputs_npz(fixture, inputs)
    env = os.environ.copy()
    env["JAX_PLATFORMS"] = "cpu"
    env["JAX_ENABLE_X64"] = "0"
    env.pop("XLA_FLAGS", None)
    env.pop("JAX_COMPILATION_CACHE_DIR", None)
    proc = subprocess.run(
        [sys.executable, "-I", str(GOLDEN_CLI), "--case", str(case_json),
         "--inputs", str(inputs), "--npz", str(tmp_path / "golden.npz"),
         "--meta", str(tmp_path / "golden.json"), "--mode", "eager",
         "--expect-backend", "npu"],
        capture_output=True, text=True, timeout=180, env=env,
    )
    assert proc.returncode == 3, f"rc={proc.returncode} stderr={proc.stderr[-400:]}"
    assert "GOLDEN_ENV_MISMATCH" in (proc.stderr or "")
    _mark_selfcheck("test_selfcheck_golden_env_guard_negative")


def test_selfcheck_golden_reject_rules(tmp_path):
    """拒绝规则全部走 golden 子进程（不在 pytest 进程执行 JAX），覆盖 CPU 可确认/不可确认两类。"""
    reject_cases = [case for case in FROZEN_SUITE["cases"]
                    if case.get("expected_error")]
    assert [case["case_id"] for case in reject_cases] == [
        "cond_invalid_predicate", "cond_branch_structure_mismatch",
        "cond_branch_shape_mismatch", "cond_branch_dtype_mismatch"]

    def run_reject(case, variant, mode, tag):
        fixture = fixtures.make_fixture(case, variant, mode)
        cdir = tmp_path / tag
        cdir.mkdir(parents=True, exist_ok=True)
        case_json = cdir / "case.json"
        inputs = cdir / "inputs.npz"
        fixtures.write_case_json(case, fixture, case_json)
        fixtures.write_inputs_npz(fixture, inputs)
        return engine._run_golden(WS, case_json, inputs, "reject", variant, cdir)

    confirmed = [
        ("cond_invalid_predicate", "bool1", "eager"),
        ("cond_invalid_predicate", "bool2", "jit"),
        ("cond_branch_structure_mismatch", "", "eager"),
        ("cond_branch_shape_mismatch", "", "jit"),
        ("cond_branch_dtype_mismatch", "", "eager"),
    ]
    for index, (case_id, variant, mode) in enumerate(confirmed):
        case = next(item for item in reject_cases if item["case_id"] == case_id)
        result = run_reject(case, variant, mode, f"reject-ok-{index}")
        assert result["ok"], (case_id, variant, result.get("detail"))
        upstream = (result["meta"] or {}).get("upstream") or {}
        assert upstream.get("type") == "TypeError", (case_id, upstream)
        assert upstream.get("message")
        assert upstream.get("sha256") == hashlib.sha256(
            upstream["message"].encode("utf-8")).hexdigest()
    # 整型 pred：JAX 0.8.1 CPU 按真值接受（不抛错）；已确认 spec 的 scalar-bool 规则
    # 仍要求 INVALID_PREDICATE。CLI 必须 rc=0，并同时保留“CPU 无异常”和“规格规则拒绝”
    # 两类证据，不能伪造上游异常，也不能把 CPU 的接受行为伪装成拒绝。
    c13 = next(item for item in reject_cases if item["case_id"] == "cond_invalid_predicate")
    for variant in ("int0", "int1"):
        result = run_reject(c13, variant, "eager", f"reject-int-{variant}")
        assert result["ok"] and result["rc"] == 0, (variant, result.get("rc"),
                                                    result.get("detail"))
        meta = json.loads((tmp_path / f"reject-int-{variant}" / "golden_reject.json")
                          .read_text(encoding="utf-8"))
        assert meta["status"] == "reject"
        evidence = meta.get("reject_evidence") or {}
        assert evidence.get("cpu_rejects") is False
        assert evidence.get("spec_rule_rejects") is True
        assert evidence.get("reject_basis") == "spec_rule"
        assert (meta.get("upstream") or {}).get("type") is None
        assert (meta.get("upstream") or {}).get("spec_rule") == "scalar_bool_pred"
    # 合法用例不得被 CLI 误判为拒绝（expect_reject 抛 ValueError → rc=4）
    legal = next(item for item in FROZEN_SUITE["cases"]
                 if item["case_id"] == "cond_true_eager")
    result = run_reject(legal, "", "eager", "reject-legal")
    assert result["ok"] is False and result["rc"] == 4, result.get("rc")
    _mark_selfcheck("test_selfcheck_golden_reject_rules")


def test_selfcheck_golden_signature_guard():
    """在独立 CPU 子进程里验证 golden 签名守卫，父进程不 import jax。"""
    golden_path = GOLDEN_DIR / "golden.py"
    code = (
        "import sys, numpy as np\n"
        f"sys.path.insert(0, {str(GOLDEN_DIR)!r})\n"
        "import importlib.util\n"
        f"spec = importlib.util.spec_from_file_location('cond_golden_selfcheck', "
        f"{str(golden_path)!r})\n"
        "module = importlib.util.module_from_spec(spec)\n"
        "spec.loader.exec_module(module)\n"
        "try:\n"
        "    module.compute_golden(np.zeros(4, np.float32), np.zeros(4, np.float32))\n"
        "except TypeError:\n"
        "    raise SystemExit(0)\n"
        "raise SystemExit(1)\n"
    )
    env = os.environ.copy()
    env["JAX_PLATFORMS"] = "cpu"
    env["JAX_ENABLE_X64"] = "0"
    env["PYTHONHASHSEED"] = "0"
    env.pop("XLA_FLAGS", None)
    env.pop("JAX_COMPILATION_CACHE_DIR", None)
    proc = subprocess.run([sys.executable, "-I", "-c", code],
                          capture_output=True, text=True, timeout=180, env=env)
    assert proc.returncode == 0, (proc.returncode, proc.stderr[-600:])
    _mark_selfcheck("test_selfcheck_golden_signature_guard")


def _numpy_reference(case: dict, fixture: dict) -> dict:
    case_id = case["case_id"]
    arrays = fixture["arrays"]
    if case_id == "cond_input_pytree":
        return {"out/a": arrays["operands/a"] + 1.0,
                "out/b": arrays["operands/b"] + 1.0}
    if case_id == "cond_output_pytree":
        x = arrays["operands/x"]
        return {"out/0": x + 1.0, "out/1/aux": x[:, 0] + 1.0}
    x = arrays[fixture["leaves"][0]["path"]]
    if case_id == "cond_jit_static_pred":
        return {"out": x * 2.0}
    if case_id == "cond_sup_grad_traces_both":
        return {"grad": np.full_like(x, 2.0)}
    return {"out": x + 1.0}


def test_selfcheck_golden_numeric_binding(tmp_path):
    """代表用例的 golden 参数绑定：golden_cli（CPU 子进程）vs 独立 numpy 语义。"""
    case_ids = ("cond_true_eager", "cond_jit_static_pred", "cond_input_pytree",
                "cond_output_pytree", "cond_sup_grad_traces_both")
    for case_id in case_ids:
        case = next(item for item in CASES if item["case_id"] == case_id)
        variant = fixtures.variants_for(case)[0]
        mode = fixtures.modes_for(case)[0]
        fixture = fixtures.make_fixture(case, variant, mode)
        cdir = tmp_path / case_id
        cdir.mkdir(parents=True, exist_ok=True)
        case_json = cdir / "case.json"
        inputs = cdir / "inputs.npz"
        fixtures.write_case_json(case, fixture, case_json)
        fixtures.write_inputs_npz(fixture, inputs)
        golden = engine._run_golden(WS, case_json, inputs,
                                    fixtures.base_mode_of(mode), variant, cdir)
        assert golden["ok"], f"{case_id}: {golden.get('detail')}"
        reference = _numpy_reference(case, fixture)
        assert set(golden["arrays"]) == set(reference), (case_id, sorted(golden["arrays"]))
        for key, want in reference.items():
            got = np.asarray(golden["arrays"][key], dtype=np.float32)
            verdict = CRITERIA_MOD.check(got, np.asarray(want, dtype=np.float32))
            assert verdict["is_pass"], (case_id, key, verdict.get("failure_reasons"))
        if case_id == "cond_sup_grad_traces_both":
            assert np.array_equal(np.asarray(golden["arrays"]["grad"]),
                                  np.full((5, 4), 2.0, np.float32))
    _mark_selfcheck("test_selfcheck_golden_numeric_binding")


def test_selfcheck_report_writer(tmp_path):
    synthetic = {
        "case_id": "selfcheck_report_writer", "variant": "", "mode": "eager",
        "origin": "selfcheck", "required": False, "expected": "match_cpu",
        "status": "NOT_RUN", "detail": "仅验证报告写入链路，不是算子执行结果",
        "dispatch": {}, "golden": {}, "leaves": [],
    }
    json_path, md_path = write_report([synthetic], out_dir=tmp_path)
    payload = json.loads(json_path.read_text(encoding="utf-8"))
    assert payload["schema"] == "opx-jax-cond-correctness/1"
    assert payload["cases"][0]["case_id"] == "selfcheck_report_writer"
    assert payload["summary"]["expected_cases"] == EXPECTED_TOTAL_CASES
    assert payload["summary"]["expected_mode_executions"] == EXPECTED_TOTAL_MODES
    assert payload["summary"]["selfcheck_count"] == len(SELFCHECK_NAMES)
    assert json_path.stat().st_size > 0 and md_path.stat().st_size > 0
    _mark_selfcheck("test_selfcheck_report_writer")


def test_selfcheck_whitebox_hook(tmp_path):
    whitebox = tmp_path / f"{OP}_whitebox_cases.json"
    whitebox.write_text(json.dumps({
        "schema": "opx-cases/1", "op": OP, "criteria_ref": "ops-precision-standard@1",
        "cases": [{
            "case_id": "cond_wb_selfcheck", "group": "白盒自检", "required": True,
            "operand": "float32[4]", "branches": ["x+1", "x-1"], "pred": True,
            "dtype": "float32", "mode": ["eager"], "expected": "match_cpu",
            "note": "仅验证白盒 loader 接入，不参与正式计数",
        }],
    }), encoding="utf-8")
    cases, suites = _load_cases(whitebox_path=whitebox)
    ids = [case["case_id"] for case in cases]
    assert "cond_wb_selfcheck" in ids
    assert suites["whitebox"]["cases"][0]["case_id"] == "cond_wb_selfcheck"
    assert len(cases) == EXPECTED_PYTEST_CASES + 1
    _mark_selfcheck("test_selfcheck_whitebox_hook")


def test_selfcheck_whitebox_cases():
    """锁定节点 3 白盒用例的身份、tile/尾块/多核/跨步/8 叶边界与字节实算。"""
    expected_ids = [
        "cond_wb_tile_exact_8192",
        "cond_wb_tile_tail_8193",
        "cond_wb_multicore_grid_stride",
        "cond_wb_strided_src_full_tile",
        "cond_wb_jax_pytree_dict",
        "cond_wb_max_leaves_8",
    ]
    cases = list(WHITEBOX_SUITE.get("cases") or [])
    assert [case["case_id"] for case in cases] == expected_ids
    assert len(fixtures.expand_items(cases)) == EXPECTED_WHITEBOX_MODES
    by_id = {case["case_id"]: case for case in cases}

    exact = fixtures.make_fixture(by_id["cond_wb_tile_exact_8192"], "", "eager_true")
    assert exact["numel"] == 8192
    assert fixtures.leaf_bytes(exact["output_shapes"]["out"], exact["dtype"]) == {
        "elements": 8192, "dtype": "float32", "dtype_size": 4,
        "bytes": 32768, "bytes_mod_32": 0}

    tail = fixtures.make_fixture(by_id["cond_wb_tile_tail_8193"], "", "eager_false")
    assert tail["numel"] == 8193
    assert (tail["numel"] + 8191) // 8192 == 2
    assert fixtures.leaf_bytes(tail["output_shapes"]["out"], tail["dtype"])["bytes_mod_32"] == 4

    multi = fixtures.make_fixture(by_id["cond_wb_multicore_grid_stride"], "", "eager_true")
    assert multi["numel"] == 524288
    assert (multi["numel"] + 8191) // 8192 == 64

    strided = _build_fixture(by_id["cond_wb_strided_src_full_tile"], "", "eager_true")
    assert strided["true_id"] == 7 and strided["false_id"] == 8
    assert strided["output_shapes"] == {"out/0": [64, 128], "out/1/aux": [64]}
    assert fixtures.leaf_bytes(strided["output_shapes"]["out/1/aux"], strided["dtype"])[
        "bytes"] == 256

    jax_tree = _build_fixture(by_id["cond_wb_jax_pytree_dict"], "", "eager_false")
    assert jax_tree["family"] == "jax" and jax_tree["kind"] == "dict"
    assert jax_tree["true_id"] == 5 and jax_tree["false_id"] == 6
    assert jax_tree["output_shapes"] == {"out/a": [3, 5], "out/b": [4]}

    leaves8 = fixtures.make_fixture(by_id["cond_wb_max_leaves_8"], "", "eager_true")
    assert len(leaves8["leaves"]) == 8 and len(leaves8["output_shapes"]) == 8
    assert leaves8["numel"] == 55
    _mark_selfcheck("test_selfcheck_whitebox_cases")


def test_selfcheck_engine_dispatch_negative():
    good = {
        "op": OP, "kernel": "jax_lax_cond_kernel", "arch": "dav-2201",
        "mode": "eager", "variant": "", "pred_kind": "python_bool",
        "selected": "true", "path": "so", "exec_mode": "selected_only",
        "abi_calls": "1", "backend": "npu", "device_id": "0",
        "trace_true": "1", "trace_false": "1", "exec_true": "1", "exec_false": "0",
        "runtime_token": "1", "compile_count": "0", "plan_id": "plan",
        "plan_cache": "n/a",
    }
    fixture = fixtures.make_fixture(
        next(item for item in FROZEN_SUITE["cases"]
             if item["case_id"] == "cond_true_eager"), "", "eager")
    assert engine._exec_problems(good, fixture) == []
    for name, patch in {
        "missing_key": {"op": None},
        "cpu_backend": {"backend": "cpu"},
        "trace_path": {"path": "trace"},
        "abi_zero": {"abi_calls": "0"},
        "wrong_selected": {"selected": "false"},
        "no_trace_false": {"trace_false": "0"},
        "exec_both": {"exec_true": "1", "exec_false": "1"},
        "bad_token": {"runtime_token": "2"},
    }.items():
        bad = dict(good)
        if patch.get("op", "") is None:
            bad.pop("op", None)
        else:
            bad.update(patch)
        assert engine._exec_problems(bad, fixture), name

    # 拒绝 dispatch（pytest 接口层）：path/exec_mode/abi_calls 三项都必须查。
    good_reject = {"path": "reject", "exec_mode": "reject", "abi_calls": 0}
    assert _reject_dispatch_problems(good_reject) == []
    for name, patch in {
        "reject_path": {"path": "so"},
        "reject_exec_mode": {"exec_mode": "selected_only"},
        "reject_abi": {"abi_calls": 1},
    }.items():
        bad = dict(good_reject)
        bad.update(patch)
        assert _reject_dispatch_problems(bad), name

    # CLI meta 契约（门禁层）：status/leaves/upstream 与 dispatch 必须逐项校验。
    good_meta = {
        "status": "reject",
        "dispatch": {"path": "reject", "abi_calls": 0},
        "leaves": [],
        "upstream": {"type": None, "message": "spec rule", "sha256": "0" * 64},
    }
    assert engine.reject_meta_problems(good_meta) == []
    for name, patch in {
        "meta_status": {"status": "ok"},
        "meta_path": {"dispatch": {"path": "so", "abi_calls": 0}},
        "meta_abi": {"dispatch": {"path": "reject", "abi_calls": 1}},
        "meta_leaves": {"leaves": [{"path": "out"}]},
        "meta_upstream": {"upstream": {}},
    }.items():
        bad = json.loads(json.dumps(good_meta))
        bad.update(patch)
        assert engine.reject_meta_problems(bad), name
    _mark_selfcheck("test_selfcheck_engine_dispatch_negative")


# ---------------------------------------------------------------------------
# workflow3 迁移专项诊断（D1–D4；设计 §5.6 / §7.4；不计入 23/71）
# ---------------------------------------------------------------------------


def _record_diagnostic(record: dict) -> None:
    _DIAGNOSTICS.append(record)
    print("COND_DIAG " + json.dumps(record, ensure_ascii=False, default=_json_default))


def _spawn_diagnostic(payload: dict, timeout: int = WORKER_TIMEOUT_S) -> dict:
    box = _spawn_worker(payload, timeout=timeout)
    if box.get("ok"):
        record = dict(box["result"])
        record["worker"] = {
            "mode": "fresh_subprocess",
            "pid": record.get("worker_pid") or record.get("pid"),
            "spawn_pid": box["spawn_pid"], "spawn_index": box["spawn_index"],
            "pid_matches_spawn": bool(record.get("worker_pid") or record.get("pid"))
            and int(record.get("worker_pid") or record.get("pid")) == int(box["spawn_pid"]),
            "returncode": box["returncode"],
        }
        return record
    return {"diagnostic": payload.get("kind", "?").upper(), "status": "ERROR",
            "detail": f"fresh worker 失败：{box.get('error')}",
            "stderr_tail": box["stderr_tail"], "spawn_pid": box["spawn_pid"]}


def _diag_hlo_flags(text: str) -> dict:
    return {op: bool(op in text) for op in UNSUPPORTED_HLO_OPS}


def _diag_lower_compile(fn, args: tuple, tag: str, workdir: Path):
    import jax  # noqa: PLC0415

    lowered = jax.jit(fn).lower(*args)
    text = lowered.as_text()
    hdir = workdir / "d2"
    hdir.mkdir(parents=True, exist_ok=True)
    path = hdir / f"hlo-{tag}.mlir"
    path.write_text(text, encoding="utf-8")
    return lowered.compile(), {"path": str(path), "sha256": hashlib.sha256(
        text.encode("utf-8")).hexdigest(), "flags": _diag_hlo_flags(text),
        "text_head": text[:400]}


def _diag_judge(actual, want, path: str) -> dict:
    got = np.ascontiguousarray(_to_numpy(actual), dtype=np.float32)
    expected = np.ascontiguousarray(np.asarray(want), dtype=np.float32)
    if got.shape != expected.shape:
        return {"path": path, "is_pass": False, "matched_ratio": 0.0,
                "max_abs_error": None,
                "detail": f"shape {got.shape} != {expected.shape}"}
    return {"path": path, **CRITERIA_MOD.check(got, expected)}


def _diag_platforms(value) -> list[str]:
    platforms, _kind = _leaf_platforms(value)
    return sorted(platforms)


def _device_scalar_info(value) -> dict:
    if callable(getattr(value, "block_until_ready", None)):
        value = value.block_until_ready()
    array = np.asarray(value).reshape(())
    return {"value": float(array), "platforms": _diag_platforms(value),
            "sha256": hashlib.sha256(array.tobytes()).hexdigest()}


def _diag_d1_worker(case: dict, workdir: Path) -> dict:
    """D1：设备标量 sum(x)>0 派生 pred；x / -x 必须翻转 token，且共享同一 plan。"""
    import jax  # noqa: PLC0415
    import jax.numpy as jnp  # noqa: PLC0415

    backend = jax.default_backend()
    if backend == "cpu":
        return {"diagnostic": "D1", "status": "BLOCKED", "device_required": True,
                "backend": backend,
                "detail": "D1 需要非 CPU 后端；CPU 自检不产生 PASS"}
    module = _candidate()
    fixture = fixtures.make_fixture(case, "", "jit_true")
    pair = fixtures.branch_pair(case)
    leaf = fixture["leaves"][0]["path"]
    x_np = np.ascontiguousarray(np.asarray(fixture["arrays"][leaf], dtype=np.float32))
    runs = []
    try:
        for name, data in (("x", x_np), ("neg_x", np.ascontiguousarray(-x_np))):
            scalar = _device_scalar_info(jnp.sum(jnp.asarray(data)))
            pred_np = np.array(scalar["value"] > 0.0, dtype=np.bool_)
            mode = "jit_true" if bool(pred_np) else "jit_false"
            data_fixture = fixtures.make_fixture(case, "", mode)
            data_fixture["arrays"] = {**data_fixture["arrays"], leaf: data}
            data_fixture["inputs_sha256"] = fixtures.compute_arrays_sha256(
                data_fixture["arrays"])
            cdir = workdir / "d1" / name
            cdir.mkdir(parents=True, exist_ok=True)
            fixtures.write_case_json(case, data_fixture, cdir / "case.json")
            fixtures.write_inputs_npz(data_fixture, cdir / "inputs.npz")
            golden = engine._run_golden(WS, cdir / "case.json",
                                        cdir / "inputs.npz", "jit", "", cdir)
            if not golden["ok"]:
                raise RuntimeError(f"golden 失败：{golden.get('detail')}")
            module._reset_evidence()
            out = module.cond_compiled(pred_np, pair["true_fn"], pair["false_fn"], data)
            dispatch = _norm_dispatch(module)
            leaves, problems = _judge_outputs(out, golden["arrays"], golden["meta"],
                                              data_fixture, dispatch)
            hits = dispatch.get("exec_hits") or {}
            trace = dispatch.get("trace_hits") or {}
            runs.append({
                "name": name, "device_scalar": scalar["value"],
                "device_scalar_platforms": scalar["platforms"],
                "device_scalar_sha256": scalar["sha256"],
                "pred": bool(pred_np), "token": dispatch.get("runtime_token"),
                "selected": dispatch.get("selected"),
                "exec_hits": hits, "trace_hits": trace,
                "plan_id": str(dispatch.get("plan_id") or ""),
                "plan_cache": dispatch.get("plan_cache"),
                "path": dispatch.get("path"), "backend": dispatch.get("backend"),
                "abi_calls": dispatch.get("abi_calls"),
                "device_id": dispatch.get("device_id"),
                "leaves": leaves, "problems": problems,
                "is_pass": bool(not problems and all(leaf.get("is_pass") for leaf in leaves)),
                "golden_sha256": golden["meta"].get("golden_sha256"),
            })
        plan_ids = {run["plan_id"] for run in runs}
        plan_same = all(plan_ids) and len(plan_ids) == 1
        ok = bool(
            runs[0]["pred"] != runs[1]["pred"]
            and runs[0]["token"] != runs[1]["token"]
            and all(run["is_pass"] for run in runs)
            and all(run["path"] == "so" and run["backend"] == "npu" for run in runs)
            and all(int(run["abi_calls"] or 0) == 1 for run in runs)
            and all(int(run["device_id"] or 0) == 0 for run in runs)
            and all(run["selected"] == ("true" if run["pred"] else "false")
                    for run in runs)
            and all({k: int((run["exec_hits"] or {}).get(k) or 0)
                     for k in ("true", "false")}
                    == ({"true": 1, "false": 0} if run["token"] == 1
                        else {"true": 0, "false": 1}) for run in runs)
            and all(int((run["trace_hits"] or {}).get(side) or 0) >= 1
                    for run in runs for side in ("true", "false"))
            and plan_same
        )
        try:
            so = module.library_fingerprint()
        except Exception as exc:  # noqa: BLE001
            so = {"error": f"{type(exc).__name__}: {exc}"}
        ok = bool(ok and so.get("sha256"))
        return {"diagnostic": "D1", "status": "PASS" if ok else "FAIL",
                "backend": backend, "runs": runs, "plan_ids": sorted(plan_ids),
                "plan_same": plan_same, "so_fingerprint": so,
                "detail": (f"sign_flip pred={[run['pred'] for run in runs]} "
                           f"token={[run['token'] for run in runs]} "
                           f"plan_same={plan_same}")}
    except Exception as exc:  # noqa: BLE001
        return {"diagnostic": "D1", "status": "FAIL", "backend": backend,
                "runs": runs, "detail": f"{type(exc).__name__}: {str(exc)[:900]}"}


def _diag_d2_worker(case: dict, pred: bool, extra: str | None, workdir: Path) -> dict:
    """D2：真实 cotangent / 扁平 VJP 诊断（每次一组，fresh worker）。"""
    import jax  # noqa: PLC0415
    import jax.numpy as jnp  # noqa: PLC0415

    backend = jax.default_backend()
    if backend == "cpu":
        return {"diagnostic": "D2", "status": "BLOCKED", "device_required": True,
                "backend": backend,
                "detail": "D2 需要非 CPU 后端；CPU 自检不产生 PASS"}
    tag = f"{case['case_id']}-{extra or 'main'}"
    try:
        module = _candidate()
        golden_case = case
        fixture = fixtures.make_fixture(case, "", "grad_jit")
        pair = fixtures.branch_pair(case)
        leaf = fixture["leaves"][0]["path"]
        x_orig = np.ascontiguousarray(np.asarray(fixture["arrays"][leaf], np.float32))
        in_shape = tuple(x_orig.shape)
        x_flat = jnp.asarray(np.ascontiguousarray(
            np.asarray(x_orig, dtype=np.float32).reshape(-1)))
        cotangent = 1.0
        want_scale = 1.0
        if extra == "cot":
            cotangent = 3.5
            want_scale = 3.5

            def loss(value):
                return jnp.sum(jnp.asarray(cotangent, dtype=value.dtype)
                               * module.cond(pred, pair["true_fn"], pair["false_fn"], value))
        elif extra == "half":
            cotangent = 2.0
            want_scale = 2.0
            golden_case = dict(case)
            # P2-4：half 组曾固定 cond(True, ...) 但记录外层 pred，导致
            # cond_grad_false-half 出现 pred=false / selected=true。这里沿用外层
            # pred，并让 golden_case 用同一谓词生成对应支的梯度，记录与实跑一致。
            golden_case.update({"case_id": f"{case['case_id']}_diag_half",
                                "group": "D2", "required": False,
                                "branches": ["x/2", "x+1"], "pred": pred,
                                "mode": ["grad_jit"]})
            fixture = fixtures.make_fixture(golden_case, "", "grad_jit")
            pair = fixtures.branch_pair(golden_case)
            x_orig = np.ascontiguousarray(np.asarray(fixture["arrays"][leaf], np.float32))
            in_shape = tuple(x_orig.shape)
            x_flat = jnp.asarray(np.ascontiguousarray(
                np.asarray(x_orig, dtype=np.float32).reshape(-1)))

            def loss(value):
                return jnp.sum(jnp.asarray(cotangent, dtype=value.dtype)
                               * module.cond(pred, pair["true_fn"], pair["false_fn"], value))
        elif extra == "large":
            golden_case = dict(case)
            golden_case.update({"case_id": f"{case['case_id']}_diag_large",
                                "group": "D2", "required": False,
                                "mode": ["grad_jit"]})
            fixture = fixtures.make_fixture(golden_case, "", "grad_jit")
            pair = fixtures.branch_pair(golden_case)
            fixture["arrays"][leaf] = np.ascontiguousarray(x_orig * 1000.0)
            fixture["inputs_sha256"] = fixtures.compute_arrays_sha256(fixture["arrays"])
            x_orig = np.ascontiguousarray(np.asarray(fixture["arrays"][leaf], np.float32))
            in_shape = tuple(x_orig.shape)
            x_flat = jnp.asarray(np.ascontiguousarray(
                np.asarray(x_orig, dtype=np.float32).reshape(-1)))

            def loss(value):
                return jnp.sum(module.cond(pred, pair["true_fn"],
                                           pair["false_fn"], value))
        else:

            def loss(value):
                return jnp.sum(module.cond(pred, pair["true_fn"],
                                           pair["false_fn"], value))

        cdir = workdir / "d2" / tag
        cdir.mkdir(parents=True, exist_ok=True)
        fixtures.write_case_json(golden_case, fixture, cdir / "case.json")
        fixtures.write_inputs_npz(fixture, cdir / "inputs.npz")
        golden = engine._run_golden(WS, cdir / "case.json", cdir / "inputs.npz",
                                    "grad_jit", "", cdir)
        if not golden["ok"]:
            raise RuntimeError(f"golden 失败：{golden.get('detail')}")
        want = np.asarray(golden["arrays"]["grad"], dtype=np.float32) * want_scale
        module._reset_evidence()
        compiled, hlo = _diag_lower_compile(jax.grad(loss), (x_flat,), tag, workdir)
        flat_out = compiled(x_flat)
        platforms = _diag_platforms(flat_out)
        # 20→5x4 的 device reshape 在镜像 #145 上不支持；比较前在 host 恢复 shape。
        got = np.asarray(flat_out.block_until_ready(), dtype=np.float32).reshape(in_shape)
        verdict = _diag_judge(got, want, "grad")
        dispatch = _norm_dispatch(module)
        trace = dispatch.get("trace_hits") or {}
        trace_ok = all(int(trace.get(side) or 0) >= 1 for side in ("true", "false"))
        hlo_clean = not any(hlo["flags"].values())
        platform_ok = bool(platforms) and all("npu" in item.lower() for item in platforms)
        got_np = np.ascontiguousarray(_to_numpy(got), dtype=np.float32)
        ok = bool(verdict["is_pass"] and trace_ok and hlo_clean
                  and platform_ok and backend != "cpu")
        return {"diagnostic": "D2", "status": "PASS" if ok else "FAIL", "tag": tag,
                "case_id": case["case_id"], "extra": extra, "pred": pred,
                "cotangent": cotangent, "flat_len": int(x_flat.size),
                "input_shape": list(in_shape), "backend": backend,
                "platforms": platforms, "verdict": verdict,
                "vjp_output_sha256": hashlib.sha256(got_np.tobytes()).hexdigest(),
                "hlo": hlo, "hlo_clean": hlo_clean,
                "trace_hits": trace, "trace_ok": trace_ok,
                "dispatch": {key: dispatch.get(key) for key in
                             ("path", "exec_mode", "abi_calls", "selected",
                              "runtime_token", "backend", "device_id")},
                "detail": (f"cotangent={cotangent} flat_len={x_flat.size} "
                           f"criteria={verdict['is_pass']} hlo_clean={hlo_clean} "
                           f"trace_ok={trace_ok} platforms={platforms}")}
    except Exception as exc:  # noqa: BLE001
        return {"diagnostic": "D2", "status": "FAIL", "tag": tag,
                "detail": f"{type(exc).__name__}: {str(exc)[:1200]}"}


def _diag_d3_worker(tag: str) -> dict:
    """D3：fresh worker 身份探针（PID/backend/jaxpr 摘要）。"""
    import jax  # noqa: PLC0415

    backend = jax.default_backend()
    try:
        jaxpr = jax.make_jaxpr(lambda value: value + 1.0)(np.ones(4, np.float32))
        text = str(jaxpr)
        sha = hashlib.sha256(text.encode("utf-8")).hexdigest()
    except Exception as exc:  # noqa: BLE001
        return {"diagnostic": "D3", "status": "FAIL", "tag": tag,
                "detail": f"jaxpr 生成失败：{type(exc).__name__}: {exc}"}
    return {"diagnostic": "D3", "status": "OK", "tag": tag, "pid": os.getpid(),
            "backend": backend, "devices": [str(d) for d in jax.devices()],
            "jaxpr_sha256": sha, "jaxpr_len": len(text)}


def _diag_d4_worker(probe: str, case: dict, workdir: Path) -> dict:
    """D4：outer-jit tracer pred / stock lax.cond conditional 能力探针。"""
    import jax  # noqa: PLC0415
    import jax.numpy as jnp  # noqa: PLC0415

    backend = jax.default_backend()
    if backend == "cpu":
        return {"diagnostic": "D4", "probe": probe, "status": "BLOCKED",
                "device_required": True, "backend": backend,
                "detail": "D4 需要非 CPU 后端；CPU 自检不产生 PASS"}
    try:
        module = _candidate()
        fixture = fixtures.make_fixture(case, "", "jit_true")
        pair = fixtures.branch_pair(case)
        leaf = fixture["leaves"][0]["path"]
        x = jnp.asarray(fixture["arrays"][leaf], dtype=jnp.float32)
        if probe == "outer_jit_tracer_pred":
            module._reset_evidence()
            jax.jit(lambda p, v: module.cond(p, pair["true_fn"], pair["false_fn"], v))(
                np.asarray(True), x).block_until_ready()
        elif probe == "stock_lax_cond_conditional":
            jax.jit(lambda p, v: jax.lax.cond(
                p, pair["true_fn"], pair["false_fn"], v))(
                    np.asarray(True), x).block_until_ready()
        else:
            return {"diagnostic": "D4", "probe": probe, "status": "ERROR",
                    "detail": f"未知 probe={probe!r}"}
    except Exception as exc:  # noqa: BLE001
        dispatch = _norm_dispatch(module) if "module" in locals() else {}
        return {"diagnostic": "D4", "probe": probe, "status": "BACKEND_LIMITATION",
                "backend": backend, "error_type": type(exc).__name__,
                "error_code": getattr(exc, "code", None),
                "error_message": str(exc)[:1200],
                "expected_block": True,
                "dispatch": {key: dispatch.get(key) for key in
                             ("path", "exec_mode", "abi_calls", "selected",
                              "runtime_token", "backend", "device_id")},
                "detail": (f"{probe}: {getattr(exc, 'code', None) or type(exc).__name__}: "
                           f"{str(exc)[:500]}")}
    return {"diagnostic": "D4", "probe": probe, "status": "UNEXPECTED_PASS",
            "backend": backend, "expected_block": False,
            "detail": f"{probe} 在设备上成功执行；后端能力可能变化，需重估路线，不能记 PASS"}


def test_diag_d1_data_derived_predicate():
    case = next(item for item in CASES
                if item["case_id"] == "cond_selected_branch_only_executes")
    record = _spawn_diagnostic({"kind": "d1", "case": case, "workdir": str(WORKDIR)})
    _record_diagnostic(record)
    if record.get("status") == "BLOCKED":
        pytest.skip(str(record.get("detail")))
    assert record.get("status") == "PASS", record.get("detail")


def test_diag_d2_real_cotangent():
    statuses = []
    for case_id, pred in (("cond_grad_true", True), ("cond_grad_false", False)):
        case = next(item for item in CASES if item["case_id"] == case_id)
        for extra in (None, "cot", "half", "large"):
            record = _spawn_diagnostic({"kind": "d2", "case": case, "pred": pred,
                                        "extra": extra, "workdir": str(WORKDIR)})
            _record_diagnostic(record)
            if record.get("status") == "PASS":
                # P2-4 回归：记录谓词必须与实跑选中支一致（half 组不得再回填外层 pred）。
                assert bool(record.get("pred")) is bool(pred), (
                    case_id, extra, record.get("pred"))
                selected = (record.get("dispatch") or {}).get("selected")
                assert selected == ("true" if pred else "false"), (
                    case_id, extra, selected)
            statuses.append(record.get("status"))
    if statuses and all(status == "BLOCKED" for status in statuses):
        pytest.skip("D2 需要非 CPU 后端；CPU 自检只记录 BLOCKED")
    assert statuses and all(status == "PASS" for status in statuses), statuses


def test_diag_d3_fresh_worker_isolation():
    records = []
    for index in range(3):
        record = _spawn_diagnostic({"kind": "d3", "tag": f"probe-{index}"})
        records.append(record)
    pids = [record.get("pid") for record in records]
    backends = [record.get("backend") for record in records]
    cpu = bool(backends) and all(str(item) == "cpu" for item in backends)
    # 71 条用例每条都在 fresh worker 中执行：PID/程序哈希必须落盘且与父进程
    # 实际 spawn 的 PID 一致；jit/grad 的首调不得命中跨 case 缓存（plan_cache=hit）。
    case_workers = [item for item in _RESULTS if (item.get("worker") or {}).get("pid")]
    jit_first = [item for item in case_workers
                 if str(item.get("mode")) in ("jit", "jit_true", "jit_false",
                                              "grad_eager", "grad_jit")
                 and item.get("status") == "PASS"]
    no_cross_cache = all(
        str((item.get("dispatch") or {}).get("plan_cache")) != "hit"
        for item in jit_first)
    case_evidence_ok = bool(
        case_workers
        and all(item["worker"].get("pid_matches_spawn") for item in case_workers)
        and all(item["worker"].get("program_sha256") for item in case_workers)
        and len({item["worker"].get("spawn_index") for item in case_workers})
        == len(case_workers)
    )
    ok = bool(
        len(records) == 3
        and all(record.get("status") == "OK" for record in records)
        and all(record.get("worker", {}).get("pid_matches_spawn") for record in records)
        and len(set(pids)) == 3
        and all(record.get("jaxpr_sha256") for record in records)
        and case_evidence_ok and no_cross_cache
        and not cpu
    )
    aggregate = {"diagnostic": "D3",
                 "status": ("PASS" if ok else ("BLOCKED" if cpu else "FAIL")),
                 "pids": pids, "backends": backends,
                 "distinct_pids": len(set(pids)) == 3,
                 "jaxpr_sha256": [record.get("jaxpr_sha256") for record in records],
                 "case_workers": len(case_workers),
                 "case_pid_matches": sum(1 for item in case_workers
                                         if item["worker"].get("pid_matches_spawn")),
                 "case_program_hashes": len({item["worker"].get("program_sha256")
                                             for item in case_workers}),
                 "no_cross_case_cache_hit": no_cross_cache,
                 "records": records,
                 "detail": (f"fresh_probes={len(records)} pids={pids} "
                            f"backends={backends} case_workers={len(case_workers)} "
                            f"case_evidence_ok={case_evidence_ok} "
                            f"no_cross_cache={no_cross_cache}")}
    _record_diagnostic(aggregate)
    if cpu:
        pytest.skip("D3 的设备后端条件未满足；CPU 只验证 fresh worker 协议，不记 PASS")
    assert aggregate["status"] == "PASS", aggregate.get("detail")


def test_diag_d4_backend_capability():
    case = next(item for item in CASES
                if item["case_id"] == "cond_jit_dynamic_pred")
    records = []
    for probe in ("outer_jit_tracer_pred", "stock_lax_cond_conditional"):
        record = _spawn_diagnostic({"kind": "d4", "probe": probe, "case": case,
                                    "workdir": str(WORKDIR)})
        _record_diagnostic(record)
        records.append(record)
    statuses = [record.get("status") for record in records]
    cpu = bool(statuses) and all(status == "BLOCKED" for status in statuses)
    ok = bool(statuses and all(status == "BACKEND_LIMITATION" for status in statuses))
    aggregate = {"diagnostic": "D4",
                 "status": ("RECORDED" if ok else ("BLOCKED" if cpu else "FAIL")),
                 "probes": statuses, "records": records,
                 "detail": (f"probes={statuses}；BACKEND_LIMITATION 不计 PASS，"
                            "stock/outer 能力变化不得写成支持")}
    _record_diagnostic(aggregate)
    if cpu:
        pytest.skip("D4 需要非 CPU 后端；CPU 自检只记录 BLOCKED")
    # 能力限制已如实记录即通过诊断；BACKEND_LIMITATION/RECORDED 都不是算子 PASS。
    assert aggregate["status"] == "RECORDED", aggregate.get("detail")


# ---------------------------------------------------------------------------
# worker / self-check 入口
# ---------------------------------------------------------------------------


def test_selfcheck_s1_jit_closure_route():
    """CPU 自检：S1 jit 必须是零参数闭包，0-d operand/pred 不得成为 jit 动态实参。

    在 fresh worker 内执行 ``_s1_jit_execute``（与正式路径同一函数），用假候选
    记录 ``cond`` 收到的实参；断言 pred/operand 均为 concrete（非 Tracer）、program
    零参数、``jax.jit`` 的 lower/调用都不收动态实参，且 ``_run_numeric`` 确实委托
    该 helper。这是契约自检，不是设备精度证据；worker 在 CPU 上运行，不冒充 NPU PASS。
    """
    case = next(item for item in SUPPLEMENT_SUITE["cases"]
                if item["case_id"] == "cond_sup_weak_type")
    records = []
    for variant in ("weak", "strong"):
        box = _spawn_worker({"kind": "selfcheck_s1_jit", "case": case,
                             "variant": variant, "workdir": str(WORKDIR)},
                            timeout=300)
        assert box["ok"], box.get("error")
        records.append(box["result"])
    for rec in records:
        assert rec["status"] == "PASS", rec
        evidence = rec["evidence"]
        assert evidence["exec_form"] == "closure_jit_zero_arg"
        assert evidence["jit_dynamic_args"] == 0
        assert evidence["program_params"] == []
        assert evidence["closure_constants"] == ["operand", "pred"]
        assert evidence["pred_source"] == "jax_device_scalar"
        assert evidence["pred_is_concrete"] is True
        assert evidence["operand_is_concrete"] is True
        assert rec["cond_calls"][0]["pred_is_tracer"] is False
        assert rec["cond_calls"][0]["operand_is_tracer"] is False
        assert rec["run_numeric_delegates_to_helper"] is True
        assert rec["run_numeric_has_outer_jit_lambda"] is False
        assert rec["output_weak_type"] == rec["expected_weak_type"]
    # 旧形态（把 pred/operand 传给 jax.jit）必须绝迹；回退时此断言会失败。
    source = inspect.getsource(_run_numeric)
    assert "jax.jit(lambda" not in source
    assert "_s1_jit_execute(" in source
    _mark_selfcheck("test_selfcheck_s1_jit_closure_route")


def test_selfcheck_worker_protocol():
    box = _spawn_worker({"kind": "selfcheck_echo", "echo": {"answer": 42}},
                        timeout=120)
    assert box["ok"], box.get("error")
    result = box["result"]
    assert result["echo"] == {"answer": 42}
    assert int(result["pid"]) == int(box["spawn_pid"])
    assert int(result["spawn_index"]) == int(box["spawn_index"])
    crash = _spawn_worker({"kind": "selfcheck_crash"}, timeout=120)
    assert crash["ok"] is False and crash["returncode"] == 9
    assert "未输出" in str(crash.get("error"))
    # 父进程只调度/聚合：候选与 golden 的 JAX 都不应在本进程 import。
    assert (_JAX_PREIMPORTED or "jax" not in sys.modules) and (
        _JAXLIB_PREIMPORTED or "jaxlib" not in sys.modules), (
        "pytest 父进程 import 了 jax/jaxlib；违反 fresh worker 隔离")
    # D2 的 HLO 判定与 D1 的标量证据工具（纯 CPU 可验证）。
    flags = _diag_hlo_flags("module { stablehlo.select ... stablehlo.conditional }")
    assert flags["stablehlo.select"] is True
    assert flags["stablehlo.conditional"] is True
    assert flags["stablehlo.convert"] is False
    verdict = _diag_judge(np.ones(4, np.float32), np.ones(4, np.float32), "out")
    assert verdict["is_pass"] is True
    scalar = _device_scalar_info(np.array(3.5, np.float32))
    assert scalar["value"] == 3.5 and len(scalar["sha256"]) == 64
    _mark_selfcheck("test_selfcheck_worker_protocol")


if __name__ == "__main__":
    import sys as _sys

    if len(_sys.argv) == 3 and _sys.argv[1] == "--worker":
        raise SystemExit(_worker_main(_sys.argv[2]))
    raise SystemExit(f"usage: {_sys.argv[0]} --worker <payload.json>")
