# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""jax_lax_associative_scan · pytest 交付件（真机运行；候选 = Python 接口，参考 = 同版本 JAX CPU 子进程）。

结构与判据（设计 §1.3 / §2.1 / §2.2.2 / §2.3）：

* 用例 = 冻结 ``cases/jax_lax_associative_scan/cases.json``（7 条）
  + 基准复用 ``benchmark.json``（2 条，仅 jit）
  + 补充 ``tests/jax_lax_associative_scan_supplement_cases.json``（8 条 L1 + 3 条 L2）
  + 白盒 ``tests/jax_lax_associative_scan_whitebox_cases.json``（6 条数值 + 1 条拒绝）
  = **27 条身份 / 48 个 ``(case_id, mode)`` 参数化项**。
  白盒由节点 3 依据当前源码分支创建，经同一 loader（``origin="whitebox"``）并入同一入口；
  ``EXPECTED_PYTEST_CASES`` / ``EXPECTED_PYTEST_MODES`` / ``EXPECTED_WHITEBOX_CASES`` 必须同步。
  白盒中无法由用例 JSON 表达的输入（axis 非整数、9 叶子 n_leaves>8 上限）由文件末尾的白盒
  CPU 分支用例覆盖；它们不进入上面的用例参数化计数，也不冒充设备数值证据。
* 候选：``ops/jax_lax_associative_scan/python/jax_lax_associative_scan.py`` 的
  ``associative_scan`` / ``associative_scan_compiled``；``eager`` = ``per_leaf`` /
  ``abi_calls=叶子数``；``jit`` = ``whole_tree`` / ``abi_calls=1`` / 第二次调用
  ``plan_cache="hit"`` 且 ``plan_id`` 稳定（设计 §1.3.2）。
* 参考：``golden_cli.py`` 独立 CPU 子进程（``JAX_PLATFORMS=cpu``）；主进程只跑候选（NPU），
  绝不为 golden 设置父进程环境变量。
* 判定：逐叶子先设备守卫，再 dtype/shape/结构精确，浮点走 ``criteria/check.py``；
  非有限位逐元素精确；结构/shape 无容差；输出不得与输入别名、输入不得被改写。
* AD：``assoc_grad_sum`` 使用测试侧 ``jax.Array``（仅测试支持），连续 3 次同码
  ``UNSUPPORTED:autodiff_unsupported`` 才允许分类为 ``UNSUPPORTED``，且要求
  ``path=trace``/``abi_calls=0``/``backend`` 非 CPU/附后端原文；``UNSUPPORTED`` 不记 PASS。
* 拒绝：L2 必须给稳定码 + ``path=reject`` + ``abi_calls=0``；golden 也按已确认规格确认拒绝。

报告：``${OPX_OUT:-/out}/assoc_scan_correctness.json`` + ``.md``；每条 ``(case_id, mode)``
另打印一行 ``ASSOC_CASE {...}``。pytest 的 junit/log 由 ``test.sh`` 第二段保存。
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

WS = Path(__file__).resolve().parents[1]
OP = "jax_lax_associative_scan"
for extra in (WS, WS / "ops" / OP / "scripts"):
    if str(extra) not in sys.path:
        sys.path.insert(0, str(extra))

np = pytest.importorskip("numpy")

import fixtures  # noqa: E402
import assoc_scan_engine as engine  # noqa: E402

CANDIDATE_PATH = WS / "ops" / OP / "python" / f"{OP}.py"
GOLDEN_DIR = WS / "ops" / OP / "scripts"
GOLDEN_CLI = GOLDEN_DIR / "golden_cli.py"
FROZEN_CASES_PATH = WS / "cases" / OP / "cases.json"
BENCH_PATH = WS / "cases" / OP / "benchmark.json"
SUPPLEMENT_PATH = WS / "tests" / f"{OP}_supplement_cases.json"
WHITEBOX_PATH = WS / "tests" / f"{OP}_whitebox_cases.json"

EXPECTED_BASE_CASES = 20
EXPECTED_BASE_MODES = 35
EXPECTED_GATE_CASES = 7
EXPECTED_GATE_MODES = 14
EXPECTED_SUPPLEMENT_CASES = 11
#: 节点 3 依据当前源码分支新增的白盒用例（6 条数值 + 1 条拒绝）。
EXPECTED_WHITEBOX_CASES = 7
EXPECTED_WHITEBOX_MODES = 13
#: 节点 3 接入白盒后同步这两个值（与 ``test_selfcheck_discovery_counts`` 的提示一致）。
EXPECTED_PYTEST_CASES = 27
EXPECTED_PYTEST_MODES = 48

CRITERIA_MOD = engine._load_module(WS / "criteria" / "check.py", "opx_assoc_pytest_criteria")

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
)


# ---------------------------------------------------------------------------
# 用例加载与发现计数
# ---------------------------------------------------------------------------


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _normalise(case: dict, origin: str) -> dict:
    item = dict(case)
    item["mode"] = list(item.get("mode") or ["jit"])
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
    cases: list[dict] = []
    for origin in ("frozen", "benchmark", "supplement"):
        cases += [_normalise(case, origin) for case in suites[origin].get("cases") or []]
    if whitebox_path.is_file():
        suites["whitebox"] = _load_json(whitebox_path)
        cases += [_normalise(case, "whitebox") for case in suites["whitebox"].get("cases") or []]
    else:
        suites["whitebox"] = {"cases": []}
    return cases, suites


def _ordered(cases: list[dict]) -> list[dict]:
    """AD 用例先跑（PJRT 与 host-driver ACL 的进程级运行时初始化顺序），其余保持原序。

    ``enumerate`` 的原始下标保证同一 case 的 ``eager`` 仍在 ``jit`` 之前，供一致性互比。
    """
    def ad_first(pair):
        index, case = pair
        return (0 if any(mode in engine.AD_MODES for mode in case["mode"]) else 1, index)

    return [case for _index, case in sorted(enumerate(cases), key=ad_first)]


CASES, SUITES = _load_cases()
CASES = _ordered(CASES)
FROZEN_SUITE = SUITES["frozen"]
BENCH_SUITE = SUITES["benchmark"]
SUPPLEMENT_SUITE = SUITES["supplement"]
WHITEBOX_SUITE = SUITES["whitebox"]
ITEMS = [(case, mode) for case in CASES for mode in case["mode"]]
MODE_COUNT = len(ITEMS)
GATE_MODE_COUNT = sum(len(case.get("mode") or ["jit"]) for case in (FROZEN_SUITE.get("cases") or []))
BASE_CASES = [case for case in CASES if case.get("_origin") != "whitebox"]
BASE_MODE_COUNT = sum(len(case["mode"]) for case in BASE_CASES)

_CANDIDATE = None
_RESULTS: list[dict] = []
_CASE_MODE_RESULTS: dict[tuple[str, str], dict] = {}
_CASE_MODE_ARRAYS: dict[tuple[str, str], dict] = {}
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
# 设备守卫（逐叶子；依据设计 §1.3.5）
# ---------------------------------------------------------------------------


def _leaf_platforms(leaf):
    """返回 ``(platforms: set[str], kind: str, devices: list)``；无法判定时 platforms 为空。"""
    if callable(getattr(leaf, "devices", None)):
        devs = list(leaf.devices())
        return {str(getattr(dev, "platform", "?")) for dev in devs}, "jax", devs
    device = getattr(leaf, "device", None)
    if getattr(device, "type", None) is not None:
        return {str(device.type)}, "torch", [device]
    if type(leaf).__module__.split(".")[0] == "numpy":
        return set(), "numpy", []
    return set(), type(leaf).__name__, []


def assert_all_leaves_on_accelerator(outputs, case_id, dispatch=None):
    """numpy 回读只在 ``path=so``/``backend=npu``/``abi_calls>0`` 证据完整时放行；
    jax 叶子要求 ``devices()`` 平台非 CPU。任一 CPU 回退抛 ``AssertionError``。
    """
    leaves = fixtures.flatten_with_paths(outputs)
    assert leaves, f"{case_id}: 输出为空，无法证明设备"
    recorded = []
    for path, leaf in leaves:
        platforms, kind, devs = _leaf_platforms(leaf)
        if kind == "numpy":
            # numpy 是 host-driver 的 I/O 形式（NPU 执行后 D2H 回读），不代表候选在 CPU 计算。
            assert dispatch and dispatch.get("path") == "so", \
                f"{case_id}:{path} host 回读缺 path=so 证据"
            assert dispatch.get("backend") == "npu", \
                f"{case_id}:{path} host 回读缺 backend=npu 证据"
            assert int(dispatch.get("abi_calls") or 0) > 0, \
                f"{case_id}:{path} host 回读缺真实 ABI 调用证据"
            recorded.append({"path": path, "kind": "numpy-host-copy",
                             "platforms": ["npu"], "device": "D2H after NPU ABI"})
            continue
        assert platforms, f"{case_id}:{path} 无法取得设备信息（kind={kind}）"
        assert all(platform != "cpu" for platform in platforms), \
            f"{case_id}:{path} 位于 CPU：{sorted(platforms)}"
        recorded.append({"path": path, "kind": kind, "platforms": sorted(platforms),
                         "device": str(devs[0]) if devs else None})
    return recorded


def _to_numpy(leaf):
    platforms, kind, _devs = _leaf_platforms(leaf)
    if kind == "numpy":
        return np.asarray(leaf)
    if kind == "torch":
        return leaf.detach().to("cpu").numpy()
    if kind == "jax":
        return np.asarray(leaf)
    raise AssertionError(f"无法转成 numpy 的叶子类型：{kind}")


# ---------------------------------------------------------------------------
# 候选输入与调用
# ---------------------------------------------------------------------------


def _workdir(case_id: str) -> Path:
    root = Path(os.environ.get("OPX_CASE_WORKDIR", "/tmp/opx-assoc-pytest-cases"))
    path = root / str(case_id)
    path.mkdir(parents=True, exist_ok=True)
    return path


def _prepare_case(case: dict) -> tuple[dict, Path, Path]:
    fixture = fixtures.make_fixture(case)
    cdir = _workdir(case["case_id"])
    case_json = cdir / "case.json"
    inputs = cdir / "inputs.npz"
    fixtures.write_case_json(case, fixture, case_json)
    fixtures.write_inputs_npz(fixture, inputs)
    return fixture, case_json, inputs


def _copy_numpy(array):
    value = np.array(array, copy=True)
    return value if value.ndim == 0 else np.ascontiguousarray(value)


def _make_numpy_tree(fixture: dict):
    """host-driver 输入（numpy）；候选仍在 NPU ``.so`` 上执行。"""
    return fixtures.tree_from_spec(fixture["elems_tree"], fixture["arrays"], convert=_copy_numpy)


def _make_jax_tree(fixture: dict):
    """测试侧 ``jax.Array``（仅 AD 用例）；import 前显式要求 NPU 后端（设计 §2.2.2）。"""
    if "jax" not in sys.modules:
        os.environ["JAX_PLATFORMS"] = "npu"
    try:
        import jax  # noqa: PLC0415
        import jax.numpy as jnp  # noqa: PLC0415
    except Exception as exc:  # noqa: BLE001
        pytest.fail(f"候选侧 jax 不可用：{type(exc).__name__}: {exc}")
    try:
        backend = jax.default_backend()
        devices = list(jax.devices())
    except Exception as exc:  # noqa: BLE001
        pytest.fail(f"候选侧 JAX 后端初始化失败：{type(exc).__name__}: {exc}")
    if backend == "cpu":
        pytest.fail(f"候选侧 JAX 默认后端是 CPU（{backend}/{devices}）：不能用 CPU 结果冒充 NPU 证据")
    npu_devices = [dev for dev in devices if str(getattr(dev, "platform", "?")) != "cpu"]
    if not npu_devices:
        pytest.fail(f"候选侧没有非 CPU JAX 设备（{backend}/{devices}）")
    dev = npu_devices[0]
    elems = fixtures.tree_from_spec(
        fixture["elems_tree"], fixture["arrays"],
        convert=lambda array: jax.device_put(jnp.asarray(np.array(array, copy=True)), dev))
    return elems, jax


def _dispatch_of(module) -> dict:
    dispatch = getattr(module, "_last_dispatch", None)
    if not isinstance(dispatch, dict):
        return {}
    keys = ("op", "mode", "path", "exec_mode", "combine_id", "abi_calls", "family",
            "backend", "device_id", "plan_id", "plan_cache", "reason")
    return {key: dispatch.get(key) for key in keys}


def _numeric_dispatch_problems(dispatch: dict, fixture: dict, mode: str,
                               second: dict | None) -> list[str]:
    expected_exec = "per_leaf" if mode == "eager" else "whole_tree"
    expected_calls = len(fixture["leaves"]) if mode == "eager" else 1
    problems = []
    if dispatch.get("path") != "so":
        problems.append(f"path={dispatch.get('path')!r}（期望 so）")
    if dispatch.get("exec_mode") != expected_exec:
        problems.append(f"exec_mode={dispatch.get('exec_mode')!r}（期望 {expected_exec}）")
    if dispatch.get("abi_calls") != expected_calls:
        problems.append(f"abi_calls={dispatch.get('abi_calls')}（期望 {expected_calls}）")
    if dispatch.get("combine_id") != fixture["combine_id"]:
        problems.append(f"combine_id={dispatch.get('combine_id')!r}（期望 {fixture['combine_id']}）")
    if dispatch.get("family") != "numpy":
        problems.append(f"family={dispatch.get('family')!r}（期望 numpy）")
    if dispatch.get("backend") != "npu":
        problems.append(f"backend={dispatch.get('backend')!r}（期望 npu）")
    if dispatch.get("device_id") != 0:
        problems.append(f"device_id={dispatch.get('device_id')!r}（期望 0）")
    if mode == "eager":
        if dispatch.get("plan_cache") is not None:
            problems.append(f"eager plan_cache={dispatch.get('plan_cache')!r}（期望 None）")
    else:
        if not str(dispatch.get("plan_id") or "").strip():
            problems.append("jit plan_id 不能为空")
        if second is None:
            problems.append("jit 缺少第二次调用的缓存证据")
        else:
            if second.get("plan_id") != dispatch.get("plan_id"):
                problems.append("jit 两次调用的 plan_id 不稳定")
            if second.get("plan_cache") != "hit":
                problems.append(f"jit 第二次调用 plan_cache={second.get('plan_cache')!r}（期望 hit）")
    return problems


def _ad_dispatch_problems(dispatch: dict) -> list[str]:
    problems = []
    path = dispatch.get("path")
    if path not in ("so", "trace"):
        problems.append(f"path={path!r}（AD 只允许 so 或 trace）")
    if path == "so":
        if int(dispatch.get("abi_calls") or 0) < 1:
            problems.append(f"kernel AD abi_calls={dispatch.get('abi_calls')!r}（期望 >=1）")
        if dispatch.get("exec_mode") not in ("jvp", "vjp"):
            problems.append(f"kernel AD exec_mode={dispatch.get('exec_mode')!r}（期望 jvp/vjp）")
    elif path == "trace":
        if dispatch.get("abi_calls") != 0:
            problems.append(f"trace AD abi_calls={dispatch.get('abi_calls')!r}（期望 0）")
        if dispatch.get("exec_mode") != "trace":
            problems.append(f"trace AD exec_mode={dispatch.get('exec_mode')!r}（期望 trace）")
    if str(dispatch.get("backend")) in ("", "None", "cpu"):
        problems.append(f"AD backend={dispatch.get('backend')!r}（必须非 CPU）")
    return problems


def _invoke_numeric(module, fixture: dict, mode: str) -> tuple[dict, dict, dict, bool, bool]:
    elems = _make_numpy_tree(fixture)
    before = {path: np.array(leaf, copy=True)
              for path, leaf in fixtures.flatten_with_paths(elems)}
    fn = fixtures.combine_fn_for(fixture["combine_name"])
    reverse = bool(fixture["reverse"])
    axis = int(fixture["axis"])
    second = None
    if mode == "eager":
        result = module.associative_scan(fn, elems, reverse=reverse, axis=axis)
        dispatch = _dispatch_of(module)
    else:
        result = module.associative_scan_compiled(fn, elems, reverse=reverse, axis=axis)
        first = _dispatch_of(module)
        result = module.associative_scan_compiled(fn, elems, reverse=reverse, axis=axis)
        dispatch = _dispatch_of(module)
        second = {"plan_id": first.get("plan_id"),
                  "plan_cache": dispatch.get("plan_cache"),
                  "first_plan_id": first.get("plan_id")}
    flat = fixtures.flatten_with_paths(result, fixtures.OUT_PATH)
    outputs = {path: leaf for path, leaf in flat}
    input_unchanged = True
    for path, leaf in fixtures.flatten_with_paths(elems):
        if not np.array_equal(np.asarray(leaf), before[path]):
            input_unchanged = False
    alias_free = True
    output_arrays = [_to_numpy(leaf) for _path, leaf in flat]
    input_arrays = [np.asarray(leaf) for _path, leaf in fixtures.flatten_with_paths(elems)]
    for out_arr in output_arrays:
        for in_arr in input_arrays:
            if np.shares_memory(out_arr, in_arr):
                alias_free = False
    return outputs, dispatch, (second or {}), input_unchanged, alias_free


def _invoke_ad(module, fixture: dict, mode: str):
    elems, jax = _make_jax_tree(fixture)
    fn = fixtures.combine_fn_for(fixture["combine_name"])
    reverse = bool(fixture["reverse"])
    axis = int(fixture["axis"])

    def f_ad(value):
        return module.associative_scan(fn, value, reverse=reverse, axis=axis)

    try:
        if mode == "jvp":
            tangent_seed = jax.tree_util.tree_map(jax.numpy.ones_like, elems)
            primal, tangent = jax.jvp(f_ad, (elems,), (tangent_seed,))
            outputs = {"jvp_primal": primal, "jvp_tangent": tangent}
        else:
            primal, vjp_fn = jax.vjp(f_ad, elems)
            cotangent_seed = jax.tree_util.tree_map(jax.numpy.ones_like, primal)
            cotangent = vjp_fn(cotangent_seed)[0]
            outputs = {"vjp_primal": primal, "vjp_cotangent": cotangent}
    except Exception:
        # candidate 只知道当前调用是 eager；外层才知道这次 trace 是 jvp 还是 vjp。
        # 在异常离开前把真实 AD mode 回填到 dispatch，供 3 次稳定分类校验。
        dispatch = getattr(module, "_last_dispatch", None)
        if isinstance(dispatch, dict):
            dispatch["mode"] = mode
        raise
    outputs = jax.tree_util.tree_map(lambda leaf: leaf.block_until_ready(), outputs)
    dispatch = _dispatch_of(module)
    if isinstance(getattr(module, "_last_dispatch", None), dict):
        module._last_dispatch["mode"] = mode
        dispatch = _dispatch_of(module)
    return outputs, dispatch


# ---------------------------------------------------------------------------
# 单 mode 判定
# ---------------------------------------------------------------------------


def _golden(case_json: Path, inputs: Path, mode: str, cdir: Path) -> dict:
    return engine._run_golden(WS, case_json, inputs, mode, cdir)


def _judge(outputs: dict, golden_arrays: dict, expected_keys: list[str], leaf_map: dict,
           fixture: dict, mode: str) -> tuple[list[dict], list[str]]:
    actual_np = {path: _to_numpy(leaf) for path, leaf in outputs.items()}
    return engine._judge_arrays(actual_np, golden_arrays, expected_keys, CRITERIA_MOD,
                                leaf_map, fixture, mode)


def _run_reject(module, case: dict, fixture: dict, mode: str, cdir: Path,
                case_json: Path, inputs: Path) -> dict:
    rec = {"mode": mode, "status": "NOT_RUN", "leaves": [], "dispatch": {},
           "attempts": [{"attempt": 1}]}
    expected_code = case.get("expect_reject_code", "invalid_argument")
    elems = _make_numpy_tree(fixture)
    fn = fixtures.combine_fn_for(fixture["combine_name"])
    try:
        module.associative_scan(fn, elems, reverse=bool(fixture["reverse"]),
                                axis=int(fixture["axis"]))
    except Exception as exc:  # noqa: BLE001
        text = f"{type(exc).__name__}: {exc}"
        rec["exception"] = text
        if expected_code not in text:
            rec.update(status="FAIL", detail=f"拒绝了但不是 {expected_code}：{text[:300]}")
            return rec
    else:
        rec.update(status="FAIL", detail=f"该拒绝却成功返回；期望 {expected_code}")
        return rec
    dispatch = _dispatch_of(module)
    rec["dispatch"] = dispatch
    if dispatch.get("path") != "reject" or dispatch.get("exec_mode") != "reject":
        rec.update(status="FAIL",
                   detail=f"拒绝路径标记不对：path={dispatch.get('path')!r} "
                          f"exec_mode={dispatch.get('exec_mode')!r}")
        return rec
    if dispatch.get("abi_calls") != 0:
        rec.update(status="FAIL",
                   detail=f"拒绝路径 abi_calls={dispatch.get('abi_calls')}（必须为 0）")
        return rec
    golden = _golden(case_json, inputs, "reject", cdir)
    rec["golden"] = {"ok": golden.get("ok"), "rc": golden.get("rc")}
    if not golden["ok"]:
        rec.update(status="ERROR", detail=f"golden 未确认该用例应被拒绝：{golden.get('detail')}")
        return rec
    meta = golden.get("meta") or {}
    rec["golden"].update({"backend": meta.get("backend"), "jax": meta.get("jax"),
                          "jaxlib": meta.get("jaxlib"), "reject_code": meta.get("reject_code")})
    rec["reject_evidence"] = meta.get("reject_evidence")
    if meta.get("reject_code") != expected_code:
        rec.update(status="FAIL",
                   detail=f"golden 拒绝码 {meta.get('reject_code')!r} != 期望 {expected_code!r}")
        return rec
    rec.update(status="PASS", detail=f"按预期拒绝：{expected_code}；abi_calls=0")
    return rec


def _run_numeric(module, case: dict, fixture: dict, mode: str, cdir: Path,
                 case_json: Path, inputs: Path) -> dict:
    rec = {"mode": mode, "status": "NOT_RUN", "leaves": [], "dispatch": {}, "attempts": []}
    try:
        outputs, dispatch, second, input_unchanged, alias_free = _invoke_numeric(module, fixture, mode)
    except Exception as exc:  # noqa: BLE001
        rec["dispatch"] = _dispatch_of(module)
        rec.update(status="ERROR", detail=f"候选执行异常：{type(exc).__name__}: {exc}"[:400])
        return rec
    rec["attempts"] = [{"attempt": 1, "rc": 0}]
    rec["dispatch"] = dispatch
    rec["second_call"] = second
    if not input_unchanged:
        rec.update(status="FAIL", detail="候选改写了输入 pytest elems（必须返回新结果、不原地修改）")
        return rec
    if not alias_free:
        rec.update(status="FAIL", detail="候选输出与输入共享存储（必须返回新数组）")
        return rec
    try:
        device_leaves = assert_all_leaves_on_accelerator(outputs, case["case_id"], dispatch)
    except AssertionError as exc:
        rec.update(status="FAIL", detail=f"设备守卫失败：{exc}")
        return rec
    rec["device_leaves"] = device_leaves
    problems = _numeric_dispatch_problems(dispatch, fixture, mode, second)
    if problems:
        rec.update(status="FAIL", detail="dispatch 断言失败：" + "；".join(problems))
        return rec

    golden = _golden(case_json, inputs, mode, cdir)
    rec["golden"] = {"ok": golden.get("ok"), "rc": golden.get("rc")}
    if not golden["ok"]:
        rec.update(status="ERROR", detail=f"golden 失败：{golden.get('detail')}")
        return rec
    meta = golden.get("meta") or {}
    rec["golden"].update({"backend": meta.get("backend"), "jax": meta.get("jax"),
                          "jaxlib": meta.get("jaxlib"), "pid": meta.get("pid")})
    expected_keys = list(fixtures.output_shapes_of(fixture, mode))
    leaf_map = {item["path"]: item for item in device_leaves}
    leaves, judge_problems = _judge(outputs, golden["arrays"], expected_keys, leaf_map, fixture, mode)
    rec["leaves"] = leaves
    rec["matched_ratio"] = min((leaf.get("matched_ratio", 1.0) or 0.0) for leaf in leaves) if leaves else 0.0
    rec["max_abs_error"] = max((leaf.get("max_abs_error") or 0.0) for leaf in leaves) if leaves else None
    rec["nonfinite"] = {
        "count": sum(int((leaf.get("nonfinite") or {}).get("count", 0)) for leaf in leaves),
        "match": all(bool((leaf.get("nonfinite") or {}).get("match", True)) for leaf in leaves),
    }
    if judge_problems:
        rec.update(status="FAIL", detail="；".join(judge_problems))
    else:
        rec.update(status="PASS", detail=f"{mode} 通过（{len(leaves)} 个叶子）")
        rec["_arrays"] = {path: _to_numpy(leaf) for path, leaf in outputs.items()}
    return rec


def _run_ad(module, case: dict, fixture: dict, mode: str, cdir: Path,
            case_json: Path, inputs: Path) -> dict:
    rec = {"mode": mode, "status": "NOT_RUN", "leaves": [], "dispatch": {}, "attempts": []}
    expected = case.get("expected", "match_cpu")
    allow_unsupported = expected == "match_cpu_or_explicit_unsupported"
    max_attempts = 3 if allow_unsupported else 1
    unsupported_texts: list[str] = []
    outputs = None
    dispatch: dict = {}
    last_exc: Exception | None = None

    for attempt in range(1, max_attempts + 1):
        rec["attempts"].append({"attempt": attempt})
        try:
            outputs, dispatch = _invoke_ad(module, fixture, mode)
        except Exception as exc:  # noqa: BLE001
            last_exc = exc
            text = f"{type(exc).__name__}: {exc}"
            dispatch = _dispatch_of(module)
            explicit = (allow_unsupported and engine.UNSUPPORTED_CODE in text)
            rec["attempts"][-1].update(rc="exception", error=text[:300],
                                       unsupported=explicit, dispatch=dispatch)
            if explicit:
                problems = engine.unsupported_attempt_problems(
                    dispatch, text, require_op=True, expected_mode=mode)
                if problems:
                    rec.update(status="INVALID",
                               detail="显式 unsupported 分类缺少硬证据，不能记 UNSUPPORTED："
                                      + "；".join(problems))
                    rec["dispatch"] = dispatch
                    return rec
                unsupported_texts.append(text)
                continue
            rec.update(status="ERROR", detail=f"候选执行异常：{text[:400]}")
            rec["dispatch"] = dispatch
            return rec
        else:
            rec["attempts"][-1].update(rc=0, unsupported=False)
            if unsupported_texts:
                rec.update(status="FAIL",
                           detail="AD 分类不稳定：先出现 UNSUPPORTED，随后又成功（必须 3 次同码）")
                rec["dispatch"] = dispatch
                return rec
            break

    if outputs is None:
        if allow_unsupported and len(unsupported_texts) == max_attempts:
            problems = engine.unsupported_attempt_problems(
                dispatch, unsupported_texts[-1], require_op=True, expected_mode=mode)
            if problems:
                rec.update(status="INVALID",
                           detail="3 次同码但 unsupported 分类缺少硬证据，不能记 UNSUPPORTED："
                                  + "；".join(problems))
                rec["dispatch"] = dispatch
                return rec
            rec.update(status="UNSUPPORTED",
                       detail="3 次均显式 UNSUPPORTED:autodiff_unsupported"
                              "（path=trace/abi_calls=0/backend 非 CPU，附后端原文）")
            rec["backend_errors"] = [text[:400] for text in unsupported_texts]
            rec["dispatch"] = dispatch
            return rec
        rec.update(status="ERROR",
                   detail=f"候选没有成功输出：{type(last_exc).__name__ if last_exc else 'unknown'}")
        rec["dispatch"] = dispatch
        return rec

    rec["dispatch"] = dispatch
    try:
        device_leaves = assert_all_leaves_on_accelerator(outputs, case["case_id"], dispatch)
    except AssertionError as exc:
        rec.update(status="FAIL", detail=f"设备守卫失败：{exc}")
        return rec
    rec["device_leaves"] = device_leaves
    problems = _ad_dispatch_problems(dispatch)
    if problems:
        rec.update(status="FAIL", detail="AD dispatch 断言失败：" + "；".join(problems))
        return rec

    golden = _golden(case_json, inputs, mode, cdir)
    rec["golden"] = {"ok": golden.get("ok"), "rc": golden.get("rc")}
    if not golden["ok"]:
        rec.update(status="ERROR", detail=f"golden 失败：{golden.get('detail')}")
        return rec
    meta = golden.get("meta") or {}
    rec["golden"].update({"backend": meta.get("backend"), "jax": meta.get("jax"),
                          "jaxlib": meta.get("jaxlib"), "pid": meta.get("pid")})
    expected_keys = list(fixtures.output_shapes_of(fixture, mode))
    leaf_map = {item["path"]: item for item in device_leaves}
    leaves, judge_problems = _judge(outputs, golden["arrays"], expected_keys, leaf_map, fixture, mode)
    rec["leaves"] = leaves
    rec["matched_ratio"] = min((leaf.get("matched_ratio", 1.0) or 0.0) for leaf in leaves) if leaves else 0.0
    rec["max_abs_error"] = max((leaf.get("max_abs_error") or 0.0) for leaf in leaves) if leaves else None
    rec["nonfinite"] = {
        "count": sum(int((leaf.get("nonfinite") or {}).get("count", 0)) for leaf in leaves),
        "match": all(bool((leaf.get("nonfinite") or {}).get("match", True)) for leaf in leaves),
    }
    if judge_problems:
        rec.update(status="FAIL", detail="；".join(judge_problems))
    else:
        rec.update(status="PASS", detail=f"{mode} 通过（{len(leaves)} 个叶子）")
        rec["_arrays"] = {path: _to_numpy(leaf) for path, leaf in outputs.items()}
    return rec


def _consistency_problems(eager: dict, jit: dict) -> list[str]:
    problems = []
    if set(eager) != set(jit):
        return [f"eager/jit 输出叶子集合不同：{sorted(eager)} vs {sorted(jit)}"]
    for key in eager:
        a, b = np.asarray(eager[key]), np.asarray(jit[key])
        if a.dtype != b.dtype or a.shape != b.shape:
            problems.append(f"{key}: eager/jit dtype/shape 不同")
            continue
        verdict = CRITERIA_MOD.check(np.ascontiguousarray(a, dtype=np.float32),
                                     np.ascontiguousarray(b, dtype=np.float32))
        if not verdict["is_pass"]:
            problems.append(f"{key}: eager 与 jit 输出不一致（" + "；".join(
                verdict.get("failure_reasons") or ["数值不一致"]) + "）")
    return problems


def _run_mode(case: dict, mode: str) -> dict:
    module = _candidate()
    fixture, case_json, inputs = _prepare_case(case)
    cdir = case_json.parent
    if case.get("expected") == "reject":
        rec = _run_reject(module, case, fixture, mode, cdir, case_json, inputs)
    elif mode in engine.AD_MODES:
        rec = _run_ad(module, case, fixture, mode, cdir, case_json, inputs)
    else:
        rec = _run_numeric(module, case, fixture, mode, cdir, case_json, inputs)

    rec.update({
        "case_id": case["case_id"],
        "group": case.get("group"),
        "origin": case.get("_origin"),
        "required": bool(case.get("required", True)),
        "expected": case.get("expected", "match_cpu"),
        "shape": case.get("shape") or (
            [int(case["length"]), int(case["width"])] if case.get("length") is not None else None),
        "geometry": fixture["geometry"],
        "dtype": case.get("dtype", fixture["dtype"]),
        "seed": case.get("seed"),
        "note": case.get("note"),
        "reject_evidence": rec.get("reject_evidence"),
    })
    # eager × jit 一致性（同一 case 两个 mode 均 PASS 时才互比；全量运行时 jit 在 eager 之后）。
    if rec["status"] == "PASS" and mode == "jit":
        eager_rec = _CASE_MODE_RESULTS.get((case["case_id"], "eager"))
        eager_arrays = _CASE_MODE_ARRAYS.get((case["case_id"], "eager"))
        jit_arrays = rec.get("_arrays")
        if eager_rec is not None and eager_rec.get("status") == "PASS" and eager_arrays and jit_arrays:
            problems = _consistency_problems(eager_arrays, jit_arrays)
            if problems:
                rec.update(status="FAIL", detail="eager/jit 一致性失败：" + "；".join(problems))
    arrays = rec.pop("_arrays", None)
    _CASE_MODE_RESULTS[(case["case_id"], mode)] = dict(rec)
    if arrays is not None:
        _CASE_MODE_ARRAYS[(case["case_id"], mode)] = arrays
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


def build_report_payload(results: list[dict]) -> dict:
    summary = {status: sum(1 for item in results if item["status"] == status)
               for status in engine.CASE_STATUSES}
    summary.update(
        TOTAL=len(results),
        identity_count=len({item["case_id"] for item in results}),
        mode_executions=len(results),
        expected_cases=EXPECTED_PYTEST_CASES,
        expected_mode_executions=EXPECTED_PYTEST_MODES,
        expected_base_cases=EXPECTED_BASE_CASES,
        expected_base_mode_executions=EXPECTED_BASE_MODES,
        selfcheck_count=len(SELFCHECK_NAMES),
        selfcheck_pass=len(_SELFCHECK_PASSED & set(SELFCHECK_NAMES)),
        UNSUPPORTED_modes=sum(1 for item in results if item["status"] == "UNSUPPORTED"),
        PASS_modes=sum(1 for item in results if item["status"] == "PASS"),
    )
    golden = next((item.get("golden") for item in results
                   if (item.get("golden") or {}).get("backend")), {})
    family_counts: dict[str, int] = {}
    for item in results:
        family = (item.get("dispatch") or {}).get("family")
        if family:
            family_counts[str(family)] = family_counts.get(str(family), 0) + 1
    return {
        "schema": "opx-jax-assoc-correctness/1",
        "op": OP,
        "cases_sha256": _sha256_file(FROZEN_CASES_PATH),
        "benchmark_sha256": _sha256_file(BENCH_PATH),
        "supplement_sha256": _sha256_file(SUPPLEMENT_PATH),
        "whitebox_sha256": _sha256_file(WHITEBOX_PATH) if WHITEBOX_PATH.is_file() else None,
        "suite_ref": os.environ.get("OPX_SUITE_REF") or None,
        "candidate": {
            "interface": f"ops/{OP}/python/{OP}.py::associative_scan",
            "compiled_interface": f"ops/{OP}/python/{OP}.py::associative_scan_compiled",
            "tensor_family": {
                "executed": dict(sorted(family_counts.items())),
                "description": "由本次真实 dispatch 生成，不对未执行的族做声明",
            },
        },
        "golden": {
            "backend": golden.get("backend", "cpu"),
            "jax": golden.get("jax"),
            "jaxlib": golden.get("jaxlib"),
            "pid": golden.get("pid"),
        },
        "cases": results,
        "summary": summary,
    }


def _render_md(payload: dict) -> str:
    lines = [
        f"# {OP} · pytest 正确性报告",
        "",
        f"- 用例集：`cases/{OP}/cases.json`（sha256 `{payload['cases_sha256'][:16]}…`）"
        f" + benchmark + supplement",
        f"- suite_ref：`{payload['suite_ref']}` ｜ golden：CPU JAX {payload['golden'].get('jax')}",
        f"- 汇总：{payload['summary']}",
        "",
        "| 用例 | 来源 | mode | 期望 | 状态 | matched_ratio | max_abs_error | 说明 |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for item in payload["cases"]:
        lines.append(
            f"| {item['case_id']} | {item.get('origin', '')} | {item.get('mode', '')} | "
            f"{item.get('expected')} | **{item['status']}** | {item.get('matched_ratio')} | "
            f"{item.get('max_abs_error')} | "
            f"{(item.get('detail') or '').replace('|', '/')[:140]} |")
    lines += [
        "",
        "> UNSUPPORTED 不是 PASS；判定口径与门禁同为 criteria/check.py；",
        "> 结构/shape/dtype 精确；非有限位逐元素精确；候选必须在非 CPU 后端执行。",
    ]
    return "\n".join(lines) + "\n"


def write_report(results: list[dict], out_dir: Path | None = None) -> tuple[Path, Path]:
    out = out_dir or Path(os.environ.get("OPX_OUT", "/out"))
    out.mkdir(parents=True, exist_ok=True)
    payload = build_report_payload(results)
    json_path = out / "assoc_scan_correctness.json"
    md_path = out / "assoc_scan_correctness.md"
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    md_path.write_text(_render_md(payload), encoding="utf-8")
    return json_path, md_path


@pytest.fixture(scope="session", autouse=True)
def _session_report_writer():
    yield
    if _RESULTS:
        paths = write_report(_RESULTS)
        print(f"\nASSOC_REPORT {paths[0]} {paths[1]}")


# ---------------------------------------------------------------------------
# 正确性用例（20 条身份 / 35 个 (case_id, mode) 参数化项）
# ---------------------------------------------------------------------------


@pytest.mark.opx_case
@pytest.mark.parametrize("case,mode", ITEMS, ids=[f"{case['case_id']}[{mode}]" for case, mode in ITEMS])
def test_assoc_scan_case(case, mode):
    rec = _run_mode(case, mode)
    _RESULTS.append(rec)
    print("ASSOC_CASE " + json.dumps(rec, ensure_ascii=False))
    if rec["status"] == "PASS":
        return
    if rec["status"] == "UNSUPPORTED":
        if case.get("expected") == "match_cpu_or_explicit_unsupported":
            pytest.xfail(f"{case['case_id']}/{mode}: 显式 UNSUPPORTED（已记录，不记 PASS）")
        pytest.fail(f"{case['case_id']}/{mode}: 不允许的 UNSUPPORTED")
    pytest.fail(f"{case['case_id']}/{mode}: {rec['status']} - {rec.get('detail')}")


# ---------------------------------------------------------------------------
# 自检（CPU 可跑；不需要候选/设备；不计入 20/35）
# ---------------------------------------------------------------------------


def test_selfcheck_discovery_counts():
    assert len(FROZEN_SUITE["cases"]) == EXPECTED_GATE_CASES, (
        f"门禁冻结用例数变成 {len(FROZEN_SUITE['cases'])}，期望 {EXPECTED_GATE_CASES}；"
        "改冻结集会改变 suite_ref，需先取得 PM/用户确认")
    assert GATE_MODE_COUNT == EXPECTED_GATE_MODES, (
        f"门禁 mode 次数变成 {GATE_MODE_COUNT}，期望 {EXPECTED_GATE_MODES}")
    assert len(SUPPLEMENT_SUITE["cases"]) == EXPECTED_SUPPLEMENT_CASES, (
        f"补充用例数变成 {len(SUPPLEMENT_SUITE['cases'])}，期望 {EXPECTED_SUPPLEMENT_CASES}；"
        "设计 §5.4 是 8 条 L1 + 3 条 L2")
    assert len(BASE_CASES) == EXPECTED_BASE_CASES, (
        f"pytest 身份数（不含白盒）变成 {len(BASE_CASES)}，期望 {EXPECTED_BASE_CASES}")
    assert BASE_MODE_COUNT == EXPECTED_BASE_MODES, (
        f"pytest mode 参数化项（不含白盒）变成 {BASE_MODE_COUNT}，期望 {EXPECTED_BASE_MODES}")
    assert len({case['case_id'] for case in BASE_CASES}) == EXPECTED_BASE_CASES, (
        "pytest 身份数必须按 case_id 去重后仍为 20")
    assert len(WHITEBOX_SUITE.get("cases") or []) == EXPECTED_WHITEBOX_CASES, (
        f"白盒用例数变成 {len(WHITEBOX_SUITE.get('cases') or [])}，期望 {EXPECTED_WHITEBOX_CASES}；"
        "白盒文件见 tests/jax_lax_associative_scan_whitebox_cases.json")
    assert sum(len(case.get("mode") or ["jit"])
               for case in (WHITEBOX_SUITE.get("cases") or [])) == EXPECTED_WHITEBOX_MODES, (
        "白盒 (case_id, mode) 次数变化；请同步 EXPECTED_WHITEBOX_MODES")
    assert len(CASES) == EXPECTED_PYTEST_CASES, (
        f"pytest 用例身份数变成 {len(CASES)}，期望 {EXPECTED_PYTEST_CASES}；"
        "请同步 EXPECTED_PYTEST_CASES 并在 1-测试工程说明 §4.2 记录发现数")
    assert MODE_COUNT == EXPECTED_PYTEST_MODES, (
        f"pytest mode 参数化项变成 {MODE_COUNT}，期望 {EXPECTED_PYTEST_MODES}；"
        "请同步 EXPECTED_PYTEST_MODES 并在 1-测试工程说明 §4.2 记录发现数")
    origins: dict[str, int] = {}
    for case in CASES:
        origins[case["_origin"]] = origins.get(case["_origin"], 0) + 1
    expected_origins = {"frozen": 7, "benchmark": 2, "supplement": 11,
                        "whitebox": EXPECTED_WHITEBOX_CASES}
    assert origins == expected_origins, f"来源分布变化：{origins}；请同步 origins 期望"
    assert len(SELFCHECK_NAMES) == len(set(SELFCHECK_NAMES)) == 11, (
        "自检名清单必须唯一且为 11 项；新增/删除自检时同步 SELFCHECK_NAMES 与报告口径")
    _mark_selfcheck("test_selfcheck_discovery_counts")


def test_selfcheck_smoke_selection():
    selected = engine.pick_smoke_case(FROZEN_SUITE["cases"])
    assert selected["case_id"] == "assoc_sum_len1", selected["case_id"]
    fixture = fixtures.make_fixture(selected)
    assert fixture["numel"] == 16, fixture["numel"]
    # 旧 elementwise 规则（expected=="match" + n_a==n_b）在 assoc 用例上命中 0 条，
    # 必须显式分派，不能靠隐式字段猜测。
    old_rule = [case for case in FROZEN_SUITE["cases"]
                if case.get("expected", "match") == "match"
                and isinstance(case.get("n_a"), int) and case.get("n_a") == case.get("n_b")]
    assert old_rule == []
    # 不允许把 AD / 拒绝用例当 smoke
    for case in FROZEN_SUITE["cases"]:
        if case["case_id"] in ("assoc_grad_sum",):
            assert "jit" not in (case.get("mode") or [])
    assert all(case.get("expected") == "match_cpu" for case in [selected])
    _mark_selfcheck("test_selfcheck_smoke_selection")


def test_selfcheck_data_generation_frozen_rules():
    f1 = "assoc_sum_len8_axis0"
    case = next(case for case in CASES if case["case_id"] == f1)
    fixture_a = fixtures.make_fixture(case)
    fixture_b = fixtures.make_fixture(case)
    assert fixture_a["inputs_sha256"] == fixture_b["inputs_sha256"], "同一 (case, seed) 必须字节级相同"
    arrays = fixture_a["arrays"]
    assert list(arrays) == ["elems"]
    arr = arrays["elems"]
    assert arr.dtype == np.float32 and arr.shape == (8, 16)
    expected = np.random.default_rng(20261026).uniform(-1.0, 1.0, size=(8, 16)).astype(np.float32)
    assert np.array_equal(arr, expected), "单叶子生成规则必须是 default_rng(seed).uniform(-1,1).astype(float32)"

    # pytree：按 elems 字符串声明顺序（a 然后 b）用同一个 rng 连续取样
    case4 = next(case for case in CASES if case["case_id"] == "assoc_pytree_add_len6")
    fixture4 = fixtures.make_fixture(case4)
    assert fixture4["leaf_order"] == ["elems/a", "elems/b"], fixture4["leaf_order"]
    rng = np.random.default_rng(20261026)
    a = rng.uniform(-1.0, 1.0, size=(6, 4)).astype(np.float32)
    b = rng.uniform(-1.0, 1.0, size=(6, 2)).astype(np.float32)
    assert np.array_equal(fixture4["arrays"]["elems/a"], a)
    assert np.array_equal(fixture4["arrays"]["elems/b"], b)

    # L2-1 按 case.dtype 生成 float16（仅用于验证拒绝）
    case_f16 = next(case for case in CASES if case["case_id"] == "assoc_l2_dtype_f16")
    fixture_f16 = fixtures.make_fixture(case_f16)
    assert fixture_f16["arrays"]["elems"].dtype == np.float16
    assert fixture_f16["combine_id"] == 1
    _mark_selfcheck("test_selfcheck_data_generation_frozen_rules")


def test_selfcheck_criteria_thresholds():
    good = CRITERIA_MOD.check(np.ones(8, np.float32), np.ones(8, np.float32))
    assert good["is_pass"] is True
    assert good["profile"] == "ops-precision-standard@1"
    assert good["rtol"] == 2 ** -10, good["rtol"]
    assert good["atol"] == 2 ** -16, good["atol"]
    assert good["required_matched_ratio"] == 0.99, good["required_matched_ratio"]
    assert good["max_abs_error_limit"] == 1e-2, good["max_abs_error_limit"]
    bad = CRITERIA_MOD.check(np.ones(8, np.float32), np.zeros(8, np.float32))
    assert bad["is_pass"] is False
    # 非有限位逐元素精确（不是被容差吞掉）
    nan_ok = CRITERIA_MOD.check(np.array([np.nan, 1.0], np.float32),
                                np.array([np.nan, 1.0], np.float32))
    assert nan_ok["is_pass"] is True and nan_ok["nonfinite"]["count"] == 1
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
        assert_all_leaves_on_accelerator(np.zeros(4, np.float32), "neg_numpy_no_dispatch")
    with pytest.raises(AssertionError):
        assert_all_leaves_on_accelerator(np.zeros(4, np.float32), "neg_numpy_trace",
                                         {"path": "trace", "backend": "npu", "abi_calls": 0})
    with pytest.raises(AssertionError):
        assert_all_leaves_on_accelerator(np.zeros(4, np.float32), "neg_numpy_cpu",
                                         {"path": "so", "backend": "cpu", "abi_calls": 2})
    with pytest.raises(AssertionError):
        assert_all_leaves_on_accelerator(_FakeJaxLeaf("cpu"), "neg_jax_cpu")
    with pytest.raises(AssertionError):
        assert_all_leaves_on_accelerator((_FakeJaxLeaf("npu"), _FakeJaxLeaf("cpu")), "neg_jax_mixed")
    recorded = assert_all_leaves_on_accelerator(
        np.zeros(4, np.float32), "pos_numpy",
        {"path": "so", "backend": "npu", "abi_calls": 2, "device_id": 0})
    assert recorded[0]["kind"] == "numpy-host-copy"

    # 显式 unsupported 硬证据（与 runner 共用同一 helper）
    good = {"op": OP, "mode": "jvp", "path": "trace", "abi_calls": 0, "backend": "npu"}
    assert engine.unsupported_attempt_problems(
        good, "RuntimeError: backend rejects autodiff", require_op=True, expected_mode="jvp") == []
    for name, dispatch in {
        "no_exec": {},
        "path_so": {**good, "path": "so"},
        "backend_cpu": {**good, "backend": "cpu"},
        "wrong_mode": {**good, "mode": "jit"},
        "wrong_op": {**good, "op": "other"},
    }.items():
        assert engine.unsupported_attempt_problems(
            dispatch, "RuntimeError: boom", require_op=True, expected_mode="jvp"), name
    assert engine.unsupported_attempt_problems(good, engine.UNSUPPORTED_CODE), "缺后端原文必须被拒"
    _mark_selfcheck("test_selfcheck_device_guard_negative")


def test_selfcheck_golden_env_guard_negative(tmp_path):
    case = next(case for case in CASES if case["case_id"] == "assoc_sum_len8_axis0")
    fixture = fixtures.make_fixture(case)
    case_json = tmp_path / "case.json"
    inputs = tmp_path / "inputs.npz"
    fixtures.write_case_json(case, fixture, case_json)
    fixtures.write_inputs_npz(fixture, inputs)
    env = os.environ.copy()
    env["JAX_PLATFORMS"] = "cpu"          # 故意只给 CPU
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


def _load_golden_module():
    spec = importlib.util.spec_from_file_location("assoc_golden_selfcheck", GOLDEN_DIR / "golden.py")
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_selfcheck_golden_reject_rules():
    golden = _load_golden_module()
    reject_cases = [case for case in CASES if case.get("expected") == "reject"]
    base_reject = [case for case in reject_cases if case["_origin"] != "whitebox"]
    whitebox_reject = [case for case in reject_cases if case["_origin"] == "whitebox"]
    assert len(base_reject) == 3, [case["case_id"] for case in base_reject]
    assert len(whitebox_reject) == 1, [case["case_id"] for case in whitebox_reject]
    for case in reject_cases:
        fixture = fixtures.make_fixture(case)
        code = golden.expect_reject(case, fixture)
        assert code == case["expect_reject_code"], (case["case_id"], code)
        evidence = golden.reject_evidence(case, fixture)
        assert evidence["reject_code"] == code
        if code == "invalid_axis":
            assert evidence.get("rule") == "spec_axis_range", evidence
            # 默认要求 JAX 也拒绝（更强的交叉证据）；仅当用例显式标注 spec_rule_only
            # （例如 axis=True：JAX 可能把 bool 当 1 接受，拒绝依据是 spec 的静态整数规则）
            # 时，JAX 交叉结果只作补充证据。
            if case.get("spec_rule_only"):
                assert evidence.get("jax_rejects") in (True, False), evidence
            else:
                assert evidence.get("jax_rejects") is True, evidence
        else:
            assert evidence.get("jax_rejects") is None, evidence
    # 合法用例不得被 expect_reject 误判
    legal = next(case for case in CASES if case["case_id"] == "assoc_sum_len8_axis0")
    with pytest.raises(RuntimeError):
        golden.expect_reject(legal, fixtures.make_fixture(legal))
    _mark_selfcheck("test_selfcheck_golden_reject_rules")


def test_selfcheck_golden_signature_guard():
    golden = _load_golden_module()
    with pytest.raises(TypeError):
        golden.compute_golden(np.zeros(4, np.float32), np.zeros(4, np.float32))
    _mark_selfcheck("test_selfcheck_golden_signature_guard")


def _numpy_prefix_reference(fixture: dict) -> dict:
    axis = int(fixture["axis"])
    reverse = bool(fixture["reverse"])
    out = {}
    for leaf in fixture["leaves"]:
        arr = np.asarray(fixture["arrays"][leaf["path"]], dtype=np.float32)
        if reverse:
            arr = np.flip(arr, axis=axis)
        if fixture["combine_name"] in ("sum", "pytree_add"):
            res = np.cumsum(arr, axis=axis, dtype=np.float32)
        elif fixture["combine_name"] == "max":
            res = np.maximum.accumulate(arr, axis=axis)
        else:
            raise AssertionError(f"golden 数值自检未覆盖 op_kind={fixture['combine_name']!r}")
        if reverse:
            res = np.flip(res, axis=axis)
        key = "out" if leaf["path"] == "elems" else f"out/{leaf['name']}"
        out[key] = np.ascontiguousarray(res)
    return out


def test_selfcheck_golden_numeric_binding(tmp_path):
    """代表用例的 golden 参数绑定：golden_cli（CPU 子进程）与 numpy 前缀语义交叉核对。

    仅验证测试工程能正确绑定 case/axis/reverse/combine；不是 NPU 精度证据。
    """
    case_ids = ("assoc_sum_len8_axis0", "assoc_max_len7_axis0",
                "assoc_reverse_sum_len9", "assoc_pytree_add_len6")
    for case_id in case_ids:
        case = next(case for case in CASES if case["case_id"] == case_id)
        fixture = fixtures.make_fixture(case)
        cdir = tmp_path / case_id
        cdir.mkdir(parents=True, exist_ok=True)
        case_json = cdir / "case.json"
        inputs = cdir / "inputs.npz"
        fixtures.write_case_json(case, fixture, case_json)
        fixtures.write_inputs_npz(fixture, inputs)
        golden = engine._run_golden(WS, case_json, inputs, "eager", cdir)
        assert golden["ok"], f"{case_id}: {golden.get('detail')}"
        reference = _numpy_prefix_reference(fixture)
        assert set(golden["arrays"]) == set(reference), (case_id, sorted(golden["arrays"]))
        for key in reference:
            got = np.asarray(golden["arrays"][key])
            want = np.asarray(reference[key])
            if fixture["combine_name"] == "max":
                assert np.array_equal(got, want), f"{case_id}/{key}: max 前缀应精确一致"
            else:
                verdict = CRITERIA_MOD.check(got.astype(np.float32), want.astype(np.float32))
                assert verdict["is_pass"], f"{case_id}/{key}: {verdict.get('failure_reasons')}"

    # AD golden 的解析核对（设计 §2.1.3 的辅助诊断，不作通过依据）
    ad_case = next(case for case in CASES if case["case_id"] == "assoc_grad_sum")
    fixture = fixtures.make_fixture(ad_case)
    cdir = tmp_path / "ad"
    cdir.mkdir(parents=True, exist_ok=True)
    case_json = cdir / "case.json"
    inputs = cdir / "inputs.npz"
    fixtures.write_case_json(ad_case, fixture, case_json)
    fixtures.write_inputs_npz(fixture, inputs)
    jvp = engine._run_golden(WS, case_json, inputs, "jvp", cdir)
    assert jvp["ok"], f"AD jvp: {jvp.get('detail')}"
    vjp = engine._run_golden(WS, case_json, inputs, "vjp", cdir)
    assert vjp["ok"], f"AD vjp: {vjp.get('detail')}"
    length = int(fixture["geometry"][0]["length"])
    tangent_expected = np.repeat(np.arange(1, length + 1, dtype=np.float32)[:, None], 4, axis=1)
    cotangent_expected = np.repeat(
        np.arange(length, 0, -1, dtype=np.float32)[:, None], 4, axis=1)
    assert np.array_equal(np.asarray(jvp["arrays"]["jvp_tangent"]), tangent_expected)
    assert np.array_equal(np.asarray(vjp["arrays"]["vjp_cotangent"]), cotangent_expected)
    _mark_selfcheck("test_selfcheck_golden_numeric_binding")


def test_selfcheck_report_writer(tmp_path):
    synthetic = {
        "case_id": "selfcheck_report_writer", "group": "自检", "origin": "selfcheck",
        "mode": "jit", "required": False, "expected": "match_cpu", "status": "NOT_RUN",
        "detail": "仅验证报告写入链路，不是算子执行结果", "dispatch": {}, "golden": {},
    }
    json_path, md_path = write_report([synthetic], out_dir=tmp_path)
    payload = json.loads(json_path.read_text(encoding="utf-8"))
    assert payload["schema"] == "opx-jax-assoc-correctness/1"
    assert payload["cases"][0]["case_id"] == "selfcheck_report_writer"
    assert payload["summary"]["expected_cases"] == EXPECTED_PYTEST_CASES
    assert payload["summary"]["expected_mode_executions"] == EXPECTED_PYTEST_MODES
    assert payload["summary"]["selfcheck_count"] == len(SELFCHECK_NAMES)
    assert json_path.stat().st_size > 0 and md_path.stat().st_size > 0
    _mark_selfcheck("test_selfcheck_report_writer")


def test_selfcheck_whitebox_hook(tmp_path):
    """预留白盒入口：白盒 JSON 与冻结/基准/补充走同一 loader 与同一 pytest 参数化入口。"""
    whitebox = tmp_path / f"{OP}_whitebox_cases.json"
    whitebox.write_text(json.dumps({
        "schema": "opx-cases/1",
        "op": OP,
        "criteria_ref": "ops-precision-standard@1",
        "cases": [{
            "case_id": "assoc_wb_selfcheck_only",
            "group": "白盒",
            "required": True,
            "op_kind": "sum",
            "shape": [4, 8],
            "axis": 0,
            "reverse": False,
            "dtype": "float32",
            "seed": 20261026,
            "mode": ["eager", "jit"],
            "expected": "match_cpu",
        }],
    }), encoding="utf-8")
    loaded, suites = _load_cases(whitebox_path=whitebox)
    assert len(loaded) == len(BASE_CASES) + 1
    assert sum(len(case["mode"]) for case in loaded) == BASE_MODE_COUNT + 2
    assert suites["whitebox"]["cases"][0]["case_id"] == "assoc_wb_selfcheck_only"
    assert loaded[-1]["_origin"] == "whitebox"
    _mark_selfcheck("test_selfcheck_whitebox_hook")


# ---------------------------------------------------------------------------
# 白盒 CPU 分支用例（节点 3；依据当前源码逐分支构造，不冒充设备数值证据）
# ---------------------------------------------------------------------------
#
# 覆盖对象：
#   * Python 接口 ``_analyze`` / ``_geometry`` / ``_combine_id_of`` / ``_flatten`` /
#     ``_rebuild`` / ``_tree_map`` / ``_leaf_kind`` / ``_is_tracer`` / ``_is_contiguous``
#     的拒绝与辅助分支（``_plan_dispatch`` 四出口，纯 CPU）；
#   * 公开入口的 9 叶子负向用例（``associative_scan`` / ``associative_scan_compiled`` 各一，
#     期望 device 前 ``invalid_argument``；当前暴露 M-02，交 ``[4]``）；
#   * ``_run_trace`` / ``_scan_leaf_trace`` 的 reverse 与 sum/max 组合（CPU JAX，判据复用
#     ``criteria/check.py``）与「泛化异常不得判 UNSUPPORTED」负向用例（暴露 M-01，交 ``[4]``）；
#   * ``_autodiff_kernel`` 的设备前守卫；
#   * host C ABI（``libjax_lax_associative_scan_host.so``）的全部参数校验分支与 buffer 辅助
#     guard（校验发生在任何 ACL 调用之前，因此本机即可执行；仍需 build 产物）；
#   * 可执行入口 ``build/jax_lax_associative_scan`` 的 argv 协议分支；
#   * host driver CLI（``jax_lax_associative_scan_cli.py``）的参数/IO/拒绝/异常分支，以及
#     ``_tree_from_spec`` 的 leaf/dict/tuple/list/未知 kind 分支。
#
# 这些用例不写入 ``EXPECTED_PYTEST_*`` 的用例参数化计数；设备数值覆盖由白盒 JSON 的
# ``assoc_wb_*`` 7 条（13 个 mode 项）与既有黑盒共同承担。


class _WbFakeJaxLeaf:
    """伪 jax 叶子：只用于 CPU 路由分支，不参与任何数值判定。"""

    __module__ = "jax"

    def __init__(self, shape=(4,), dtype="float32"):
        self.shape = tuple(shape)
        self.dtype = dtype


class _WbFakeJaxTracer(_WbFakeJaxLeaf):
    """类型名含 ``Tracer`` → ``_is_tracer`` 为 True。"""

    __module__ = "jax"


class _WbFakeTorchLeaf:
    __module__ = "torch"

    def __init__(self):
        self.shape = (4,)
        self.dtype = "float32"

    def data_ptr(self):
        return 0x1000


class _WbDeviceLeaf:
    """无 torch/jax 模块名、但有 ``devices()``：命中 ``_leaf_kind`` 的 devices 分支。"""

    def __init__(self):
        self.shape = (4,)
        self.dtype = "float32"

    def devices(self):
        return []


class _WbPtrLeaf:
    """无 torch/jax 模块名、但有 ``data_ptr()``：命中 ``_leaf_kind`` 的 data_ptr 分支。"""

    def __init__(self):
        self.shape = (4,)
        self.dtype = "float32"

    def data_ptr(self):
        return 0


class _WbNumpyLikeNoFlags:
    """模块名伪装 numpy 且无 ``flags``：命中 ``_is_contiguous`` 的 flags is None 分支。"""

    __module__ = "numpy"

    def __init__(self):
        self.shape = (4,)
        self.dtype = "float32"


def _wb_plain_sum(a, b):
    return a + b


def _wb_assert_reject(plan: dict, code: str, label: str) -> None:
    assert plan["path"] == "reject", f"{label}: path={plan['path']!r}（期望 reject）"
    assert plan["exec_mode"] == "reject", f"{label}: exec_mode={plan['exec_mode']!r}"
    assert plan["abi_calls"] == 0, f"{label}: 拒绝路径 abi_calls={plan['abi_calls']}（必须 0）"
    assert plan["reason"] == code, f"{label}: reason={plan['reason']!r}（期望 {code}）"


def test_whitebox_route_branches():
    """P5/P6/P7/P8/P9/P10/P18：``_analyze`` 拒绝出口与 ``_plan_dispatch`` 四出口（纯 CPU）。"""
    cand = _candidate()
    fn = fixtures.combine_sum
    ok_leaf = np.zeros((8, 16), np.float32)

    plan = cand._plan_dispatch(fn, ok_leaf)
    assert plan["path"] == "so" and plan["family"] == "numpy" and plan["combine_id"] == cand.COMBINE_SUM
    plan = cand._plan_dispatch(fn, {"a": np.zeros((6, 4), np.float32),
                                    "b": np.zeros((6, 2), np.float32)})
    assert plan["path"] == "so" and plan["combine_id"] == cand.COMBINE_SUM

    _wb_assert_reject(cand._plan_dispatch(fn, {}), "invalid_argument", "empty dict")
    _wb_assert_reject(cand._plan_dispatch(fn, []), "invalid_argument", "empty list")
    _wb_assert_reject(cand._plan_dispatch(fn, np.float32(1.0)), "invalid_argument", "rank-0")
    _wb_assert_reject(cand._plan_dispatch(fn, ok_leaf, axis=True), "invalid_axis", "axis bool")
    _wb_assert_reject(cand._plan_dispatch(fn, ok_leaf, axis=1.0), "invalid_axis", "axis float")
    _wb_assert_reject(cand._plan_dispatch(fn, ok_leaf, axis=2), "invalid_axis", "axis oob")
    _wb_assert_reject(cand._plan_dispatch(fn, np.asfortranarray(np.zeros((8, 16), np.float32))),
                      "invalid_argument", "non-contiguous")
    _wb_assert_reject(cand._plan_dispatch(fn, np.zeros((8, 16), np.float16)),
                      "dtype_mismatch", "float16")
    _wb_assert_reject(cand._plan_dispatch(fn, {"a": np.zeros((6, 4), np.float32),
                                               "b": np.zeros((5, 2), np.float32)}),
                      "invalid_argument", "shape[axis] mismatch")
    _wb_assert_reject(cand._plan_dispatch(fn, {"a": np.zeros((4, 2), np.float32),
                                               "b": np.zeros((4,), np.float32)}, axis=1),
                      "invalid_axis", "per-leaf axis")
    _wb_assert_reject(cand._plan_dispatch(fn, [np.zeros((4,), np.float32), "x"]),
                      "invalid_argument", "mixed leaf kinds")
    _wb_assert_reject(cand._plan_dispatch(fn, [object(), object()]),
                      "invalid_argument", "all-other leaf kinds")
    _wb_assert_reject(cand._plan_dispatch(fn, [_WbFakeTorchLeaf()]),
                      "invalid_argument", "torch family not registered")

    # 未注册 callable → trace 路由（不得静默数值化）
    plan = cand._plan_dispatch(_wb_plain_sum, ok_leaf)
    assert plan["path"] == "trace" and plan["reason"] == "unsupported_combine_fn"
    # jax family（具体数组）→ trace 路由
    before = dict(cand._last_dispatch)
    plan = cand._plan_dispatch(fn, [_WbFakeJaxLeaf(shape=(4,))])
    assert plan["path"] == "trace" and plan["family"] == "jax"
    # ``_plan_dispatch`` 不得污染 ``_last_dispatch``（与调用前逐字段一致）
    assert dict(cand._last_dispatch) == before


def test_whitebox_tree_and_combine_helpers():
    """P2/P3/P4/P11/P12：树展开/重建、族识别、几何、combine 解析（纯 CPU）。"""
    cand = _candidate()

    paths = [path for path, _leaf in cand._flatten(
        {"a": np.zeros(2), "b": (np.ones(1), [np.zeros(3)])})]
    assert paths == ["a", "b/0", "b/1/0"], paths
    consumed = iter(["A", "B", "C", "D"])
    rebuilt = cand._rebuild({"a": (1, [2, 3]), "b": 4}, consumed)
    assert rebuilt == {"a": ("A", ["B", "C"]), "b": "D"}, rebuilt
    mapped = cand._tree_map(lambda value: value * 2, {"a": (1, [2]), "b": 3})
    assert mapped == {"a": (2, [4]), "b": 6}, mapped

    assert cand._geometry((4, 3, 5), 0) == {"outer": 1, "length": 4, "inner": 15}
    assert cand._geometry((4, 3, 5), 1) == {"outer": 4, "length": 3, "inner": 5}
    assert cand._geometry((4, 3, 5), 2) == {"outer": 12, "length": 5, "inner": 1}

    assert cand._leaf_kind(np.zeros(2)) == "numpy"
    assert cand._leaf_kind(_WbFakeJaxLeaf()) == "jax"
    assert cand._leaf_kind(_WbFakeTorchLeaf()) == "torch"
    assert cand._leaf_kind(_WbDeviceLeaf()) == "jax"
    assert cand._leaf_kind(_WbPtrLeaf()) == "torch"
    assert cand._leaf_kind(object()) == "other"
    assert cand._is_tracer(_WbFakeJaxTracer()) is True
    assert cand._is_tracer(_WbFakeJaxLeaf()) is False
    assert cand._is_tracer(np.zeros(2)) is False
    assert cand._is_contiguous(np.zeros((4, 6), np.float32)) is True
    assert cand._is_contiguous(np.asfortranarray(np.zeros((4, 6), np.float32))) is False
    assert cand._is_contiguous(_WbFakeJaxLeaf()) is True
    assert cand._is_contiguous(_WbNumpyLikeNoFlags()) is True

    with pytest.raises(cand._AssocReject) as excinfo:
        cand._shape_of(object())
    assert excinfo.value.code == "invalid_argument"
    assert cand._dtype_name(object()) == "unknown"

    assert cand._combine_id_of(fixtures.combine_sum) == cand.COMBINE_SUM
    assert cand._combine_id_of(fixtures.combine_max) == cand.COMBINE_MAX
    _wb_plain_sum.__name__ = "pytree_add"
    try:
        assert cand._combine_id_of(_wb_plain_sum) == cand.COMBINE_TREE_ADD
    finally:
        _wb_plain_sum.__name__ = "_wb_plain_sum"
    bad = lambda a, b: a + b  # noqa: E731 —— 只测解析分支
    bad.combine_id = 99
    bad.__name__ = "not_registered"
    assert cand._combine_id_of(bad) is None
    assert cand._combine_id_of(5) is None


def test_whitebox_trace_path_reverse_and_combine():
    """P16：trace 路径的 reverse/sum/max 语义（CPU 后端）与设备后端的显式分类。

    目标设备镜像的 ``xla-npu`` 后端无法 lowering ``jnp.take``（与 ``assoc_grad_sum`` 连续 3 次
    同码 ``UNSUPPORTED`` 同因），因此**数值语义只在 CPU 后端断言**；在设备后端上本用例断言的是
    真实发生的显式分类（``path=trace`` / ``abi_calls=0`` / ``reason=UNSUPPORTED``），不是 skip。
    """
    cand = _candidate()
    try:
        import jax  # noqa: PLC0415
        import jax.numpy as jnp  # noqa: PLC0415
    except Exception as exc:  # noqa: BLE001
        pytest.fail(f"本机 JAX 不可用：{type(exc).__name__}: {exc}")

    backend = str(jax.default_backend())
    x = np.random.default_rng(20261026).uniform(-1.0, 1.0, size=(5, 4)).astype(np.float32)

    if backend == "cpu":
        for name, combine_id in (("sum", cand.COMBINE_SUM), ("max", cand.COMBINE_MAX)):
            for reverse in (False, True):
                for axis in (0, 1):
                    got = np.asarray(cand._scan_leaf_trace(
                        jnp.asarray(x), combine_id, axis, reverse))
                    base = np.flip(x, axis=axis) if reverse else x
                    if name == "sum":
                        want = np.cumsum(base, axis=axis, dtype=np.float32)
                    else:
                        want = np.maximum.accumulate(base, axis=axis)
                    if reverse:
                        want = np.flip(want, axis=axis)
                    verdict = CRITERIA_MOD.check(np.ascontiguousarray(got, np.float32),
                                                 np.ascontiguousarray(want, np.float32))
                    assert verdict["is_pass"], (name, reverse, axis,
                                                verdict.get("failure_reasons"))
        traced = cand._run_trace({"a": jnp.asarray(x)}, cand.COMBINE_SUM, True, 0)
        assert isinstance(traced, dict) and list(traced) == ["a"]
        assert traced["a"].shape == (5, 4)

    # 显式分类路径（固定 fake npu + 真实后端错误签名，避免机器差异）：必须是
    # trace/abi_calls=0/显式 unsupported + 后端原文。
    saved_scan = cand._scan_leaf_trace
    saved_backend = cand._backend_name
    try:
        cand._backend_name = lambda: "npu"

        def _backend_boom(leaf, combine_id, axis, reverse):
            # 与设备真实 trace 失败同型同文本（jax.errors.JaxRuntimeError 是 RuntimeError 子类），
            # 使“后端签名收窄”无论按异常类型还是按文本判断，都应仍分类为 UNSUPPORTED。
            raise jax.errors.JaxRuntimeError(
                "INTERNAL: Failed to parse AFIR string: take lowering unsupported")

        cand._scan_leaf_trace = _backend_boom
        with pytest.raises(RuntimeError) as excinfo:
            cand._run_trace({"a": jnp.asarray(x)}, cand.COMBINE_SUM, True, 0)
    finally:
        cand._scan_leaf_trace = saved_scan
        cand._backend_name = saved_backend
    assert cand.UNSUPPORTED_CODE in str(excinfo.value)
    dispatch = cand._last_dispatch
    assert dispatch["path"] == "trace" and dispatch["exec_mode"] == "trace"
    assert dispatch["abi_calls"] == 0 and dispatch["reason"] == cand.UNSUPPORTED_CODE
    assert dispatch["backend"] == "npu", dispatch["backend"]

    if backend != "cpu":
        # 设备后端：真实 trace 以同一显式码失败 —— 断言分类而不是要求数值 PASS
        with pytest.raises(RuntimeError) as excinfo:
            cand._run_trace({"a": jnp.asarray(x)}, cand.COMBINE_SUM, True, 0)
        assert cand.UNSUPPORTED_CODE in str(excinfo.value)
        device_dispatch = cand._last_dispatch
        assert device_dispatch["path"] == "trace" and device_dispatch["abi_calls"] == 0
        assert device_dispatch["reason"] == cand.UNSUPPORTED_CODE
        assert str(excinfo.value).strip() != cand.UNSUPPORTED_CODE, "必须附后端原文"


def test_whitebox_trace_error_classification_negative():
    """P16 负向：非后端 unsupported 的 shape/编程异常不得被包装成 UNSUPPORTED。

    D-02（M-01）回归锁定：``_run_trace`` 只允许把非 CPU 后端 runtime 错误的
    AFIR/lowering/未实现签名包装为 ``UNSUPPORTED:autodiff_unsupported``；实现侧
    shape/编程异常必须保留原始类型/文本原样抛出，不能用“任意异常都 UNSUPPORTED”冒充通过。
    """
    cand = _candidate()
    try:
        import jax.numpy as jnp  # noqa: PLC0415
    except Exception as exc:  # noqa: BLE001
        pytest.fail(f"本机 JAX 不可用：{type(exc).__name__}: {exc}")

    class _WbShapeError(ValueError):
        """模拟实现侧 shape/axis 编程错误，不是后端 lowering 不支持。"""

    saved_scan = cand._scan_leaf_trace
    saved_backend = cand._backend_name
    try:
        cand._backend_name = lambda: "npu"

        def _shape_bug(leaf, combine_id, axis, reverse):
            raise _WbShapeError("shape mismatch: axis 越界（编程错误，不是后端 unsupported）")

        cand._scan_leaf_trace = _shape_bug
        with pytest.raises(_WbShapeError) as excinfo:
            cand._run_trace({"a": jnp.zeros((4,), np.float32)}, cand.COMBINE_SUM, False, 0)
    finally:
        cand._scan_leaf_trace = saved_scan
        cand._backend_name = saved_backend
    assert cand.UNSUPPORTED_CODE not in str(excinfo.value)


def test_whitebox_autodiff_kernel_guards():
    """P17：``_autodiff_kernel`` 在 device 前的参数守卫（纯 CPU）。"""
    cand = _candidate()
    try:
        import jax.numpy as jnp  # noqa: PLC0415
    except Exception as exc:  # noqa: BLE001
        pytest.fail(f"本机 JAX 不可用：{type(exc).__name__}: {exc}")
    x = np.zeros((5, 4), np.float32)

    with pytest.raises(ValueError, match="mode="):
        cand._autodiff_kernel(fixtures.combine_sum, {"a": x}, mode="nope")
    with pytest.raises(ValueError, match="只接受 numpy family"):
        cand._autodiff_kernel(fixtures.combine_sum, {"a": jnp.asarray(x)})
    with pytest.raises(ValueError, match="单叶子"):
        cand._autodiff_kernel(fixtures.combine_sum, {"a": x, "b": x})
    with pytest.raises(ValueError, match="unsupported_combine_fn"):
        cand._autodiff_kernel(fixtures.combine_max, {"a": x})
    with pytest.raises(cand._AssocReject) as excinfo:
        cand._autodiff_kernel(fixtures.combine_sum, {"a": x}, axis=2)
    assert excinfo.value.code == "invalid_axis"


def test_whitebox_host_abi_validation():
    """H1–H10/H15–H20：host C ABI 的参数校验与 buffer guard（校验先于任何 ACL 调用）。"""
    import ctypes  # noqa: PLC0415

    cand = _candidate()
    lib = cand._load_lib()
    assert ctypes.sizeof(cand.JaxLaxAssociativeScanArgs) == cand._ARGS_EXPECTED_SIZE
    assert lib.jax_lax_associative_scan_abi_version().decode("utf-8") == cand.ABI_TAG

    assert lib.jax_lax_associative_scan_run(None) == 2, "null args 必须返回 invalid_argument(2)"

    def _args():
        args = cand.JaxLaxAssociativeScanArgs()
        args.abi_version = cand.ABI_VERSION
        args.combine_id = cand.COMBINE_SUM
        args.reverse = 0
        args.n_leaves = 1
        args.device_id = 0
        args.dtype[0] = 0
        args.outer[0] = 2
        args.length[0] = 3
        args.inner[0] = 4
        args.numel[0] = 24
        args.in_ptr[0] = 0x1000
        args.out_ptr[0] = 0x2000
        err = ctypes.create_string_buffer(256)
        args.err = ctypes.cast(err, ctypes.c_void_p)
        args.err_len = 256
        return args, err

    def _abi_version(args, _err):
        args.abi_version = 99

    def _combine(args, _err):
        args.combine_id = 99

    def _reverse(args, _err):
        args.reverse = 2

    def _n_leaves_zero(args, _err):
        args.n_leaves = 0

    def _n_leaves_nine(args, _err):
        args.n_leaves = 9

    def _device(args, _err):
        args.device_id = -1

    def _dtype(args, _err):
        args.dtype[0] = 1

    def _geometry(args, _err):
        args.outer[0] = 0

    def _numel(args, _err):
        args.numel[0] = 23

    def _null_ptr(args, _err):
        args.in_ptr[0] = None
        args.out_ptr[0] = None

    guards = [
        ("abi_version", _abi_version, "abi_version=99"),
        ("combine_id", _combine, "combine_id=99"),
        ("reverse", _reverse, "reverse=2"),
        ("n_leaves=0", _n_leaves_zero, "n_leaves=0"),
        ("n_leaves=9", _n_leaves_nine, "n_leaves=9"),
        ("device_id", _device, "device_id=-1"),
        ("dtype", _dtype, "dtype=1"),
        ("geometry", _geometry, "几何非法"),
        ("numel", _numel, "numel=23"),
        ("null ptr", _null_ptr, "设备地址为空"),
    ]
    problems = []
    for label, mutate, expected in guards:
        args, err = _args()
        mutate(args, err)
        rc = int(lib.jax_lax_associative_scan_run(ctypes.byref(args)))
        text = err.value.decode("utf-8", "replace")
        if rc != 2:
            problems.append(f"{label}: rc={rc}（期望 2）")
        if not text.startswith("invalid_argument:"):
            problems.append(f"{label}: err={text!r}（期望 invalid_argument: ...）")
        if expected not in text:
            problems.append(f"{label}: err 缺少 {expected!r}（实际 {text!r}）")
    assert not problems, "；".join(problems)

    # WriteErr 的防御出口：err 指针为空 / err_len<=0 时不得崩溃，仍返回 2
    args, _err = _args()
    args.abi_version = 99
    args.err = None
    args.err_len = 0
    assert int(lib.jax_lax_associative_scan_run(ctypes.byref(args))) == 2
    args, _err = _args()
    args.combine_id = 99
    args.err_len = 0
    assert int(lib.jax_lax_associative_scan_run(ctypes.byref(args))) == 2

    # H17–H20：buffer 辅助函数的空指针/非正字节 guard
    import ctypes as _ct  # noqa: PLC0415
    assert int(lib.jax_lax_associative_scan_malloc(None, 0)) == 2
    ptr = _ct.c_void_p()
    assert int(lib.jax_lax_associative_scan_malloc(_ct.byref(ptr), 0)) == 2
    assert int(lib.jax_lax_associative_scan_free(None)) == 0
    assert int(lib.jax_lax_associative_scan_h2d(None, None, 0)) == 2
    assert int(lib.jax_lax_associative_scan_d2h(None, None, 0)) == 2


def test_whitebox_exec_argv_protocol():
    """E2/E3/E1：可执行入口的 argv 协议分支与 ``--self-check``（需要 build 产物，不 skip）。"""
    exe = WS / "ops" / OP / "build" / OP
    if not exe.is_file() or not os.access(exe, os.X_OK):
        pytest.fail(f"可执行入口不存在或不可执行：{exe}（先 build；本用例不是 skip）")

    proc = subprocess.run([str(exe)], capture_output=True, text=True, timeout=120)
    assert proc.returncode == 2, (proc.returncode, proc.stdout[-400:], proc.stderr[-400:])
    assert "用法" in (proc.stderr or "") + (proc.stdout or "")

    proc = subprocess.run([str(exe), "--bogus", "1"], capture_output=True, text=True, timeout=120)
    assert proc.returncode == 2
    assert "REJECT_BAD_ARGS unknown key --bogus" in (proc.stdout or "")

    proc = subprocess.run([str(exe), "--case-json"], capture_output=True, text=True, timeout=120)
    assert proc.returncode == 2
    assert "missing value for --case-json" in (proc.stdout or "")

    proc = subprocess.run([str(exe), "--self-check"], capture_output=True, text=True, timeout=120)
    assert proc.returncode == 0, (proc.returncode, proc.stdout[-800:], proc.stderr[-400:])
    assert "SELF_CHECK so=" in (proc.stdout or "")
    assert "loadable=1" in (proc.stdout or "")
    assert f"abi={_candidate().ABI_TAG}" in (proc.stdout or "")


_CLI_MODULE = None


def _wb_cli_module():
    global _CLI_MODULE
    if _CLI_MODULE is not None:
        return _CLI_MODULE
    path = WS / "ops" / OP / "python" / f"{OP}_cli.py"
    if not path.is_file():
        pytest.fail(f"host driver CLI 不存在：{path}")
    spec = importlib.util.spec_from_file_location(f"{OP}_cli_under_test", path)
    if spec is None or spec.loader is None:
        pytest.fail(f"host driver CLI 无法加载：{path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    _CLI_MODULE = module
    return module


def _wb_write_cli_case(case: dict, directory: Path) -> tuple[Path, Path]:
    directory.mkdir(parents=True, exist_ok=True)
    fixture = fixtures.make_fixture(case)
    case_json = directory / "case.json"
    inputs = directory / "inputs.npz"
    fixtures.write_case_json(case, fixture, case_json)
    fixtures.write_inputs_npz(fixture, inputs)
    return case_json, inputs


def test_whitebox_cli_guard_branches(tmp_path):
    """C1–C7/C9/C11：host driver CLI 的参数/IO/候选加载/拒绝/通用异常出口（前两个不需设备）。"""
    cli = _wb_cli_module()
    plain_case = {"case_id": "assoc_wb_cli_plain", "group": "白盒", "required": False,
                  "op_kind": "sum", "shape": [4, 2], "axis": 0, "reverse": False,
                  "dtype": "float32", "seed": 20261026, "mode": ["eager"],
                  "expected": "match_cpu"}

    def _call(case_json, inputs, outdir, mode="eager"):
        return cli.main(["--case-json", str(case_json), "--inputs", str(inputs),
                         "--mode", mode, "--outdir", str(outdir)])

    with pytest.raises(SystemExit) as excinfo:
        cli.main([])
    assert excinfo.value.code == 2

    outdir = tmp_path / "out-missing"
    assert _call(tmp_path / "nope.json", tmp_path / "nope.npz", outdir) == 2

    case_json, inputs = _wb_write_cli_case(plain_case, tmp_path / "plain")
    assert _call(case_json, tmp_path / "missing.npz", outdir) == 2

    bad = tmp_path / "bad"
    bad.mkdir()
    (bad / "case.json").write_text("{not json", encoding="utf-8")
    assert _call(bad / "case.json", inputs, outdir) == 2
    (bad / "case.json").write_text(json.dumps({"case_id": "no_resolved"}), encoding="utf-8")
    assert _call(bad / "case.json", inputs, outdir) == 2
    (bad / "inputs.npz").write_bytes(b"not a npz")
    assert _call(case_json, bad / "inputs.npz", outdir) == 2

    weird = tmp_path / "weird"
    weird.mkdir()
    np.savez(weird / "inputs.npz", elems=np.zeros((4, 2), np.float32))
    (weird / "case.json").write_text(json.dumps({
        "case_id": "assoc_wb_cli_weird_tree",
        "_resolved": {"combine_name": "sum", "combine_id": 1, "axis": 0, "reverse": False,
                      "elems_tree": {"kind": "weird"}, "output_paths": ["out"]},
    }), encoding="utf-8")
    assert _call(weird / "case.json", weird / "inputs.npz", outdir) == 2

    import unittest.mock as mock  # noqa: PLC0415
    with mock.patch.object(cli, "CANDIDATE_PATH", tmp_path / "missing_candidate.py"):
        assert _call(case_json, inputs, outdir) == 1

    # C9：真实候选在 device 前拒绝 float16 → rc=3 + meta.status=reject + 无 outputs.npz
    reject_case = dict(plain_case, case_id="assoc_wb_cli_reject_f16", dtype="float16",
                       expected="reject", expect_reject_code="dtype_mismatch")
    reject_case_json, reject_inputs = _wb_write_cli_case(reject_case, tmp_path / "f16")
    reject_out = tmp_path / "out-reject"
    assert _call(reject_case_json, reject_inputs, reject_out) == 3
    meta = json.loads((reject_out / "meta.json").read_text(encoding="utf-8"))
    assert meta["status"] == "reject" and meta["reject_code"] == "dtype_mismatch"
    assert meta["dispatch"]["path"] == "reject" and meta["dispatch"]["abi_calls"] == 0
    assert not (reject_out / "outputs.npz").exists(), "拒绝路径不得写 outputs.npz"

    # C11：未注册 combine 的普通异常 → rc=1（不得混成 rc=3 拒绝）
    unregistered = lambda a, b: a + b  # noqa: E731 —— 名称不在注册表，只测异常出口
    with mock.patch.dict(cli._COMBINE_FUNCS, {"sum": unregistered}):
        assert _call(case_json, inputs, tmp_path / "out-error") == 1


def test_whitebox_lib_missing_guard(tmp_path):
    """P1：``_load_lib`` 在共享库缺失时明确失败（不得静默回退/伪造指纹）。"""
    cand = _candidate()
    saved_path = cand.LIB_PATH
    saved_lib = cand._LIB
    try:
        cand.LIB_PATH = tmp_path / "libmissing.so"
        cand._LIB = None
        with pytest.raises(RuntimeError, match="缺少 Python 接口共享库"):
            cand._load_lib()
    finally:
        cand.LIB_PATH = saved_path
        cand._LIB = saved_lib


def test_whitebox_cli_tree_from_spec():
    """C6/C13：host driver CLI 的 ``_tree_from_spec`` 的 leaf/dict/tuple/list 分支（纯 CPU）。

    用例格式（``fixtures.parse_elems_spec``）只表达 ``dict`` + 叶子；CLI 本身支持
    tuple/list 嵌套（cli L105-116）。这里直接调用同一函数，锁定结构重建与未知 kind
    的拒绝出口，不伪造设备证据。
    """
    cli = _wb_cli_module()
    arrays = {
        "elems/a": np.arange(4, dtype=np.float32),
        "elems/b/0": np.arange(2, dtype=np.float32),
        "elems/b/1": np.arange(3, dtype=np.float32),
    }
    spec = {
        "kind": "dict",
        "keys": ["a", "b"],
        "children": {
            "a": {"kind": "leaf", "path": "elems/a"},
            "b": {"kind": "tuple", "children": [
                {"kind": "leaf", "path": "elems/b/0"},
                {"kind": "list", "children": [{"kind": "leaf", "path": "elems/b/1"}]},
            ]},
        },
    }
    converted: list[str] = []

    def _convert(value):
        converted.append(str(np.asarray(value).shape))
        return np.array(value, copy=True)

    tree = cli._tree_from_spec(spec, arrays, _convert)
    assert isinstance(tree, dict) and list(tree) == ["a", "b"], tree
    assert isinstance(tree["b"], tuple) and isinstance(tree["b"][1], list), tree
    assert np.array_equal(tree["a"], arrays["elems/a"])
    assert np.array_equal(tree["b"][0], arrays["elems/b/0"])
    assert np.array_equal(tree["b"][1][0], arrays["elems/b/1"])
    assert converted == ["(4,)", "(2,)", "(3,)"], converted
    assert tree["a"] is not arrays["elems/a"], "convert 必须对每个叶子各调用一次（copy 语义）"

    with pytest.raises(ValueError, match="未知树节点类型"):
        cli._tree_from_spec({"kind": "weird"}, arrays, _convert)


_WB_NINE_LEAF_SHAPE = (3, 4)


def _wb_nine_leaf_elems():
    """9 个同 shape 的 float32 叶子：只用于叶子数边界，不参与数值比对。"""
    return {f"leaf{index}": np.zeros(_WB_NINE_LEAF_SHAPE, np.float32) for index in range(9)}


def _wb_call_kinds(calls):
    """按 ABI 入口聚合调用次数，便于在失败信息里给出可核对的清单。"""
    kinds = {}
    for call in calls:
        kind = call[0] if isinstance(call, tuple) else call
        kinds[kind] = kinds.get(kind, 0) + 1
    return kinds


def _wb_run_leaves(calls):
    """取出所有 ``run`` 调用的 ``n_leaves`` 实参（证明 per_leaf/whole_tree 的语义差异）。"""
    return [call[1] for call in calls if isinstance(call, tuple) and call[0] == "run"]


class _WbRecordingLib:
    """只记录 ABI 调用序列的假 ``.so``（不碰设备、不计算），用于锁定 device 前后的控制流。

    ``malloc`` 返回非零假地址，让 ``_run_so`` 在需要时能走到 ``_run_once``（per_leaf
    正向用例仍要真实经过 malloc/h2d/run/d2h）；对 D-01 的拒绝路径则用于证明**没有任何**
    ABI 调用。``run`` 额外记录 ``n_leaves``：per_leaf 恒为 1，whole_tree 为整树叶子数
    （``docs/…/design.md`` §3.2.2、``pyapi.asc`` L21-22）。
    """

    def __init__(self):
        self.calls: list = []
        self._next_ptr = 0x10000

    def jax_lax_associative_scan_acl_init(self):
        self.calls.append("acl_init")
        return 0

    def jax_lax_associative_scan_malloc(self, ptr_ref, nbytes):
        self.calls.append(("malloc", int(nbytes)))
        ptr_ref._obj.value = self._next_ptr
        self._next_ptr += 0x1000
        return 0

    def jax_lax_associative_scan_h2d(self, dev_ptr, host_ptr, nbytes):
        self.calls.append(("h2d", int(nbytes)))
        return 0

    def jax_lax_associative_scan_d2h(self, host_ptr, dev_ptr, nbytes):
        self.calls.append(("d2h", int(nbytes)))
        return 0

    def jax_lax_associative_scan_free(self, dev_ptr):
        self.calls.append("free")
        return 0

    def jax_lax_associative_scan_run(self, args_ref):
        # 记录本次 launch 的 n_leaves：per_leaf 恒为 1，whole_tree 为整树叶子数。
        self.calls.append(("run", int(args_ref._obj.n_leaves)))
        return 0


def test_whitebox_whole_tree_nine_leaves_reject():
    """P21/H4/M-02（仅 whole_tree 侧）：整树入口 9 叶子必须在 device 前稳定拒绝。

    ``ASSOC_MAX_LEAVES=8`` 是**单次 launch** 的 ABI 上限（``pyapi.asc`` L83-87）；
    ``associative_scan_compiled`` 走 ``whole_tree``，一次 launch 带全部叶子（``pyapi.asc``
    L21-22、``design.md`` §3.2.2），9 叶子因此真实违反该上限。D-01（M-02）已在
    ``_analyze(whole_tree=True)`` 加同口径前置检查：必须在进入 ``_run_so`` / 任何 ABI
    调用之前拒绝，不得再落到 ``_run_once`` 的定长 ``MAX_LEAVES`` ctypes 数组越界。

    期望：``invalid_argument`` + ``path=reject`` + ``abi_calls=0`` + 零 ABI 调用。
    本用例是 D-01 的回归锁定：若再次失败，即说明拒绝时机/稳定码发生回归，不能把
    底层 ``IndexError`` 当预期通过。

    eager 入口不在此列：``associative_scan`` 走 ``per_leaf``，每次 launch 只带 1 个叶子，
    不受该上限约束，其 9 叶子行为由 ``test_whitebox_per_leaf_nine_leaves_ok`` 正向锁定。
    """
    cand = _candidate()
    fn = fixtures.combine_sum
    elems = _wb_nine_leaf_elems()
    fake_lib = _WbRecordingLib()
    saved_load_lib = cand._load_lib
    cand._load_lib = lambda: fake_lib
    cand._set_dispatch()
    try:
        try:
            result = cand.associative_scan_compiled(fn, elems, axis=0)
        except cand._AssocReject as exc:
            reject = exc
        except BaseException as exc:  # noqa: BLE001 —— 回归失败时的真实异常要保留为证据
            raise AssertionError(
                f"whole_tree 9 叶子未在 device 前拒绝：ABI 记录共 {len(fake_lib.calls)} 项 "
                f"{_wb_call_kinds(fake_lib.calls)}；其中 run={_wb_run_leaves(fake_lib.calls)}；"
                f"候选真实异常={type(exc).__name__}: {exc}") from exc
        else:
            pytest.fail(
                f"whole_tree 9 叶子未拒绝且成功返回：{type(result).__name__}；"
                f"ABI 记录={_wb_call_kinds(fake_lib.calls)}")
    finally:
        cand._load_lib = saved_load_lib
    assert fake_lib.calls == [], (
        f"拒绝必须发生在任何 ABI 调用之前（path=reject、abi_calls=0、不写输出），"
        f"实际调用={fake_lib.calls[:6]}")
    assert reject.code == "invalid_argument", reject.code
    dispatch = cand._last_dispatch
    assert dispatch["path"] == "reject" and dispatch["exec_mode"] == "reject", dispatch
    assert dispatch["abi_calls"] == 0 and dispatch["reason"] == "invalid_argument", dispatch


def test_whitebox_per_leaf_nine_leaves_ok():
    """P13/P20：per_leaf 入口每次 launch 只带 1 个叶子，9 叶子是合法输入（正向）。

    ``associative_scan``（eager）逐叶子调用 ``_run_once``，每次 ``n_leaves == 1``
    （``design.md`` §3.2.2、``pyapi.asc`` L20-21）；ABI 的 ``n_leaves ∈ [1, 8]``
    只约束单次 launch，故 9 叶子**不应**被拒绝。期望：9 次 ``run`` 且每次
    ``n_leaves == 1``、``abi_calls == 9``、``exec_mode == "per_leaf"``、返回同构 dict。

    本用例把「9 叶子在 eager 下成功」固定成可执行语义，避免把 eager 的成功返回
    误记成 M-02 的 ``IndexError``（V3-06/V3-07）。
    """
    cand = _candidate()
    fn = fixtures.combine_sum
    elems = _wb_nine_leaf_elems()
    fake_lib = _WbRecordingLib()
    saved_load_lib = cand._load_lib
    cand._load_lib = lambda: fake_lib
    cand._set_dispatch()
    try:
        result = cand.associative_scan(fn, elems, axis=0)
        dispatch = dict(cand._last_dispatch)
    finally:
        cand._load_lib = saved_load_lib
    assert isinstance(result, dict), type(result).__name__
    assert list(result) == list(elems), list(result)
    for key, value in result.items():
        assert value.shape == _WB_NINE_LEAF_SHAPE, (key, value.shape)
        assert value.dtype == np.float32, (key, value.dtype)
    leaves = _wb_run_leaves(fake_lib.calls)
    assert len(leaves) == 9, (
        f"per_leaf 应有 9 次 run，实际 {len(leaves)}：{_wb_call_kinds(fake_lib.calls)}")
    assert set(leaves) == {1}, (
        f"per_leaf 每次 launch 只带 1 个叶子，实际 n_leaves={sorted(set(leaves))}")
    assert dispatch["path"] == "so" and dispatch["exec_mode"] == "per_leaf", dispatch
    assert dispatch["abi_calls"] == 9, dispatch
