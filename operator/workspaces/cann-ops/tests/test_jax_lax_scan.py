# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""jax_lax_scan · pytest 交付件（真机运行；候选 = Python 接口，参考 = 同版本 JAX CPU 子进程）。

结构与判据（设计 §2.2.2 / §2.3 / §5.3）：

* 用例 = 冻结 ``cases/jax_lax_scan/cases.json``（7 条）+ 基准复用
  ``benchmark.json``（2 条，仅 jit）+ 补充 ``tests/jax_lax_scan_supplement_cases.json``（6 条）
  + 白盒 ``tests/jax_lax_scan_whitebox_cases.json``（4 条）
  = **19 条 / 33 次逻辑 mode 执行**（§7）。
* 白盒分支用例（源码分支覆盖）分两类，都进本文件同一入口：
  ① 设备数据驱动用例 = ``jax_lax_scan_whitebox_cases.json`` 的 4 条（宽度对齐边界 + body_id=3 内核分支）；
  ② CPU 可跑的分支单测（``test_whitebox_*``）= 路由/长度校验/树与 dtype 辅助/入口守卫/host driver
  参数分支/非 AD dispatch 断言（含 ``body_id``）。
* 候选：``ops/jax_lax_scan/python/jax_lax_scan.py`` 的 ``scan`` / ``scan_compiled``；
  ``eager`` / ``jit`` 的可执行定义与 ``abi_calls`` 断言见 §1.3.4 / §5.3。
* 参考：``golden_cli.py`` 独立 CPU 子进程（``JAX_PLATFORMS=cpu``），主进程只跑候选（NPU）。
* 判定：逐叶子先设备守卫，再 dtype/shape，浮点走 ``criteria/check.py``，int32 精确，
  非有限位逐元素精确；结构/shape 无容差。
* AD：``scan_jvp_vjp`` 使用测试侧 ``jax.Array``（仅测试支持），3 次同码
  ``UNSUPPORTED:autodiff_unsupported`` 才允许分类为 ``UNSUPPORTED``，不记 PASS；每次异常都要求
  ``_last_dispatch`` 给出 ``path=trace``/``abi_calls=0``/``backend!=cpu``，3 次同码后再要求
  ``reason=so_no_autodiff`` 且异常文本含后端原文，否则判 ``INVALID``（设计 §1.3.4/§5.3）。
* 拒拒绝：``carry_shape_mismatch`` + ``abi_calls==0`` + 不得写 ``outputs.npz``。

报告：``${OPX_OUT:-/out}/scan_correctness.json`` + ``scan_correctness.md``，每条用例另打印一行
``SCAN_CASE {...}``。pytest 的 junit 日志由 ``test.sh`` 第二段保存。
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
for extra in (WS, WS / "ops" / "jax_lax_scan" / "scripts"):
    if str(extra) not in sys.path:
        sys.path.insert(0, str(extra))

np = pytest.importorskip("numpy")

import fixtures  # noqa: E402
import scan_engine  # noqa: E402

CANDIDATE_PATH = WS / "ops" / "jax_lax_scan" / "python" / "jax_lax_scan.py"
GOLDEN_CLI = WS / "ops" / "jax_lax_scan" / "scripts" / "golden_cli.py"
FROZEN_CASES_PATH = WS / "cases" / "jax_lax_scan" / "cases.json"
BENCH_PATH = WS / "cases" / "jax_lax_scan" / "benchmark.json"
SUPPLEMENT_PATH = WS / "tests" / "jax_lax_scan_supplement_cases.json"
WHITEBOX_PATH = WS / "tests" / "jax_lax_scan_whitebox_cases.json"

EXPECTED_PYTEST_CASES = 19
EXPECTED_PYTEST_MODES = 33
EXPECTED_GATE_CASES = 7
EXPECTED_GATE_MODES = 13

CRITERIA_MOD = scan_engine._load_module(WS / "criteria" / "check.py", "opx_pytest_criteria")


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


def _load_cases() -> tuple[list[dict], dict, dict, dict, dict]:
    frozen = _load_json(FROZEN_CASES_PATH)
    bench = _load_json(BENCH_PATH)
    supplement = _load_json(SUPPLEMENT_PATH)
    whitebox = _load_json(WHITEBOX_PATH)
    cases = [_normalise(c, "frozen") for c in frozen.get("cases") or []]
    cases += [_normalise(c, "benchmark") for c in bench.get("cases") or []]
    cases += [_normalise(c, "supplement") for c in supplement.get("cases") or []]
    cases += [_normalise(c, "whitebox") for c in whitebox.get("cases") or []]
    return cases, frozen, bench, supplement, whitebox


CASES, FROZEN_SUITE, BENCH_SUITE, SUPPLEMENT_SUITE, WHITEBOX_SUITE = _load_cases()
# PJRT 与 host-driver ACL 都会管理进程级运行时。AD 用例先跑，避免在
# 同一 pytest 进程里先由 host driver aclInit 后再初始化 PJRT 而冲突。
CASES.sort(key=lambda item: 0 if item.get("case_id") == "scan_jvp_vjp" else 1)
MODE_COUNT = sum(len(c["mode"]) for c in CASES)
GATE_MODE_COUNT = sum(len(c.get("mode") or ["jit"]) for c in (FROZEN_SUITE.get("cases") or []))

# 发现数不在 import 期硬失败：新增白盒用例而忘记同步计数时，应在
# test_selfcheck_discovery_counts 得到“请同步 EXPECTED_PYTEST_CASES / EXPECTED_PYTEST_MODES”的明确提示，
# 而不是 import 期无提示崩溃（见 1-测试工程说明 §12 白盒接入顺序）。

_CANDIDATE = None


def _candidate():
    """候选 Python 接口（节点 2 交付）。缺失时直接失败，不能 skip 成“通过”。"""
    global _CANDIDATE
    if _CANDIDATE is not None:
        return _CANDIDATE
    if not CANDIDATE_PATH.is_file():
        pytest.fail(f"候选 Python 接口不存在：{CANDIDATE_PATH}（节点 2 必须先交付）")
    spec = importlib.util.spec_from_file_location("jax_lax_scan_candidate", CANDIDATE_PATH)
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
# 设备守卫（逐叶子；参考实现来自设计 §2.3）
# ---------------------------------------------------------------------------


def _leaf_platforms(leaf):
    """返回 (platforms: set[str], kind: str, devices: list)。无法判定时 platforms 为空。"""
    if callable(getattr(leaf, "devices", None)):
        devs = list(leaf.devices())
        return {str(getattr(d, "platform", "?")) for d in devs}, "jax", devs
    device = getattr(leaf, "device", None)
    if getattr(device, "type", None) is not None:
        return {str(device.type)}, "torch", [device]
    if type(leaf).__module__.split(".")[0] == "numpy":
        return set(), "numpy", []
    return set(), type(leaf).__name__, []


def assert_all_leaves_on_accelerator(outputs, case_id, dispatch=None):
    leaves = fixtures.flatten_with_paths(outputs)
    assert leaves, f"{case_id}: 输出为空，无法证明设备"
    recorded = []
    for path, leaf in leaves:
        platforms, kind, devs = _leaf_platforms(leaf)
        if kind == "numpy":
            # JAX 镜像里的 host-driver 契约是 NPU `.so` 执行后 D2H 回读；
            # 返回 numpy 是 I/O 形式，不代表候选在 CPU 计算。只有 ABI 证据
            # 完整时才放行，避免普通 host fallback 冒充。
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
        assert all(p != "cpu" for p in platforms), f"{case_id}:{path} 位于 CPU：{sorted(platforms)}"
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
    root = Path(os.environ.get("OPX_CASE_WORKDIR", "/tmp/opx-pytest-cases"))
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


def _make_torch_tree(fixture: dict):
    try:
        import torch  # noqa: PLC0415
        import torch_npu  # noqa: F401,PLC0415
    except Exception as exc:  # noqa: BLE001
        pytest.fail(f"torch/torch_npu 不可用，无法跑交付接口：{type(exc).__name__}: {exc}")

    def convert(array):
        return torch.from_numpy(np.array(array, copy=True)).npu()

    init = fixtures.tree_from_spec(fixture["carry_tree"], fixture["arrays"], convert=convert)
    xs = None if fixture["xs_tree"] is None else fixtures.tree_from_spec(
        fixture["xs_tree"], fixture["arrays"], convert=convert)
    return init, xs


def _make_numpy_tree(fixture: dict):
    """JAX 专用镜像的 host-driver 输入；候选仍在 NPU `.so` 上执行。"""
    def convert(array):
        value = np.array(array, copy=True)
        return value if value.ndim == 0 else np.ascontiguousarray(value)

    init = fixtures.tree_from_spec(fixture["carry_tree"], fixture["arrays"], convert=convert)
    xs = None if fixture["xs_tree"] is None else fixtures.tree_from_spec(
        fixture["xs_tree"], fixture["arrays"], convert=convert)
    return init, xs


def _make_jax_tree(fixture: dict):
    # 只在 AD 用例里 import jax；镜像若需要 JAX_PLATFORMS=npu，在 import 前设好（设计 §2.2.2）。
    if "jax" not in sys.modules:
        # 设计 §2.2.2：AD 用例必须显式让候选 backend 成为 npu；这里在 import 前设置。
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
    npu_devices = [d for d in devices if str(getattr(d, "platform", "?")) != "cpu"]
    if not npu_devices:
        pytest.fail(f"候选侧没有非 CPU JAX 设备（{backend}/{devices}）")
    dev = npu_devices[0]

    def convert(array):
        return jax.device_put(jnp.asarray(np.array(array, copy=True)), dev)

    init = fixtures.tree_from_spec(fixture["carry_tree"], fixture["arrays"], convert=convert)
    xs = None if fixture["xs_tree"] is None else fixtures.tree_from_spec(
        fixture["xs_tree"], fixture["arrays"], convert=convert)
    return init, xs, jax


def _dispatch_of(module) -> dict:
    dispatch = getattr(module, "_last_dispatch", None)
    if not isinstance(dispatch, dict):
        return {}
    return {key: dispatch.get(key) for key in ("op", "mode", "path", "exec_mode", "body_id",
                                               "abi_calls", "family", "reason", "backend")}


def _invoke_host_driver(module, fixture: dict, mode: str):
    # lax.scan 的目标镜像是 JAX NPU 镜像，不强制捆绑 torch_npu。
    # numpy 只负责 host I/O；module.scan/scan_compiled 会调 libjax_lax_scan_host.so，
    # 真实数值计算由 NPU kernel 完成，后续设备守卫仍校验 path=so/backend=npu。
    init, xs = _make_numpy_tree(fixture)
    body = fixtures.get_body(fixture["body_id"])
    fn = module.scan if mode == "eager" else module.scan_compiled
    result = fn(body, init, xs, length=int(fixture["length"]), reverse=bool(fixture["reverse"]))
    if not isinstance(result, tuple) or len(result) != 2:
        pytest.fail(f"候选返回结构不是 (final_carry, ys)：{type(result).__name__}")
    final_carry, ys = result
    outputs = {"final_carry": final_carry, "ys": ys}
    return outputs, _dispatch_of(module)


def _invoke_jax_ad(module, fixture: dict, mode: str):
    init, xs, jax = _make_jax_tree(fixture)
    body = fixtures.get_body(fixture["body_id"])

    def f_ad(x):
        result = module.scan(body, init, x, length=int(fixture["length"]),
                             reverse=bool(fixture["reverse"]))
        return result[0]

    try:
        if mode == "jvp":
            tangent_seed = jax.tree_util.tree_map(jax.numpy.ones_like, xs)
            primal, tangent = jax.jvp(f_ad, (xs,), (tangent_seed,))
            outputs = {"jvp_primal": primal, "jvp_tangent": tangent}
        else:
            primal, vjp_fn = jax.vjp(f_ad, xs)
            cotangent_seed = jax.tree_util.tree_map(jax.numpy.ones_like, primal)
            cotangent = vjp_fn(cotangent_seed)[0]
            outputs = {"vjp_primal": primal, "vjp_cotangent": cotangent}
    except Exception:
        # module.scan 只知道 eager/jit；外层才知道这次 trace 是 jvp 还是
        # vjp。在异常离开前把真实 AD mode 回填到 dispatch，供 3 次
        # 稳定 unsupported 分类校验，不改原始后端异常。
        dispatch = getattr(module, "_last_dispatch", None)
        if isinstance(dispatch, dict):
            dispatch["mode"] = mode
        raise
    outputs = jax.tree_util.tree_map(lambda leaf: leaf.block_until_ready(), outputs)
    dispatch = getattr(module, "_last_dispatch", None)
    if isinstance(dispatch, dict):
        dispatch["mode"] = mode
    return outputs, _dispatch_of(module)


# ---------------------------------------------------------------------------
# 单 mode / 单 case 判定
# ---------------------------------------------------------------------------


def _golden(case_json: Path, inputs: Path, mode: str, cdir: Path) -> dict:
    return scan_engine._run_golden(WS, case_json, inputs, mode, cdir)


def _judge(actual: dict, golden_arrays: dict, expected_keys: list[str], leaf_map: dict,
           fixture: dict, mode: str) -> tuple[list[dict], list[str]]:
    # actual 可能是 {"final_carry": (int32 counter, f32 acc), "ys": ...}；
    # 必须先按树展开成叶子路径，不能对 tuple 直接调 _to_numpy（自检 D 已暴露同类问题）。
    actual_np = {path: _to_numpy(leaf) for path, leaf in fixtures.flatten_with_paths(actual)}
    return scan_engine._judge_arrays(actual_np, golden_arrays, expected_keys, CRITERIA_MOD,
                                     leaf_map, fixture, mode)


def _run_reject_mode(module, case: dict, fixture: dict, case_json: Path, inputs: Path,
                     mode: str, cdir: Path) -> dict:
    rec = {"mode": mode, "status": "NOT_RUN", "leaves": [], "dispatch": {}, "attempts": []}
    expected_code = case.get("expect_reject_code", scan_engine.REJECT_CODE)
    init, xs = _make_numpy_tree(fixture)
    body = fixtures.get_body(fixture["body_id"])
    try:
        module.scan_compiled(body, init, xs, length=int(fixture["length"]),
                             reverse=bool(fixture["reverse"]))
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
                   detail=f"拒绝路径标记不对：path={dispatch.get('path')!r} exec_mode={dispatch.get('exec_mode')!r}")
        return rec
    if dispatch.get("abi_calls") != 0:
        rec.update(status="FAIL", detail=f"拒绝路径 abi_calls={dispatch.get('abi_calls')}（必须为 0）")
        return rec
    rec["attempts"] = [{"attempt": 1, "rc": "python-exception", "abi_calls": dispatch.get("abi_calls")}]
    golden = _golden(case_json, inputs, "reject", cdir)
    rec["golden"] = {"ok": golden.get("ok"), "rc": golden.get("rc")}
    if not golden["ok"]:
        rec.update(status="ERROR", detail=f"golden 未确认该用例应被拒绝：{golden.get('detail')}")
        return rec
    meta = golden.get("meta") or {}
    rec["golden"].update({"backend": meta.get("backend"), "jax": meta.get("jax"),
                          "jaxlib": meta.get("jaxlib"), "reject_code": meta.get("reject_code")})
    rec.update(status="PASS", detail=f"按预期拒绝：{expected_code}；abi_calls=0")
    return rec


def _numeric_dispatch_problems(dispatch: dict, fixture: dict, mode: str) -> list[str]:
    """非 AD 数值 mode 的 dispatch 断言（设计 §1.3.4 / §5.3）。

    ``body_id`` 必须等于本条用例 fixture 的 body：K14（``SCAN_BODY_QUAD``，body 3）等内核
    分支的覆盖结论以该字段为硬证据；缺该断言会让「设计断言 body_id=3、实现未校验」脱节
    （``3-验收报告.md`` O-1）。
    """
    expected_exec = "stepwise" if mode == "eager" else "compiled_whole_loop"
    expected_calls = int(fixture["length"]) if mode == "eager" else 1
    problems = []
    if dispatch.get("path") != "so":
        problems.append(f"path={dispatch.get('path')!r}（期望 so）")
    if dispatch.get("exec_mode") != expected_exec:
        problems.append(f"exec_mode={dispatch.get('exec_mode')!r}（期望 {expected_exec}）")
    if dispatch.get("abi_calls") != expected_calls:
        problems.append(f"abi_calls={dispatch.get('abi_calls')}（期望 {expected_calls}）")
    if dispatch.get("body_id") != fixture["body_id"]:
        problems.append(f"body_id={dispatch.get('body_id')!r}（期望 {fixture['body_id']}）")
    return problems


def _run_numeric_mode(module, case: dict, fixture: dict, case_json: Path, inputs: Path,
                      mode: str, cdir: Path) -> dict:
    rec = {"mode": mode, "status": "NOT_RUN", "leaves": [], "dispatch": {}, "attempts": []}
    expected = case.get("expected", "match_cpu")
    allow_unsupported = expected == "match_cpu_or_explicit_unsupported" and mode in scan_engine.AD_MODES
    max_attempts = 3 if allow_unsupported else 1
    unsupported_texts: list[str] = []
    outputs = None
    dispatch: dict = {}
    last_exc: Exception | None = None

    for attempt in range(1, max_attempts + 1):
        rec["attempts"].append({"attempt": attempt})
        try:
            if mode in scan_engine.AD_MODES:
                outputs, dispatch = _invoke_jax_ad(module, fixture, mode)
            else:
                outputs, dispatch = _invoke_host_driver(module, fixture, mode)
        except Exception as exc:  # noqa: BLE001
            last_exc = exc
            text = f"{type(exc).__name__}: {exc}"
            # 异常路径也必须从候选取 _last_dispatch：只打印 marker、没有 trace/abi_calls/backend 证据
            # 的“unsupported”不算显式分类（设计 §1.3.4/§5.3）。
            dispatch = _dispatch_of(module)
            explicit_unsupported = (
                allow_unsupported
                and any(token in text.lower() for token in
                        ("unimplemented", "unsupported", "not implemented", "not supported"))
                and dispatch.get("path") == "trace"
                and dispatch.get("reason") == "so_no_autodiff"
                and dispatch.get("abi_calls") == 0
                and dispatch.get("backend") not in (None, "", "cpu")
            )
            rec["attempts"][-1].update(rc="exception", error=text[:300],
                                       unsupported=(scan_engine.UNSUPPORTED_CODE in text
                                                    or explicit_unsupported),
                                       dispatch=dispatch)
            if allow_unsupported and (scan_engine.UNSUPPORTED_CODE in text or explicit_unsupported):
                problems = scan_engine.unsupported_attempt_problems(
                    dispatch, text, require_reason=False, require_op=True, expected_mode=mode)
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
            problems = scan_engine.unsupported_attempt_problems(
                dispatch, unsupported_texts[-1], require_op=True, expected_mode=mode)
            if problems:
                rec.update(status="INVALID",
                           detail="3 次同码但 unsupported 分类缺少硬证据，不能记 UNSUPPORTED："
                                  + "；".join(problems))
                rec["dispatch"] = dispatch
                return rec
            rec.update(status="UNSUPPORTED",
                       detail="3 次均显式 UNSUPPORTED:autodiff_unsupported（path=trace/abi_calls=0/"
                              "reason=so_no_autodiff/backend 非 CPU，附后端原文）")
            rec["backend_errors"] = [text[:400] for text in unsupported_texts]
            rec["dispatch"] = dispatch
            return rec
        rec.update(status="ERROR",
                   detail=f"候选没有成功输出：{type(last_exc).__name__ if last_exc else 'unknown'}")
        return rec

    rec["dispatch"] = dispatch
    try:
        device_leaves = assert_all_leaves_on_accelerator(outputs, case["case_id"], dispatch)
    except AssertionError as exc:
        rec.update(status="FAIL", detail=f"设备守卫失败：{exc}")
        return rec
    rec["device_leaves"] = device_leaves

    # dispatch 断言（设计 §1.3.4 / §5.3）
    if mode in scan_engine.AD_MODES:
        if dispatch.get("path") != "trace":
            rec.update(status="FAIL", detail=f"AD 必须走 trace，收到 path={dispatch.get('path')!r}")
            return rec
        if dispatch.get("reason") != "so_no_autodiff":
            rec.update(status="FAIL",
                       detail=f"AD 的 reason 必须是 so_no_autodiff，收到 {dispatch.get('reason')!r}")
            return rec
        if dispatch.get("abi_calls") != 0:
            rec.update(status="FAIL",
                       detail=f"AD 走 trace 时 .so 调用次数必须为 0，收到 abi_calls={dispatch.get('abi_calls')}")
            return rec
    else:
        problems = _numeric_dispatch_problems(dispatch, fixture, mode)
        if problems:
            rec.update(status="FAIL", detail="dispatch 断言失败：" + "；".join(problems))
            return rec

    golden = _golden(case_json, inputs, mode, cdir)
    rec["golden"] = {"ok": golden.get("ok"), "rc": golden.get("rc")}
    if not golden["ok"]:
        rec.update(status="ERROR", detail=f"golden 失败：{golden.get('detail')}")
        return rec
    golden_meta = golden.get("meta") or {}
    rec["golden"].update({"backend": golden_meta.get("backend"),
                          "jax": golden_meta.get("jax"),
                          "jaxlib": golden_meta.get("jaxlib"),
                          "pid": golden_meta.get("pid")})
    expected_keys = scan_engine._expected_golden_keys(mode, fixture)
    leaf_map = {item["path"]: item for item in device_leaves}
    leaves, problems = _judge(outputs, golden["arrays"], expected_keys, leaf_map, fixture, mode)
    rec["leaves"] = leaves
    rec["matched_ratio"] = min((leaf.get("matched_ratio", 1.0) or 0.0) for leaf in leaves) if leaves else 0.0
    rec["max_abs_error"] = max((leaf.get("max_abs_error") or 0.0) for leaf in leaves) if leaves else None
    rec["nonfinite"] = {
        "count": sum(int((leaf.get("nonfinite") or {}).get("count", 0)) for leaf in leaves),
        "match": all(bool((leaf.get("nonfinite") or {}).get("match", True)) for leaf in leaves),
    }
    if problems:
        rec.update(status="FAIL", detail="；".join(problems))
    else:
        rec.update(status="PASS", detail=f"{mode} 通过（{len(leaves)} 个叶子）")
        # 供 eager × jit 一致性互比（不进 JSON 报告）；同样按叶子路径展开。
        rec["_arrays"] = {path: _to_numpy(leaf) for path, leaf in fixtures.flatten_with_paths(outputs)}
    return rec


def _run_case(case: dict) -> dict:
    module = _candidate()
    fixture, case_json, inputs = _prepare_case(case)
    cdir = case_json.parent
    mode_records = []
    for mode in case["mode"]:
        if case.get("expected") == "reject":
            mode_records.append(_run_reject_mode(module, case, fixture, case_json, inputs, mode, cdir))
        else:
            mode_records.append(_run_numeric_mode(module, case, fixture, case_json, inputs, mode, cdir))

    statuses = [m["status"] for m in mode_records]
    overall, detail = "NOT_RUN", ""
    if all(status == "PASS" for status in statuses):
        overall, detail = "PASS", "全部 mode 通过"
    elif statuses and all(status == "UNSUPPORTED" for status in statuses):
        overall = "UNSUPPORTED"
        detail = "全部 mode 为显式 UNSUPPORTED（不是数值 PASS）"
    elif "UNSUPPORTED" in statuses:
        overall = "FAIL"
        detail = "mode 分类混合（UNSUPPORTED 与其它状态并存）"
    else:
        for label in ("ERROR", "INVALID", "FAIL", "NOT_RUN"):
            if label in statuses:
                overall = label
                break
        detail = "；".join(f"{m['mode']}:{m['status']}:{(m.get('detail') or '')[:160]}"
                           for m in mode_records if m["status"] != "PASS")

    consistency = _consistency_problems(case, mode_records)
    if consistency and overall == "PASS":
        overall, detail = "FAIL", "；".join(consistency)

    for rec in mode_records:
        rec.pop("_arrays", None)
    golden_info = next((m.get("golden") for m in mode_records if m.get("golden")), {})
    return {
        "case_id": case["case_id"],
        "group": case.get("group"),
        "origin": case.get("_origin"),
        "required": bool(case.get("required", True)),
        "expected": case.get("expected", "match_cpu"),
        "shape": case.get("shape") or [fixture["length"], fixture["width"]],
        "length": fixture["length"],
        "width": fixture["width"],
        "dtype": case.get("dtype", fixture["dtype"]),
        "seed": case.get("seed"),
        "note": case.get("note"),
        "status": overall,
        "detail": detail,
        "modes": mode_records,
        "dispatch": mode_records[0].get("dispatch") if mode_records else {},
        "golden": {key: golden_info.get(key) for key in ("backend", "jax", "jaxlib", "pid")},
    }


def _consistency_problems(case: dict, mode_records: list[dict]) -> list[str]:
    arrays = {m["mode"]: m.get("_arrays") for m in mode_records if m.get("_arrays") is not None}
    if "eager" not in arrays or "jit" not in arrays:
        return []
    eager, jit = arrays["eager"], arrays["jit"]
    if set(eager) != set(jit):
        return [f"eager/jit 输出叶子集合不同：{sorted(eager)} vs {sorted(jit)}"]
    problems = []
    for key in eager:
        a, b = np.asarray(eager[key]), np.asarray(jit[key])
        if a.dtype != b.dtype or a.shape != b.shape:
            problems.append(f"{key}: eager/jit dtype/shape 不同")
            continue
        if a.dtype == np.int32:
            if not np.array_equal(a, b):
                problems.append(f"{key}: eager/jit int32 不一致")
            continue
        verdict = CRITERIA_MOD.check(np.ascontiguousarray(a, dtype=np.float32),
                                     np.ascontiguousarray(b, dtype=np.float32))
        if not verdict["is_pass"]:
            problems.append(f"{key}: eager 与 jit 输出不一致")
    return problems


# ---------------------------------------------------------------------------
# 报告
# ---------------------------------------------------------------------------

_RESULTS: list[dict] = []
#: 补充镜像（有 torch_npu）上的交付接口正向结果；只在 torch 探针实际执行时非空。
_TORCH_RESULTS: list[dict] = []


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def build_report_payload(results: list[dict], *, collection_only: bool = False) -> dict:
    summary = {status: sum(1 for r in results if r["status"] == status)
               for status in scan_engine.CASE_STATUSES}
    summary.update(
        TOTAL=len(results),
        mode_executions=sum(len(r.get("modes") or []) for r in results),
        expected_cases=EXPECTED_PYTEST_CASES,
        expected_mode_executions=EXPECTED_PYTEST_MODES,
        collection_only=collection_only,
    )
    golden = next((r.get("golden") for r in results if r.get("golden", {}).get("backend")), {})
    # tensor_family 必须由本次真实 dispatch 生成，不能硬编码：JAX 主镜像（#145）没有
    # torch_npu，so 用例走 numpy host driver；只有补充镜像（#80）才会出现 torch。
    family_counts: dict[str, int] = {}
    for result in results:
        for mode_rec in result.get("modes") or []:
            family = (mode_rec.get("dispatch") or {}).get("family")
            if family:
                family_counts[str(family)] = family_counts.get(str(family), 0) + 1
    return {
        "schema": "opx-jax-correctness/1",
        "op": "jax_lax_scan",
        "cases_sha256": _sha256_file(FROZEN_CASES_PATH),
        "benchmark_sha256": _sha256_file(BENCH_PATH),
        "supplement_sha256": _sha256_file(SUPPLEMENT_PATH),
        "suite_ref": os.environ.get("OPX_SUITE_REF") or None,
        "candidate": {
            "interface": "ops/jax_lax_scan/python/jax_lax_scan.py::scan",
            "compiled_interface": "ops/jax_lax_scan/python/jax_lax_scan.py::scan_compiled",
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
        "# jax_lax_scan · pytest 正确性报告",
        "",
        f"- 用例集：`cases/jax_lax_scan/cases.json`（sha256 `{payload['cases_sha256'][:16]}…`）"
        f" + benchmark + supplement",
        f"- suite_ref：`{payload['suite_ref']}` ｜ golden：CPU JAX {payload['golden'].get('jax')}",
        f"- 汇总：{payload['summary']}",
        "",
        "| 用例 | 来源 | 期望 | 状态 | mode 状态 | 说明 |",
        "|---|---|---|---|---|---|",
    ]
    for r in payload["cases"]:
        modes = ", ".join(f"{m['mode']}={m['status']}" for m in (r.get("modes") or []))
        lines.append(
            f"| {r['case_id']} | {r.get('origin', '')} | {r.get('expected')} | **{r['status']}** | "
            f"{modes} | {(r.get('detail') or '').replace('|', '/')[:140]} |")
    lines += ["", "> UNSUPPORTED 不是 PASS；判定口径与门禁同为 criteria/check.py。"]
    return "\n".join(lines) + "\n"


def write_report(results: list[dict], out_dir: Path | None = None) -> tuple[Path, Path]:
    out = out_dir or Path(os.environ.get("OPX_OUT", "/out"))
    out.mkdir(parents=True, exist_ok=True)
    payload = build_report_payload(results, collection_only=not results)
    json_path = out / "scan_correctness.json"
    md_path = out / "scan_correctness.md"
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    md_path.write_text(_render_md(payload), encoding="utf-8")
    return json_path, md_path


def build_torch_report_payload(results: list[dict]) -> dict:
    """补充镜像 torch_npu 正向探针的机器可读报告（与 JAX 主门禁报告分开存放）。"""
    summary = {"PASS": 0, "FAIL": 0, "TOTAL": len(results)}
    for result in results:
        status = str(result.get("status") or "FAIL")
        summary[status] = summary.get(status, 0) + 1
    torch = pytest.importorskip("torch")
    torch_npu = pytest.importorskip("torch_npu")
    try:
        device_name = torch_npu.npu.get_device_name(0)
    except Exception:  # noqa: BLE001
        device_name = None
    return {
        "schema": "opx-scan-torch-correctness/1",
        "op": "jax_lax_scan",
        "suite_ref": os.environ.get("OPX_SUITE_REF") or None,
        "candidate": {
            "interface": "ops/jax_lax_scan/python/jax_lax_scan.py::scan / scan_compiled",
            "tensor_family": "torch_npu",
        },
        "environment": {
            "torch": getattr(torch, "__version__", None),
            "torch_npu": getattr(torch_npu, "__version__", None),
            "device_name": device_name,
            "device_count": int(torch_npu.npu.device_count()),
            "image": os.environ.get("OPX_TORCH_IMAGE") or None,
        },
        "cases": results,
        "summary": summary,
    }


def write_torch_report(results: list[dict], out_dir: Path | None = None) -> tuple[Path, Path]:
    out = out_dir or Path(os.environ.get("OPX_TORCH_OUT", "/out/torch"))
    out.mkdir(parents=True, exist_ok=True)
    payload = build_torch_report_payload(results)
    json_path = out / "scan_correctness.json"
    md_path = out / "scan_correctness.md"
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    lines = [
        "# jax_lax_scan · torch_npu 交付接口补充验证",
        "",
        f"- 环境：{payload['environment']}",
        f"- 汇总：{payload['summary']}",
        "",
        "| 用例 | mode | 状态 | abi_calls | max_abs_error |",
        "|---|---|---|---|---|",
    ]
    for result in results:
        dispatch = result.get("dispatch") or {}
        lines.append(
            f"| {result.get('case_id')} | {result.get('mode')} | **{result.get('status')}** | "
            f"{dispatch.get('abi_calls')} | {result.get('max_abs_error')} |")
    lines += ["", "> 本报告只来自补充镜像 #80（torch_npu）；#145 无 torch_npu，不产生该族结果。"]
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return json_path, md_path


@pytest.fixture(scope="session", autouse=True)
def _session_report_writer():
    yield
    if _RESULTS:
        paths = write_report(_RESULTS)
        print(f"\nSCAN_REPORT {paths[0]} {paths[1]}")
    if _TORCH_RESULTS:
        paths = write_torch_report(_TORCH_RESULTS)
        print(f"\nSCAN_TORCH_REPORT {paths[0]} {paths[1]}")


# ---------------------------------------------------------------------------
# 正确性用例（15 条 / 25 次逻辑执行）
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("case", CASES, ids=[case["case_id"] for case in CASES])
def test_scan_case(case):
    rec = _run_case(case)
    _RESULTS.append(rec)
    print("SCAN_CASE " + json.dumps(rec, ensure_ascii=False))
    if rec["status"] == "PASS":
        return
    if rec["status"] == "UNSUPPORTED":
        if case.get("expected") == "match_cpu_or_explicit_unsupported":
            pytest.xfail(f"{case['case_id']}: 显式 UNSUPPORTED（已记录，不记 PASS）")
        pytest.fail(f"{case['case_id']}: 不允许的 UNSUPPORTED")
    pytest.fail(f"{case['case_id']}: {rec['status']} - {rec.get('detail')}")


# JAX 主镜像不捆绑 torch_npu，因此该交付接口探针只在真正安装了
# torch_npu 的补充验收镜像中收集；未安装时不注册此 test，避免把主镜像
# 的正常环境差异写成 pytest skip。补充探针不更改冻结 cases/criteria。
#
# 覆盖（P-10/P-11 的完成判据）：
#   * 5 条冻结数值用例 × eager/jit：path=so、family=torch、backend=npu、abi_calls 匹配；
#   * 逐叶与 NumPy 参考（同一 body 函数、CPU）按 criteria/check.py 同口径比较；
#   * 输入 init/xs 调用前后逐元素不变，输出与输入不别名；
#   * 额外 length=2 用例覆盖“偶数长度 final_carry 是输入别名”的回归点。
TORCH_CASE_IDS = (
    "scan_sum_len8_f32",
    "scan_reverse_len7_f32",
    "scan_pytree_carry",
    "scan_pytree_xs",
    "scan_xs_none_length5",
)
_TORCH_LEN2_CASE = {
    "case_id": "scan_torch_len2_input_immutability",
    "fixture": "B1_add",
    "shape": [2, 16],
    "dtype": "float32",
    "seed": 20261026,
    "mode": ["eager", "jit"],
    "expected": "match_cpu",
}


def _tree_index(tree, index: int):
    if isinstance(tree, dict):
        return {key: _tree_index(value, index) for key, value in tree.items()}
    if isinstance(tree, tuple):
        return tuple(_tree_index(value, index) for value in tree)
    if isinstance(tree, list):
        return [_tree_index(value, index) for value in tree]
    return tree[index]


def _restore_tree_dtypes(value, template):
    """NumPy 0-D int32 + python int 会升成 int64；JAX carry 的 dtype 不变式要求还原。"""
    if isinstance(template, dict):
        return {key: _restore_tree_dtypes(value[key], template[key]) for key in template}
    if isinstance(template, tuple):
        return tuple(_restore_tree_dtypes(item, ref)
                     for item, ref in zip(value, template))
    if isinstance(template, list):
        return [_restore_tree_dtypes(item, ref) for item, ref in zip(value, template)]
    got = np.asarray(value)
    want = np.asarray(template).dtype
    return got.astype(want, copy=False) if got.dtype != want else got


def _numpy_reference(fixture: dict):
    """补充镜像没有 JAX；用同一 body 函数在 NumPy CPU 上逐步执行参考语义。"""
    body = fixtures.get_body(fixture["body_id"])
    copy = lambda value: np.array(value, copy=True)  # noqa: E731
    carry = fixtures.tree_from_spec(fixture["carry_tree"], fixture["arrays"], convert=copy)
    template = carry
    xs = None if fixture["xs_tree"] is None else fixtures.tree_from_spec(
        fixture["xs_tree"], fixture["arrays"], convert=copy)
    length = int(fixture["length"])
    reverse = bool(fixture["reverse"])
    ys = [None] * length
    for step in range(length):
        index = length - 1 - step if reverse else step
        x_t = None if xs is None else _tree_index(xs, index)
        carry, y = body(carry, x_t)
        carry = _restore_tree_dtypes(carry, template)
        ys[index] = np.asarray(y)
    return carry, np.stack(ys, axis=0)


def _frozen_case(case_id: str) -> dict:
    return next(case for case in FROZEN_SUITE["cases"] if case["case_id"] == case_id)


def _run_torch_case(case: dict, mode: str) -> dict:
    fixture = fixtures.make_fixture(case)
    init, xs = _make_torch_tree(fixture)
    init_flat = fixtures.flatten_with_paths(init)
    xs_flat = [] if xs is None else fixtures.flatten_with_paths(xs)
    init_before = [leaf.clone() for _path, leaf in init_flat]
    xs_before = [leaf.clone() for _path, leaf in xs_flat]
    input_ptrs = {int(leaf.data_ptr()) for _path, leaf in init_flat + xs_flat}

    module = _candidate()
    body = fixtures.get_body(fixture["body_id"])
    fn = module.scan if mode == "eager" else module.scan_compiled
    final_carry, ys = fn(body, init, xs, length=int(fixture["length"]),
                         reverse=bool(fixture["reverse"]))
    dispatch = _dispatch_of(module)

    assert dispatch.get("path") == "so", dispatch
    assert dispatch.get("family") == "torch", dispatch
    assert dispatch.get("backend") == "npu", dispatch
    expected_exec = "stepwise" if mode == "eager" else "compiled_whole_loop"
    assert dispatch.get("exec_mode") == expected_exec, dispatch
    assert dispatch.get("abi_calls") == (int(fixture["length"]) if mode == "eager" else 1), dispatch
    device_leaves = assert_all_leaves_on_accelerator(
        (final_carry, ys), f"torch_delivery_{case['case_id']}_{mode}", dispatch)

    output_ptrs = {int(leaf.data_ptr())
                   for _path, leaf in fixtures.flatten_with_paths((final_carry, ys))}
    assert input_ptrs.isdisjoint(output_ptrs), "输出不得与 init/xs 输入张量别名"
    for (_path, leaf), before in zip(init_flat, init_before):
        assert bool((leaf == before).all().item()), "init 在调用后被改写"
    for (_path, leaf), before in zip(xs_flat, xs_before):
        assert bool((leaf == before).all().item()), "xs 在调用后被改写"

    expected_carry, expected_ys = _numpy_reference(fixture)
    expected = {"final_carry": expected_carry, "ys": expected_ys}
    expected_flat = {path: np.asarray(leaf)
                     for path, leaf in fixtures.flatten_with_paths(expected)}
    actual = {"final_carry": final_carry, "ys": ys}
    leaf_map = {item["path"]: item for item in device_leaves}
    leaves, problems = _judge(actual, expected_flat,
                              scan_engine._expected_golden_keys(mode, fixture),
                              leaf_map, fixture, mode)
    assert not problems, f"{case['case_id']}/{mode}: " + "；".join(problems)
    assert leaves and all(leaf["is_pass"] for leaf in leaves), leaves

    record = {
        "case_id": case["case_id"],
        "mode": mode,
        "status": "PASS",
        "length": int(fixture["length"]),
        "reverse": bool(fixture["reverse"]),
        "dispatch": dispatch,
        "leaves": leaves,
        "max_abs_error": max((leaf.get("max_abs_error") or 0.0) for leaf in leaves),
        "input_unchanged": True,
        "output_alias_free": True,
    }
    _TORCH_RESULTS.append(record)
    return record


if importlib.util.find_spec("torch_npu") is not None:

    @pytest.mark.parametrize("mode", ["eager", "jit"])
    @pytest.mark.parametrize("case_id", TORCH_CASE_IDS)
    def test_torch_family_delivery(case_id, mode):
        """torch_npu 公开接口：真实 NPU 计算、输入只读、输出不别名。"""
        _run_torch_case(_frozen_case(case_id), mode)

    @pytest.mark.parametrize("mode", ["eager", "jit"])
    def test_torch_length2_input_immutability(mode):
        """偶数长度回归：final_carry 必须是新缓冲，不能是 init 别名。"""
        _run_torch_case(_TORCH_LEN2_CASE, mode)


# ---------------------------------------------------------------------------
# 自检（CPU 可跑；不需要候选/设备）
# ---------------------------------------------------------------------------


def test_selfcheck_discovery_counts():
    # 白盒接入提示：新增用例后若本测试失败，请同步 EXPECTED_PYTEST_CASES / EXPECTED_PYTEST_MODES
    # （以及下方 origins 期望），再按 1-测试工程说明 §12 记录新的发现数。
    assert len(FROZEN_SUITE["cases"]) == EXPECTED_GATE_CASES, (
        f"门禁用例数变成 {len(FROZEN_SUITE['cases'])}，期望 {EXPECTED_GATE_CASES}；"
        "改冻结集会改变 suite_ref，需先取得 PM/用户确认")
    assert GATE_MODE_COUNT == EXPECTED_GATE_MODES, (
        f"门禁 mode 次数变成 {GATE_MODE_COUNT}，期望 {EXPECTED_GATE_MODES}")
    assert len(CASES) == EXPECTED_PYTEST_CASES, (
        f"pytest 用例数变成 {len(CASES)}，期望 {EXPECTED_PYTEST_CASES}；"
        "请同步 EXPECTED_PYTEST_CASES 并在 1-测试工程说明 §4.2 记录发现数")
    assert MODE_COUNT == EXPECTED_PYTEST_MODES, (
        f"pytest mode 次数变成 {MODE_COUNT}，期望 {EXPECTED_PYTEST_MODES}；"
        "请同步 EXPECTED_PYTEST_MODES 并在 1-测试工程说明 §4.2 记录发现数")
    origins = {}
    for case in CASES:
        origins[case["_origin"]] = origins.get(case["_origin"], 0) + 1
    assert origins == {"frozen": 7, "benchmark": 2, "supplement": 6, "whitebox": 4}, (
        f"来源分布变化：{origins}；请同步 origins 期望")
    assert len(WHITEBOX_SUITE["cases"]) == 4, (
        f"白盒用例数变成 {len(WHITEBOX_SUITE['cases'])}；"
        "请同步 3-白盒测试设计.md 的源码分支映射与覆盖记录")


def test_selfcheck_smoke_selection():
    selected = scan_engine.pick_smoke_case(FROZEN_SUITE["cases"])
    assert selected["case_id"] == "scan_xs_none_length5", selected["case_id"]
    # 反例：旧规则 expected=="match" 在当前冻结集命中 0 条；只补 n_a=length 会选中拒绝用例。
    old_rule = [c for c in FROZEN_SUITE["cases"]
                if c.get("expected", "match") == "match" and isinstance(c.get("n_a"), int)
                and c.get("n_a") == c.get("n_b")]
    assert old_rule == []
    naive = min((c for c in FROZEN_SUITE["cases"] if c.get("required", True)),
                key=lambda c: (int(c.get("length") or (c.get("shape") or [0])[0]), c["case_id"]))
    assert naive["case_id"] == "scan_bad_carry_shape"
    assert naive.get("expected") == "reject"


def test_selfcheck_device_guard_negative():
    with pytest.raises(AssertionError):
        assert_all_leaves_on_accelerator(np.zeros(4, np.float32), "neg_numpy")
    try:
        import torch  # noqa: PLC0415
    except Exception:  # noqa: BLE001 —— 本机无 torch 时只跑 numpy 负例
        return
    with pytest.raises(AssertionError):
        assert_all_leaves_on_accelerator(torch.zeros(4), "neg_torch_cpu")
    with pytest.raises(AssertionError):
        assert_all_leaves_on_accelerator((torch.zeros(4), np.zeros(4)), "neg_mixed")

    # AD unsupported 硬证据（与 runner 共用 scan_engine.unsupported_attempt_problems）：
    # 无 EXEC / path=so / backend=cpu / 缺 reason / 缺后端原文 都必须被拒。
    good = {"op": "jax_lax_scan", "mode": "jvp", "path": "trace", "abi_calls": 0,
            "backend": "npu", "reason": "so_no_autodiff"}
    assert scan_engine.unsupported_attempt_problems(
        good, "RuntimeError: boom", require_op=True, expected_mode="jvp") == []
    bad_dispatches = {
        "no_exec": {},
        "path_so": {**good, "path": "so"},
        "backend_cpu": {**good, "backend": "cpu"},
        "no_reason": {**good, "reason": None},
        "wrong_reason": {**good, "reason": "backend_unsupported"},
        "wrong_mode": {**good, "mode": "jit"},
        "wrong_op": {**good, "op": "other_op"},
    }
    for name, dispatch in bad_dispatches.items():
        assert scan_engine.unsupported_attempt_problems(
            dispatch, "RuntimeError: boom", require_op=True, expected_mode="jvp"), name
    assert scan_engine.unsupported_attempt_problems(good, ""), "缺后端原文必须被拒"
    assert scan_engine.strip_unsupported_marker(scan_engine.UNSUPPORTED_CODE).strip() == ""


def test_selfcheck_golden_signature_guard():
    spec = importlib.util.spec_from_file_location(
        "jax_lax_scan_golden_selfcheck", WS / "ops" / "jax_lax_scan" / "scripts" / "golden.py")
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    with pytest.raises(TypeError):
        module.compute_golden(np.zeros(4, np.float32), np.zeros(4, np.float32))


def test_selfcheck_golden_env_guard_negative(tmp_path):
    case = CASES[0]
    fixture = fixtures.make_fixture(case)
    case_json = tmp_path / "case.json"
    inputs = tmp_path / "inputs.npz"
    fixtures.write_case_json(case, fixture, case_json)
    fixtures.write_inputs_npz(fixture, inputs)
    env = os.environ.copy()
    env["JAX_PLATFORMS"] = "npu"          # 故意请求非 CPU 后端
    env["JAX_ENABLE_X64"] = "0"
    env.pop("XLA_FLAGS", None)
    env.pop("JAX_COMPILATION_CACHE_DIR", None)
    proc = subprocess.run(
        [sys.executable, "-I", str(GOLDEN_CLI), "--case", str(case_json),
         "--inputs", str(inputs), "--npz", str(tmp_path / "golden.npz"),
         "--meta", str(tmp_path / "golden.json"), "--mode", "eager"],
        capture_output=True, text=True, timeout=180, env=env,
    )
    assert proc.returncode == 3, f"rc={proc.returncode} stderr={proc.stderr[-400:]}"
    assert "GOLDEN_ENV_MISMATCH" in (proc.stderr or "")


def test_selfcheck_judge_flattens_tuple_carry():
    """回归：tuple carry 必须先展开成 final_carry/0、final_carry/1 再逐叶子判定。"""
    case = next(c for c in CASES if c["case_id"] == "scan_pytree_carry")
    fixture = fixtures.make_fixture(case)
    xs = fixture["arrays"]["xs"]
    acc = np.sum(xs.astype(np.float32), axis=0, dtype=np.float32)
    counter = np.asarray(fixture["length"], dtype=np.int32)
    ys = np.cumsum(xs.astype(np.float32), axis=0, dtype=np.float32)
    actual = {"final_carry": (counter, acc), "ys": ys}
    golden = {path: np.asarray(leaf) for path, leaf in fixtures.flatten_with_paths(actual)}
    leaves, problems = _judge(actual, golden, scan_engine._expected_golden_keys("jit", fixture),
                              {}, fixture, "jit")
    assert not problems, problems
    assert [leaf["path"] for leaf in leaves] == ["final_carry/0", "final_carry/1", "ys"]
    assert all(leaf["is_pass"] for leaf in leaves), leaves


def test_selfcheck_report_writer(tmp_path):
    synthetic = {
        "case_id": "selfcheck_report_writer", "group": "自检", "origin": "selfcheck",
        "required": False, "expected": "match_cpu", "status": "NOT_RUN",
        "detail": "仅验证报告写入链路，不是算子执行结果", "modes": [],
        "dispatch": {}, "golden": {},
    }
    json_path, md_path = write_report([synthetic], out_dir=tmp_path)
    payload = json.loads(json_path.read_text(encoding="utf-8"))
    assert payload["schema"] == "opx-jax-correctness/1"
    assert payload["cases"][0]["case_id"] == "selfcheck_report_writer"
    assert payload["summary"]["expected_cases"] == EXPECTED_PYTEST_CASES
    assert json_path.stat().st_size > 0 and md_path.stat().st_size > 0


# ---------------------------------------------------------------------------
# 白盒分支用例（源码分支覆盖；CPU 可跑，不需要设备）
# ---------------------------------------------------------------------------
#
# 依据 `$WORK_DIR/3-白盒测试设计.md` 的源码分支清单：设备数据驱动用例在
# `tests/jax_lax_scan_whitebox_cases.json`（宽度对齐边界 + body_id=3 内核分支），
# 本段覆盖 `python/jax_lax_scan.py` 与 `python/jax_lax_scan_cli.py` 中**不依赖 device**
# 的有效分支：_plan_dispatch 的 4 个出口、length/前导轴校验、树与 dtype 辅助、
# 入口守卫、host driver 参数分支。本段只证明源码分支被走到，不是 NPU 精度证据。


class _WbTorchLeaf:
    """鸭子类型叶子：data_ptr 可调用 + device 非 None → _leaf_kind 判 torch。"""

    def __init__(self, shape=(4,), device="npu:0"):
        self.shape = tuple(shape)
        self.device = device

    def data_ptr(self):
        return 0


class _WbJaxLeaf:
    """鸭子类型叶子：devices 可调用 → _leaf_kind 判 jax。"""

    def __init__(self, shape=(4,)):
        self.shape = tuple(shape)

    def devices(self):
        return []


class _WbJaxTracer(_WbJaxLeaf):
    """伪装 JAX tracer：module/类名同时满足 _has_tracer 的判定条件。"""


_WbJaxTracer.__module__ = "jax"
_WbJaxTracer.__name__ = "DynamicJaxprTracer"


def _wb_body(name: str):
    """无 body_id 属性的 callable，用于覆盖 _body_id_of 的名称回退分支。"""

    def body(carry, x):
        return carry, carry

    body.__name__ = name
    return body


def test_whitebox_plan_dispatch_branches():
    """_plan_dispatch 的 4 个出口：reject / so / trace(具体 jax) / trace(tracer 或未注册)。"""
    module = _candidate()
    init = np.zeros(4, np.float32)
    xs = np.zeros((5, 4), np.float32)
    body = fixtures.get_body(1)

    plan = module._plan_dispatch(body, init, xs, 5, False)
    assert (plan["path"], plan["exec_mode"], plan["abi_calls"], plan["family"]) == (
        "so", "stepwise", None, "numpy"), plan

    torch_plan = module._plan_dispatch(body, _WbTorchLeaf(), _WbTorchLeaf((5, 4)), 5, True)
    assert (torch_plan["path"], torch_plan["family"]) == ("so", "torch"), torch_plan

    jax_plan = module._plan_dispatch(body, _WbJaxLeaf(), _WbJaxLeaf((5, 4)), 5, False)
    assert (jax_plan["path"], jax_plan["exec_mode"], jax_plan["reason"]) == (
        "trace", "jax_jit", "so_no_autodiff"), jax_plan

    tracer_plan = module._plan_dispatch(body, _WbJaxTracer(), _WbJaxTracer((5, 4)), 5, False)
    assert (tracer_plan["path"], tracer_plan["exec_mode"], tracer_plan["reason"]) == (
        "trace", "jax_eager", "so_no_autodiff"), tracer_plan

    unregistered = _wb_body("wb_plain_body")
    unreg_plan = module._plan_dispatch(unregistered, init, xs, 5, False)
    assert (unreg_plan["path"], unreg_plan["exec_mode"], unreg_plan["reason"],
            unreg_plan["body_id"]) == ("trace", "jax_eager", "unregistered_body", None), unreg_plan

    reject_plan = module._plan_dispatch(fixtures.get_body(4), init, xs, 3, False)
    assert (reject_plan["path"], reject_plan["exec_mode"], reject_plan["abi_calls"],
            reject_plan["reason"]) == ("reject", "reject", 0, "carry_shape_mismatch"), reject_plan


def test_whitebox_length_guard_branches():
    """_tree_leading_size / _resolve_length 的 5 个校验出口。"""
    module = _candidate()
    xs = {"a": np.zeros((5, 4), np.float32), "b": np.zeros((5, 4), np.float32)}
    init = np.zeros(4, np.float32)

    assert module._tree_leading_size(None) is None
    assert module._tree_leading_size(xs) == 5
    assert module._resolve_length(init, xs, None) == 5
    assert module._resolve_length(init, xs, 5) == 5
    with pytest.raises(ValueError, match="length_mismatch"):
        module._resolve_length(init, None, None)
    with pytest.raises(ValueError, match="length_mismatch"):
        module._resolve_length(init, xs, 0)
    with pytest.raises(ValueError, match="length_mismatch"):
        module._resolve_length(init, xs, 6)
    with pytest.raises(ValueError, match="length_mismatch"):
        module._tree_leading_size({"a": np.zeros((5, 4), np.float32),
                                   "b": np.zeros((6, 4), np.float32)})
    with pytest.raises(ValueError, match="invalid_argument"):
        module._tree_leading_size(np.asarray(3.0, np.float32))


def test_whitebox_tree_dtype_helpers():
    """_leaf_kind / _family_of / _flatten / _rebuild / _body_id_of / _dtype_code 的有效分支。"""
    module = _candidate()

    assert module._leaf_kind(np.zeros(2, np.float32)) == "numpy"
    assert module._leaf_kind(_WbTorchLeaf()) == "torch"
    assert module._leaf_kind(_WbJaxLeaf()) == "jax"
    assert module._leaf_kind(3) == "other"

    assert module._family_of(np.zeros(2, np.float32), None) == "numpy"
    assert module._family_of(_WbTorchLeaf(), None) == "torch"
    assert module._family_of(_WbJaxLeaf(), None) == "jax"
    assert module._family_of(_WbTorchLeaf(), {"a": np.zeros(2, np.float32)}) == "numpy+torch"
    assert module._family_of([], None) == "unknown"

    flatten = module._flatten({"a": (1, [2]), "b": 3})
    assert [path for path, _leaf in flatten] == ["a/0", "a/1/0", "b"], flatten
    rebuilt = module._rebuild({"a": (1, [2]), "b": 3}, iter([10, 20, 30]))
    assert rebuilt == {"a": (10, [20]), "b": 30}, rebuilt

    assert module._body_id_of(fixtures.get_body(7)) == 7
    assert module._body_id_of(_wb_body("b1_add")) == 1          # 名称回退
    bad = _wb_body("wb_unknown_body")
    bad.body_id = 99                                            # 越界 body_id 不认
    assert module._body_id_of(bad) is None
    assert module._body_id_of(123) is None

    assert module._dtype_code("float32") == 0
    assert module._dtype_code("int32") == 1
    with pytest.raises(ValueError, match="dtype_mismatch"):
        module._dtype_code("float16")


def test_whitebox_entry_guards():
    """scan/scan_compiled 的 unroll 守卫、_execute 的非 callable 守卫、_require_npu_backend 的设备守卫。"""
    module = _candidate()
    init = np.zeros(4, np.float32)
    xs = np.zeros((2, 4), np.float32)
    body = fixtures.get_body(1)

    with pytest.raises(TypeError, match="invalid_argument"):
        module.scan(123, init)
    with pytest.raises(ValueError, match="unroll=1"):
        module.scan(body, init, xs, length=2, unroll=2)
    with pytest.raises(ValueError, match="unroll=1"):
        module.scan_compiled(body, init, xs, length=2, unroll=2)

    import jax  # noqa: PLC0415 —— 只为判定当前 backend，不执行用例
    backend = jax.default_backend()
    if backend == "cpu":
        with pytest.raises(RuntimeError, match="不允许用 CPU 结果冒充 NPU"):
            module._require_npu_backend(jax)
    else:
        got, devices = module._require_npu_backend(jax)
        assert got == backend and devices, (got, devices)


def _load_cli_module():
    spec = importlib.util.spec_from_file_location(
        "jax_lax_scan_cli_whitebox",
        WS / "ops" / "jax_lax_scan" / "python" / "jax_lax_scan_cli.py")
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _write_cli_case(directory: Path, resolved: dict, arrays: dict):
    directory.mkdir(parents=True, exist_ok=True)
    case_json = directory / "case.json"
    case_json.write_text(json.dumps({"case_id": "wb_cli_case", "_resolved": resolved}),
                         encoding="utf-8")
    inputs = directory / "inputs.npz"
    np.savez(inputs, **arrays)
    return case_json, inputs


def test_whitebox_cli_guard_branches(tmp_path):
    """host driver（jax_lax_scan_cli.py）不进入 device 的参数/拒绝分支。"""
    cli = _load_cli_module()
    base = {"body_id": 4, "length": 3, "reverse": False, "carry_paths": ["init"],
            "xs_paths": ["xs"], "inputs_sha256": "x", "body_name": "B5_bad_shape"}
    arrays = {"init": np.zeros(4, np.float32), "xs": np.zeros((3, 4), np.float32)}

    # ① 缺 _resolved：rc=1，且不进入算子路径
    missing = tmp_path / "missing_resolved"
    missing.mkdir()
    case_json = missing / "case.json"
    case_json.write_text(json.dumps({"case_id": "wb_no_resolved"}), encoding="utf-8")
    rc = cli.main(["--case-json", str(case_json), "--inputs", str(missing / "inputs.npz"),
                   "--mode", "eager", "--outdir", str(missing / "out")])
    assert rc == 1, rc

    # ② 拒绝 body（4/5/6）：device 前 rc=3 + carry_shape_mismatch，不写 outputs.npz
    reject_dir = tmp_path / "reject"
    reject_case, reject_inputs = _write_cli_case(reject_dir, base, arrays)
    reject_out = reject_dir / "out"
    rc = cli.main(["--case-json", str(reject_case), "--inputs", str(reject_inputs),
                   "--mode", "jit", "--outdir", str(reject_out)])
    assert rc == 3, rc
    assert not (reject_out / "outputs.npz").exists()

    # ③ AD 只接受 body_id=3：参数分支，rc=1
    ad_case, ad_inputs = _write_cli_case(
        tmp_path / "ad_body", {**base, "body_id": 1, "xs_paths": ["xs"]}, arrays)
    rc = cli.main(["--case-json", str(ad_case), "--inputs", str(ad_inputs),
                   "--mode", "jvp", "--outdir", str(tmp_path / "ad_body" / "out")])
    assert rc == 1, rc

    # ④ AD 需要非 None 的 xs：参数分支，rc=1
    no_xs_case, no_xs_inputs = _write_cli_case(
        tmp_path / "ad_no_xs", {**base, "body_id": 3, "xs_paths": []}, arrays)
    rc = cli.main(["--case-json", str(no_xs_case), "--inputs", str(no_xs_inputs),
                   "--mode", "vjp", "--outdir", str(tmp_path / "ad_no_xs" / "out")])
    assert rc == 1, rc


def test_whitebox_numeric_dispatch_assertions():
    """非 AD dispatch 断言（含 ``body_id``）的 CPU 分支用例。

    ``3-验收报告.md`` O-1 要求把「断言 ``body_id=3``」写进实现；本用例锁定断言本身：
    stepwise/compiled 的合法 dispatch 必须零问题，``body_id`` 不符必须被判失败，
    原有 path / exec_mode / abi_calls 断言不得回归。
    """
    fixture = {"length": 6, "body_id": 3}  # scan_wb_quad_len6_w8：K14 SCAN_BODY_QUAD
    eager_ok = {"path": "so", "exec_mode": "stepwise", "abi_calls": 6, "body_id": 3}
    jit_ok = {"path": "so", "exec_mode": "compiled_whole_loop", "abi_calls": 1, "body_id": 3}
    assert _numeric_dispatch_problems(eager_ok, fixture, "eager") == []
    assert _numeric_dispatch_problems(jit_ok, fixture, "jit") == []
    for bad in ({**eager_ok, "body_id": 1},
                {**eager_ok, "body_id": None},
                {**eager_ok, "body_id": "3"}):
        problems = _numeric_dispatch_problems(bad, fixture, "eager")
        assert any("body_id" in item for item in problems), (bad, problems)
    assert any("path" in item for item in _numeric_dispatch_problems(
        {**eager_ok, "path": "trace"}, fixture, "eager"))
    assert any("exec_mode" in item for item in _numeric_dispatch_problems(
        {**eager_ok, "exec_mode": "compiled_whole_loop"}, fixture, "eager"))
    assert any("abi_calls" in item for item in _numeric_dispatch_problems(
        {**jit_ok, "abi_calls": 2}, fixture, "jit"))


# ---------------------------------------------------------------------------
# 候选路由单测（节点 2 交付后才有意义；不是 selfcheck 子集）
# ---------------------------------------------------------------------------


def test_candidate_plan_dispatch():
    module = _candidate()
    plan_fn = getattr(module, "_plan_dispatch", None)
    if not callable(plan_fn):
        pytest.fail("候选没有 _plan_dispatch（设计 §1.3.2 要求提供可单测的路由函数）")
    fixture = fixtures.make_fixture({"case_id": "scan_sum_len8_f32", "shape": [8, 16],
                                     "dtype": "float32", "seed": 20261026})
    init, xs = _make_numpy_tree(fixture)
    body = fixtures.get_body(fixture["body_id"])
    plan = plan_fn(body, init, xs, int(fixture["length"]), False)
    assert plan.get("path") == "so", plan
    assert int(plan.get("body_id")) == fixture["body_id"], plan
    assert plan.get("exec_mode") == "stepwise", plan

    reject_fixture = fixtures.make_fixture({"case_id": "scan_bad_carry_shape", "length": 3,
                                           "dtype": "float32", "seed": 20261026})
    reject_init, reject_xs = _make_numpy_tree(reject_fixture)
    reject_body = fixtures.get_body(reject_fixture["body_id"])
    reject_plan = plan_fn(reject_body, reject_init, reject_xs, int(reject_fixture["length"]), False)
    assert reject_plan.get("path") == "reject", reject_plan
    assert int(reject_plan.get("abi_calls", -1)) == 0, reject_plan
