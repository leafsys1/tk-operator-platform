# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""jax_lax_while_loop 的 while 分支执行核心（共享 runner 的唯一门禁实现）。

被 ``tools/run_cases.py``（test）与 ``tools/smoke_case.py``（smoke）按 ``op`` 显式分派复用。
while 的字段模型（``carry/condition/body/limits/runtime_limits/tangent``）与 elementwise 的
``n_a/n_b``、``jax_lax_scan`` 的 carry、``jax_lax_associative_scan`` 的 combine、``jax_lax_cond``
的 pred/branch 都不同，但**统一入口与判据只有一个**：本模块把 while 特有的输入、候选调用、
golden 子进程、逐叶判定、设备证据与报告组装集中实现；旧 op 分支原样保留。

候选可执行文件契约（设计 §2.2 / §5.3）
--------------------------------------

``ops/jax_lax_while_loop/build/jax_lax_while_loop \\
  --case-json <case.json> --inputs <inputs.npz> --mode <eager|jit|jvp_eager|jvp_jit> \\
  [--variant <name>] --outdir <dir>``

* 成功：rc=0；``<outdir>/outputs.npz`` + ``<outdir>/meta.json``；stdout 含 ``EXEC ...``。
* 参数校验拒绝（c13–c15）：rc=3；输出 ``REJECT code=<稳定码>`` 与
  ``UPSTREAM type=<类型> sha256=<摘要> summary=<截断消息>``；``EXEC ... path=reject
  exec_mode=reject abi_calls=0``；不得写 ``outputs.npz``。
* 设备安全阀拒绝（c16）：rc=3；``code=ITERATION_LIMIT_EXCEEDED``、``path=so``、
  ``iterations=cond_hits=body_hits=64``、``limit_flag=1``、``abi_calls`` 为原始整数（>0）；
  ``upstream.type`` 为 null，但候选异常类型/摘要非空可复算。
* 其他非 0：ERROR（不得因“别的原因失败”算拒绝通过）。本任务没有合法 ``UNSUPPORTED``。
* ``meta.json`` 成功时至少含：``op/kernel/arch/mode/variant/path/exec_mode/route/
  abi_calls/backend/platforms/device_id/iterations/cond_hits/body_hits/trace_hits/
  exec_hits/runtime_token/counter_source/compile_count/plan_id/plan_cache/limit_flag/
  leaves/tree``。``abi_calls`` 必须是宿主 ABI 原始整数（字符串不接受）；``platforms``
  必须含 ``npu``；``leaves`` 为 ``{path,dtype,shape,weak_type,sha256}`` 列表；
  ``tree`` 为 flatten 前的容器指纹（``fixtures.tree_signature``；JVP 顶层是
  ``(primal, tangent)`` 二元组），用于区分 tuple/list。

设备真实性门禁（设计 §2.4，PASS 的唯一路径）
-------------------------------------------

数值用例与 c16 必须同时满足：候选 ``path=so`` + ``backend!=cpu`` + platforms 含 npu +
``abi_calls`` 原始整数 + ``counter_source=kernel`` + 合法 route + 容器 device 0 +
runtime_token 非空，并通过**独立**后端探测（runner 子进程 ``jax.default_backend()``）与
容器设备节点探测；pytest 路径还要求 fresh worker 进程真实 backend 非 CPU。c13–c15 是
device 之前的参数校验拒绝，只校验身份 + ``path=reject``/``abi_calls=0``，不要求 NPU。
身份（image #145 / host device 5 / container device 0）优先读环境变量，其次读
``nm.sh`` 从控制台 plan 写出的 ``<WS>/.opx-run-identity.json``；缺失或冲突记 INVALID。

golden 一律走独立 CPU 子进程 ``golden_cli.py``；本模块**不在 runner 进程 import jax**，
避免候选侧 backend 被 golden 的 CPU 环境限制；独立后端探测只在 runner 的**子进程**里
import jax，不污染 golden/候选。
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

import fixtures  # noqa: E402

WHILE_OP = "jax_lax_while_loop"
KERNEL_NAME = "jax_lax_while_loop_kernel"
ARCH = "dav-2201"
HOST_LIB = f"lib{WHILE_OP}_host.so"
GOLDEN_CLI = _HERE / "golden_cli.py"
CANDIDATE_EXIT_OK = 0
CANDIDATE_EXIT_REJECT = 3
GOLDEN_TIMEOUT_S = 180
CASE_STATUSES = ("PASS", "FAIL", "ERROR", "INVALID", "NOT_RUN")
JVP_MODES = ("jvp_eager", "jvp_jit")
JIT_MODES = ("jit", "jvp_jit")
EXPECTED_SPEC_VERSION_ID = "7"
EXPECTED_SUITE_REF = "cases@288876910a5e2843"
EXPECTED_CRITERIA_REF = "ops-precision-standard@1"

EXEC_REQUIRED_KEYS = (
    "op", "kernel", "arch", "mode", "variant", "path", "exec_mode", "abi_calls",
    "backend", "platforms", "device_id", "iterations", "cond_hits", "body_hits",
    "trace_hits", "exec_hits", "runtime_token", "counter_source", "route",
    "compile_count", "plan_id", "plan_cache", "limit_flag",
)

#: 本轮冻结身份（设计 §2.4 / 环境信息：镜像 #145、宿主设备 5、容器设备 0）。
#: 期望值可由 ``OPX_EXPECT_*`` 覆盖（仅用于负向自检/未来任务），实际值优先读环境变量，
#: 其次读 ``nm.sh`` 从控制台 plan 写出的 ``<WS>/.opx-run-identity.json``。
IDENTITY_DEFAULTS = {
    "image_id": "145",
    "host_device_id": "5",
    "container_device_id": "0",
}
IDENTITY_FILE = ".opx-run-identity.json"
EXPECTED_JAX_VERSION = "0.8.1"
_BACKEND_PROBE_CACHE: dict | None = None
REJECT_RE = re.compile(r"REJECT\s+code=([A-Za-z0-9_]+)")
UPSTREAM_RE = re.compile(r"UPSTREAM\s+type=(\S*)\s+sha256=([0-9a-fA-F]*)\s+summary=(.*)")
EXEC_RE = re.compile(r"\bEXEC\b(.*)")
TOKEN_RE = re.compile(r"([A-Za-z_][A-Za-z0-9_]*)=([^\s]+)")


def is_while_operator(op: str, suite: dict | None = None) -> bool:
    """显式分派判据：只认 op 名，不用隐式字段漂移。"""
    if op != WHILE_OP:
        return False
    return suite is None or suite.get("op") in (None, WHILE_OP)


# ---------------------------------------------------------------------------
# 小工具
# ---------------------------------------------------------------------------


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _sha256_text(text: str) -> str:
    return hashlib.sha256((text or "").encode("utf-8")).hexdigest()


def _rel(ws: Path, path: Path) -> str:
    try:
        return str(Path(path).resolve().relative_to(ws.resolve()))
    except (ValueError, OSError):
        return str(path)


def _load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec and spec.loader, f"模块加载失败：{path}"
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _as_int(value):
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _as_bool(value):
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        if value.lower() in ("true", "1", "yes"):
            return True
        if value.lower() in ("false", "0", "no"):
            return False
        if value.lower() in ("none", "null", ""):
            return None
    return None


def _raw_int(value):
    """只接受 JSON/通信协议里的原始整数；字符串、bool、浮点一律返回 None。"""
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    return None


def _text(value) -> str | None:
    if value in (None, ""):
        return None
    return str(value)


def _identity_file_payload(ws: Path | None = None) -> dict:
    if ws is None:
        # while_engine.py → <ws>/ops/jax_lax_while_loop/scripts
        ws = _HERE.parents[2]
    path = Path(ws) / IDENTITY_FILE
    if not path.is_file():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001 —— 身份文件损坏按缺失处理，由 problems 报告
        return {"_error": f"{path} 解析失败"}
    return payload if isinstance(payload, dict) else {}


def expected_identity() -> dict:
    """当前任务期望身份；默认取设计 §2.4 冻结值，可用 OPX_EXPECT_* 覆盖。"""
    env = os.environ
    return {
        "image_id": env.get("OPX_EXPECT_IMAGE_ID") or IDENTITY_DEFAULTS["image_id"],
        "host_device_id": env.get("OPX_EXPECT_HOST_DEVICE_ID") or IDENTITY_DEFAULTS["host_device_id"],
        "container_device_id": (env.get("OPX_EXPECT_CONTAINER_DEVICE_ID")
                                or IDENTITY_DEFAULTS["container_device_id"]),
    }


def observed_identity(ws: Path | None = None) -> dict:
    """实际身份：环境变量优先，其次 run identity 文件；记录来源便于回指 run manifest。"""
    env = os.environ
    payload = _identity_file_payload(ws)
    image = env.get("OPX_IMAGE_ID") or payload.get("image_id")
    host = env.get("OPX_HOST_DEVICE_ID") or payload.get("host_device_id")
    devices = env.get("OPX_DEVICES")
    if not host and devices:
        host = devices
    container = env.get("OPX_DEVICE_ID")
    if container in (None, ""):
        container = payload.get("container_device_id")
    if container in (None, ""):
        # 容器内只暴露锁定的单张宿主卡，映射到逻辑 device 0。
        container = "0" if host else None
    source = {
        "image_id": "env:OPX_IMAGE_ID" if env.get("OPX_IMAGE_ID") else (
            "file:" + IDENTITY_FILE if payload.get("image_id") else "missing"),
        "host_device_id": "env:OPX_HOST_DEVICE_ID" if env.get("OPX_HOST_DEVICE_ID") else (
            "env:OPX_DEVICES" if env.get("OPX_DEVICES") else (
                "file:" + IDENTITY_FILE if payload.get("host_device_id") else "missing")),
        "container_device_id": "env:OPX_DEVICE_ID" if env.get("OPX_DEVICE_ID") else (
            "file:" + IDENTITY_FILE if payload.get("container_device_id") not in (None, "")
            else ("derived:single_visible_device" if container else "missing")),
    }
    if payload.get("_error"):
        source["identity_file_error"] = payload["_error"]
    return {
        "image_id": _text(image),
        "host_device_id": _text(host),
        "container_device_id": _text(container),
        "source": source,
    }


def identity_problems(identity: dict | None = None,
                      expected: dict | None = None) -> list[str]:
    """身份缺项/冲突一律 INVALID；不把 prompt 里的值当作已观测证据。"""
    obs = identity if identity is not None else observed_identity()
    exp = expected if expected is not None else expected_identity()
    problems: list[str] = []
    for field in ("image_id", "host_device_id", "container_device_id"):
        actual = _text(obs.get(field))
        wanted = _text(exp.get(field))
        if actual is None:
            problems.append(f"身份字段缺失：{field}")
        elif wanted is not None and actual != wanted:
            problems.append(f"{field}={actual!r} != 已确认 {wanted!r}")
    return problems


def backend_probe(timeout: int = 180) -> dict:
    """在不 import jax 的 runner 进程外做一次独立后端探测；结果缓存。

    这是 runner 侧唯一不依赖候选自报的 NPU 证据来源：CPU 合成桩即使伪造
    ``meta.backend=npu``，本探测仍返回 ``cpu``，设备门禁必须失败。
    """
    global _BACKEND_PROBE_CACHE
    if _BACKEND_PROBE_CACHE is not None:
        return _BACKEND_PROBE_CACHE
    # 结果写临时文件而不是 stdout：xla-npu 初始化日志会写 stdout 且可能分片交错，
    # 直接解析 stdout 的“最后一行”在真机上不稳定（workflow3 真机 smoke 实测
    # JSONDecodeError）。写文件不削弱门禁：backend/platforms/jax 仍由独立子进程产生。
    code = (
        "import json,jax,jaxlib,sys;"
        "payload={'backend':jax.default_backend(),"
        "'platforms':sorted({str(d.platform) for d in jax.devices()}),"
        "'jax':getattr(jax,'__version__',None),"
        "'jaxlib':getattr(jaxlib,'__version__',None)};"
        "open(sys.argv[1],'w',encoding='utf-8').write(json.dumps(payload))"
    )
    tmpdir = tempfile.mkdtemp(prefix="opx-backend-probe-")
    payload_path = Path(tmpdir) / "probe.json"
    result: dict = {"rc": None, "backend": None, "platforms": [], "jax": None,
                    "jaxlib": None, "problems": [], "cmd": None,
                    "stdout_tail": None, "stderr_tail": None}
    cmd = [sys.executable, "-I", "-c", code, str(payload_path)]
    result["cmd"] = cmd
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        result["rc"] = proc.returncode
        result["stdout_tail"] = _tail(proc.stdout or "", 1500)
        result["stderr_tail"] = _tail(proc.stderr or "", 1500)
        if proc.returncode != 0:
            result["problems"].append(
                f"独立后端探测失败 rc={proc.returncode}：{_tail(proc.stderr, 300)}")
        elif not payload_path.is_file():
            result["problems"].append(
                "独立后端探测退出 0 但没有产出 JSON 文件（stdout 尾部已记录）")
        else:
            data = json.loads(payload_path.read_text(encoding="utf-8"))
            result.update({key: data.get(key) for key in
                           ("backend", "platforms", "jax", "jaxlib")})
    except Exception as exc:  # noqa: BLE001
        result["problems"].append(f"独立后端探测异常：{type(exc).__name__}: {exc}")
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)
    backend = str(result.get("backend") or "")
    platforms = [str(item) for item in (result.get("platforms") or [])]
    if not result["problems"]:
        if backend in ("", "cpu"):
            result["problems"].append(f"独立后端探测 backend={backend!r}（必须非 CPU）")
        if not any("npu" in item.lower() for item in platforms):
            result["problems"].append(f"独立后端探测 platforms={platforms}（未包含 npu）")
        if (result.get("jax"), result.get("jaxlib")) != (EXPECTED_JAX_VERSION,
                                                         EXPECTED_JAX_VERSION):
            result["problems"].append(
                f"独立后端探测 jax/jaxlib={result.get('jax')}/{result.get('jaxlib')}")
    _BACKEND_PROBE_CACHE = result
    return result


def host_device_probe() -> dict:
    """检查容器里是否真的暴露了 Ascend 设备节点与可见设备号（不依赖候选）。"""
    problems: list[str] = []
    try:
        nodes = sorted(path.name for path in Path("/dev").glob("davinci*"))
    except OSError:
        nodes = []
    if not nodes:
        problems.append("未发现 /dev/davinci*（容器未暴露 Ascend 设备）")
    visible = os.environ.get("ASCEND_RT_VISIBLE_DEVICES") or os.environ.get("OPX_DEVICES")
    if not visible:
        problems.append("ASCEND_RT_VISIBLE_DEVICES/OPX_DEVICES 为空（无从确认设备可见性）")
    return {
        "ok": not problems,
        "device_nodes": nodes,
        "npu_smi": shutil.which("npu-smi"),
        "visible_devices": visible,
        "problems": problems,
    }


def combined_device_probe() -> dict:
    backend = backend_probe()
    host = host_device_probe()
    problems = list(backend.get("problems") or []) + list(host.get("problems") or [])
    return {
        "backend": backend.get("backend"),
        "platforms": backend.get("platforms") or [],
        "jax": backend.get("jax"),
        "jaxlib": backend.get("jaxlib"),
        "backend_probe": backend,
        "host_probe": host,
        "problems": problems,
    }


def device_evidence(*, dispatch: dict, worker: dict | None = None,
                    require_npu: bool = True, identity: dict | None = None,
                    expected: dict | None = None, probe: dict | None = None
                    ) -> tuple[dict, list[str]]:
    """设备真实性证据（设计 §2.4 硬门禁）。

    参数校验拒绝（c13–c15）用 ``require_npu=False`` 只校验身份；数值用例与 c16 必须同时
    满足：``path=so``、候选 backend 非 CPU、platforms 含 npu、``abi_calls`` 为原始整数、
    ``counter_source=kernel``、合法 route、``device_id`` 等于容器 0、runtime_token 非空，
    并通过**独立**后端/设备节点探测；pytest 路径还要求 worker 进程真实 backend 非 CPU。
    """
    dispatch = dispatch or {}
    meta = dispatch.get("_meta") or {}
    obs = identity if identity is not None else observed_identity()
    exp = expected if expected is not None else expected_identity()
    problems: list[str] = list(identity_problems(obs, exp))
    platforms_raw = dispatch.get("platforms")
    if platforms_raw in (None, ""):
        platforms_raw = meta.get("platforms")
    if isinstance(platforms_raw, str):
        platforms_raw = [platforms_raw]
    platforms = [str(item) for item in (platforms_raw or [])]
    probe_info = probe if probe is not None else (combined_device_probe() if require_npu else {})
    info: dict = {
        "required_npu": bool(require_npu),
        "path": dispatch.get("path"),
        "backend": dispatch.get("backend"),
        "platforms": platforms,
        "counter_source": dispatch.get("counter_source"),
        "route": dispatch.get("route"),
        "abi_calls": dispatch.get("abi_calls"),
        "abi_calls_integer": _raw_int(dispatch.get("abi_calls")),
        "device_id": dispatch.get("device_id"),
        "runtime_token": dispatch.get("runtime_token"),
        "identity": obs,
        "expected_identity": exp,
        "worker": worker or None,
        "probe": probe_info,
        "worker_backend_npu": None,
        "probe_backend_npu": None,
        "real_backend_npu": False,
        "problems": problems,
        "ok": False,
    }
    if not require_npu:
        info["ok"] = not problems
        return info, problems

    path = str(dispatch.get("path") or "")
    if path != "so":
        problems.append(f"path={path!r}（PASS 只认 path=so；trace/CPU 不计 PASS）")
    backend = str(dispatch.get("backend") or "")
    if backend in ("", "cpu"):
        problems.append(f"backend={backend!r}（必须非 CPU）")
    if not platforms:
        problems.append("platforms 缺失（设备记录必须写候选后端平台列表）")
    elif not any("npu" in item.lower() for item in platforms):
        problems.append(f"platforms={platforms}（未包含 npu）")
    abi_calls = _raw_int(dispatch.get("abi_calls"))
    if abi_calls is None:
        problems.append(f"abi_calls={dispatch.get('abi_calls')!r}（必须是宿主 ABI 原始整数，"
                        "不接受字符串/推算值）")
    elif abi_calls < 1:
        problems.append(f"abi_calls={abi_calls}（数值/设备证据必须 >=1）")
    if str(dispatch.get("counter_source") or "") != "kernel":
        problems.append(f"counter_source={dispatch.get('counter_source')!r}（必须为 kernel）")
    if not str(dispatch.get("runtime_token") or ""):
        problems.append("runtime_token 为空（必须可回指本设备次执行）")
    route = str(dispatch.get("route") or "")
    if route not in ("device_loop", "host_step"):
        problems.append(f"route={route!r}（必须为 device_loop/host_step）")
    device_id = _raw_int(dispatch.get("device_id"))
    container_text = _text(obs.get("container_device_id"))
    container_id = None
    if container_text is not None:
        try:
            container_id = int(container_text)
        except (TypeError, ValueError):
            problems.append(f"容器设备号不可解析：{container_text!r}")
    if device_id is None:
        problems.append(f"device_id={dispatch.get('device_id')!r}（必须是容器逻辑设备整数）")
    elif container_id is not None and device_id != container_id:
        problems.append(f"device_id={device_id} != 容器设备 {container_id}")

    probe_backend = str(probe_info.get("backend") or "")
    info["probe_backend_npu"] = bool(probe_backend and probe_backend != "cpu")
    problems.extend(probe_info.get("problems") or [])
    if worker is not None:
        worker_backend = str(worker.get("backend") or "")
        worker_platforms = [str(item) for item in (worker.get("platforms") or [])]
        info["worker_backend_npu"] = bool(worker_backend and worker_backend != "cpu")
        if worker_backend in ("", "cpu"):
            problems.append(f"worker 真实 backend={worker_backend!r}（不得只信候选自报）")
        if not any("npu" in item.lower() for item in worker_platforms):
            problems.append(f"worker 真实 devices={worker_platforms}（未包含 npu）")
        if (worker.get("jax"), worker.get("jaxlib")) != (EXPECTED_JAX_VERSION,
                                                         EXPECTED_JAX_VERSION):
            problems.append(f"worker jax/jaxlib={worker.get('jax')}/{worker.get('jaxlib')}"
                            f"（必须 {EXPECTED_JAX_VERSION}）")
    info["real_backend_npu"] = bool(info.get("worker_backend_npu")
                                    or info.get("probe_backend_npu"))
    info["problems"] = problems
    info["ok"] = not problems
    return info, problems


def _iter_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path) as data:
        return {key: data[key] for key in data.files}


def _tail(text: str, limit: int = 4000) -> str:
    text = text or ""
    return text[-limit:]


def _json_dump(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _dtype_text(dtype) -> str:
    text = str(dtype)
    if "float0" in text:
        return "[('float0', 'V')]"
    return text


def _dtype_matches(expected: str, actual: str) -> bool:
    if "float0" in str(expected):
        return "float0" in str(actual)
    return str(actual) == str(expected)


# ---------------------------------------------------------------------------
# 候选 .so preflight
# ---------------------------------------------------------------------------


def ensure_host_so(ws: Path, op: str, log_dir: Path | None = None) -> tuple[bool, str]:
    """检查 ``build/lib<op>_host.so``；缺失时按同一份源码补编一次（不静默跳过）。"""
    so = Path(ws) / "ops" / op / "build" / HOST_LIB
    if so.is_file():
        return True, str(so)
    build = Path(ws) / "build.sh"
    if not build.is_file():
        return False, f"缺少 {so} 且找不到 {build}"
    log_dir = Path(log_dir) if log_dir else Path(ws) / "ops" / op / "build"
    log_dir.mkdir(parents=True, exist_ok=True)
    log = log_dir / "while-host-so-build.log"
    env = os.environ.copy()
    env["OPX_OP"] = op
    env["OPX_WS"] = str(ws)
    try:
        proc = subprocess.run(["bash", str(build)], cwd=str(ws), env=env,
                              capture_output=True, text=True, timeout=1800)
    except Exception as exc:  # noqa: BLE001
        log.write_text(f"build exception: {type(exc).__name__}: {exc}\n", encoding="utf-8")
        return False, f".so 补编异常：{type(exc).__name__}: {exc}"
    log.write_text((proc.stdout or "") + "\n" + (proc.stderr or ""), encoding="utf-8")
    if not so.is_file():
        return False, f".so 补编失败 rc={proc.returncode}（日志 {log}）"
    return True, str(so)


def library_fingerprint(ws: Path, op: str) -> dict:
    so = Path(ws) / "ops" / op / "build" / HOST_LIB
    if not so.is_file():
        return {"path": _rel(Path(ws), so), "exists": False, "sha256": None}
    return {"path": _rel(Path(ws), so), "exists": True, "sha256": _sha256_file(so)}


# ---------------------------------------------------------------------------
# 候选调用
# ---------------------------------------------------------------------------


def _candidate_cmd(exe: Path, case_json: Path, inputs: Path, mode: str, variant: str,
                   outdir: Path, timeout: int = 0) -> list[str]:
    cmd = [str(exe), "--case-json", str(case_json), "--inputs", str(inputs),
           "--mode", str(mode), "--outdir", str(outdir)]
    if variant:
        cmd += ["--variant", str(variant)]
    return cmd


def _parse_exec(stdout: str) -> dict:
    """解析 stdout 里的 ``EXEC k=v ...``（最后一行优先）。"""
    for line in reversed((stdout or "").splitlines()):
        match = EXEC_RE.search(line)
        if not match:
            continue
        values = {key: value for key, value in TOKEN_RE.findall(match.group(1))}
        if values:
            values["_raw"] = line.strip()
            return values
    return {}


def _parse_reject(stdout: str) -> dict:
    text = stdout or ""
    code_match = REJECT_RE.search(text)
    upstream_match = UPSTREAM_RE.search(text)
    return {
        "code": code_match.group(1) if code_match else None,
        "upstream_type": upstream_match.group(1) if upstream_match else None,
        "upstream_sha256": upstream_match.group(2) if upstream_match else None,
        "upstream_summary": upstream_match.group(3).strip() if upstream_match else None,
    }


def _exec_once(exe: Path, case_json: Path, inputs: Path, mode: str, variant: str,
               outdir: Path, timeout: int) -> dict:
    outdir = Path(outdir)
    if outdir.exists():
        shutil.rmtree(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    cmd = _candidate_cmd(exe, case_json, inputs, mode, variant, outdir, timeout)
    started = time.time()
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        timed_out = False
        rc, stdout, stderr = proc.returncode, proc.stdout, proc.stderr
    except subprocess.TimeoutExpired as exc:
        timed_out = True
        rc, stdout, stderr = None, (exc.stdout or ""), (exc.stderr or "")
    duration_ms = round((time.time() - started) * 1000.0, 3)
    meta_path = outdir / "meta.json"
    meta = {}
    meta_error = None
    if meta_path.is_file():
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
        except Exception as exc:  # noqa: BLE001
            meta_error = f"meta.json 解析失败：{type(exc).__name__}: {exc}"
    return {
        "cmd": cmd,
        "rc": rc,
        "timed_out": timed_out,
        "stdout": stdout or "",
        "stderr": stderr or "",
        "duration_ms": duration_ms,
        "exec": _parse_exec(stdout or ""),
        "reject": _parse_reject((stdout or "") + "\n" + (stderr or "")),
        "meta": meta,
        "meta_error": meta_error,
        "outputs": outdir / "outputs.npz",
        "outdir": outdir,
    }


def _run_golden(ws: Path, case_json: Path, inputs: Path, mode: str, variant: str,
                cdir: Path, timeout: int = GOLDEN_TIMEOUT_S) -> dict:
    env = dict(os.environ)
    env["JAX_PLATFORMS"] = "cpu"
    env["JAX_ENABLE_X64"] = "0"
    env["PYTHONHASHSEED"] = "0"
    env.pop("XLA_FLAGS", None)
    env.pop("JAX_COMPILATION_CACHE_DIR", None)
    npz = cdir / "golden.npz"
    meta = cdir / "golden.json"
    cmd = [sys.executable, "-I", str(GOLDEN_CLI), "--case", str(case_json),
           "--inputs", str(inputs), "--npz", str(npz), "--meta", str(meta),
           "--mode", str(mode), "--variant", str(variant)]
    started = time.time()
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, env=env)
        rc, stdout, stderr = proc.returncode, proc.stdout, proc.stderr
    except subprocess.TimeoutExpired as exc:
        rc, stdout, stderr = None, (exc.stdout or ""), (exc.stderr or "")
    duration_ms = round((time.time() - started) * 1000.0, 3)
    payload = {}
    if meta.is_file():
        try:
            payload = json.loads(meta.read_text(encoding="utf-8"))
        except Exception as exc:  # noqa: BLE001
            payload = {"error": f"golden.json 解析失败：{type(exc).__name__}: {exc}"}
    return {
        "cmd": cmd, "rc": rc, "duration_ms": duration_ms,
        "stdout": stdout or "", "stderr": stderr or "",
        "npz": npz, "meta": payload, "ok": rc == 0,
    }


# ---------------------------------------------------------------------------
# 逐叶判定
# ---------------------------------------------------------------------------


def _leaves_map(raw) -> dict[str, dict]:
    """把 golden（path→meta dict）与候选（[{path,...}] list）统一成 path→meta。"""
    if isinstance(raw, dict):
        return {str(key): value for key, value in raw.items() if isinstance(value, dict)}
    if isinstance(raw, list):
        out: dict[str, dict] = {}
        for item in raw:
            if isinstance(item, dict) and item.get("path") not in (None, ""):
                out[str(item["path"])] = item
        return out
    return {}


def _tree_problems(label: str, expected, actual) -> list[str]:
    if actual is None:
        return [f"{label} 缺少 tree（无法核对 tuple/list 容器类型）"]
    if actual != expected:
        return [f"{label} tree={json.dumps(actual, ensure_ascii=False)} != 期望 "
                f"{json.dumps(expected, ensure_ascii=False)}"]
    return []


def judge_outputs(*, case: dict, variant: str, mode: str, candidate_npz: Path,
                  golden_npz: Path, candidate_meta: dict, golden_meta: dict,
                  criteria) -> tuple[list[dict], list[str]]:
    """逐叶独立判定；返回 ``(leaves, problems)``。任一不等直接 FAIL。

    判定顺序（设计 §2.3 第 7 条）：树容器类型 → 叶子路径集合 → shape/dtype →
    ``weak_type``（jax family 逐叶精确；numpy family 记 n/a）→ 数值容差。
    """
    problems: list[str] = []
    expected = fixtures.output_paths(case, variant, mode)
    try:
        candidate = _iter_npz(candidate_npz)
    except Exception as exc:  # noqa: BLE001
        return [], [f"候选 outputs.npz 读取失败：{type(exc).__name__}: {exc}"]
    try:
        gold = _iter_npz(golden_npz)
    except Exception as exc:  # noqa: BLE001
        return [], [f"golden.npz 读取失败：{type(exc).__name__}: {exc}"]
    if sorted(candidate) != sorted(expected):
        problems.append(f"候选输出键 {sorted(candidate)} != 期望 {sorted(expected)}")
    if sorted(gold) != sorted(expected):
        problems.append(f"golden 输出键 {sorted(gold)} != 期望 {sorted(expected)}")

    expected_tree = fixtures.expected_output_tree(case, variant, mode)
    problems.extend(_tree_problems("候选", expected_tree, candidate_meta.get("tree")))
    problems.extend(_tree_problems("golden", expected_tree, golden_meta.get("tree")))

    specs = fixtures.output_leaf_specs(case, variant, mode)
    candidate_leaves = _leaves_map(candidate_meta.get("leaves"))
    gold_leaves = _leaves_map(golden_meta.get("leaves"))
    if not candidate_leaves:
        problems.append("候选 meta.leaves 缺失（无法核对逐叶结构/weak_type）")
    if not gold_leaves:
        problems.append("golden meta.leaves 缺失（无法核对逐叶 weak_type）")
    leaves: list[dict] = []
    for key in expected:
        spec = specs[key]
        item = candidate_leaves.get(key) or {}
        gold_item = gold_leaves.get(key) or {}
        record = {
            "path": key, "dtype": spec["dtype"], "shape": list(spec["shape"]),
            "weak_type": item.get("weak_type"),
            "candidate_weak_type": item.get("weak_type"),
            "golden_weak_type": gold_item.get("weak_type"),
            "weak_type_compared": False,
            "is_pass": False, "detail": "",
        }
        if not item:
            record["detail"] = "候选 meta.leaves 缺少该叶子"
            leaves.append(record)
            continue
        if not gold_item:
            problems.append(f"golden meta.leaves 缺少 {key}")
        dtype = _dtype_text(item.get("dtype"))
        if not _dtype_matches(spec["dtype"], dtype):
            record["dtype"] = dtype
            record["detail"] = f"dtype={dtype} != {spec['dtype']}"
            leaves.append(record)
            continue
        shape = [int(x) for x in (item.get("shape") or [])]
        if shape != [int(x) for x in spec["shape"]]:
            record["shape"] = shape
            record["detail"] = f"shape={shape} != {spec['shape']}"
            leaves.append(record)
            continue
        # weak_type：golden 为 jax.Array 时逐叶精确；numpy family（None）记 n/a。
        gold_wt = gold_item.get("weak_type")
        cand_wt = item.get("weak_type")
        if gold_wt is not None:
            if cand_wt is None:
                record["detail"] = "无法观测 weak_type（候选未提供 jax.Array weak_type）"
                problems.append(f"{key} 无法观测 weak_type（golden={gold_wt}）")
                leaves.append(record)
                continue
            record["weak_type_compared"] = True
            if bool(cand_wt) != bool(gold_wt):
                record["detail"] = f"weak_type={cand_wt} != golden {gold_wt}"
                problems.append(f"{key} weak_type={cand_wt} != golden {gold_wt}")
                leaves.append(record)
                continue
        else:
            record["detail"] = "weak_type n/a（golden 非 jax family）"
        if key not in candidate or key not in gold:
            record["detail"] = "输出键缺失"
            leaves.append(record)
            continue
        actual, reference = candidate[key], gold[key]
        record["dtype"] = _dtype_text(actual.dtype)
        record["shape"] = [int(x) for x in actual.shape]
        if "float0" in spec["dtype"]:
            if actual.nbytes != 0 or reference.nbytes != 0:
                record["detail"] = (f"float0 尺寸非 0：candidate.nbytes={actual.nbytes} "
                                    f"golden.nbytes={reference.nbytes}")
            else:
                record["is_pass"] = True
                record["detail"] = "float0 结构精确（dtype/shape/nbytes=0）"
            leaves.append(record)
            continue
        if spec["dtype"] == "int32":
            ok = bool(np.array_equal(actual, reference))
            record["is_pass"] = ok
            record["detail"] = "int32 逐元素精确相等" if ok else "int32 存在不等元素"
            leaves.append(record)
            continue
        try:
            judged = criteria.check(actual, reference)
            record.update({
                "is_pass": bool(judged["is_pass"]),
                "matched_ratio": judged.get("matched_ratio"),
                "max_abs_error": judged.get("max_abs_error"),
                "nonfinite": judged.get("nonfinite"),
                "failure_reasons": judged.get("failure_reasons") or [],
            })
            record["detail"] = "混合容差通过" if judged["is_pass"] else "混合容差未通过"
        except Exception as exc:  # noqa: BLE001
            record["detail"] = f"判据异常：{type(exc).__name__}: {exc}"
        leaves.append(record)
    for leaf in leaves:
        if not leaf["is_pass"]:
            problems.append(f"{leaf['path']} 未通过：{leaf['detail']}")
    return leaves, problems


# ---------------------------------------------------------------------------
# 证据校验
# ---------------------------------------------------------------------------


def _merged_dispatch(run: dict) -> dict:
    """meta.json 优先，EXEC 行补缺；返回统一字段表。"""
    meta = run.get("meta") or {}
    exec_line = run.get("exec") or {}
    merged = {}
    for key in EXEC_REQUIRED_KEYS:
        if key in meta:
            merged[key] = meta[key]
        elif key in exec_line:
            merged[key] = exec_line[key]
    merged["_exec_raw"] = exec_line.get("_raw")
    merged["_meta"] = meta
    return merged


def _validate_dispatch(*, case: dict, variant: str, mode: str, run: dict,
                       worker: dict | None = None) -> tuple[dict, list[str]]:
    """校验设备真实性、ABI 计数、设备计数、plan/token 证据；返回 ``(dispatch, problems)``。

    设备真实性（path/backend/platforms/abi_calls/counter_source/route/device_id/token +
    独立后端探测 + worker 真实 backend）统一由 :func:`device_evidence` 判定；本函数只保留
    字段完整性、循环计数与 jit plan 证据。
    """
    problems: list[str] = []
    if run.get("timed_out"):
        return {"path": None}, ["候选执行超时"]
    if run.get("rc") != CANDIDATE_EXIT_OK:
        return {"path": None}, [f"候选 rc={run.get('rc')}（数值用例应为 0）"]
    if run.get("meta_error"):
        problems.append(run["meta_error"])
    dispatch = _merged_dispatch(run)
    required = [key for key in EXEC_REQUIRED_KEYS
                if key not in ("plan_id", "plan_cache", "compile_count") or mode in JIT_MODES]
    missing = [key for key in required
               if key not in dispatch or (key != "variant" and dispatch.get(key) in (None, ""))]
    if missing:
        problems.append(f"EXEC/meta 缺少字段：{missing}")
    evidence, evidence_problems = device_evidence(dispatch=dispatch, worker=worker,
                                                  require_npu=True)
    dispatch["_device_evidence"] = evidence
    problems.extend(evidence_problems)
    counts = fixtures.expected_counts(case, variant)
    for key in ("iterations", "cond_hits", "body_hits", "limit_flag"):
        actual = _as_int(dispatch.get(key))
        expected = counts.get(key)
        if actual is None:
            problems.append(f"{key} 不是整数")
        elif expected is not None and actual != expected:
            problems.append(f"{key}={actual} != {expected}")
    if mode in JIT_MODES:
        plan_id = str(dispatch.get("plan_id") or "")
        if not plan_id:
            problems.append("jit/jvp_jit 缺少 plan_id")
        compile_count = _as_int(dispatch.get("compile_count"))
        if compile_count != 1:
            problems.append(f"compile_count={dispatch.get('compile_count')!r}（fresh worker 首调应为 1）")
        if str(dispatch.get("plan_cache") or "") not in ("miss", "hit"):
            problems.append(f"plan_cache={dispatch.get('plan_cache')!r}")
    return dispatch, problems


def reject_meta_problems(*, case: dict, variant: str, run: dict, golden: dict,
                         dispatch: dict, worker: dict | None = None) -> list[str]:
    """c13–c16 拒绝证据校验（五码 + 路径 + 上游摘要 + c16 设备真实性）。

    c13–c15 是 device 之前的参数校验拒绝：只校验身份与 ``path=reject``/``abi_calls=0``，
    不要求 NPU 平台。c16 是设备安全阀拒绝：必须通过完整的 NPU 设备门禁（含独立后端探测
    与 worker 真实 backend），CPU/合成桩记 INVALID 而不是 PASS。
    """
    problems: list[str] = []
    case_id = str(case.get("case_id"))
    expected_code = fixtures.expected_reject_code(case, variant)
    reject = run.get("reject") or {}
    if run.get("timed_out"):
        return ["候选拒绝执行超时（不得死循环）"]
    if run.get("rc") == CANDIDATE_EXIT_OK:
        return [f"该拒绝却成功返回（rc=0）；期望 {expected_code}"]
    if run.get("rc") != CANDIDATE_EXIT_REJECT:
        return [f"拒绝了但不是约定退出码：rc={run.get('rc')}（期望 {CANDIDATE_EXIT_REJECT}）"]
    if reject.get("code") != expected_code:
        problems.append(f"拒绝码 {reject.get('code')!r} != {expected_code!r}")
    path = str(dispatch.get("path") or "")
    exec_mode = str(dispatch.get("exec_mode") or "")
    abi_calls = _raw_int(dispatch.get("abi_calls"))
    outputs = run.get("outputs")
    has_outputs = bool(outputs and Path(outputs).is_file())
    golden_evidence = (golden.get("meta") or {}).get("reject_evidence") or {}
    cpu_rejects = bool(golden_evidence.get("cpu_rejects"))
    spec_rule_rejects = bool(golden_evidence.get("spec_rule_rejects"))
    candidate_type = reject.get("upstream_type")
    candidate_summary = reject.get("upstream_summary") or ""
    candidate_sha = reject.get("upstream_sha256") or ""
    if candidate_summary:
        if candidate_sha != _sha256_text(candidate_summary):
            problems.append("候选 UPSTREAM sha256 与 summary 不可复算")
    c16 = case_id == "while_iteration_limit_exceeded"
    evidence, device_problems = device_evidence(dispatch=dispatch, worker=worker,
                                                require_npu=c16)
    dispatch["_device_evidence"] = evidence
    problems.extend(device_problems)
    if c16:
        counts = fixtures.expected_counts(case, variant)
        for key in ("iterations", "cond_hits", "body_hits", "limit_flag"):
            actual = _as_int(dispatch.get(key))
            if actual != counts.get(key):
                problems.append(f"c16 {key}={dispatch.get(key)!r} != {counts.get(key)}")
        if candidate_type not in (None, "", "None", "null"):
            problems.append(f"c16 upstream.type={candidate_type!r}（JAX 无该异常，必须为 null）")
        if not candidate_summary:
            problems.append("c16 候选异常摘要为空（必须非空可复算）")
        if not golden_evidence.get("spec_rule_rejects"):
            problems.append("golden 未确认 c16 为已确认规格规则拒绝")
        if has_outputs:
            problems.append("c16 拒绝不得写 outputs.npz")
        return problems

    # c13–c15：参数校验拒绝，发生在 device 之前。
    if path != "reject":
        problems.append(f"path={path!r}（参数校验拒绝必须 path=reject）")
    if exec_mode != "reject":
        problems.append(f"exec_mode={exec_mode!r}（必须为 reject）")
    if abi_calls is None or abi_calls != 0:
        problems.append(f"abi_calls={dispatch.get('abi_calls')!r}（参数校验拒绝必须为原始整数 0）")
    if has_outputs:
        problems.append("参数校验拒绝不得写 outputs.npz")
    if cpu_rejects:
        cpu_type = golden_evidence.get("upstream_exception_type")
        if not candidate_type:
            problems.append("候选未保留上游异常类型（CPU 该输入真实抛错）")
        elif candidate_type != cpu_type:
            problems.append(f"候选上游类型 {candidate_type!r} != CPU {cpu_type!r}")
        if not candidate_summary:
            problems.append("候选上游异常摘要为空")
    elif spec_rule_rejects:
        if candidate_type not in (None, "", "None", "null"):
            problems.append("规格规则拒绝不得伪造上游异常类型")
        meta = dispatch.get("_meta") or {}
        if str(meta.get("spec_rule") or "") != str(golden_evidence.get("spec_rule") or ""):
            problems.append(f"spec_rule={meta.get('spec_rule')!r} 与 golden "
                            f"{golden_evidence.get('spec_rule')!r} 不一致")
        if not candidate_summary:
            problems.append("规格规则拒绝说明为空（必须非空可复算）")
    else:
        problems.append("CPU 未确认该输入应被拒绝（既无 CPU 异常，也不属于规格规则）")
    return problems


# ---------------------------------------------------------------------------
# 单条用例
# ---------------------------------------------------------------------------


class WhileRun:
    def __init__(self, *, ws: Path, op: str, exe: Path, criteria, workdir: Path,
                 evidence_dir: Path, timeout: int):
        self.ws = Path(ws)
        self.op = op
        self.exe = Path(exe)
        self.criteria = criteria
        self.workdir = Path(workdir)
        self.evidence_dir = Path(evidence_dir)
        self.timeout = int(timeout)


def _base_dir(ctx: WhileRun, case_id: str, variant: str, mode: str) -> Path:
    safe = "__".join([case_id, variant or "single", mode])
    return ctx.workdir / safe


def _run_case(ctx: WhileRun, case: dict, variant: str, mode: str) -> dict:
    case_id = str(case.get("case_id"))
    display = "reject" if fixtures.is_reject(case) else mode
    rec: dict = {
        "case_id": case_id,
        "variant": variant,
        "mode": display,
        "requested_mode": mode,
        "required": bool(case.get("required", True)),
        "status": "ERROR",
        "detail": "",
        "bytes": fixtures.bytes_info(case, variant),
        "expected_iterations": fixtures.expected_iterations(case, variant),
        "expected_counts": fixtures.expected_counts(case, variant),
    }
    cdir = _base_dir(ctx, case_id, variant, mode)
    cdir.mkdir(parents=True, exist_ok=True)
    case_json = cdir / "case.json"
    inputs = cdir / "inputs.npz"
    try:
        fixtures.write_case_json(case, variant, mode, case_json)
        inputs_sha = fixtures.write_inputs_npz(fixtures.input_arrays(case, variant), inputs)
    except Exception as exc:  # noqa: BLE001
        rec.update(status="ERROR", detail=f"输入生成失败：{type(exc).__name__}: {exc}")
        return rec
    rec["inputs_sha256"] = inputs_sha
    rec["case_json"] = _rel(ctx.ws, case_json)
    rec["inputs"] = _rel(ctx.ws, inputs)

    if fixtures.is_reject(case):
        rec = _run_reject(ctx, case, variant, mode, cdir, case_json, inputs, rec)
    else:
        rec = _run_numeric(ctx, case, variant, mode, cdir, case_json, inputs, rec)
    _persist_evidence(ctx, rec, cdir)
    return rec


def _persist_evidence(ctx: WhileRun, rec: dict, cdir: Path) -> None:
    """把逐 (case, variant, mode) 的原始输入/候选 meta/golden/record 落到 /out/evidence。"""
    tag = "__".join([str(rec["case_id"]), rec.get("variant") or "single",
                     str(rec.get("requested_mode") or rec.get("mode"))])
    dst = ctx.evidence_dir / tag
    try:
        dst.mkdir(parents=True, exist_ok=True)
        _json_dump(dst / "record.json", rec)
        for name in ("case.json", "inputs.npz"):
            src = cdir / name
            if src.is_file():
                shutil.copy2(src, dst / name)
        for name in ("golden.json", "golden.npz"):
            src = cdir / name
            if src.is_file():
                shutil.copy2(src, dst / name)
        for name in ("meta.json", "outputs.npz"):
            src = cdir / "out" / name
            if src.is_file():
                shutil.copy2(src, dst / f"candidate_{name}")
        if rec.get("stdout_tail"):
            (dst / "stdout_tail.txt").write_text(rec["stdout_tail"], encoding="utf-8")
        if rec.get("stderr_tail"):
            (dst / "stderr_tail.txt").write_text(rec["stderr_tail"], encoding="utf-8")
    except Exception as exc:  # noqa: BLE001 —— 证据落盘失败必须让用例可见
        rec["evidence_error"] = f"{type(exc).__name__}: {exc}"
        rec["status"] = "ERROR"
        rec["detail"] = (rec.get("detail") or "") + f"；证据落盘失败：{exc}"


def _run_numeric(ctx: WhileRun, case: dict, variant: str, mode: str, cdir: Path,
                 case_json: Path, inputs: Path, rec: dict) -> dict:
    run = _exec_once(ctx.exe, case_json, inputs, mode, variant, cdir / "out", ctx.timeout)
    rec.update({
        "cmd": run["cmd"],
        "duration_ms": run["duration_ms"],
        "exec_marker": (run["exec"] or {}).get("_raw"),
        "stdout_tail": _tail(run["stdout"]),
        "stderr_tail": _tail(run["stderr"]),
    })
    golden = _run_golden(ctx.ws, case_json, inputs, mode, variant, cdir)
    rec["golden"] = {"ok": golden["ok"], "rc": golden["rc"], "cmd": golden["cmd"],
                     "backend": (golden["meta"] or {}).get("backend"),
                     "jax": (golden["meta"] or {}).get("jax"),
                     "jaxlib": (golden["meta"] or {}).get("jaxlib")}
    if not golden["ok"]:
        rec.update(status="ERROR", detail=f"golden 执行失败：{_tail(golden['stderr'], 800)}")
        return rec
    if run["timed_out"]:
        rec.update(status="ERROR", detail=f"候选执行超时（{ctx.timeout}s）")
        return rec
    if run["rc"] != CANDIDATE_EXIT_OK:
        if run["rc"] == CANDIDATE_EXIT_REJECT:
            rec.update(status="FAIL", detail="数值用例被候选拒绝（rc=3）")
        else:
            rec.update(status="ERROR", detail=f"候选 rc={run['rc']}")
        return rec
    dispatch, dispatch_problems = _validate_dispatch(case=case, variant=variant, mode=mode,
                                                     run=run)
    rec["identity"] = observed_identity(ctx.ws)
    rec["device_evidence"] = dispatch.get("_device_evidence") or {}
    rec["dispatch"] = {
        key: dispatch.get(key) for key in EXEC_REQUIRED_KEYS
    }
    rec["dispatch"]["path"] = dispatch.get("path")
    rec["dispatch"]["backend"] = dispatch.get("backend")
    if dispatch_problems:
        rec.update(status="INVALID", detail="；".join(dispatch_problems))
        return rec
    if not run["outputs"].is_file():
        rec.update(status="INVALID", detail="候选未写 outputs.npz")
        return rec
    leaves, judge_problems = judge_outputs(
        case=case, variant=variant, mode=mode,
        candidate_npz=run["outputs"], golden_npz=golden["npz"],
        candidate_meta=run["meta"], golden_meta=golden["meta"], criteria=ctx.criteria)
    rec["leaves"] = leaves
    rec["outputs_sha256"] = _sha256_file(run["outputs"])
    if judge_problems:
        rec.update(status="FAIL", detail="；".join(judge_problems))
    else:
        rec.update(status="PASS", detail=f"{mode} 通过（{len(leaves)} 个叶子）")
    return rec


def _run_reject(ctx: WhileRun, case: dict, variant: str, mode: str, cdir: Path,
                case_json: Path, inputs: Path, rec: dict) -> dict:
    run = _exec_once(ctx.exe, case_json, inputs, mode, variant, cdir / "out", ctx.timeout)
    golden = _run_golden(ctx.ws, case_json, inputs, "reject", variant, cdir)
    rec.update({
        "cmd": run["cmd"],
        "duration_ms": run["duration_ms"],
        "exec_marker": (run["exec"] or {}).get("_raw"),
        "stdout_tail": _tail(run["stdout"]),
        "stderr_tail": _tail(run["stderr"]),
        "golden": {"ok": golden["ok"], "rc": golden["rc"], "cmd": golden["cmd"]},
    })
    if not golden["ok"]:
        rec.update(status="ERROR", detail=f"golden 拒绝确认失败：{_tail(golden['stderr'], 800)}")
        return rec
    rec["golden"].update({
        "reject_evidence": (golden["meta"] or {}).get("reject_evidence"),
        "backend": (golden["meta"] or {}).get("backend"),
        "jax": (golden["meta"] or {}).get("jax"),
        "jaxlib": (golden["meta"] or {}).get("jaxlib"),
    })
    dispatch = _merged_dispatch(run)
    rec["identity"] = observed_identity(ctx.ws)
    rec["dispatch"] = {key: dispatch.get(key) for key in EXEC_REQUIRED_KEYS}
    rec["dispatch"]["path"] = dispatch.get("path")
    rec["dispatch"]["backend"] = dispatch.get("backend")
    rec["reject"] = run["reject"]
    c16 = str(case.get("case_id")) == "while_iteration_limit_exceeded"
    problems = reject_meta_problems(case=case, variant=variant, run=run,
                                    golden=golden, dispatch=dispatch)
    rec["device_evidence"] = dispatch.get("_device_evidence") or {}
    device_problems = list((rec["device_evidence"] or {}).get("problems") or []) if c16 else []
    if device_problems:
        rec.update(status="INVALID", detail="设备门禁未通过：" + "；".join(device_problems))
    elif problems:
        # 已约定 rc=3 且 code 正确但证据缺失，仍记 FAIL；非拒绝/异常退出已由 helper 返回。
        status = "FAIL" if run.get("rc") == CANDIDATE_EXIT_REJECT else "ERROR"
        rec.update(status=status, detail="；".join(problems))
    else:
        rec.update(status="PASS", detail=f"{fixtures.expected_reject_code(case, variant)} 拒绝证据完整")
    return rec


# ---------------------------------------------------------------------------
# 报告
# ---------------------------------------------------------------------------


def _by_mode(records: list[dict]) -> dict:
    out = {mode: {status: 0 for status in CASE_STATUSES} | {"total": 0}
           for mode in ("eager", "jit", "jvp_eager", "jvp_jit", "reject")}
    for rec in records:
        mode = rec.get("mode") if rec.get("mode") in out else "reject"
        status = rec.get("status") if rec.get("status") in CASE_STATUSES else "ERROR"
        out[mode][status] += 1
        out[mode]["total"] += 1
    return out


def _authenticity(records: list[dict]) -> dict:
    """真实性聚合：PASS 设备证据只认真实 backend/独立探测，不认候选自报字段。"""
    numeric = [rec for rec in records if rec.get("mode") != "reject"]
    rejects = [rec for rec in records if rec.get("mode") == "reject"]
    c16 = [rec for rec in rejects if rec.get("case_id") == "while_iteration_limit_exceeded"]
    param_rejects = [rec for rec in rejects
                     if rec.get("case_id") != "while_iteration_limit_exceeded"]

    def evidence(rec):
        return rec.get("device_evidence") or {}

    def real_backend_npu(rec):
        return bool(evidence(rec).get("real_backend_npu"))

    return {
        "numeric_cases": len(numeric),
        "path_so": sum(1 for rec in numeric if (rec.get("dispatch") or {}).get("path") == "so"),
        # 只统计 worker 真实 backend / 独立后端探测，不再统计候选自报 dispatch.backend
        "backend_npu": sum(1 for rec in numeric if real_backend_npu(rec)),
        "worker_backend_npu": sum(1 for rec in numeric
                                  if evidence(rec).get("worker_backend_npu")),
        "probe_backend_npu": sum(1 for rec in numeric
                                 if evidence(rec).get("probe_backend_npu")),
        "device_evidence_ok": sum(1 for rec in numeric if evidence(rec).get("ok")),
        "abi_calls_integer": sum(1 for rec in numeric
                                 if isinstance((rec.get("dispatch") or {}).get("abi_calls"), int)
                                 and not isinstance((rec.get("dispatch") or {}).get("abi_calls"),
                                                    bool)),
        "counter_source_kernel": sum(1 for rec in numeric
                                     if (rec.get("dispatch") or {}).get("counter_source") == "kernel"),
        "reject_records": len(rejects),
        "parameter_reject_records": len(param_rejects),
        "reject_device_records": len(c16),
        "reject_device_evidence_ok": sum(1 for rec in c16 if evidence(rec).get("ok")),
        "reject_path_so": sum(1 for rec in c16 if (rec.get("dispatch") or {}).get("path") == "so"),
    }


def _cross_checks(records: list[dict]) -> list[dict]:
    """c07 同一 plan 多 runtime limit + c12 设备 token 随 limit 变化。"""
    checks: list[dict] = []
    plan_ids = [str((rec.get("dispatch") or {}).get("plan_id") or "")
                for rec in records if rec["case_id"] == "while_jit_runtime_limit"]
    checks.append({
        "id": "c07_same_plan_id",
        "ok": bool(plan_ids) and all(plan_ids) and len(set(plan_ids)) == 1,
        "plan_ids": plan_ids,
        "detail": "同一 compiled plan 接受 0/1/8 三个 runtime limit",
    })
    for mode in ("eager", "jit"):
        tokens = [str((rec.get("dispatch") or {}).get("runtime_token") or "")
                  for rec in records
                  if rec["case_id"] == "while_iteration_token_and_hits"
                  and rec["requested_mode"] == mode]
        checks.append({
            "id": f"c12_token_varies_{mode}",
            "ok": len(tokens) == 3 and all(tokens) and len(set(tokens)) == 3,
            "tokens": tokens,
            "detail": f"{mode} 下 limit=0/1/7 的设备 token 必须不同",
        })
    return checks


def _exception_record(item: dict, exc: BaseException) -> dict:
    case = item["case"]
    try:
        byte_info = fixtures.bytes_info(case, item["variant"])
    except Exception:  # noqa: BLE001
        byte_info = {}
    return {
        "case_id": item["case_id"], "variant": item["variant"],
        "mode": item["display_mode"], "requested_mode": item["mode"],
        "required": bool(case.get("required", True)),
        "status": "ERROR", "detail": f"执行器异常：{type(exc).__name__}: {exc}",
        "bytes": byte_info,
        "expected_iterations": fixtures.expected_iterations(case, item["variant"]),
        "expected_counts": fixtures.expected_counts(case, item["variant"]),
        "dispatch": {}, "leaves": [],
    }


def _render_md(report: dict) -> str:
    summary = report["summary"]
    candidate = report.get("candidate") or {}
    candidate_name = candidate.get("exe") or candidate.get("python_interface") or "n/a"
    candidate_sha = str(candidate.get("sha256") or "")
    lines = [
        f"# 正确性报告 · {report['op']}（while 分支）",
        "",
        f"- 用例集：`{report['suite_ref']}`；判据档：`{report['criteria_ref']}`；"
        f"spec_version_id={report['spec_version_id']}",
        f"- 候选：`{candidate_name}` sha256 `{candidate_sha[:16]}…`",
        f"- 结果：TOTAL={summary['TOTAL']} PASS={summary['PASS']} FAIL={summary['FAIL']} "
        f"ERROR={summary['ERROR']} INVALID={summary['INVALID']} NOT_RUN={summary['NOT_RUN']}",
        "",
        "## 逐 mode 计数",
        "",
        "| mode | total | PASS | FAIL | ERROR | INVALID | NOT_RUN |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for mode, counts in report["by_mode"].items():
        lines.append(f"| {mode} | {counts['total']} | {counts['PASS']} | {counts['FAIL']} | "
                     f"{counts['ERROR']} | {counts['INVALID']} | {counts['NOT_RUN']} |")
    lines += [
        "",
        "## 逐用例",
        "",
        "| case_id | variant | mode | status | iterations/body/cond | abi_calls | path | detail |",
        "|---|---|---|---|---|---:|---|---|",
    ]
    for rec in report["cases"]:
        dispatch = rec.get("dispatch") or {}
        counts = "/".join(str(dispatch.get(key, "—"))
                          for key in ("iterations", "body_hits", "cond_hits"))
        lines.append(f"| {rec['case_id']} | {rec['variant'] or '—'} | {rec['mode']} | "
                     f"{rec['status']} | {counts} | {dispatch.get('abi_calls', '—')} | "
                     f"{dispatch.get('path', '—')} | {str(rec.get('detail', ''))[:120]} |")
    lines += ["", "## 交叉核对", ""]
    for check in report.get("cross_checks") or []:
        lines.append(f"- `{check['id']}`：{'PASS' if check['ok'] else 'FAIL'}（{check['detail']}）")
    lines += [
        "",
        "> PASS 只认 `path=so` + 非 CPU backend + platforms 含 npu + 原始整数 `abi_calls` + "
        "`counter_source=kernel` + 容器 device 0 + worker/独立探测真实非 CPU + "
        "设备侧 `iterations/cond_hits/body_hits`；`trace`/CPU/`xfail`/`UNSUPPORTED`/`NOT_RUN` 不计 PASS。",
        "> c13–c15 参数校验拒绝按 `path=reject`+`abi_calls=0` 单独判定；c16 设备安全阀与数值用例"
        "同口径要求完整设备证据。",
        "> 本文件是给人看的伴生品；机判只认同目录的 `correctness.json`。",
    ]
    return "\n".join(lines) + "\n"


def write_report(*, out: Path, md: Path, report: dict) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    _json_dump(out, report)
    md.parent.mkdir(parents=True, exist_ok=True)
    md.write_text(_render_md(report), encoding="utf-8")


def _report_envelope(ctx: WhileRun, *, spec_version_id: str, criteria_ref: str,
                     code_snapshot_sha256: str, suite_ref: str, run_id: str,
                     records: list[dict], case_source: Path, case_source_sha256: str) -> dict:
    summary = {status: sum(1 for rec in records if rec["status"] == status)
               for status in CASE_STATUSES}
    summary["TOTAL"] = len(records)
    identity = observed_identity(ctx.ws)
    env = {
        "host_device_id": identity.get("host_device_id"),
        "container_device_id": identity.get("container_device_id"),
        "image_id": identity.get("image_id"),
        "identity_source": identity.get("source"),
        "expected_identity": expected_identity(),
        "backend": os.environ.get("OPX_BACKEND"),
        "lock_mode": os.environ.get("OPX_LOCK_MODE"),
    }
    return {
        "schema": "opx-jax-while-loop-correctness/1",
        "op": ctx.op,
        "action": "test",
        "run_id": run_id or None,
        "spec_version_id": int(spec_version_id),
        "criteria_ref": criteria_ref,
        "code_snapshot_sha256": code_snapshot_sha256,
        "suite_ref": suite_ref,
        "candidate": {
            "exe": _rel(ctx.ws, ctx.exe),
            "sha256": _sha256_file(ctx.exe),
            "exec_path_marker_required": True,
            "host_so": library_fingerprint(ctx.ws, ctx.op),
        },
        "case_source": _rel(ctx.ws, case_source),
        "case_source_sha256": case_source_sha256,
        "environment": env,
        "expected": {
            "identities": 16,
            "mode_executions": 51,
            "by_mode": {"eager": 14, "jit": 17, "jvp_eager": 2, "jvp_jit": 2, "reject": 16},
        },
        "cases": records,
        "summary": summary,
        "by_mode": _by_mode(records),
        "authenticity": _authenticity(records),
        "cross_checks": _cross_checks(records),
        "golden": {"backend": "cpu", "jax": "0.8.1", "jaxlib": "0.8.1",
                   "cli": _rel(ctx.ws, GOLDEN_CLI)},
    }


def _identity_problems(*, spec_version_id: str, criteria_ref: str,
                       code_snapshot_sha256: str, suite_ref: str) -> list[str]:
    problems = []
    if str(spec_version_id) != EXPECTED_SPEC_VERSION_ID:
        problems.append(f"spec_version_id={spec_version_id!r} != {EXPECTED_SPEC_VERSION_ID}")
    if criteria_ref != EXPECTED_CRITERIA_REF:
        problems.append(f"criteria_ref={criteria_ref!r} != {EXPECTED_CRITERIA_REF}")
    if suite_ref != EXPECTED_SUITE_REF:
        problems.append(f"suite_ref={suite_ref!r} != {EXPECTED_SUITE_REF}")
    if not code_snapshot_sha256:
        problems.append("code_snapshot_sha256 为空")
    return problems


def run_suite(*, ws: Path, op: str, exe: Path, cases_path: Path, out: Path, md: Path,
              workdir: Path, evidence_dir: Path, timeout: int, spec_version_id: str,
              criteria_ref: str, code_snapshot_sha256: str, suite_ref: str,
              run_id: str) -> int:
    if op != WHILE_OP:
        print(f"[run_cases] while 分支拒绝执行：op={op!r}", file=sys.stderr)
        return 2
    if not cases_path.is_file():
        print(f"[run_cases] 用例矩阵不存在：{cases_path}", file=sys.stderr)
        return 2
    suite = json.loads(cases_path.read_text(encoding="utf-8"))
    if not is_while_operator(op, suite):
        print(f"[run_cases] while 分支拒绝执行：suite.op={suite.get('op')!r}", file=sys.stderr)
        return 2
    identity = _identity_problems(spec_version_id=spec_version_id, criteria_ref=criteria_ref,
                                  code_snapshot_sha256=code_snapshot_sha256, suite_ref=suite_ref)
    if identity:
        print(f"[run_cases] 身份门禁失败：{identity}", file=sys.stderr)
        return 2
    ok, detail = ensure_host_so(ws, op, log_dir=out.parent / "evidence")
    if not ok:
        print(f"[run_cases] MISSING_SO {detail}（不在缺共享库时降级为通过）", file=sys.stderr)
        return 2
    criteria = _load_module(ws / "criteria" / "check.py", "opx_criteria_check")
    ctx = WhileRun(ws=ws, op=op, exe=exe, criteria=criteria, workdir=workdir,
                   evidence_dir=evidence_dir, timeout=timeout)
    workdir.mkdir(parents=True, exist_ok=True)
    out.parent.mkdir(parents=True, exist_ok=True)
    evidence_dir.mkdir(parents=True, exist_ok=True)

    items = fixtures.expand_items(suite.get("cases") or [])
    print(f"[run_cases] while 分支：冻结 {len(suite.get('cases') or [])} 条 / "
          f"{len(items)} 个 (case_id, variant, mode) 项")
    if len(items) != 51:
        print(f"[run_cases] 发现数 {len(items)} != 预期 51（请同步用例/展开规则）", file=sys.stderr)
        return 2
    records = []
    for item in items:
        try:
            rec = _run_case(ctx, item["case"], item["variant"], item["mode"])
        except Exception as exc:  # noqa: BLE001 —— 单条执行器异常不得中断全量/丢证据
            rec = _exception_record(item, exc)
        records.append(rec)
        dispatch = rec.get("dispatch") or {}
        print(f"[run_cases] {rec['case_id']}/{rec['variant'] or '-'}/{rec['mode']}: "
              f"{rec['status']} path={dispatch.get('path')} "
              f"iter={dispatch.get('iterations')} abi={dispatch.get('abi_calls')} "
              f"{str(rec.get('detail', ''))[:120]}")

    # c07/c12 交叉核对失败时，把影响项改为 FAIL（不隐藏）。
    cross = _cross_checks(records)
    for check in cross:
        if check["ok"]:
            continue
        for rec in records:
            if check["id"] == "c07_same_plan_id" and rec["case_id"] == "while_jit_runtime_limit":
                rec["status"] = "FAIL"
                rec["detail"] = (rec.get("detail") or "") + "；c07 plan_id 不稳定"
            if check["id"].startswith("c12_token_varies_") and \
                    rec["case_id"] == "while_iteration_token_and_hits" and \
                    rec["requested_mode"] == check["id"].rsplit("_", 1)[-1]:
                rec["status"] = "FAIL"
                rec["detail"] = (rec.get("detail") or "") + "；c12 device token 未随 limit 变化"

    report = _report_envelope(ctx, spec_version_id=spec_version_id,
                              criteria_ref=criteria_ref,
                              code_snapshot_sha256=code_snapshot_sha256,
                              suite_ref=suite_ref, run_id=run_id, records=records,
                              case_source=cases_path,
                              case_source_sha256=_sha256_file(cases_path))
    write_report(out=out, md=md, report=report)
    summary = report["summary"]
    print(f"[run_cases] {summary}  → {out}")
    failed = [rec["case_id"] for rec in records if rec["status"] != "PASS"]
    if failed:
        print(f"[run_cases] 未通过：{failed}", file=sys.stderr)
    return 0 if summary["PASS"] == summary["TOTAL"] else 1


def pick_smoke_case(cases: list[dict]) -> dict:
    for case in cases:
        if str(case.get("case_id")) == "while_one_iteration":
            return case
    raise SystemExit("smoke 需要 while_one_iteration 用例（设计 §7.2）")


def run_smoke(*, ws: Path, op: str, exe: Path, cases_path: Path, out: Path, md: Path,
              workdir: Path, evidence_dir: Path, timeout: int, spec_version_id: str,
              criteria_ref: str, code_snapshot_sha256: str, suite_ref: str,
              run_id: str) -> int:
    if not cases_path.is_file():
        print(f"[smoke] 用例矩阵不存在：{cases_path}", file=sys.stderr)
        return 2
    suite = json.loads(cases_path.read_text(encoding="utf-8"))
    if not is_while_operator(op, suite):
        print(f"[smoke] while 分支拒绝执行：suite.op={suite.get('op')!r}", file=sys.stderr)
        return 2
    ok, detail = ensure_host_so(ws, op, log_dir=out.parent / "evidence")
    if not ok:
        print(f"[smoke] MISSING_SO {detail}（不在缺共享库时降级为通过）", file=sys.stderr)
        return 2
    criteria = _load_module(ws / "criteria" / "check.py", "opx_criteria_check")
    ctx = WhileRun(ws=ws, op=op, exe=exe, criteria=criteria, workdir=workdir,
                   evidence_dir=evidence_dir, timeout=timeout)
    workdir.mkdir(parents=True, exist_ok=True)
    out.parent.mkdir(parents=True, exist_ok=True)
    evidence_dir.mkdir(parents=True, exist_ok=True)
    case = pick_smoke_case(suite.get("cases") or [])
    print(f"[smoke] 最小用例：{case['case_id']} mode=eager（while 分支）")
    rec = _run_case(ctx, case, "", "eager")
    passed = rec.get("status") == "PASS"
    report = {
        "schema": "opx-smoke/1",
        "op": op,
        "action": "smoke",
        "run_id": run_id or None,
        "spec_version_id": int(spec_version_id),
        "criteria_ref": criteria_ref,
        "code_snapshot_sha256": code_snapshot_sha256,
        "suite_ref": suite_ref,
        "candidate": {
            "exe": _rel(ws, exe),
            "sha256": _sha256_file(exe),
            "exec_path_marker_required": True,
            "host_so": library_fingerprint(ws, op),
        },
        "case_source": _rel(ws, cases_path),
        "case_source_sha256": _sha256_file(cases_path),
        "environment": {
            **{key: value for key, value in observed_identity(ws).items()
               if key != "source"},
            "identity_source": observed_identity(ws).get("source"),
            "expected_identity": expected_identity(),
        },
        "case": rec,
        "verdict": "PASS" if passed else "FAIL",
    }
    _json_dump(out, report)
    lines = [
        f"# 冒烟结果：{op}（while 分支）",
        "",
        f"- 用例：`{rec.get('case_id')}`（mode=eager）",
        f"- 结论：**{report['verdict']}**",
        f"- 候选：`{report['candidate']['exe']}` sha256 `{report['candidate']['sha256'][:16]}…`",
        f"- 判据档：`{criteria_ref}`；用例集：`{suite_ref}`",
    ]
    if not passed:
        lines += ["", f"- 失败原因：{rec.get('detail') or rec.get('status')}"]
    md.parent.mkdir(parents=True, exist_ok=True)
    md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[smoke] SMOKE_VERDICT {report['verdict']}  → {out}")
    return 0 if passed else 1


__all__ = [
    "WHILE_OP", "KERNEL_NAME", "ARCH", "HOST_LIB", "CASE_STATUSES",
    "EXEC_REQUIRED_KEYS", "IDENTITY_DEFAULTS", "EXPECTED_JAX_VERSION",
    "is_while_operator", "ensure_host_so", "library_fingerprint",
    "expected_identity", "observed_identity", "identity_problems",
    "backend_probe", "host_device_probe", "combined_device_probe", "device_evidence",
    "judge_outputs", "reject_meta_problems", "write_report",
    "run_suite", "run_smoke", "pick_smoke_case",
]
