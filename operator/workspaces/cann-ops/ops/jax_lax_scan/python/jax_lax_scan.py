# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""jax_lax_scan · Python 调用接口（ctypes 封装 host C ABI）。

公开入口与 ``jax.lax.scan`` 对齐：

    scan(f, init, xs=None, length=None, reverse=False, *, unroll=1) -> (final_carry, ys)
    scan_compiled(...)   # 编译态等价入口（测试/门禁用；一次 .so 调用走完整循环）

契约（`0-黑盒测试设计.md` §1.3）：

* **已注册 body**（带 ``body_id`` 属性的 callable，即 ``ops/jax_lax_scan/scripts/fixtures.py``
  的 8 个规范循环体）→ 走 ``libjax_lax_scan_host.so`` 的 AscendC kernel：
  ``scan`` 逐步（``exec_mode="stepwise"``，``abi_calls == length``），
  ``scan_compiled`` 整循环（``exec_mode="compiled_whole_loop"``，``abi_calls == 1``）。
* **未注册 callable / JAX tracer（含 JVP/VJP）** → 走 JAX trace（``path="trace"``，``abi_calls=0``）。
  AD 的 ``reason`` 固定 ``"so_no_autodiff"``：``.so`` 没有自动微分，这是候选侧的事实，不是"后端不支持"。
* **拒绝 body**（4/5/6：改变 carry 的 shape/dtype/结构）→ device 执行前抛
  ``ValueError("carry_shape_mismatch: ...")``，``abi_calls=0``。

C ABI 结构体见 ``op_host/jax_lax_scan_abi.h``（布局版本 2）；本模块只做参数校验、设备缓冲管理与
调用编排，**不做任何数值计算**（编码规则 A1）。
"""
from __future__ import annotations

import ctypes
import hashlib
import math
import os
import sys
from pathlib import Path

OP_ROOT = Path(__file__).resolve().parents[1]
LIB_PATH = OP_ROOT / "build" / "libjax_lax_scan_host.so"
OP_NAME = "jax_lax_scan"
KERNEL_NAME = "jax_lax_scan_kernel"
NPU_ARCH = "dav-2201"

ABI_VERSION = 2
ABI_TAG = "opx-scan-python-abi/2"

# 与 op_kernel/jax_lax_scan_tiling.h 保持一致
SCAN_BODY_ADD = 1
SCAN_BODY_PYTREE = 2
SCAN_BODY_QUAD = 3
SCAN_BODY_PYTREE_XS = 7
SCAN_BODY_XS_NONE = 8
BODY_SO = (SCAN_BODY_ADD, SCAN_BODY_PYTREE, SCAN_BODY_QUAD, SCAN_BODY_PYTREE_XS, SCAN_BODY_XS_NONE)
BODY_REJECT = (4, 5, 6)

SCAN_DTYPE_F32 = 0
SCAN_DTYPE_I32 = 1

REJECT_CODE = "carry_shape_mismatch"
UNSUPPORTED_CODE = "UNSUPPORTED:autodiff_unsupported"
TRACE_REASON = "so_no_autodiff"

_RETURN_CODES = {
    0: "OK",
    1: "device_error",
    2: "invalid_argument",
    3: REJECT_CODE,
    4: "autodiff_unsupported",
}


class JaxLaxScanArgs(ctypes.Structure):
    """与 op_host/jax_lax_scan_abi.h 的 jax_lax_scan_args_t 逐字段对应（布局版本 2）。"""

    _fields_ = [
        ("body_id", ctypes.c_int32),
        ("reverse", ctypes.c_int32),
        ("length", ctypes.c_int64),
        ("n_carry", ctypes.c_int32),
        ("n_xs", ctypes.c_int32),
        ("carry_dtype", ctypes.c_int32 * 8),
        ("xs_dtype", ctypes.c_int32 * 8),
        ("carry_in", ctypes.c_void_p * 8),
        ("carry_out", ctypes.c_void_p * 8),
        ("carry_numel", ctypes.c_int64 * 8),
        ("xs", ctypes.c_void_p * 8),
        ("xs_numel", ctypes.c_int64 * 8),
        ("ys_out", ctypes.c_void_p),
        ("ys_numel", ctypes.c_int64),
        ("ys_dtype", ctypes.c_int32),
        ("device_id", ctypes.c_int32),
        ("err", ctypes.c_void_p),
        ("err_len", ctypes.c_int32),
        ("abi_version", ctypes.c_int32),
        ("total_length", ctypes.c_int64),
        ("step_start", ctypes.c_int64),
        ("step_count", ctypes.c_int64),
    ]


# 布局自检：ctypes 与 C 结构体必须同尺寸（472 = 59*8），不一致说明 ABI 头被改过而 Python 没跟上。
_ARGS_EXPECTED_SIZE = 472
if ctypes.sizeof(JaxLaxScanArgs) != _ARGS_EXPECTED_SIZE:  # pragma: no cover - 只有 ABI 漂移才触发
    raise RuntimeError(
        f"JaxLaxScanArgs 布局漂移：ctypes={ctypes.sizeof(JaxLaxScanArgs)} != {_ARGS_EXPECTED_SIZE}"
    )


_last_dispatch: dict = {
    "op": OP_NAME,
    "mode": None,
    "path": None,
    "exec_mode": None,
    "body_id": None,
    "abi_calls": 0,
    "family": None,
    "reason": None,
    "backend": None,
}

_LIB = None
_LIB_FINGERPRINT = None


# ---------------------------------------------------------------------------
# 共享库加载
# ---------------------------------------------------------------------------


def library_fingerprint() -> dict:
    """返回运行期实际加载的 .so 指纹（绝对路径 / size / sha256 / abi）。

    代码检视 P-03：exe 只是薄壳，kernel 与 C ABI 的载体是 .so。构建日志由
    CMakeLists.txt 的 POST_BUILD 打印同一组指纹，这里打印运行期加载值，二者可逐字比对。
    """
    _load_lib()
    return dict(_LIB_FINGERPRINT)


def _load_lib():
    global _LIB, _LIB_FINGERPRINT
    if _LIB is not None:
        return _LIB
    if not LIB_PATH.is_file():
        raise RuntimeError(f"缺少 Python 接口共享库：{LIB_PATH}（先跑 build.sh）")
    raw = LIB_PATH.read_bytes()
    fingerprint = {
        "path": str(LIB_PATH.resolve()),
        "size_bytes": len(raw),
        "sha256": hashlib.sha256(raw).hexdigest(),
    }
    lib = ctypes.CDLL(str(LIB_PATH))

    lib.jax_lax_scan_run.argtypes = [ctypes.POINTER(JaxLaxScanArgs)]
    lib.jax_lax_scan_run.restype = ctypes.c_int
    lib.jax_lax_scan_abi_version.argtypes = []
    lib.jax_lax_scan_abi_version.restype = ctypes.c_char_p
    lib.jax_lax_scan_acl_init.argtypes = []
    lib.jax_lax_scan_acl_init.restype = ctypes.c_int
    lib.jax_lax_scan_malloc.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_int64]
    lib.jax_lax_scan_malloc.restype = ctypes.c_int
    lib.jax_lax_scan_free.argtypes = [ctypes.c_void_p]
    lib.jax_lax_scan_free.restype = ctypes.c_int
    lib.jax_lax_scan_h2d.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int64]
    lib.jax_lax_scan_h2d.restype = ctypes.c_int
    lib.jax_lax_scan_d2h.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int64]
    lib.jax_lax_scan_d2h.restype = ctypes.c_int

    tag = lib.jax_lax_scan_abi_version().decode("utf-8")
    if tag != ABI_TAG:
        raise RuntimeError(f"共享库 ABI 版本不匹配：{tag} != {ABI_TAG}")
    fingerprint["abi"] = tag
    _LIB_FINGERPRINT = fingerprint
    # 用 stderr 输出协议行之外的诊断：stdout 只放 EXEC/OK/ERROR，避免污染 runner 的标记行解析。
    print(
        "[jax_lax_scan] SO path={path} size_bytes={size_bytes} sha256={sha256} abi={abi}".format(
            **fingerprint
        ),
        file=sys.stderr,
    )
    _LIB = lib
    return lib


def abi_version() -> str:
    """返回共享库自报的 ABI 版本（供构建后自检与 README 对齐）。"""
    return _load_lib().jax_lax_scan_abi_version().decode("utf-8")


# ---------------------------------------------------------------------------
# 族识别 / body 识别（鸭子类型，不依赖 import torch 或 jax）
# ---------------------------------------------------------------------------


def _contig(array):
    """0-d 叶子必须保持 rank-0（np.ascontiguousarray 会把 0-d 变成 shape=(1,)）。"""
    import numpy as np

    array = np.asarray(array)
    return array if array.ndim == 0 else np.ascontiguousarray(array)


def _type_module(value) -> str:
    return type(value).__module__.split(".")[0]


def _leaf_kind(leaf) -> str:
    if callable(getattr(leaf, "data_ptr", None)) and getattr(leaf, "device", None) is not None:
        return "torch"
    if callable(getattr(leaf, "devices", None)):
        return "jax"
    if _type_module(leaf) == "jax":
        return "jax"
    if _type_module(leaf) == "numpy":
        return "numpy"
    return "other"


def _flatten(value, prefix: str = ""):
    """把 pytree 展开成 [(path, leaf)]；与 fixtures.flatten_with_paths 同口径。"""
    if isinstance(value, tuple):
        out = []
        for index, item in enumerate(value):
            key = f"{prefix}/{index}" if prefix else str(index)
            out.extend(_flatten(item, key))
        return out
    if isinstance(value, list):
        out = []
        for index, item in enumerate(value):
            key = f"{prefix}/{index}" if prefix else str(index)
            out.extend(_flatten(item, key))
        return out
    if isinstance(value, dict):
        out = []
        for key, item in value.items():
            sub = f"{prefix}/{key}" if prefix else str(key)
            out.extend(_flatten(item, sub))
        return out
    return [(prefix, value)]


def _family_of(init, xs) -> str:
    leaves = [leaf for _p, leaf in _flatten(init)]
    if xs is not None:
        leaves += [leaf for _p, leaf in _flatten(xs)]
    kinds = {_leaf_kind(leaf) for leaf in leaves}
    if not kinds:
        return "unknown"
    if kinds == {"torch"}:
        return "torch"
    if kinds == {"jax"}:
        return "jax"
    if kinds == {"numpy"}:
        return "numpy"
    return "+".join(sorted(kinds))


def _body_id_of(f) -> int | None:
    if not callable(f):
        return None
    body_id = getattr(f, "body_id", None)
    if isinstance(body_id, int) and 1 <= body_id <= 8:
        return body_id
    name = getattr(f, "__name__", "")
    return _BODY_NAME_TO_ID.get(name)


_BODY_NAME_TO_ID = {
    "b1_add": 1,
    "b3_pytree": 2,
    "b4_quad": 3,
    "b5_bad_shape": 4,
    "b6_bad_dtype": 5,
    "b7_bad_struct": 6,
    "pytree_xs_add_scale": 7,
    "none_input_increment": 8,
}


def _has_tracer(init, xs) -> bool:
    leaves = [leaf for _p, leaf in _flatten(init)]
    if xs is not None:
        leaves += [leaf for _p, leaf in _flatten(xs)]
    for leaf in leaves:
        if _type_module(leaf) == "jax" and "Tracer" in type(leaf).__name__:
            return True
    return False


def _tree_leading_size(xs) -> int | None:
    if xs is None:
        return None
    sizes = set()
    for _path, leaf in _flatten(xs):
        shape = getattr(leaf, "shape", None)
        if shape is None or len(shape) == 0:
            raise ValueError("invalid_argument: xs 的叶子必须至少 1 维（前导轴 = length）")
        sizes.add(int(shape[0]))
    if len(sizes) != 1:
        raise ValueError(f"length_mismatch: xs 各叶子的前导轴不一致：{sorted(sizes)}")
    return sizes.pop()


def _resolve_length(init, xs, length) -> int:
    leading = _tree_leading_size(xs)
    if length is None:
        if leading is None:
            raise ValueError("length_mismatch: xs=None 时必须显式给出 length")
        return leading
    length = int(length)
    if length < 1:
        raise ValueError(f"length_mismatch: length={length} 必须 >= 1")
    if leading is not None and leading != length:
        raise ValueError(f"length_mismatch: length={length} 与 xs 前导轴 {leading} 不一致")
    return length


def _set_dispatch(**fields) -> dict:
    _last_dispatch.update(fields)
    return dict(_last_dispatch)


# ---------------------------------------------------------------------------
# 路由（可单测；不执行 device）
# ---------------------------------------------------------------------------


def _plan_dispatch(f, init, xs, length=None, reverse=False) -> dict:
    """按设计 §1.3.4 返回路由计划；不执行 device、不修改 ``_last_dispatch``。"""
    del reverse  # reverse 不改路由，只改 kernel 取步方向
    body_id = _body_id_of(f)
    family = _family_of(init, xs)
    if body_id in BODY_REJECT:
        return {
            "op": OP_NAME,
            "path": "reject",
            "exec_mode": "reject",
            "body_id": body_id,
            "abi_calls": 0,
            "family": family,
            "reason": REJECT_CODE,
            "backend": None,
        }
    tracing = _has_tracer(init, xs)
    if body_id in BODY_SO and not tracing and family in ("torch", "numpy"):
        return {
            "op": OP_NAME,
            "path": "so",
            "exec_mode": "stepwise",
            "body_id": body_id,
            "abi_calls": None,  # 由调用方按 eager/jit 决定（length / 1）
            "family": family,
            "reason": None,
            "backend": None,
        }
    if body_id in BODY_SO and not tracing and family == "jax":
        # 具体 JAX 数组：trace 路由（.so 不接收 JAX 设备指针），reason 记录为“无 .so AD”
        return {
            "op": OP_NAME,
            "path": "trace",
            "exec_mode": "jax_jit",
            "body_id": body_id,
            "abi_calls": 0,
            "family": family,
            "reason": TRACE_REASON,
            "backend": None,
        }
    return {
        "op": OP_NAME,
        "path": "trace",
        "exec_mode": "jax_eager",
        "body_id": body_id,
        "abi_calls": 0,
        "family": family,
        "reason": TRACE_REASON if tracing else "unregistered_body",
        "backend": None,
    }


# ---------------------------------------------------------------------------
# .so 调用（torch / numpy 两族共用同一 ABI）
# ---------------------------------------------------------------------------


def _dtype_code(dtype_name) -> int:
    if dtype_name in ("float32", "torch.float32"):
        return SCAN_DTYPE_F32
    if dtype_name in ("int32", "torch.int32"):
        return SCAN_DTYPE_I32
    raise ValueError(f"dtype_mismatch: 只支持 float32/int32，收到 {dtype_name}")


def _run_so(body_id, carry_in, carry_out, xs_ptrs, length, reverse, device_id, ys_ptr,
            ys_numel, step_start, step_count):
    """一次 ``jax_lax_scan_run``；返回 (rc, detail)。"""
    lib = _load_lib()
    args = JaxLaxScanArgs()
    args.body_id = int(body_id)
    args.reverse = 1 if reverse else 0
    args.length = int(length)
    args.n_carry = len(carry_in)
    args.n_xs = len(xs_ptrs)
    args.ys_dtype = SCAN_DTYPE_F32
    args.device_id = int(device_id)
    args.abi_version = ABI_VERSION
    args.total_length = int(length)
    args.step_start = int(step_start)
    args.step_count = int(step_count)
    args.ys_out = ctypes.c_void_p(ys_ptr)
    args.ys_numel = int(ys_numel)
    for index, (src, dst) in enumerate(zip(carry_in, carry_out)):
        args.carry_dtype[index] = int(src.dtype_code)
        args.carry_numel[index] = int(src.numel)
        args.carry_in[index] = ctypes.c_void_p(int(src.ptr))
        args.carry_out[index] = ctypes.c_void_p(int(dst.ptr))
    for index, item in enumerate(xs_ptrs):
        args.xs_dtype[index] = int(item.dtype_code)
        args.xs_numel[index] = int(item.numel)
        args.xs[index] = ctypes.c_void_p(int(item.ptr))
    err = ctypes.create_string_buffer(256)
    args.err = ctypes.cast(err, ctypes.c_void_p)
    args.err_len = 256
    rc = int(lib.jax_lax_scan_run(ctypes.byref(args)))
    detail = err.value.decode("utf-8", "replace") if err.value else ""
    return rc, detail


class _DevLeaf:
    """numpy 路径用：一个叶子在 device 上的一份缓冲。"""

    __slots__ = ("ptr", "bytes", "dtype", "shape", "numel", "dtype_code")

    def __init__(self, array):
        import numpy as np

        self.dtype = np.dtype(str(array.dtype))
        self.shape = tuple(int(v) for v in array.shape)
        self.numel = int(array.size)
        self.bytes = int(array.nbytes)
        self.dtype_code = _dtype_code(str(self.dtype))
        self.ptr = 0


def _dev_alloc(lib, array):
    import numpy as np

    leaf = _DevLeaf(_contig(array))
    ptr = ctypes.c_void_p()
    if lib.jax_lax_scan_malloc(ctypes.byref(ptr), leaf.bytes) != 0 or not ptr.value:
        raise RuntimeError(f"device_error: aclrtMalloc({leaf.bytes}B) 失败")
    leaf.ptr = int(ptr.value)
    return leaf


def _dev_upload(lib, leaf, array):
    import numpy as np

    host = _contig(array)
    if lib.jax_lax_scan_h2d(ctypes.c_void_p(leaf.ptr), ctypes.c_void_p(host.ctypes.data),
                            leaf.bytes) != 0:
        raise RuntimeError("device_error: aclrtMemcpy H2D 失败")


def _dev_download(lib, leaf):
    import numpy as np

    host = np.empty(leaf.shape, dtype=leaf.dtype)
    if lib.jax_lax_scan_d2h(ctypes.c_void_p(host.ctypes.data), ctypes.c_void_p(leaf.ptr),
                            leaf.bytes) != 0:
        raise RuntimeError("device_error: aclrtMemcpy D2H 失败")
    return host


def _dev_free(lib, leaf):
    if leaf is not None and leaf.ptr:
        lib.jax_lax_scan_free(ctypes.c_void_p(leaf.ptr))
        leaf.ptr = 0


def _run_so_numpy(body_id, carry_arrays, xs_arrays, length, reverse, compiled, device_id=0):
    """numpy 路径：host driver（CLI）用，返回 (carry_out_arrays, ys)。"""
    import numpy as np

    lib = _load_lib()
    if lib.jax_lax_scan_acl_init() != 0:
        raise RuntimeError("device_error: aclInit 失败")
    if lib.jax_lax_scan_run is None:  # pragma: no cover - 防御性
        raise RuntimeError("device_error: 共享库缺少 jax_lax_scan_run")

    carry_arrays = [_contig(item) for item in carry_arrays]
    xs_arrays = [_contig(item) for item in xs_arrays]
    float_array = next((item for item in carry_arrays if item.dtype == np.float32), None)
    if float_array is None:
        raise ValueError("carry_shape_mismatch: carry 缺少 float32 累加器叶子")
    ys_shape = (int(length),) + float_array.shape
    ys = np.zeros(ys_shape, dtype=np.float32)

    pinned = []  # 持有所有 device 叶子，避免提前释放
    src = []  # carry 输入缓冲（每叶子两份轮换）: [[a, b], ...]
    xs_dev = []
    ys_dev = None
    try:
        for array in carry_arrays:
            first = _dev_alloc(lib, array)
            _dev_upload(lib, first, array)
            second = _dev_alloc(lib, array)
            pinned.extend([first, second])
            src.append([first, second])
        for array in xs_arrays:
            leaf = _dev_alloc(lib, array)
            _dev_upload(lib, leaf, array)
            pinned.append(leaf)
            xs_dev.append(leaf)
        ys_dev = _dev_alloc(lib, ys)
        pinned.append(ys_dev)

        calls = int(length) if (not compiled) else 1
        state = 0  # 当前 carry 结果所在缓冲下标
        for call in range(calls):
            if compiled:
                start, count = 0, int(length)
            else:
                start, count = call, 1
            carry_in = [pair[state] for pair in src]
            carry_out = [pair[1 - state] for pair in src]
            rc, detail = _run_so(
                body_id, carry_in, carry_out, xs_dev, length, reverse, device_id,
                ys_dev.ptr, ys_dev.numel, start, count,
            )
            if rc != 0:
                raise _SoError(rc, detail)
            state = 1 - state
        out = [_dev_download(lib, pair[state]) for pair in src]
        return out, _dev_download(lib, ys_dev)
    finally:
        for leaf in pinned:
            _dev_free(lib, leaf)


class _SoError(RuntimeError):
    def __init__(self, rc, detail):
        self.rc = rc
        self.detail = detail
        super().__init__(f"{_RETURN_CODES.get(rc, f'UNKNOWN({rc})')}: {detail}")


# ---------------------------------------------------------------------------
# JAX trace 路径（AD 与未注册 callable）
# ---------------------------------------------------------------------------


def _require_npu_backend(jax):
    backend = jax.default_backend()
    if backend == "cpu":
        devices = [str(d) for d in jax.devices()]
        raise RuntimeError(
            f"device_error: 候选 JAX 后端是 CPU（{backend}/{devices}）：不允许用 CPU 结果冒充 NPU"
        )
    return backend, [str(d) for d in jax.devices()]


def _run_trace(f, init, xs, length, reverse, compiled):
    import jax

    backend, devices = _require_npu_backend(jax)
    if compiled:
        if xs is None:
            fn = jax.jit(lambda carry: jax.lax.scan(f, carry, None, length=length,
                                                    reverse=reverse))
            final_carry, ys = fn(init)
        else:
            fn = jax.jit(lambda carry, seq: jax.lax.scan(f, carry, seq, length=length,
                                                         reverse=reverse))
            final_carry, ys = fn(init, xs)
        exec_mode = "jax_jit"
    else:
        final_carry, ys = jax.lax.scan(f, init, xs, length=length, reverse=reverse)
        exec_mode = "jax_eager"
    final_carry = jax.tree_util.tree_map(lambda leaf: leaf.block_until_ready()
                                         if hasattr(leaf, "block_until_ready") else leaf,
                                         final_carry)
    if hasattr(ys, "block_until_ready"):
        ys = ys.block_until_ready()
    return final_carry, ys, exec_mode, backend, devices


# ---------------------------------------------------------------------------
# torch 路径（交付接口默认族；镜像缺 torch_npu 时由调用方改走 trace/CLI）
# ---------------------------------------------------------------------------


def _run_so_torch(body_id, init, xs, length, reverse, compiled):
    import torch  # noqa: PLC0415 —— 只在 torch 路径导入

    carry_leaves = [leaf for _p, leaf in _flatten(init)]
    if not carry_leaves:
        raise ValueError("carry_shape_mismatch: carry 不能为空")
    for leaf in carry_leaves:
        if not leaf.is_contiguous():
            raise ValueError("invalid_argument: carry 叶子必须是连续张量")
    xs_leaves = [] if xs is None else [leaf for _p, leaf in _flatten(xs)]
    for leaf in xs_leaves:
        if not leaf.is_contiguous():
            raise ValueError("invalid_argument: xs 叶子必须是连续张量")
    if xs_leaves:
        leading = {int(leaf.shape[0]) for leaf in xs_leaves}
        if len(leading) != 1 or leading.pop() != int(length):
            raise ValueError("length_mismatch: xs 叶子前导轴必须等于 length")

    # 设备类型/一致性校验必须先于任何 data_ptr()：CPU 张量的 data_ptr 不是设备地址，
    # 直接传给 kernel 会造成非法设备访问（代码检视 P-05）。
    device_set = {leaf.device for leaf in carry_leaves + xs_leaves}
    for item in device_set:
        if str(item.type) != "npu":
            raise ValueError(f"invalid_argument: 只接受 npu 设备上的 torch 张量，收到 {item}")
    if len(device_set) != 1:
        raise ValueError(f"invalid_argument: carry/xs 张量必须在同一设备，收到 {sorted(str(d) for d in device_set)}")

    # 公开 torch 接口的 AD：`.so` 未注册 torch autograd，JAX 也无法穿过 torch→jax 的 DLPack 边界。
    # 必须在任何 kernel 启动前显式拒绝，不能返回一个没有 grad_fn 的结果冒充“支持”（设计 §1.3.5/§3.5）。
    differentiable = [leaf for leaf in carry_leaves + xs_leaves
                      if bool(getattr(leaf, "requires_grad", False))]
    if differentiable:
        raise ValueError(
            "UNSUPPORTED:autodiff_unsupported: 公开 torch 接口的 JVP/VJP 无法穿过 .so 的 "
            "autograd/DLPack 边界；本批 AD 用例使用测试侧 jax.Array 走 JAX trace，"
            "不得用 jax-family 结果冒充 torch 接口 AD 通过。"
        )

    device = carry_leaves[0].device
    device_id = int(device.index) if device.index is not None else 0
    float_leaves = [leaf for leaf in carry_leaves if leaf.dtype == torch.float32]
    if not float_leaves:
        raise ValueError("carry_shape_mismatch: carry 缺少 float32 累加器叶子")
    ys_shape = (int(length),) + tuple(int(v) for v in float_leaves[0].shape)
    ys = torch.empty(ys_shape, dtype=torch.float32, device=device)

    # 输入 carry 只读。不能把 init 本身当作 ping-pong 的其中一块
    # 输出缓冲：eager 且 length>=2 时那会在偶数步写回 init，违反
    # Python 交付接口“返回新 tensor”且会悄然篡改调用方输入。
    # 因此为每个叶子独立分配两块输出缓冲，首步从 init
    # 读、之后只在两块新缓冲间交替。
    buffers = [[torch.empty_like(leaf), torch.empty_like(leaf)] for leaf in carry_leaves]

    class _Leaf:
        __slots__ = ("ptr", "numel", "dtype_code")

        def __init__(self, tensor):
            self.ptr = int(tensor.data_ptr())
            self.numel = int(tensor.numel())
            self.dtype_code = _dtype_code(str(tensor.dtype))

    xs_wrap = [_Leaf(leaf) for leaf in xs_leaves]
    # 输入可能仍在 torch 当前 stream 上生产；.so 把 kernel 投到自建的 ACL stream，二者没有
    # 事件依赖。启动前同步调用方 stream，确保 .so 读到的输入已完成写入（P-05）。
    torch.npu.current_stream().synchronize()
    calls = int(length) if not compiled else 1
    current = carry_leaves
    abi_calls = 0
    for call in range(calls):
        start, count = (0, int(length)) if compiled else (call, 1)
        target = [pair[call % 2] for pair in buffers]
        carry_in = [_Leaf(leaf) for leaf in current]
        carry_out = [_Leaf(leaf) for leaf in target]
        rc, detail = _run_so(
            body_id, carry_in, carry_out, xs_wrap, length, reverse, device_id,
            int(ys.data_ptr()), int(ys.numel()), start, count,
        )
        abi_calls += 1
        if rc != 0:
            raise _SoError(rc, detail)
        current = target

    final_leaves = current
    # 还原 carry 的 pytree 结构（叶子顺序与 _.flatten 一致）
    final_carry = _rebuild(init, iter(final_leaves))
    return final_carry, ys, abi_calls


def _rebuild(template, leaves):
    if isinstance(template, tuple):
        return tuple(_rebuild(item, leaves) for item in template)
    if isinstance(template, list):
        return [_rebuild(item, leaves) for item in template]
    if isinstance(template, dict):
        return {key: _rebuild(item, leaves) for key, item in template.items()}
    return next(leaves)


# ---------------------------------------------------------------------------
# 公开入口
# ---------------------------------------------------------------------------


def _execute(f, init, xs, length, reverse, compiled):
    if not callable(f):
        raise TypeError("invalid_argument: f 必须是 callable：f(carry, x_t) -> (new_carry, y)")
    mode = "jit" if compiled else "eager"
    length = _resolve_length(init, xs, length)
    plan = _plan_dispatch(f, init, xs, length, reverse)
    body_id = plan["body_id"]
    family = plan["family"]

    if plan["path"] == "reject":
        _set_dispatch(mode=mode, path="reject", exec_mode="reject", body_id=body_id,
                      abi_calls=0, family=family, reason=REJECT_CODE, backend=None)
        raise ValueError(
            f"{REJECT_CODE}: body_id={body_id} 在 trace 期改变了 carry 的结构/shape/dtype，"
            "违反 carry 不变式（device 执行前拒绝，abi_calls=0）"
        )

    if plan["path"] == "so":
        if family == "torch":
            final_carry, ys, abi_calls = _run_so_torch(body_id, init, xs, length, reverse, compiled)
            _set_dispatch(mode=mode, path="so",
                          exec_mode="compiled_whole_loop" if compiled else "stepwise",
                          body_id=body_id, abi_calls=abi_calls, family=family, reason=None,
                          backend="npu")
            return final_carry, ys
        if family == "numpy":
            carry_leaves = [leaf for _p, leaf in _flatten(init)]
            xs_leaves = [] if xs is None else [leaf for _p, leaf in _flatten(xs)]
            carry_out, ys = _run_so_numpy(body_id, carry_leaves, xs_leaves, length, reverse, compiled)
            _set_dispatch(mode=mode, path="so",
                          exec_mode="compiled_whole_loop" if compiled else "stepwise",
                          body_id=body_id, abi_calls=int(length) if not compiled else 1,
                          family=family, reason=None, backend="npu")
            return _rebuild(init, iter(carry_out)), ys
        raise ValueError(f"unsupported_body: 不支持的张量族 {family}")

    try:
        final_carry, ys, exec_mode, backend, devices = _run_trace(
            f, init, xs, length, reverse, compiled
        )
    except Exception:
        # 后端明确不支持 scan/AD 时也必须留下可机判的路由证据；
        # 否则上层只看到原始异常，无法区分「显式 unsupported」与普通失败。
        try:
            import jax
            failed_backend = jax.default_backend()
        except Exception:  # noqa: BLE001
            failed_backend = "npu" if os.environ.get("JAX_PLATFORMS") == "npu" else None
        _set_dispatch(mode=mode, path="trace", exec_mode="trace", body_id=body_id,
                      abi_calls=0, family=family, reason=TRACE_REASON,
                      backend=failed_backend)
        raise
    _set_dispatch(mode=mode, path="trace", exec_mode=exec_mode, body_id=body_id, abi_calls=0,
                  family=family, reason=TRACE_REASON, backend=backend)
    return final_carry, ys


def scan(f, init, xs=None, length=None, reverse=False, *, unroll=1):
    """静态循环逐步执行（``exec_mode="stepwise"``，已注册 body 时 ``abi_calls == length``）。"""
    if unroll != 1:
        raise ValueError("invalid_argument: 本批只支持 unroll=1")
    return _execute(f, init, xs, length, reverse, compiled=False)


def scan_compiled(f, init, xs=None, length=None, reverse=False, *, unroll=1):
    """编译态整循环入口（``exec_mode="compiled_whole_loop"``，已注册 body 时 ``abi_calls == 1``）。

    这是测试/门禁的编译态入口，不是第二个公开 API；README 必须写明该边界。
    """
    if unroll != 1:
        raise ValueError("invalid_argument: 本批只支持 unroll=1")
    return _execute(f, init, xs, length, reverse, compiled=True)


__all__ = [
    "scan",
    "scan_compiled",
    "abi_version",
    "JaxLaxScanArgs",
    "_last_dispatch",
    "_plan_dispatch",
    "OP_NAME",
    "KERNEL_NAME",
    "NPU_ARCH",
    "REJECT_CODE",
    "UNSUPPORTED_CODE",
    "TRACE_REASON",
]
