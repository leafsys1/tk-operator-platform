# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""jax_lax_associative_scan · Python 调用接口（ctypes 封装 host C ABI）。

公开入口与 ``jax.lax.associative_scan`` 对齐：

    associative_scan(fn, elems, reverse=False, axis=0) -> pytree
    associative_scan_compiled(fn, elems, reverse=False, axis=0) -> pytree

契约（``0-黑盒测试设计.md`` §1.3）：

* **已注册 combine**（``combine_sum``/``combine_max``/``combine_tree_add``，按对象身份或
  ``fn.combine_id`` 解析）+ **numpy pytree** → 走 ``libjax_lax_associative_scan_host.so``
  的 AscendC kernel：
  ``associative_scan`` 逐叶子（``exec_mode="per_leaf"``，``abi_calls == 叶子数``），
  ``associative_scan_compiled`` 整树（``exec_mode="whole_tree"``，``abi_calls == 1``）。
* **JAX 数组 / JAX tracer（含 JVP/VJP）** → 走 JAX trace（``path="trace"``，``abi_calls=0``）。
  这是 AD 的显式路径：kernel 不接收 PyObject/JAX tracer；trace 路径在 NPU 后端上执行
  JAX 自动微分，不冒充 kernel AD。
* **非法输入**（dtype/axis/结构/连续性）→ device 执行前抛稳定 code（``path="reject"``，
  ``abi_calls=0``）。

C ABI 结构体见 ``op_host/jax_lax_associative_scan_abi.h``（布局版本 1）；本模块只做参数校验、
设备缓冲管理与调用编排，**不做任何数值计算**（编码规则 A1）。
"""
from __future__ import annotations

import ctypes
import hashlib
import json
import numbers
import sys
from pathlib import Path

OP_ROOT = Path(__file__).resolve().parents[1]
LIB_PATH = OP_ROOT / "build" / "libjax_lax_associative_scan_host.so"
OP_NAME = "jax_lax_associative_scan"
KERNEL_NAME = "jax_lax_associative_scan_kernel"
ARCH = "dav-2201"

ABI_VERSION = 1
ABI_TAG = "opx-assoc-scan-host-abi/1"

#: 单次 launch 的叶子数上限，与 op_kernel/jax_lax_associative_scan_tiling.h 的
#: ``ASSOC_MAX_LEAVES`` 一致。只约束一次 ABI 调用：``per_leaf`` 每次只带 1 个叶子，
#: ``whole_tree`` 一次带整树叶子，故该上限只对 ``associative_scan_compiled`` 生效。
MAX_LEAVES = 8

# 与 op_kernel/jax_lax_associative_scan_tiling.h 保持一致
COMBINE_SUM = 1
COMBINE_MAX = 2
COMBINE_TREE_ADD = 3
COMBINE_IDS = {"sum": COMBINE_SUM, "max": COMBINE_MAX, "pytree_add": COMBINE_TREE_ADD}
COMBINE_NAMES = {value: key for key, value in COMBINE_IDS.items()}

REJECT_CODES = ("dtype_mismatch", "invalid_axis", "invalid_argument")
UNSUPPORTED_CODE = "UNSUPPORTED:autodiff_unsupported"

_RETURN_CODES = {
    0: "OK",
    1: "device_error",
    2: "invalid_argument",
    3: "rejected",
    4: "autodiff_unsupported",
}

#: 兼容 jax_lax_scan 的同名字段命名（测试与 README 引用 TRACE_REASON）
TRACE_REASON = "so_no_autodiff"

_DISPATCH_DEFAULTS = {
    "op": OP_NAME,
    "mode": None,
    "path": None,
    "exec_mode": None,
    "combine_id": None,
    "abi_calls": 0,
    "family": None,
    "backend": None,
    "device_id": 0,
    "plan_id": None,
    "plan_cache": None,
    "reason": None,
}

_last_dispatch: dict = dict(_DISPATCH_DEFAULTS)

_PLAN_CACHE: set = set()
_LIB = None
_LIB_FINGERPRINT = None


class JaxLaxAssociativeScanArgs(ctypes.Structure):
    """与 op_host/jax_lax_associative_scan_abi.h 逐字段对应（布局版本 1）。"""

    _fields_ = [
        ("combine_id", ctypes.c_int32),
        ("reverse", ctypes.c_int32),
        ("n_leaves", ctypes.c_int32),
        ("dtype", ctypes.c_int32 * MAX_LEAVES),
        ("outer", ctypes.c_int64 * MAX_LEAVES),
        ("length", ctypes.c_int64 * MAX_LEAVES),
        ("inner", ctypes.c_int64 * MAX_LEAVES),
        ("in_ptr", ctypes.c_void_p * MAX_LEAVES),
        ("out_ptr", ctypes.c_void_p * MAX_LEAVES),
        ("numel", ctypes.c_int64 * MAX_LEAVES),
        ("device_id", ctypes.c_int32),
        ("err", ctypes.c_void_p),
        ("err_len", ctypes.c_int32),
        ("abi_version", ctypes.c_int32),
    ]


# 布局自检：ctypes 与 C 结构体必须同尺寸（456 = 57*8），不一致说明 ABI 头被改过而 Python 没跟上。
_ARGS_EXPECTED_SIZE = 456
if ctypes.sizeof(JaxLaxAssociativeScanArgs) != _ARGS_EXPECTED_SIZE:  # pragma: no cover
    raise RuntimeError(
        f"JaxLaxAssociativeScanArgs 布局漂移：ctypes={ctypes.sizeof(JaxLaxAssociativeScanArgs)} "
        f"!= {_ARGS_EXPECTED_SIZE}"
    )


class _AssocReject(ValueError):
    """预校验期拒绝；异常文本以稳定 code 开头（Python / CLI / 报告共用）。"""

    def __init__(self, code: str, detail: str = ""):
        self.code = code
        self.detail = detail
        super().__init__(f"{code}: {detail}" if detail else code)


class _SoError(RuntimeError):
    def __init__(self, rc: int, detail: str):
        self.rc = int(rc)
        self.detail = detail
        super().__init__(f"{_RETURN_CODES.get(rc, f'UNKNOWN({rc})')}: {detail}")


# ---------------------------------------------------------------------------
# 共享库加载
# ---------------------------------------------------------------------------


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

    lib.jax_lax_associative_scan_run.argtypes = [ctypes.POINTER(JaxLaxAssociativeScanArgs)]
    lib.jax_lax_associative_scan_run.restype = ctypes.c_int
    lib.jax_lax_associative_scan_abi_version.argtypes = []
    lib.jax_lax_associative_scan_abi_version.restype = ctypes.c_char_p
    lib.jax_lax_associative_scan_acl_init.argtypes = []
    lib.jax_lax_associative_scan_acl_init.restype = ctypes.c_int
    lib.jax_lax_associative_scan_malloc.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_int64]
    lib.jax_lax_associative_scan_malloc.restype = ctypes.c_int
    lib.jax_lax_associative_scan_free.argtypes = [ctypes.c_void_p]
    lib.jax_lax_associative_scan_free.restype = ctypes.c_int
    lib.jax_lax_associative_scan_h2d.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int64]
    lib.jax_lax_associative_scan_h2d.restype = ctypes.c_int
    lib.jax_lax_associative_scan_d2h.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int64]
    lib.jax_lax_associative_scan_d2h.restype = ctypes.c_int

    tag = lib.jax_lax_associative_scan_abi_version().decode("utf-8")
    if tag != ABI_TAG:
        raise RuntimeError(f"共享库 ABI 版本不匹配：{tag} != {ABI_TAG}")
    fingerprint["abi"] = tag
    _LIB_FINGERPRINT = fingerprint
    # stdout 只放 EXEC/OK/ERROR 协议行；诊断走 stderr，避免污染 runner 的标记行解析。
    print(
        "[jax_lax_associative_scan] SO path={path} size_bytes={size_bytes} "
        "sha256={sha256} abi={abi}".format(**fingerprint),
        file=sys.stderr,
    )
    _LIB = lib
    return lib


def library_fingerprint() -> dict:
    """返回运行期实际加载的 .so 指纹（绝对路径 / size / sha256 / abi）。"""
    _load_lib()
    return dict(_LIB_FINGERPRINT)


def abi_version() -> str:
    """返回共享库自报的 ABI 版本（供构建后自检与 README 对齐）。"""
    return _load_lib().jax_lax_associative_scan_abi_version().decode("utf-8")


# ---------------------------------------------------------------------------
# pytree / dtype / 几何
# ---------------------------------------------------------------------------


def _flatten(value, prefix: str = ""):
    """把 pytree 展开成 [(path, leaf)]；dict 按插入序、tuple/list 按位置。"""
    if isinstance(value, dict):
        out = []
        for key, item in value.items():
            out.extend(_flatten(item, f"{prefix}/{key}" if prefix else str(key)))
        return out
    if isinstance(value, (tuple, list)):
        out = []
        for index, item in enumerate(value):
            out.extend(_flatten(item, f"{prefix}/{index}" if prefix else str(index)))
        return out
    return [(prefix, value)]


def _rebuild(template, leaves):
    if isinstance(template, dict):
        return {key: _rebuild(item, leaves) for key, item in template.items()}
    if isinstance(template, tuple):
        return tuple(_rebuild(item, leaves) for item in template)
    if isinstance(template, list):
        return [_rebuild(item, leaves) for item in template]
    return next(leaves)


def _tree_map(fn, tree):
    if isinstance(tree, dict):
        return {key: _tree_map(fn, item) for key, item in tree.items()}
    if isinstance(tree, tuple):
        return tuple(_tree_map(fn, item) for item in tree)
    if isinstance(tree, list):
        return [_tree_map(fn, item) for item in tree]
    return fn(tree)


def _shape_of(leaf):
    shape = getattr(leaf, "shape", None)
    if shape is None:
        raise _AssocReject("invalid_argument", f"叶子缺少 shape：{type(leaf).__name__}")
    return tuple(int(dim) for dim in shape)


def _dtype_name(leaf) -> str:
    dtype = getattr(leaf, "dtype", None)
    if dtype is None:
        return "unknown"
    return str(dtype)


def _leaf_kind(leaf) -> str:
    module = type(leaf).__module__.split(".")[0]
    if module in ("jax", "jaxlib"):
        return "jax"
    if module == "numpy":
        return "numpy"
    if callable(getattr(leaf, "devices", None)):
        return "jax"
    if callable(getattr(leaf, "data_ptr", None)):
        return "torch"
    return "other"


def _is_tracer(leaf) -> bool:
    return type(leaf).__module__.split(".")[0] in ("jax", "jaxlib") and "Tracer" in type(leaf).__name__


def _is_contiguous(leaf) -> bool:
    if _leaf_kind(leaf) != "numpy":
        return True
    flags = getattr(leaf, "flags", None)
    if flags is None:
        return True
    return bool(flags["C_CONTIGUOUS"])


def _geometry(shape, axis: int) -> dict:
    outer = 1
    for dim in shape[:axis]:
        outer *= int(dim)
    inner = 1
    for dim in shape[axis + 1:]:
        inner *= int(dim)
    return {"outer": int(outer), "length": int(shape[axis]), "inner": int(inner)}


def _combine_id_of(fn):
    if not callable(fn):
        return None
    combine_id = getattr(fn, "combine_id", None)
    if isinstance(combine_id, int) and not isinstance(combine_id, bool):
        if combine_id in (COMBINE_SUM, COMBINE_MAX, COMBINE_TREE_ADD):
            return combine_id
    name = getattr(fn, "__name__", "")
    return _COMBINE_NAME_TO_ID.get(name)


_COMBINE_NAME_TO_ID = {
    "combine_sum": COMBINE_SUM,
    "sum": COMBINE_SUM,
    "add": COMBINE_SUM,
    "combine_max": COMBINE_MAX,
    "max": COMBINE_MAX,
    "maximum": COMBINE_MAX,
    "combine_tree_add": COMBINE_TREE_ADD,
    "tree_add": COMBINE_TREE_ADD,
    "pytree_add": COMBINE_TREE_ADD,
}


def _set_dispatch(**fields) -> dict:
    _last_dispatch.clear()
    _last_dispatch.update(_DISPATCH_DEFAULTS)
    _last_dispatch.update(fields)
    return dict(_last_dispatch)


def _reject(code: str, detail: str = "", *, family=None, combine_id=None):
    _set_dispatch(path="reject", exec_mode="reject", abi_calls=0, combine_id=combine_id,
                  family=family, backend=None, reason=code)
    raise _AssocReject(code, detail)


# ---------------------------------------------------------------------------
# 参数分析（任何 device 操作之前）
# ---------------------------------------------------------------------------


def _analyze(fn, elems, axis, *, whole_tree=False):
    """校验输入并返回叶子 / 几何 / family / combine_id；非法输入抛 ``_AssocReject``。

    ``whole_tree=True`` 时额外检查整树叶子数不超过单次 launch 上限 ``MAX_LEAVES``：
    ``associative_scan_compiled`` 一次 ABI 调用带全部叶子，必须在任何 device 操作之前
    稳定拒绝；``per_leaf``（``associative_scan``）每次只带 1 个叶子，不受该上限约束。
    """
    leaves = _flatten(elems)
    if not leaves:
        _reject("invalid_argument", "elems 不能是空 pytree")

    kinds = {_leaf_kind(leaf) for _path, leaf in leaves}
    if len(kinds) != 1:
        _reject("invalid_argument", f"pytree 叶子类型不一致：{sorted(kinds)}")
    family = kinds.pop()
    if family == "torch":
        _reject("invalid_argument", "本批 family 只支持 numpy / jax；torch 路径未注册")
    if family == "other":
        _reject("invalid_argument", "elems 叶子必须是 numpy 或 jax 数组")

    combine_id = _combine_id_of(fn)
    if combine_id is None:
        _set_dispatch(path="trace", exec_mode="trace", abi_calls=0, family=family,
                      backend=None, reason="unsupported_combine_fn")
        raise ValueError(
            f"unsupported_combine_fn: 未注册 callable {getattr(fn, '__name__', fn)!r}；"
            f"本批支持 {sorted(COMBINE_IDS)}"
        )

    if whole_tree and len(leaves) > MAX_LEAVES:
        _reject("invalid_argument",
                f"whole_tree 单次 launch 最多 {MAX_LEAVES} 个叶子，收到 {len(leaves)}",
                family=family, combine_id=combine_id)

    if isinstance(axis, bool) or not isinstance(axis, numbers.Integral):
        _reject("invalid_axis", f"axis 必须是静态整数，收到 {axis!r}",
                family=family, combine_id=combine_id)
    axis = int(axis)

    shapes = []
    geoms = []
    lengths = set()
    for path, leaf in leaves:
        shape = _shape_of(leaf)
        shapes.append(shape)
        if len(shape) < 1:
            _reject("invalid_argument", f"叶子 {path!r} 是 rank-0 标量（本批要求 rank>=1）",
                    family=family, combine_id=combine_id)
        if axis < 0 or axis >= len(shape):
            _reject("invalid_axis", f"叶子 {path!r} rank={len(shape)}，axis={axis} 越界",
                    family=family, combine_id=combine_id)
        if _dtype_name(leaf) != "float32":
            _reject("dtype_mismatch", f"叶子 {path!r} dtype={_dtype_name(leaf)}，本批只支持 float32",
                    family=family, combine_id=combine_id)
        if not _is_contiguous(leaf):
            _reject("invalid_argument", f"叶子 {path!r} 不是 C-contiguous",
                    family=family, combine_id=combine_id)
        lengths.add(int(shape[axis]))
        geoms.append(_geometry(shape, axis))
    if len(lengths) != 1:
        _reject("invalid_argument", f"pytree 各叶子 shape[axis] 不一致：{sorted(lengths)}",
                family=family, combine_id=combine_id)

    return {
        "leaves": leaves,
        "shapes": shapes,
        "geometry": geoms,
        "family": family,
        "combine_id": combine_id,
    }


# ---------------------------------------------------------------------------
# .so 调用（numpy family）
# ---------------------------------------------------------------------------


def _contig(array):
    import numpy as np

    array = np.asarray(array)
    return array if array.ndim == 0 else np.ascontiguousarray(array)


class _DevLeaf:
    __slots__ = ("ptr", "bytes", "dtype", "shape", "numel")

    def __init__(self, array):
        import numpy as np

        self.dtype = np.dtype(str(array.dtype))
        self.shape = tuple(int(v) for v in array.shape)
        self.numel = int(array.size)
        self.bytes = int(array.nbytes)
        self.ptr = 0


def _dev_alloc(lib, array):
    leaf = _DevLeaf(_contig(array))
    ptr = ctypes.c_void_p()
    if lib.jax_lax_associative_scan_malloc(ctypes.byref(ptr), leaf.bytes) != 0 or not ptr.value:
        raise RuntimeError(f"device_error: aclrtMalloc({leaf.bytes}B) 失败")
    leaf.ptr = int(ptr.value)
    return leaf


def _dev_upload(lib, leaf, array):
    host = _contig(array)
    if lib.jax_lax_associative_scan_h2d(
            ctypes.c_void_p(leaf.ptr), ctypes.c_void_p(host.ctypes.data), leaf.bytes) != 0:
        raise RuntimeError("device_error: aclrtMemcpy H2D 失败")


def _dev_download(lib, leaf):
    import numpy as np

    host = np.empty(leaf.shape, dtype=leaf.dtype)
    if lib.jax_lax_associative_scan_d2h(
            ctypes.c_void_p(host.ctypes.data), ctypes.c_void_p(leaf.ptr), leaf.bytes) != 0:
        raise RuntimeError("device_error: aclrtMemcpy D2H 失败")
    return host


def _dev_free(lib, leaf):
    if leaf is not None and leaf.ptr:
        lib.jax_lax_associative_scan_free(ctypes.c_void_p(leaf.ptr))
        leaf.ptr = 0


def _run_once(lib, combine_id, ins, outs, geoms, reverse):
    args = JaxLaxAssociativeScanArgs()
    args.combine_id = int(combine_id)
    args.reverse = 1 if reverse else 0
    args.n_leaves = len(ins)
    args.device_id = 0
    args.abi_version = ABI_VERSION
    for index, (src, dst, geom) in enumerate(zip(ins, outs, geoms)):
        args.dtype[index] = 0  # float32
        args.outer[index] = int(geom["outer"])
        args.length[index] = int(geom["length"])
        args.inner[index] = int(geom["inner"])
        args.in_ptr[index] = ctypes.c_void_p(int(src.ptr))
        args.out_ptr[index] = ctypes.c_void_p(int(dst.ptr))
        args.numel[index] = int(src.numel)
    err = ctypes.create_string_buffer(256)
    args.err = ctypes.cast(err, ctypes.c_void_p)
    args.err_len = 256
    rc = int(lib.jax_lax_associative_scan_run(ctypes.byref(args)))
    detail = err.value.decode("utf-8", "replace") if err.value else ""
    return rc, detail


def _run_so(combine_id, arrays, reverse, axis, compiled):
    """在 NPU 上执行一次（per_leaf）或整树（whole_tree）；返回 (输出数组, abi_calls)。"""
    import numpy as np

    lib = _load_lib()
    if lib.jax_lax_associative_scan_acl_init() != 0:
        raise RuntimeError("device_error: aclInit 失败")

    geoms = [_geometry(tuple(int(v) for v in arr.shape), axis) for arr in arrays]
    pinned = []
    ins = []
    outs = []
    try:
        for array in arrays:
            host = _contig(array)
            src = _dev_alloc(lib, host)
            pinned.append(src)
            _dev_upload(lib, src, host)
            dst = _dev_alloc(lib, np.zeros_like(host))
            pinned.append(dst)
            ins.append(src)
            outs.append(dst)
        calls = 0
        if compiled:
            rc, detail = _run_once(lib, combine_id, ins, outs, geoms, reverse)
            if rc != 0:
                raise _SoError(rc, detail)
            calls = 1
        else:
            for index in range(len(arrays)):
                rc, detail = _run_once(lib, combine_id, [ins[index]], [outs[index]],
                                       [geoms[index]], reverse)
                if rc != 0:
                    raise _SoError(rc, detail)
                calls += 1
        return [_dev_download(lib, leaf) for leaf in outs], calls
    finally:
        for leaf in pinned:
            _dev_free(lib, leaf)


# ---------------------------------------------------------------------------
# JAX trace 路径（AD 与 jax family）
# ---------------------------------------------------------------------------


def _backend_name():
    try:
        import jax  # noqa: PLC0415

        return str(jax.default_backend())
    except Exception:  # noqa: BLE001
        return None


def _combine_jax(combine_id, left, right):
    import jax.numpy as jnp  # noqa: PLC0415

    if combine_id == COMBINE_MAX:
        return jnp.maximum(left, right)
    return left + right


def _scan_leaf_trace(x, combine_id, axis, reverse):
    """用 JAX 原语逐步展开前缀/后缀扫描（trace 路径，供 AD 与 jax family 使用）。

    不依赖 ``lax.associative_scan`` 在 xla-npu 后端上的 lowering：只使用 take/stack
    与 elementwise combine，JVP/VJP 由 JAX 自动微分直接穿过。数值口径与 kernel 相同：
    顺序折叠，允许与 JAX CPU oracle 的不同括号化（按 ops-precision-standard@1 判定）。
    """
    import jax.numpy as jnp  # noqa: PLC0415

    length = int(x.shape[axis])
    order = range(length - 1, -1, -1) if reverse else range(length)
    pieces = [None] * length
    carry = None
    for step in order:
        current = jnp.take(x, step, axis=axis)
        carry = current if carry is None else _combine_jax(combine_id, carry, current)
        pieces[step] = carry
    return jnp.stack(pieces, axis=axis)


#: 后端 runtime 错误里属于「真实不支持」的签名（AFIR / lowering / 未实现）。
#: 只有命中这些签名、且异常类型是 JAX 后端 runtime 错误时，才允许包装成
#: ``UNSUPPORTED:autodiff_unsupported``；实现侧 shape/编程异常必须原样抛出。
_UNSUPPORTED_SIGNATURES = (
    "failed to parse afir",
    "afir",
    "lowering",
    "not implemented",
    "unimplemented",
    "unsupported",
)


def _is_backend_unsupported(exc, backend):
    """判断异常是否来自非 CPU 后端的真实 unsupported/AFIR/lowering 失败。"""
    if backend is None or str(backend).lower() == "cpu":
        return False
    try:
        import jax  # noqa: PLC0415
    except Exception:  # noqa: BLE001
        return False
    runtime_errors = []
    for name in ("JaxRuntimeError", "XlaRuntimeError"):
        cls = getattr(getattr(jax, "errors", None), name, None)
        if isinstance(cls, type) and issubclass(cls, BaseException):
            runtime_errors.append(cls)
    if runtime_errors:
        is_backend_error = isinstance(exc, tuple(runtime_errors))
    else:  # pragma: no cover —— 兼容 JAX 版本未导出错误类时的保守回退
        is_backend_error = (
            type(exc).__module__.split(".")[0] in ("jax", "jaxlib")
            and "RuntimeError" in type(exc).__name__
        )
    if not is_backend_error:
        return False
    text = f"{type(exc).__name__}: {exc}".lower()
    return any(signature in text for signature in _UNSUPPORTED_SIGNATURES)


def _run_trace(elems, combine_id, reverse, axis):
    try:
        result = _tree_map(lambda leaf: _scan_leaf_trace(leaf, combine_id, axis, reverse), elems)
        return result
    except Exception as exc:  # noqa: BLE001
        backend = _backend_name()
        if _is_backend_unsupported(exc, backend):
            _set_dispatch(path="trace", exec_mode="trace", abi_calls=0, combine_id=combine_id,
                          family="jax", backend=backend, reason=UNSUPPORTED_CODE)
            raise RuntimeError(f"{UNSUPPORTED_CODE}: {exc}") from exc
        # 非后端 unsupported 签名：保留原始异常类型/文本，不伪装成 AD 不支持。
        _set_dispatch(path="trace", exec_mode="trace", abi_calls=0, combine_id=combine_id,
                      family="jax", backend=backend, reason="trace_error")
        raise


# ---------------------------------------------------------------------------
# plan（编译计划稳定标识）
# ---------------------------------------------------------------------------


def _plan_id_of(combine_id, shapes, reverse, axis):
    payload = json.dumps(
        {
            "op": OP_NAME,
            "combine_id": int(combine_id),
            "dtype": "float32",
            "axis": int(axis),
            "reverse": bool(reverse),
            "shapes": [list(shape) for shape in shapes],
        },
        sort_keys=True,
        ensure_ascii=True,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


# ---------------------------------------------------------------------------
# 公开入口
# ---------------------------------------------------------------------------


def _execute(fn, elems, reverse, axis, compiled):
    mode = "jit" if compiled else "eager"
    info = _analyze(fn, elems, axis, whole_tree=compiled)
    combine_id = info["combine_id"]
    family = info["family"]

    if family == "numpy":
        arrays = [leaf for _path, leaf in info["leaves"]]
        outputs, calls = _run_so(combine_id, arrays, reverse, axis, compiled)
        plan_id = None
        plan_cache = None
        if compiled:
            plan_id = _plan_id_of(combine_id, info["shapes"], reverse, axis)
            plan_cache = "hit" if plan_id in _PLAN_CACHE else "miss"
            _PLAN_CACHE.add(plan_id)
        _set_dispatch(mode=mode, path="so", exec_mode="whole_tree" if compiled else "per_leaf",
                      abi_calls=calls, combine_id=combine_id, family="numpy", backend="npu",
                      device_id=0, plan_id=plan_id, plan_cache=plan_cache, reason=None)
        return _rebuild(elems, iter(outputs))

    result = _run_trace(elems, combine_id, reverse, axis)
    _set_dispatch(mode=mode, path="trace", exec_mode="trace", abi_calls=0,
                  combine_id=combine_id, family=family, backend=_backend_name(),
                  device_id=0, plan_id=None, plan_cache=None, reason=None)
    return result


def associative_scan(fn, elems, reverse=False, axis=0):
    """逐叶子执行（``exec_mode="per_leaf"``，已注册 combine + numpy 时 ``abi_calls == 叶子数``）。"""
    return _execute(fn, elems, bool(reverse), axis, compiled=False)


def associative_scan_compiled(fn, elems, reverse=False, axis=0):
    """整树一次 ``.so`` 调用（``exec_mode="whole_tree"``，``abi_calls == 1``）。

    这是测试/门禁的编译态入口，不是第二个公开 API；README 必须写明该边界。
    """
    return _execute(fn, elems, bool(reverse), axis, compiled=True)


def _autodiff_kernel(fn, elems, reverse=False, axis=0, mode="jvp"):
    """host-driver 的 AD 入口（numpy family；kernel 路径）。

    AD 定义与 CPU oracle（``golden.compute_golden_autodiff``）一致：tangent/cotangent
    seed 均为全 1。对 sum：
      * JVP：primal = scan(x)，tangent = scan(ones)（同 reverse）；
      * VJP：primal = scan(x)，cotangent = scan(ones)（反向翻转）。
    两次 ``.so`` 调用都在 AscendC kernel 上完成；Python 层只生成 ones 与搬运。
    """
    import numpy as np

    if mode not in ("jvp", "vjp"):
        raise ValueError(f"invalid_argument: mode={mode!r} 只能是 jvp/vjp")
    info = _analyze(fn, elems, axis)
    if info["family"] != "numpy":
        raise ValueError("invalid_argument: host-driver AD 入口只接受 numpy family")
    if len(info["leaves"]) != 1:
        raise ValueError("invalid_argument: AD 只定义在单叶子用例上")
    if info["combine_id"] != COMBINE_SUM:
        raise ValueError(f"unsupported_combine_fn: AD 只注册 sum，收到 combine_id={info['combine_id']}")

    arrays = [leaf for _path, leaf in info["leaves"]]
    primal, calls_primal = _run_so(info["combine_id"], arrays, reverse, axis, compiled=True)
    seeds = [np.ones_like(array, dtype=np.float32) for array in arrays]
    seed_reverse = bool(reverse) if mode == "jvp" else (not bool(reverse))
    second, calls_second = _run_so(info["combine_id"], seeds, seed_reverse, axis, compiled=True)

    _set_dispatch(mode=mode, path="so", exec_mode=mode,
                  abi_calls=int(calls_primal + calls_second), combine_id=info["combine_id"],
                  family="numpy", backend="npu", device_id=0, plan_id=None, plan_cache=None,
                  reason=None)
    if mode == "jvp":
        return {"jvp_primal": primal[0], "jvp_tangent": second[0]}
    return {"vjp_primal": primal[0], "vjp_cotangent": second[0]}


def _plan_dispatch(fn, elems, reverse=False, axis=0):
    """纯路由（不执行 device、不修改 ``_last_dispatch``）；供 CPU 路由单测。"""
    saved = dict(_last_dispatch)
    try:
        info = _analyze(fn, elems, axis)
    except _AssocReject as exc:
        return {
            "op": OP_NAME, "mode": None, "path": "reject", "exec_mode": "reject",
            "combine_id": None, "abi_calls": 0, "family": None, "backend": None,
            "device_id": 0, "plan_id": None, "plan_cache": None, "reason": exc.code,
        }
    except ValueError as exc:
        return {
            "op": OP_NAME, "mode": None, "path": "trace", "exec_mode": "trace",
            "combine_id": None, "abi_calls": 0, "family": None, "backend": None,
            "device_id": 0, "plan_id": None, "plan_cache": None, "reason": str(exc).split(":")[0],
        }
    finally:
        _last_dispatch.clear()
        _last_dispatch.update(saved)
    if info["family"] == "numpy":
        return {
            "op": OP_NAME, "mode": None, "path": "so", "exec_mode": "per_leaf",
            "combine_id": info["combine_id"], "abi_calls": None, "family": "numpy",
            "backend": "npu", "device_id": 0, "plan_id": None, "plan_cache": None, "reason": None,
        }
    return {
        "op": OP_NAME, "mode": None, "path": "trace", "exec_mode": "trace",
        "combine_id": info["combine_id"], "abi_calls": 0, "family": info["family"],
        "backend": _backend_name(), "device_id": 0, "plan_id": None, "plan_cache": None,
        "reason": None,
    }


__all__ = [
    "associative_scan",
    "associative_scan_compiled",
    "library_fingerprint",
    "abi_version",
    "JaxLaxAssociativeScanArgs",
    "_last_dispatch",
    "_plan_dispatch",
    "_autodiff_kernel",
    "_AssocReject",
    "OP_NAME",
    "KERNEL_NAME",
    "ARCH",
    "COMBINE_IDS",
    "UNSUPPORTED_CODE",
]
