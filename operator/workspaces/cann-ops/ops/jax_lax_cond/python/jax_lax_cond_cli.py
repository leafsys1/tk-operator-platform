# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""jax_lax_cond · host driver（四动作可执行入口 `build/jax_lax_cond` 的协议实现）。

调用契约（0-黑盒测试设计.md §1.3.3，共享 runner `tools/run_cases.py` 的 cond 分支）：

    jax_lax_cond_cli.py --case-json <case.json> --inputs <inputs.npz> \
                        --mode <eager|jit|grad_eager|grad_jit> [--variant <name>] --outdir <dir>

* 成功：rc=0；``<outdir>/outputs.npz`` + ``<outdir>/meta.json``；stdout 首行 ``EXEC ...``。
* 拒绝：rc=3；stdout 含 ``REJECT code=<四码之一>`` 与
  ``UPSTREAM type=<类型> sha256=<摘要> summary=<截断消息>``；
  ``EXEC ... path=reject exec_mode=reject abi_calls=0``；**不写** ``outputs.npz``。
* 其它非 0：``ERROR``（不算拒绝通过）；本任务没有合法 ``UNSUPPORTED``/rc=4。

数值路径：
* ``eager`` / ``jit``：``libjax_lax_cond_host.so`` → ``jax_lax_cond_kernel``（AscendC，NPU 执行）。
* ``grad_eager`` / ``grad_jit``：JAX trace（``.so`` 无自动微分），候选仍在 NPU 后端执行。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import jax_lax_cond as iface  # noqa: E402

OP_NAME = iface.OP_NAME
KERNEL_NAME = iface.KERNEL_NAME
NPU_ARCH = iface.NPU_ARCH

BASE_MODES = ("eager", "jit", "grad_eager", "grad_jit")


# ---------------------------------------------------------------------------
# 分支词表（与 ops/jax_lax_cond/scripts/fixtures.py 同名同数学语义；供 CLI 自持输入）
# 这些 callable 只在 trace/校验与 grad 路径上使用；eager/jit 数值由 .so kernel 计算。
# ---------------------------------------------------------------------------


def _tree_map(fn, value):
    if isinstance(value, dict):
        return {key: _tree_map(fn, child) for key, child in value.items()}
    if isinstance(value, tuple):
        return tuple(_tree_map(fn, child) for child in value)
    if isinstance(value, list):
        return [_tree_map(fn, child) for child in value]
    return fn(value)


def _branch_x_plus_1(x):
    return x + 1.0


def _branch_x_minus_1(x):
    return x - 1.0


def _branch_x_times_2(x):
    return x * 2.0


def _branch_x_div_2(x):
    return x / 2.0


def _branch_tree_add(tree):
    return _tree_map(lambda leaf: leaf + 1.0, tree)


def _branch_tree_sub(tree):
    return _tree_map(lambda leaf: leaf - 1.0, tree)


def _branch_out_tree_true(x):
    return (x + 1.0, {"aux": x[:, 0] + 1.0})


def _branch_out_tree_false(x):
    return (x - 1.0, {"aux": x[:, 0] - 1.0})


def _branch_grad_true(x):
    return x * 2.0


def _branch_grad_false(x):
    return x * 3.0


def _branch_list_inc(xs):
    return [leaf + 1.0 for leaf in xs]


def _branch_list_dec(xs):
    return [leaf - 1.0 for leaf in xs]


def _branch_p02_true(x):
    return x * 1.125 + 0.5


def _branch_p02_false(x):
    return x * 0.875 - 0.5


def _astype_int32(value):
    module = type(value).__module__.split(".")[0]
    if module == "numpy":
        return value.astype(np.int32)
    import jax.numpy as jnp  # noqa: PLC0415

    return value.astype(jnp.int32)


def _branch_struct_true(x):
    return (x, x)


def _branch_struct_false(x):
    return {"x": x, "y": x}


def _branch_shape_true(x):
    return x[:4]


def _branch_shape_false(x):
    return x


def _branch_dtype_true(x):
    return x


def _branch_dtype_false(x):
    return _astype_int32(x)


_BRANCH_FUNCS = {
    "x+1": _branch_x_plus_1,
    "x-1": _branch_x_minus_1,
    "x*2": _branch_x_times_2,
    "x/2": _branch_x_div_2,
    "tree_add": _branch_tree_add,
    "tree_sub": _branch_tree_sub,
    "out_tree_true": _branch_out_tree_true,
    "out_tree_false": _branch_out_tree_false,
    "grad_true": _branch_grad_true,
    "grad_false": _branch_grad_false,
    "list_inc": _branch_list_inc,
    "list_dec": _branch_list_dec,
    "p02_true": _branch_p02_true,
    "p02_false": _branch_p02_false,
    "struct_true": _branch_struct_true,
    "struct_false": _branch_struct_false,
    "shape_true": _branch_shape_true,
    "shape_false": _branch_shape_false,
    "dtype_true": _branch_dtype_true,
    "dtype_false": _branch_dtype_false,
}

#: fixtures.branch_pair 的 canonical 名（CLI 的 case.json 已由 fixtures 写成这些名字）
_BRANCH_ALIASES = {
    "x*1.125+0.5": "p02_true",
    "x*0.875-0.5": "p02_false",
}


def _branch_fn(name: str):
    canonical = _BRANCH_ALIASES.get(str(name).replace(" ", ""), str(name).replace(" ", ""))
    try:
        return _BRANCH_FUNCS[canonical]
    except KeyError as exc:
        raise SystemExit(f"ERROR invalid_argument: 未知 branch {name!r}") from exc


# ---------------------------------------------------------------------------
# 数据/证据小工具（独立实现，不 import 测试侧 fixtures / golden）
# ---------------------------------------------------------------------------


def _contig(array):
    array = np.asarray(array)
    # 0-d 叶子必须保持 rank-0（np.ascontiguousarray 会把 0-d 变成 shape=(1,)）。
    return array if array.ndim == 0 else np.ascontiguousarray(array)


def _arrays_sha256(arrays: dict) -> str:
    digest = hashlib.sha256()
    for key in sorted(arrays):
        arr = _contig(arrays[key])
        digest.update(key.encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(arr.dtype).encode("utf-8"))
        digest.update(b"\0")
        digest.update(json.dumps(list(arr.shape)).encode("utf-8"))
        digest.update(b"\0")
        digest.update(arr.tobytes())
        digest.update(b"\0")
    return digest.hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_text(text: str) -> str:
    return hashlib.sha256((text or "").encode("utf-8")).hexdigest()


def _flatten(value, prefix: str = "") -> list[tuple[str, object]]:
    if isinstance(value, dict):
        out: list[tuple[str, object]] = []
        for key, child in value.items():
            out.extend(_flatten(child, f"{prefix}/{key}" if prefix else str(key)))
        return out
    if isinstance(value, (tuple, list)):
        out = []
        for index, child in enumerate(value):
            out.extend(_flatten(child, f"{prefix}/{index}" if prefix else str(index)))
        return out
    return [(prefix, value)]


def _tree_from_spec(spec: dict, leaves: dict):
    kind = spec["kind"]
    if kind == "leaf":
        return leaves[spec["path"]]
    if kind == "tuple":
        return tuple(_tree_from_spec(child, leaves) for child in spec["children"])
    if kind == "list":
        return [_tree_from_spec(child, leaves) for child in spec["children"]]
    if kind == "dict":
        return {key: _tree_from_spec(spec["children"][key], leaves)
                for key in spec["keys"]}
    raise ValueError(f"未知树节点类型：{kind}")


def _pred_from_descriptor(descriptor: dict):
    kind = str(descriptor.get("kind"))
    if kind == "python_bool":
        return bool(descriptor["value"])
    if kind == "scalar_bool_array":
        return np.array(bool(descriptor["value"]), dtype=np.bool_)
    if kind == "bool_array":
        return np.array(descriptor["values"], dtype=np.bool_).reshape(descriptor["shape"])
    if kind == "int32":
        return np.array(int(descriptor["value"]), dtype=np.int32)
    raise ValueError(f"未知 pred 描述符：{descriptor!r}")


def _as_numpy(leaf):
    if callable(getattr(leaf, "block_until_ready", None)):
        leaf = leaf.block_until_ready()
    return np.asarray(leaf)


def _leaf_record(path: str, value) -> dict:
    array = _contig(value)
    weak_type = getattr(value, "weak_type", None)
    return {
        "path": path,
        "dtype": str(array.dtype),
        "shape": list(array.shape),
        "weak_type": bool(weak_type) if weak_type is not None else None,
        "sha256": hashlib.sha256(array.tobytes()).hexdigest(),
    }


def _fmt_exec(dispatch: dict, extra: dict) -> str:
    fields = {
        "op": OP_NAME,
        "kernel": KERNEL_NAME,
        "arch": NPU_ARCH,
        "mode": extra.get("mode") or dispatch.get("mode"),
        "variant": extra.get("variant") or "default",
        "pred_kind": dispatch.get("pred_kind") or extra.get("pred_kind"),
        "selected": dispatch.get("selected"),
        "path": dispatch.get("path"),
        "exec_mode": dispatch.get("exec_mode"),
        "abi_calls": dispatch.get("abi_calls"),
        "backend": dispatch.get("backend"),
        "device_id": dispatch.get("device_id"),
        "trace_true": (dispatch.get("trace_hits") or {}).get("true"),
        "trace_false": (dispatch.get("trace_hits") or {}).get("false"),
        "exec_true": (dispatch.get("exec_hits") or {}).get("true"),
        "exec_false": (dispatch.get("exec_hits") or {}).get("false"),
        "runtime_token": dispatch.get("runtime_token"),
        "compile_count": dispatch.get("compile_count"),
        "plan_id": dispatch.get("plan_id"),
        "plan_cache": dispatch.get("plan_cache"),
    }
    parts = []
    for key, value in fields.items():
        if value is None or value == "":
            value = "n/a"
        parts.append(f"{key}={value}")
    return "EXEC " + " ".join(parts)


def _write_meta(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _write_outputs(path: Path, outputs: dict) -> dict:
    arrays = {}
    for key, value in outputs.items():
        array = _contig(_as_numpy(value))
        if array.dtype == np.float16:
            array = array.astype(np.float32)
        arrays[key] = array
    np.savez(path, **arrays)
    return arrays


# ---------------------------------------------------------------------------
# 执行
# ---------------------------------------------------------------------------


def _dispatch_snapshot() -> dict:
    snapshot = dict(iface._last_dispatch)  # 触发 pending device token 的最终解析
    return snapshot


def _run_numeric(resolved: dict, mode: str, pred, operands):
    if str(resolved.get("family") or "numpy") == "jax":
        import jax  # noqa: PLC0415
        import jax.numpy as jnp  # noqa: PLC0415

        true_fn = _branch_fn(resolved["true_name"])
        false_fn = _branch_fn(resolved["false_name"])
        variant = str(resolved.get("variant") or "")

        def to_jax(value):
            # S1 的 weak 变体必须由 Python 标量构造，weak_type 才与 CPU golden 一致；
            # 其余 jax family 用显式 float32 构造（strong）。
            if variant == "weak":
                return jnp.asarray(float(np.asarray(value)))
            return jnp.asarray(value, dtype=jnp.float32)

        operands = tuple(jax.tree_util.tree_map(to_jax, operand) for operand in operands)
        if mode == "jit":
            # 镜像 #145 的 AFIR 不能把 0-d operand / tracer pred 当 jit 动态实参；
            # 走 workflow2 #202 已验证路线：pred 是 concrete 闭包常量；0-d operand
            # 也放闭包（零参数程序），非 0-d 才作为动态实参。
            pred_concrete = pred
            if any(tuple(getattr(leaf, "shape", ())) == ()
                   for leaf in _flatten(operands, "")):
                def _program_zero_arg():
                    return iface.cond(pred_concrete, true_fn, false_fn, *operands)
                outputs = jax.jit(_program_zero_arg).lower().compile()()
            else:
                def _program(*args):
                    return iface.cond(pred_concrete, true_fn, false_fn, *args)
                outputs = jax.jit(_program)(*operands)
        else:
            outputs = iface.cond(pred, true_fn, false_fn, *operands)
        return outputs

    true_fn = _branch_fn(resolved["true_name"])
    false_fn = _branch_fn(resolved["false_name"])
    entry = iface.cond_compiled if mode == "jit" else iface.cond
    return entry(pred, true_fn, false_fn, *operands)


def _restore_shape(value, shape):
    """把扁平梯度在 host 上恢复成原 shape（不改变数值，不引入设备 reshape）。"""
    import jax.numpy as jnp  # noqa: PLC0415

    if tuple(value.shape) == tuple(shape):
        return value
    array = np.asarray(value.block_until_ready(), dtype=np.float32).reshape(shape)
    return jnp.asarray(array)


def _run_grad(resolved: dict, mode: str, pred, operands, arrays):
    import jax  # noqa: PLC0415
    import jax.numpy as jnp  # noqa: PLC0415

    true_fn = _branch_fn(resolved["true_name"])
    false_fn = _branch_fn(resolved["false_name"])
    leaf_path = resolved["leaf_order"][0]
    x_np = np.array(arrays[leaf_path], copy=True).astype(np.float32, copy=False)
    orig_shape = tuple(x_np.shape)
    # 镜像 #145：AFIR 对 2-D auto-fuse 输出只写首列，且不支持 20→5x4 reshape。
    # jit grad 以扁平 [N] operand 编译（workflow2 #202 / workflow3 探针），梯度值
    # 仍是同一逐元素结果；仅比较/落盘前在 host 恢复原 shape，不做数值变换。
    # 扁平化在 host 完成，避免在设备上执行 5x4→20 的 reshape。
    if mode == "grad_jit":
        x_np = np.ascontiguousarray(x_np.reshape(-1))
    x = jnp.asarray(x_np)

    def loss(value):
        return jnp.sum(iface.cond(pred, true_fn, false_fn, value))

    if mode == "grad_jit":
        grad = jax.jit(jax.grad(loss))(x)
        grad = _restore_shape(grad, orig_shape)
    else:
        grad = jax.grad(loss)(x)
    return {"grad": grad}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="jax_lax_cond host driver")
    parser.add_argument("--case-json", required=True)
    parser.add_argument("--inputs", required=True)
    parser.add_argument("--mode", required=True, choices=BASE_MODES)
    parser.add_argument("--variant", default="")
    parser.add_argument("--outdir", required=True)
    args = parser.parse_args(argv)

    case_path = Path(args.case_json)
    inputs_path = Path(args.inputs)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    if not case_path.is_file() or not inputs_path.is_file():
        print(f"ERROR invalid_argument: case/inputs 不存在：{case_path} {inputs_path}")
        return 1
    case = json.loads(case_path.read_text(encoding="utf-8"))
    resolved = case.get("_resolved")
    if not isinstance(resolved, dict):
        print("ERROR invalid_argument: case.json 缺少 _resolved")
        return 1
    with np.load(inputs_path, allow_pickle=False) as npz:
        arrays = {key: npz[key] for key in npz.files}
    pred = _pred_from_descriptor(resolved["pred"])
    operands = (_tree_from_spec(resolved["input_tree"], arrays),)
    variant = args.variant or str(resolved.get("variant") or "") or "default"
    mode = args.mode

    try:
        if mode in ("grad_eager", "grad_jit"):
            outputs = _run_grad(resolved, mode, pred, operands, arrays)
        else:
            outputs = _run_numeric(resolved, mode, pred, operands)
    except iface.CondError as exc:
        if exc.code not in iface.REJECT_CODES:
            # device_error / invalid_argument / internal_error 一律是 ERROR，不得
            # 伪装成约定拒绝码（否则“别的原因失败”会被算作拒绝通过）。
            print(f"ERROR {exc.code}: {exc}")
            return 1
        return _handle_reject(exc, outdir, resolved, mode, variant, arrays)

    dispatch = _dispatch_snapshot()
    flat = {}
    if mode in ("grad_eager", "grad_jit"):
        flat = {"grad": outputs["grad"]}
    else:
        flat = dict(_flatten(outputs, "out"))
    arrays_out = _write_outputs(outdir / "outputs.npz", flat)
    leaves = [_leaf_record(path, value) for path, value in flat.items()]
    outputs_sha = {
        path: hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()
        for path, array in arrays_out.items()
    }
    try:
        fingerprint = iface.library_fingerprint()
    except Exception as exc:  # noqa: BLE001 - 指纹缺失不应把成功路径变成静默失败
        print(f"ERROR device_error: 无法读取 .so 指纹：{type(exc).__name__}: {exc}")
        return 1

    meta = {
        "case_id": resolved.get("case_id") or case.get("case_id"),
        "variant": variant,
        "mode": mode,
        "status": "ok",
        "dispatch": {
            "path": dispatch.get("path"),
            "exec_mode": dispatch.get("exec_mode"),
            "abi_calls": dispatch.get("abi_calls"),
            "selected": dispatch.get("selected"),
            "branch_id": dispatch.get("branch_id"),
            "compile_count": dispatch.get("compile_count"),
            "plan_id": dispatch.get("plan_id"),
            "plan_cache": dispatch.get("plan_cache"),
            "backend": dispatch.get("backend"),
            "device_id": dispatch.get("device_id"),
        },
        "evidence": {
            "trace_hits": dispatch.get("trace_hits"),
            "exec_hits": dispatch.get("exec_hits"),
            "runtime_token": dispatch.get("runtime_token"),
            "evidence_level": dispatch.get("evidence_level"),
            "token_source": dispatch.get("token_source"),
            "pred_source": dispatch.get("pred_source"),
            "pred_platform": dispatch.get("pred_platform"),
            "pred_sha256": dispatch.get("pred_sha256"),
            "cotangent_route": dispatch.get("cotangent_route"),
            "grad_input_layout": ("flat[N]" if mode == "grad_jit" else "as_is"),
        },
        "leaves": leaves,
        "inputs_sha256": resolved.get("inputs_sha256"),
        "outputs_sha256": outputs_sha,
        "so_fingerprint": fingerprint,
    }
    _write_meta(outdir / "meta.json", meta)
    print(_fmt_exec(dispatch, {"variant": variant, "mode": mode,
                               "pred_kind": dispatch.get("pred_kind")}))
    print(f"OK case={meta['case_id']} mode={mode} leaves={len(leaves)} "
          f"abi_calls={dispatch.get('abi_calls')}")
    return 0


def _handle_reject(exc, outdir: Path, resolved: dict, mode: str, variant: str,
                   arrays: dict) -> int:
    dispatch = _dispatch_snapshot()
    code = exc.code
    message = exc.upstream_message or str(exc)
    summary = message.strip().splitlines()[0][:200] if message.strip() else ""
    upstream = {
        "type": exc.upstream_exception_type,
        "message": message,
        "sha256": exc.upstream_message_sha256,
        "summary": summary,
        "keyword_hit": any(word in message.lower() for word in
                           ("scalar", "bool", "pred", "structure", "tree", "pytree",
                            "identical", "types", "shape", "dtype", "type")),
    }
    evidence = exc.upstream_evidence or {}
    if evidence:
        upstream.update(evidence)
    outputs_path = outdir / "outputs.npz"
    if outputs_path.is_file():
        outputs_path.unlink()
    meta = {
        "case_id": resolved.get("case_id"),
        "variant": variant,
        "mode": mode,
        "status": "reject",
        "dispatch": {
            "path": "reject",
            "exec_mode": "reject",
            "abi_calls": 0,
            "selected": dispatch.get("selected"),
            "branch_id": dispatch.get("branch_id"),
            "compile_count": 0,
            "plan_id": "",
            "plan_cache": "reject",
            "backend": dispatch.get("backend"),
            "device_id": 0,
        },
        "evidence": {
            "trace_hits": dispatch.get("trace_hits"),
            "exec_hits": dispatch.get("exec_hits"),
            "runtime_token": dispatch.get("runtime_token"),
            "evidence_level": "reject",
        },
        "leaves": [],
        "inputs_sha256": resolved.get("inputs_sha256"),
        "upstream": upstream,
    }
    _write_meta(outdir / "meta.json", meta)
    print(f"REJECT code={code}")
    print(f"UPSTREAM type={upstream['type'] if upstream['type'] is not None else 'None'} "
          f"sha256={upstream['sha256'] or ''} summary={summary}")
    print(_fmt_exec(dispatch, {"variant": variant, "mode": mode,
                               "pred_kind": dispatch.get("pred_kind")}))
    return 3


if __name__ == "__main__":
    sys.exit(main())
