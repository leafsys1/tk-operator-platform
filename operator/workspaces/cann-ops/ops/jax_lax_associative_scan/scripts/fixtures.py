# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""jax_lax_associative_scan 测试夹具：确定性输入、pytree 描述、combine 注册表。

本模块是**测试输入**的唯一来源：golden（JAX CPU 子进程）与候选（NPU 可执行文件 /
Python 接口）都从同一份 ``inputs.npz`` 读取数组，不各自生成随机数。设计依据
``$WORK_DIR/0-黑盒测试设计.md`` §1.3.3 / §2.1.2 / §5.4。

约定
----
* 随机源：``numpy.random.default_rng(int(seed))``；单叶子 ``uniform(-1,1)``；pytree 按
  ``elems`` 字符串的声明顺序（F4/S3/S6 均为 ``a`` 然后 ``b``）用**同一个 rng** 连续取样。
  同一 ``(case_id, seed)`` 每次得到字节级相同的输入（``inputs_sha256`` 可复核）。
* ``inputs.npz`` / ``outputs.npz`` / golden 的叶子键：单叶子 ``elems`` → 输出 ``out``；
  dict pytree ``elems/a``、``elems/b`` → 输出 ``out/a``、``out/b``；
  AD 输出 ``jvp_primal``/``jvp_tangent``/``vjp_primal``/``vjp_cotangent``。
* 树描述（``elems_tree``）是纯 JSON：``{"kind":"leaf","path":...}``、``{"kind":"dict",
  "keys":[...],"children":{...}}``；候选/测试都用 ``tree_from_spec`` 还原，不靠猜。
* combine 注册表：``sum``=1（``a+b``）、``max``=2（``maximum(a,b)``）、``pytree_add``=3
  （逐叶子 ``+``）。与冻结 ``cases.json`` 的 ``op_kind`` 一一对应。

明确边界：本模块只负责**测试输入**，不是 oracle；逐叶子判定见 ``criteria/check.py``。
"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

import numpy as np

DEFAULT_SEED = 20261026
SCHEMA = "opx-assoc-scan-fixture/1"
OP_NAME = "jax_lax_associative_scan"
ELEM_KEY = "elems"
OUT_PATH = "out"

#: combine 注册表（id 与设计 §1.3.1 一致；顺序即 op_kind 的稳定顺序）
COMBINE_IDS = {"sum": 1, "max": 2, "pytree_add": 3}
COMBINE_NAMES = {value: key for key, value in COMBINE_IDS.items()}


# ---------------------------------------------------------------------------
# combine 函数（fixture 冻结；节点 2 的 .so 注册表按 combine_id 实现同一语义）
# ---------------------------------------------------------------------------


def combine_sum(a, b):
    """逐元素加法；``a``、``b`` 可为 numpy / JAX / torch 数组。"""
    return a + b


def combine_max(a, b):
    """逐元素最大值。"""
    return np.maximum(a, b) if isinstance(a, np.ndarray) else _maximum(a, b)


def combine_tree_add(a, b):
    """pytree 逐叶子加法（同一树结构）。"""
    return tree_map_binary(lambda x, y: x + y, a, b)


def _maximum(a, b):
    """数组 API 中立的 maximum：numpy 用 np.maximum，其余模块用 ``where``。"""
    module = type(a).__module__.split(".")[0]
    if module in ("numpy",):
        return np.maximum(a, b)
    if module == "jax" or module == "jaxlib":
        import jax.numpy as jnp  # noqa: PLC0415

        return jnp.maximum(a, b)
    if module == "torch":
        import torch  # noqa: PLC0415

        return torch.maximum(a, b)
    # 未知数组类型：用比较 + where（JAX tracer 也支持）
    return a * (a >= b) + b * (b > a)


for _name, _fn in (("sum", combine_sum), ("max", combine_max), ("pytree_add", combine_tree_add)):
    _fn.combine_id = COMBINE_IDS[_name]
    _fn.combine_name = _name

COMBINE_FUNCS = {"sum": combine_sum, "max": combine_max, "pytree_add": combine_tree_add}


def combine_fn_for(name: str):
    """按 ``op_kind`` 取 combine 函数；未知名字显式失败，不静默猜。"""
    try:
        return COMBINE_FUNCS[str(name)]
    except KeyError as exc:
        raise ValueError(f"未知 op_kind：{name!r}（支持 {sorted(COMBINE_FUNCS)}）") from exc


# ---------------------------------------------------------------------------
# 用例 → 叶子 / 几何
# ---------------------------------------------------------------------------

_LEAF_RE = re.compile(r"([A-Za-z_]\w*)\s*:\s*(float32|float16)\s*\[([0-9,\s]+)\]")


def parse_elems_spec(text: str) -> dict[str, dict]:
    """解析 ``dict(a:float32[6,4], b:float32[6,2])`` → 有序叶子表。

    只接受本批已确认的 dict 形式；键序、dtype、shape 全部显式保留，供生成输入时
    按声明顺序取样（设计 §2.1.2）。
    """
    body = str(text).strip()
    if not body.startswith("dict(") or not body.endswith(")"):
        raise ValueError(f"elems 描述必须是 dict(...)：{text!r}")
    inner = body[len("dict("):-1]
    leaves: dict[str, dict] = {}
    consumed = ""
    for match in _LEAF_RE.finditer(inner):
        key, dtype, shape_text = match.group(1), match.group(2), match.group(3)
        if key in leaves:
            raise ValueError(f"elems 描述里叶子重复：{key!r}")
        shape = [int(part) for part in shape_text.split(",") if part.strip() != ""]
        if not shape or any(dim < 1 for dim in shape):
            raise ValueError(f"elems 描述里 {key!r} 的 shape 非法：{shape}")
        leaves[key] = {"dtype": dtype, "shape": shape}
        consumed += match.group(0)
    if not leaves:
        raise ValueError(f"elems 描述里没有可解析的叶子：{text!r}")
    # 去掉已解析片段后只允许空白/逗号，避免把拼写错误静默吞掉
    residue = _LEAF_RE.sub("", inner).replace(" ", "").replace(",", "")
    if residue:
        raise ValueError(f"elems 描述里有无法解析的内容：{residue!r}（原文 {text!r}）")
    return leaves


def case_leaves(case: dict) -> tuple[list[dict], str]:
    """返回 ``([{path,name,shape,dtype}], kind)``；kind 为 ``leaf`` 或 ``dict``。

    * 冻结 pytree 用例用 ``elems`` 字符串描述；
    * 单叶子用 ``shape``；benchmark 用 ``length``/``width``（等价 shape ``[L,W]``）。
    """
    case_id = case.get("case_id")
    if case.get("elems"):
        parsed = parse_elems_spec(case["elems"])
        case_dtype = str(case.get("dtype", "float32"))
        leaves = []
        for name, item in parsed.items():
            if item["dtype"] != case_dtype:
                raise ValueError(
                    f"{case_id}: 叶子 {name!r} dtype={item['dtype']} 与 case.dtype={case_dtype} 不一致")
            leaves.append({"path": f"{ELEM_KEY}/{name}", "name": name,
                           "shape": item["shape"], "dtype": item["dtype"]})
        return leaves, "dict"

    shape = case.get("shape")
    if shape is None:
        if case.get("length") is None or case.get("width") is None:
            raise ValueError(f"{case_id}: 缺少 shape 或 length/width，无法生成输入")
        shape = [int(case["length"]), int(case["width"])]
    shape = [int(dim) for dim in shape]
    if not shape or any(dim < 1 for dim in shape):
        raise ValueError(f"{case_id}: shape 非法：{shape}")
    dtype = str(case.get("dtype", "float32"))
    return [{"path": ELEM_KEY, "name": ELEM_KEY, "shape": shape, "dtype": dtype}], "leaf"


def combine_name_of(case: dict) -> str:
    """用例 ``op_kind`` → combine 名（缺省 sum；未知显式失败）。"""
    name = case.get("op_kind") or "sum"
    if name not in COMBINE_FUNCS:
        raise ValueError(f"{case.get('case_id')}: 未知 op_kind={name!r}（支持 {sorted(COMBINE_FUNCS)}）")
    return str(name)


def case_geometry(case: dict) -> tuple[list[dict], bool]:
    """逐叶子 ``(outer, length, inner)``；axis 不合法时返回 ``([], False)``。

    几何抽象（设计 §1.1）：``outer=prod(shape[:axis])``、``length=shape[axis]``、
    ``inner=prod(shape[axis+1:])``。拒绝用例也需要合法输入数据，所以调用方不要在
    生成输入前用本函数拦截；它只用于 ``case.json`` 的解析元数据与规模统计。
    """
    leaves, _kind = case_leaves(case)
    axis = case.get("axis", 0)
    if not isinstance(axis, int) or isinstance(axis, bool):
        return [], False
    geometry = []
    for leaf in leaves:
        shape = leaf["shape"]
        if axis < 0 or axis >= len(shape):
            return [], False
        outer = int(np.prod(shape[:axis], dtype=np.int64)) if axis else 1
        inner = int(np.prod(shape[axis + 1:], dtype=np.int64)) if axis + 1 < len(shape) else 1
        geometry.append({"path": leaf["path"], "leaf": leaf["name"],
                         "outer": outer, "length": int(shape[axis]), "inner": inner})
    return geometry, True


def output_paths_of(fixture: dict) -> list[str]:
    """数值模式的输出叶子路径（单叶子 ``out``；pytree ``out/<name>``）。"""
    paths = []
    for leaf in fixture["leaves"]:
        if leaf["path"] == ELEM_KEY:
            paths.append(OUT_PATH)
        else:
            paths.append(f"{OUT_PATH}/{leaf['name']}")
    return paths


def output_shapes_of(fixture: dict, mode: str) -> dict[str, list[int]]:
    """按 mode 给出期望输出叶子 → shape（用于结构/shape 精确断言）。"""
    if mode == "jvp":
        keys = ("jvp_primal", "jvp_tangent")
    elif mode == "vjp":
        keys = ("vjp_primal", "vjp_cotangent")
    else:
        keys = None
    if keys is None:
        return {path: list(leaf["shape"])
                for path, leaf in zip(output_paths_of(fixture), fixture["leaves"])}
    if len(fixture["leaves"]) != 1:
        raise ValueError("AD 输出只定义在单叶子用例上")
    return {key: list(fixture["leaves"][0]["shape"]) for key in keys}


def fixture_numel(fixture: dict) -> int:
    """全部输入元素数（smoke 选例的规模口径）。"""
    return int(sum(int(np.prod(leaf["shape"], dtype=np.int64)) for leaf in fixture["leaves"]))


def case_modes(case: dict) -> list[str]:
    modes = list(case.get("mode") or ["jit"])
    if not modes:
        raise ValueError(f"{case.get('case_id')}: mode 不能为空")
    return modes


# ---------------------------------------------------------------------------
# pytree 编解码（与 jax_lax_scan 同一格式，便于测试/候选复用）
# ---------------------------------------------------------------------------


def tree_spec(value, prefix: str = "") -> dict:
    """把 pytree 描述成 JSON（叶子只记路径，不记数据）。"""
    if isinstance(value, dict):
        return {"kind": "dict", "keys": list(value.keys()),
                "children": {k: tree_spec(v, _join(prefix, k)) for k, v in value.items()}}
    if isinstance(value, tuple):
        return {"kind": "tuple",
                "children": [tree_spec(v, _join(prefix, str(i))) for i, v in enumerate(value)]}
    if isinstance(value, list):
        return {"kind": "list",
                "children": [tree_spec(v, _join(prefix, str(i))) for i, v in enumerate(value)]}
    return {"kind": "leaf", "path": prefix}


def tree_leaf_paths(spec: dict) -> list[str]:
    kind = spec["kind"]
    if kind == "leaf":
        return [spec["path"]]
    if kind in ("tuple", "list"):
        out: list[str] = []
        for child in spec["children"]:
            out.extend(tree_leaf_paths(child))
        return out
    if kind == "dict":
        out = []
        for key in spec["keys"]:
            out.extend(tree_leaf_paths(spec["children"][key]))
        return out
    raise ValueError(f"未知树节点类型：{kind}")


def tree_from_spec(spec: dict, leaves: dict, convert=None):
    """按树描述还原 pytree；``leaves`` 是 路径 → 数组。"""
    kind = spec["kind"]
    if kind == "leaf":
        value = leaves[spec["path"]]
        return convert(value) if convert is not None else value
    if kind == "tuple":
        return tuple(tree_from_spec(c, leaves, convert) for c in spec["children"])
    if kind == "list":
        return [tree_from_spec(c, leaves, convert) for c in spec["children"]]
    if kind == "dict":
        return {k: tree_from_spec(spec["children"][k], leaves, convert) for k in spec["keys"]}
    raise ValueError(f"未知树节点类型：{kind}")


def flatten_with_paths(value, prefix: str = "") -> list[tuple[str, object]]:
    """递归展开 tuple/list/dict，返回 ``[(路径, 叶子)]``；路径与 npz 键一致。"""
    if isinstance(value, dict):
        out: list[tuple[str, object]] = []
        for key, child in value.items():
            out.extend(flatten_with_paths(child, _join(prefix, str(key))))
        return out
    if isinstance(value, (tuple, list)):
        out = []
        for i, child in enumerate(value):
            out.extend(flatten_with_paths(child, _join(prefix, str(i))))
        return out
    return [(prefix, value)]


def tree_map_binary(fn, a, b):
    """对两棵同结构 pytree 逐叶子调用 ``fn``（dict 按 a 的键序，list/tuple 按位置）。"""
    if isinstance(a, dict):
        return {key: tree_map_binary(fn, a[key], b[key]) for key in a}
    if isinstance(a, tuple):
        return tuple(tree_map_binary(fn, x, y) for x, y in zip(a, b))
    if isinstance(a, list):
        return [tree_map_binary(fn, x, y) for x, y in zip(a, b)]
    return fn(a, b)


def _join(prefix: str, key: str) -> str:
    return f"{prefix}/{key}" if prefix else key


# ---------------------------------------------------------------------------
# 输入生成（设计 §2.1.2 冻结规则）
# ---------------------------------------------------------------------------


def make_fixture(case: dict) -> dict:
    """生成一条用例的确定性输入与元数据（不落盘）。

    规则（看到结果后不得调整）：
    * ``rng = np.random.default_rng(int(seed))``；
    * 按 ``case_leaves`` 的声明顺序逐叶子 ``rng.uniform(-1.0, 1.0, size=shape)``；
    * 单叶子 ``astype(np.float32)``；L2-1 的 float16 用例按 case.dtype 生成（仅用于验证拒绝）。
    """
    leaves, kind = case_leaves(case)
    combine_name = combine_name_of(case)
    axis = case.get("axis", 0)
    reverse = bool(case.get("reverse", False))
    seed = int(case.get("seed", DEFAULT_SEED))
    dtype = leaves[0]["dtype"]
    if any(leaf["dtype"] != dtype for leaf in leaves):
        raise ValueError(f"{case.get('case_id')}: 本批要求所有叶子同 dtype")
    if dtype not in ("float32", "float16"):
        raise ValueError(f"{case.get('case_id')}: 本批只生成 float32/float16，收到 {dtype}")

    rng = np.random.default_rng(seed)
    arrays: dict[str, np.ndarray] = {}
    for leaf in leaves:
        arrays[leaf["path"]] = rng.uniform(-1.0, 1.0, size=tuple(leaf["shape"])).astype(np.dtype(dtype))

    if kind == "dict":
        elems_value = {leaf["name"]: arrays[leaf["path"]] for leaf in leaves}
    else:
        elems_value = arrays[ELEM_KEY]
    elems_tree = tree_spec(elems_value, ELEM_KEY)

    geometry, valid_axis = case_geometry(case)
    fixture = {
        "schema": SCHEMA,
        "op": OP_NAME,
        "case_id": case.get("case_id"),
        "op_kind": combine_name,
        "combine_id": COMBINE_IDS[combine_name],
        "combine_name": combine_name,
        "kind": kind,
        "axis": axis,
        "reverse": reverse,
        "seed": seed,
        "dtype": dtype,
        "family": "numpy",
        "leaves": leaves,
        "leaf_order": [leaf["path"] for leaf in leaves],
        "elems_tree": elems_tree,
        "geometry": geometry,
        "valid_axis": valid_axis,
        "arrays": arrays,
    }
    fixture["output_paths"] = output_paths_of(fixture)
    fixture["numel"] = fixture_numel(fixture)
    fixture["inputs_sha256"] = compute_arrays_sha256(arrays)
    return fixture


def compute_arrays_sha256(arrays: dict[str, np.ndarray]) -> str:
    """对一组叶子做稳定指纹（键序、dtype、shape、字节全部纳入）。"""
    h = hashlib.sha256()
    for key in sorted(arrays):
        arr = np.ascontiguousarray(arrays[key])
        h.update(key.encode("utf-8"))
        h.update(b"\0")
        h.update(str(arr.dtype).encode("utf-8"))
        h.update(b"\0")
        h.update(json.dumps(list(arr.shape)).encode("utf-8"))
        h.update(b"\0")
        h.update(arr.tobytes())
        h.update(b"\0")
    return h.hexdigest()


def write_inputs_npz(fixture: dict, path: str | Path) -> str:
    """把 fixture 的数组写进 npz，返回写盘后的 sha256（与 fixture 内记录交叉核对）。"""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    arrays = fixture["arrays"]
    np.savez(path, **{key: arrays[key] for key in arrays})
    sha = compute_arrays_sha256(load_inputs_npz(path))
    if sha != fixture["inputs_sha256"]:
        raise RuntimeError(f"inputs.npz 往返指纹不一致：{sha} != {fixture['inputs_sha256']}")
    return sha


def load_inputs_npz(path: str | Path) -> dict[str, np.ndarray]:
    path = Path(path)
    with np.load(path, allow_pickle=False) as npz:
        return {key: npz[key] for key in npz.files}


def write_case_json(case: dict, fixture: dict, path: str | Path) -> dict:
    """写候选/golden 共用的 case.json（原用例 + `_resolved` 元数据），不改源用例文件。"""
    resolved = {
        "op": OP_NAME,
        "op_kind": fixture["op_kind"],
        "combine_id": fixture["combine_id"],
        "combine_name": fixture["combine_name"],
        "kind": fixture["kind"],
        "axis": fixture["axis"],
        "reverse": fixture["reverse"],
        "seed": fixture["seed"],
        "dtype": fixture["dtype"],
        "family": fixture["family"],
        "leaves": fixture["leaves"],
        "leaf_order": fixture["leaf_order"],
        "elems_tree": fixture["elems_tree"],
        "geometry": fixture["geometry"],
        "valid_axis": fixture["valid_axis"],
        "output_paths": fixture["output_paths"],
        "numel": fixture["numel"],
        "inputs_sha256": fixture["inputs_sha256"],
    }
    payload = dict(case)
    payload["_resolved"] = resolved
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return payload


def fixture_from_case(case_obj: dict, arrays: dict[str, np.ndarray]) -> dict:
    """golden / 候选桩用：从 case.json（含 `_resolved`）+ inputs.npz 还原 fixture。"""
    resolved = case_obj.get("_resolved")
    if not isinstance(resolved, dict):
        raise ValueError("case.json 缺少 _resolved（应由 runner / pytest 写入）")
    leaves = [dict(item) for item in resolved["leaves"]]
    fixture = {
        "schema": SCHEMA,
        "op": OP_NAME,
        "case_id": case_obj.get("case_id"),
        "op_kind": resolved["op_kind"],
        "combine_id": int(resolved["combine_id"]),
        "combine_name": resolved["combine_name"],
        "kind": resolved["kind"],
        "axis": resolved["axis"],
        "reverse": bool(resolved["reverse"]),
        "seed": int(resolved["seed"]),
        "dtype": resolved["dtype"],
        "family": resolved.get("family", "numpy"),
        "leaves": leaves,
        "leaf_order": list(resolved["leaf_order"]),
        "elems_tree": resolved["elems_tree"],
        "geometry": list(resolved.get("geometry") or []),
        "valid_axis": bool(resolved.get("valid_axis", True)),
        "arrays": {key: np.asarray(value) for key, value in arrays.items()},
    }
    fixture["output_paths"] = list(resolved.get("output_paths") or output_paths_of(fixture))
    fixture["numel"] = fixture_numel(fixture)
    fixture["inputs_sha256"] = compute_arrays_sha256(fixture["arrays"])
    expected = resolved.get("inputs_sha256")
    if expected and expected != fixture["inputs_sha256"]:
        raise ValueError(
            f"inputs.npz 指纹与 case.json 不一致：{fixture['inputs_sha256']} != {expected}")
    missing = [path for path in fixture["leaf_order"] if path not in fixture["arrays"]]
    if missing:
        raise ValueError(f"inputs.npz 缺叶子：{missing}")
    return fixture


if __name__ == "__main__":  # 自检：同一 (case, seed) 必须字节级相同
    demo = {"case_id": "assoc_sum_len8_axis0", "shape": [8, 16],
            "axis": 0, "reverse": False, "dtype": "float32", "seed": DEFAULT_SEED}
    f1, f2 = make_fixture(demo), make_fixture(demo)
    print("deterministic:", f1["inputs_sha256"] == f2["inputs_sha256"], f1["inputs_sha256"][:16])
    print("geometry:", json.dumps(f1["geometry"], ensure_ascii=False))
