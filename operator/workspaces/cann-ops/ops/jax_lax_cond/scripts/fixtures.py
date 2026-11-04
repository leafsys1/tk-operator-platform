# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""jax_lax_cond 测试夹具：确定性输入、变体展开、pred 形态、分支词表与 pytree 编解码。

本模块是**测试输入**的唯一来源：golden（JAX CPU 子进程）与候选（Ascend C 直调可执行文件 /
Python 接口）都从同一份 ``inputs.npz`` 读取数组，不各自生成随机数。设计依据
``$WORK_DIR/0-黑盒测试设计.md`` §0.6 / §1.3 / §2.1.2 / §2.1.3 / §5.3–§5.4。

约定
----
* 基础种子 ``20261027``；单变体 ``default_rng(seed)``，多变体 ``default_rng(seed + offset)``：
  ``w17=0``、``w33=1``；``weak=0``、``strong=1``；``bool1=0``、``bool2=1``、``int0=2``、``int1=3``。
* 叶子顺序：dict 按 key 字典序；list/tuple 按位置；单叶子只有一个。同一个 rng 按该顺序连续取样。
* ``inputs.npz`` 键：单叶子 ``operands/x``；dict ``operands/a``、``operands/b``；
  list ``operands/0``、``operands/1``。pred 不写进 npz（由 case.json 的描述符重建）。
* 输出键：单叶子 ``out``；dict ``out/a``、``out/b``；输出 pytree ``out/0``、``out/1/aux``；
  list ``out/0``、``out/1``；grad ``grad``。
* 分支词表（设计 §2.1.3）：``x+1``/``x-1``/``x*2``/``x/2``/``tree_add``/``tree_sub``/
  ``out_tree``/``grad_true``/``grad_false``/``list_inc``/``list_dec``/``p02_true``/``p02_false``；
  ``branch_id`` 只用于候选分派与证据，不改变数学定义。

明确边界：本模块只负责**测试输入/分支定义**，不是 oracle；逐叶子数值判定见
``criteria/check.py``，CPU JAX 参考见 ``golden.py``。
"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

import numpy as np

OP_NAME = "jax_lax_cond"
SCHEMA = "opx-jax-cond-fixture/1"
DEFAULT_SEED = 20261027
ELEM_KEY = "operands"

#: 变体偏移（设计 §2.1.2，看到结果后不得调整）
VARIANT_OFFSETS = {
    "": 0,
    "w17": 0,
    "w33": 1,
    "weak": 0,
    "strong": 1,
    "bool1": 0,
    "bool2": 1,
    "int0": 2,
    "int1": 3,
}

#: 设计 §2.1.3 的 branch_id（out_tree 拆 true=7 / false=8）
BRANCH_IDS = {
    "x+1": 1,
    "x-1": 2,
    "x*2": 3,
    "x/2": 4,
    "tree_add": 5,
    "tree_sub": 6,
    "out_tree_true": 7,
    "out_tree_false": 8,
    "grad_true": 9,
    "grad_false": 10,
    "list_inc": 11,
    "list_dec": 12,
    "p02_true": 13,
    "p02_false": 14,
    "struct_true": 15,
    "struct_false": 16,
    "shape_true": 17,
    "shape_false": 18,
    "dtype_true": 19,
    "dtype_false": 20,
}

#: 冻结用例未给 branches 时的默认（设计 §0.6）
DEFAULT_BRANCHES = ["x+1", "x-1"]

#: L2 拒绝用例的输入规格（设计 §5.3 精确构造；case 本身没有 operand 字段）
REJECT_INPUT_SPECS = {
    "cond_invalid_predicate": "float32[8]",
    "cond_branch_structure_mismatch": "float32[4]",
    "cond_branch_shape_mismatch": "float32[5]",
    "cond_branch_dtype_mismatch": "float32[4]",
}

#: S5 特殊值（设计 §2.1.2 / §4.4；顺序与字节都冻结）
NONFINITE_VALUES = np.array(
    [np.nan, np.inf, -np.inf, -0.0, 0.0, 1.0, -1.0, 3.4028235e38], dtype=np.float32)


# ---------------------------------------------------------------------------
# 分支词表（numpy / JAX tracer 都可调用；不 import 候选、不 import torch）
# ---------------------------------------------------------------------------


def _tree_map(fn, value):
    if isinstance(value, dict):
        return {key: _tree_map(fn, child) for key, child in value.items()}
    if isinstance(value, tuple):
        return tuple(_tree_map(fn, child) for child in value)
    if isinstance(value, list):
        return [_tree_map(fn, child) for child in value]
    return fn(value)


def _astype_int32(value):
    module = type(value).__module__.split(".")[0]
    if module == "numpy":
        return value.astype(np.int32)
    import jax.numpy as jnp  # noqa: PLC0415

    return value.astype(jnp.int32)


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


BRANCH_FUNCS = {
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


#: 冻结用例里可能直接写数学表达式（如 benchmark p02），映射到分支词表名
BRANCH_ALIASES = {
    "x*1.125+0.5": "p02_true",
    "x*0.875-0.5": "p02_false",
}


def _canonical_branch(name: str) -> str:
    text = str(name).replace(" ", "")
    return BRANCH_ALIASES.get(text, text)


def branch_fn(name: str):
    """按分支表达式取 callable；未知名字显式失败，不静默猜。"""
    try:
        return BRANCH_FUNCS[str(name)]
    except KeyError as exc:
        raise ValueError(f"未知 branch：{name!r}（支持 {sorted(BRANCH_FUNCS)}）") from exc


def branch_pair(case: dict) -> dict:
    """返回 true/false 分支、branch_id 与表达式；设计 §0.6 / §5.3 的冻结映射。"""
    case_id = str(case.get("case_id"))
    branches = [str(item).replace(" ", "") for item in (case.get("branches") or [])]
    if branches:
        if len(branches) != 2:
            raise ValueError(f"{case_id}: branches 必须恰好 2 项，收到 {branches!r}")
        true_name, false_name = (_canonical_branch(item) for item in branches)
        if true_name == "out_tree" and false_name == "out_tree":
            true_name, false_name = "out_tree_true", "out_tree_false"
    elif case_id == "cond_invalid_predicate":
        true_name, false_name = "x+1", "x-1"
    elif case_id == "cond_branch_structure_mismatch":
        true_name, false_name = "struct_true", "struct_false"
    elif case_id == "cond_branch_shape_mismatch":
        true_name, false_name = "shape_true", "shape_false"
    elif case_id == "cond_branch_dtype_mismatch":
        true_name, false_name = "dtype_true", "dtype_false"
    elif str(case_id).startswith("cond_grad"):
        true_name, false_name = "grad_true", "grad_false"
    elif case_id in ("cond_output_pytree", "cond_sup_output_pytree"):
        true_name, false_name = "out_tree_true", "out_tree_false"
    elif case_id == "cond_sup_input_pytree_list":
        true_name, false_name = "list_inc", "list_dec"
    else:
        true_name, false_name = DEFAULT_BRANCHES
    return {
        "true_name": true_name,
        "false_name": false_name,
        "true_id": BRANCH_IDS[true_name],
        "false_id": BRANCH_IDS[false_name],
        "true_fn": branch_fn(true_name),
        "false_fn": branch_fn(false_name),
    }


# ---------------------------------------------------------------------------
# 变体 / pred 形态（设计 §2.1.2 / §5.3）
# ---------------------------------------------------------------------------


def variants_for(case: dict) -> list[str]:
    """一条用例的变体列表；单变体返回 ``[""]``（case.json 里 variant 为空串）。"""
    variants = case.get("variants")
    if variants:
        return [str(item) for item in variants]
    case_id = str(case.get("case_id"))
    if case_id == "cond_unaligned_tail_widths":
        return ["w17", "w33"]
    if case_id == "cond_invalid_predicate":
        return ["bool1", "bool2", "int0", "int1"]
    return [""]


def variant_offset(variant: str) -> int:
    if variant not in VARIANT_OFFSETS:
        raise ValueError(f"未知 variant：{variant!r}（支持 {sorted(VARIANT_OFFSETS)}）")
    return VARIANT_OFFSETS[variant]


def pred_value(case: dict, variant: str, mode: str) -> tuple[object, str, bool]:
    """返回 ``(pred 对象, pred_kind, selected_bool)``；逐用例/变体/mode 冻结。"""
    case_id = str(case.get("case_id"))
    if case_id == "cond_invalid_predicate":
        table = {
            "bool1": (np.array([True], dtype=np.bool_), "bool_array_1"),
            "bool2": (np.array([True, False], dtype=np.bool_), "bool_array_2"),
            "int0": (np.array(0, dtype=np.int32), "int32_scalar"),
            "int1": (np.array(1, dtype=np.int32), "int32_scalar"),
        }
        try:
            pred, kind = table[variant]
        except KeyError as exc:
            raise ValueError(f"c13 变体必须属于 {sorted(table)}，收到 {variant!r}") from exc
        return pred, kind, bool(pred.reshape(-1)[0]) if pred.ndim else bool(pred)

    if case_id in ("cond_branch_structure_mismatch", "cond_branch_shape_mismatch",
                   "cond_branch_dtype_mismatch"):
        # 设计 §5.3：eager=True（Python bool），jit=0-d np.array(False)
        if mode == "eager":
            return True, "python_bool", True
        return np.array(False, dtype=np.bool_), "scalar_bool_array", False

    selected = _selected_bool(case, mode)
    pred_field = case.get("pred")
    if isinstance(pred_field, bool):
        return pred_field, "python_bool", pred_field
    if pred_field == "static_scalar_bool":
        return True, "python_bool", True
    if pred_field == "scalar_bool_array":
        return np.array(selected, dtype=np.bool_), "scalar_bool_array", selected
    if pred_field in ("dynamic_scalar_bool", "scalar_bool") or pred_field is None:
        if "_" in mode and mode.rsplit("_", 1)[-1] in ("true", "false"):
            return np.array(selected, dtype=np.bool_), "scalar_bool_array", selected
        return selected, "python_bool", selected
    if pred_field in ("true", "false"):
        value = pred_field == "true"
        return value, "python_bool", value
    raise ValueError(f"{case_id}: 无法解析 pred={pred_field!r}（mode={mode!r}）")


def _selected_bool(case: dict, mode: str) -> bool:
    if mode.endswith("_true"):
        return True
    if mode.endswith("_false"):
        return False
    pred_field = case.get("pred")
    if isinstance(pred_field, bool):
        return pred_field
    if pred_field == "static_scalar_bool":
        return True
    # 设计 §2.1.2：c07/c09/c10/S1/S4 的 Python bool 为 True；c14–c16 的 eager 为 True
    return bool(case.get("pred_selected", True))


def pred_descriptor(pred, kind: str) -> dict:
    """把 pred 对象编码成 case.json 可读的描述符（golden 子进程据此重建）。"""
    if isinstance(pred, (bool, np.bool_)):
        return {"kind": "python_bool", "value": bool(pred)}
    arr = np.asarray(pred)
    if arr.dtype == np.bool_ and arr.shape == ():
        return {"kind": "scalar_bool_array", "value": bool(arr)}
    if arr.dtype == np.bool_:
        return {"kind": "bool_array", "shape": list(arr.shape),
                "values": [bool(item) for item in arr.reshape(-1)]}
    if arr.dtype == np.int32 and arr.shape == ():
        return {"kind": "int32", "value": int(arr)}
    raise ValueError(f"无法编码 pred：kind={kind!r} dtype={arr.dtype} shape={arr.shape}")


def pred_from_descriptor(descriptor: dict):
    """从 case.json 的描述符重建 pred 对象。"""
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


def pred_name(pred) -> str:
    return "true" if bool(np.asarray(pred).reshape(-1)[0]) else "false"


# ---------------------------------------------------------------------------
# 输入 / 输出规格解析
# ---------------------------------------------------------------------------


_LEAF_RE = re.compile(r"(?:([A-Za-z_]\w*)\s*:\s*)?float32\s*\[([0-9,\s]*)\]")


def _parse_shape(text: str) -> list[int]:
    parts = [part.strip() for part in text.split(",") if part.strip()]
    shape = [int(part) for part in parts]
    if any(dim < 0 for dim in shape):
        raise ValueError(f"shape 非法：{text!r}")
    return shape


def _leaf(path: str, name: str, shape: list[int]) -> dict:
    return {"path": path, "name": name, "shape": shape, "dtype": "float32"}


def parse_operand_spec(text: str) -> tuple[list[dict], str]:
    """解析 operand 描述，返回 ``(leaves, kind)``；kind 为 leaf/dict/list。"""
    body = str(text).strip()
    if body.startswith("dict(") and body.endswith(")"):
        inner = body[len("dict("):-1]
        leaves = []
        for match in _LEAF_RE.finditer(inner):
            name, shape_text = match.group(1), match.group(2)
            if not name:
                raise ValueError(f"dict operand 的叶子缺少键名：{text!r}")
            leaves.append(_leaf(f"{ELEM_KEY}/{name}", name, _parse_shape(shape_text)))
        residue = _LEAF_RE.sub("", inner).replace(" ", "").replace(",", "")
        if not leaves or residue:
            raise ValueError(f"dict operand 无法解析：{text!r}（残留 {residue!r}）")
        names = [leaf["name"] for leaf in leaves]
        if names != sorted(names):
            raise ValueError(f"dict operand 必须按 key 字典序声明：{text!r}")
        return leaves, "dict"
    if body.startswith("[") and body.endswith("]"):
        inner = body[1:-1]
        leaves = []
        for index, match in enumerate(_LEAF_RE.finditer(inner)):
            if match.group(1):
                raise ValueError(f"list operand 的叶子不能带键名：{text!r}")
            leaves.append(_leaf(f"{ELEM_KEY}/{index}", str(index), _parse_shape(match.group(2))))
        residue = _LEAF_RE.sub("", inner).replace(" ", "").replace(",", "")
        if not leaves or residue:
            raise ValueError(f"list operand 无法解析：{text!r}（残留 {residue!r}）")
        return leaves, "list"
    match = _LEAF_RE.fullmatch(body)
    if not match or match.group(1):
        raise ValueError(f"operand 描述无法解析：{text!r}")
    return [_leaf(f"{ELEM_KEY}/x", "x", _parse_shape(match.group(2)))], "leaf"


def case_input_leaves(case: dict, variant: str) -> tuple[list[dict], str]:
    """返回输入叶子表与结构 kind；L2 用例按设计 §5.3 构造。"""
    case_id = str(case.get("case_id"))
    if case.get("operand"):
        return parse_operand_spec(case["operand"])
    if case.get("elements") is not None:
        elements = case["elements"]
        if case_id == "cond_unaligned_tail_widths":
            table = {"w17": 17, "w33": 33}
            try:
                count = table[variant]
            except KeyError as exc:
                raise ValueError(f"c11 变体必须是 w17/w33，收到 {variant!r}") from exc
        else:
            count = int(elements[0] if isinstance(elements, (list, tuple)) else elements)
        return [_leaf(f"{ELEM_KEY}/x", "x", [count])], "leaf"
    if case_id in REJECT_INPUT_SPECS:
        return parse_operand_spec(REJECT_INPUT_SPECS[case_id])
    raise ValueError(f"{case_id}: 缺少 operand/elements，无法生成输入")


def input_kind_of(case: dict, variant: str) -> str:
    return case_input_leaves(case, variant)[1]


def modes_for(case: dict) -> list[str]:
    modes = list(case.get("mode") or ["eager"])
    if not modes:
        raise ValueError(f"{case.get('case_id')}: mode 不能为空")
    return modes


def expand_items(cases: list[dict]) -> list[tuple[dict, str, str]]:
    """按 ``(case, variant, mode)`` 展开；门禁 16 条应得 49 项，pytest 23 条应得 71 项。"""
    items: list[tuple[dict, str, str]] = []
    for case in cases:
        for variant in variants_for(case):
            for mode in modes_for(case):
                items.append((case, variant, mode))
    return items


def count_summary(cases: list[dict]) -> dict:
    items = expand_items(cases)
    return {
        "identities": len(cases),
        "variants": len({(case.get("case_id"), variant)
                         for case, variant, _mode in items}),
        "mode_executions": len(items),
    }


def output_spec(case: dict, fixture: dict) -> dict[str, list[int]]:
    """输出叶子路径 → shape；拒绝用例返回空表（不产生输出）。"""
    if case.get("expected_error"):
        return {}
    case_id = str(case.get("case_id"))
    modes = list(case.get("mode") or [])
    if any(mode in ("grad_eager", "grad_jit") for mode in modes):
        leaves = fixture["leaves"]
        if len(leaves) != 1:
            raise ValueError(f"{case_id}: grad 只定义在单叶子用例上")
        return {"grad": list(leaves[0]["shape"])}
    if fixture["kind"] == "dict":
        return {f"out/{leaf['name']}": list(leaf["shape"]) for leaf in fixture["leaves"]}
    if fixture["kind"] == "list":
        return {f"out/{index}": list(leaf["shape"])
                for index, leaf in enumerate(fixture["leaves"])}
    if case_id in ("cond_output_pytree", "cond_sup_output_pytree"):
        shape = list(fixture["leaves"][0]["shape"])
        return {"out/0": shape, "out/1/aux": [shape[0]]}
    return {"out": list(fixture["leaves"][0]["shape"])}


# ---------------------------------------------------------------------------
# pytree 编解码（与 jax_lax_scan / assoc 同一格式）
# ---------------------------------------------------------------------------


def tree_spec(value, prefix: str = "") -> dict:
    if isinstance(value, dict):
        return {"kind": "dict", "keys": list(value.keys()),
                "children": {key: tree_spec(child, _join(prefix, key))
                             for key, child in value.items()}}
    if isinstance(value, tuple):
        return {"kind": "tuple",
                "children": [tree_spec(child, _join(prefix, str(index)))
                             for index, child in enumerate(value)]}
    if isinstance(value, list):
        return {"kind": "list",
                "children": [tree_spec(child, _join(prefix, str(index)))
                             for index, child in enumerate(value)]}
    return {"kind": "leaf", "path": prefix}


def tree_from_spec(spec: dict, leaves: dict, convert=None):
    kind = spec["kind"]
    if kind == "leaf":
        value = leaves[spec["path"]]
        return convert(value) if convert is not None else value
    if kind == "tuple":
        return tuple(tree_from_spec(child, leaves, convert) for child in spec["children"])
    if kind == "list":
        return [tree_from_spec(child, leaves, convert) for child in spec["children"]]
    if kind == "dict":
        return {key: tree_from_spec(spec["children"][key], leaves, convert)
                for key in spec["keys"]}
    raise ValueError(f"未知树节点类型：{kind}")


def flatten_with_paths(value, prefix: str = "") -> list[tuple[str, object]]:
    if isinstance(value, dict):
        out: list[tuple[str, object]] = []
        for key, child in value.items():
            out.extend(flatten_with_paths(child, _join(prefix, str(key))))
        return out
    if isinstance(value, (tuple, list)):
        out = []
        for index, child in enumerate(value):
            out.extend(flatten_with_paths(child, _join(prefix, str(index))))
        return out
    return [(prefix, value)]


def _join(prefix: str, key: str) -> str:
    return f"{prefix}/{key}" if prefix else key


# ---------------------------------------------------------------------------
# 输入生成（设计 §2.1.2 冻结规则）
# ---------------------------------------------------------------------------


def make_fixture(case: dict, variant: str = "", mode: str | None = None) -> dict:
    """生成一条 ``(case, variant, mode)`` 的确定性输入与元数据（不落盘）。"""
    leaves, kind = case_input_leaves(case, variant)
    case_id = str(case.get("case_id"))
    modes = list(case.get("mode") or ["eager"])
    mode = mode or modes[0]
    if mode not in modes:
        raise ValueError(f"{case_id}: mode={mode!r} 不在 {modes!r}")

    seed = int(case.get("seed", DEFAULT_SEED)) + variant_offset(variant)
    arrays: dict[str, np.ndarray] = {}
    if case.get("special_values") or case_id == "cond_sup_nonfinite":
        base = NONFINITE_VALUES.copy()
    else:
        base = None
    if base is not None:
        if len(leaves) != 1:
            raise ValueError(f"{case_id}: special_values 只支持单叶子")
        arrays[leaves[0]["path"]] = np.ascontiguousarray(base.reshape(leaves[0]["shape"]))
    else:
        rng = np.random.default_rng(seed)
        for leaf in leaves:
            arrays[leaf["path"]] = rng.uniform(
                -1.0, 1.0, size=tuple(leaf["shape"])).astype(np.float32)

    pred, pred_kind, selected = pred_value(case, variant, mode)
    # 设计 §1.3.3：pred 变体（0-d/1-d bool 或 int32）同时写入 inputs.npz 的 `pred` 键；
    # `build_operands` 只按 input_tree/leaves 取 operand，不会把 pred 当输入叶子。
    if isinstance(pred, np.ndarray):
        arrays["pred"] = np.array(pred, copy=True)
    pair = branch_pair(case)
    family = str(case.get("family") or "numpy")
    fixture = {
        "schema": SCHEMA,
        "op": OP_NAME,
        "case_id": case_id,
        "variant": variant,
        "mode": mode,
        "group": case.get("group"),
        "required": bool(case.get("required", True)),
        "expected": case.get("expected", "match_cpu"),
        "expected_error": case.get("expected_error"),
        "dtype": str(case.get("dtype", "float32")),
        "family": family,
        "seed": seed,
        "kind": kind,
        "leaves": leaves,
        "leaf_order": [leaf["path"] for leaf in leaves],
        "arrays": arrays,
        "pred": pred,
        "pred_kind": pred_kind,
        "pred_descriptor": pred_descriptor(pred, pred_kind),
        "selected": selected,
        "true_name": pair["true_name"],
        "false_name": pair["false_name"],
        "true_id": pair["true_id"],
        "false_id": pair["false_id"],
    }
    fixture["output_shapes"] = output_spec(case, fixture)
    fixture["output_paths"] = list(fixture["output_shapes"])
    fixture["inputs_sha256"] = compute_arrays_sha256(arrays)
    fixture["numel"] = int(sum(int(np.prod(leaf["shape"], dtype=np.int64))
                               for leaf in leaves))
    fixture["base_mode"] = base_mode_of(mode)
    fixture["input_tree"] = _input_tree_spec(fixture)
    return fixture


def base_mode_of(mode: str) -> str:
    """CLI 只认设计 §1.3.3 的四个基础 mode；``*_true/*_false`` 由 pred 描述符区分。"""
    if mode in ("grad_eager", "grad_jit"):
        return mode
    if mode == "reject":
        return "reject"
    if mode == "eager" or mode.startswith("eager_"):
        return "eager"
    if mode == "jit" or mode.startswith("jit_"):
        return "jit"
    raise ValueError(f"未知 mode：{mode!r}")


def _input_tree_spec(fixture: dict) -> dict:
    leaves = fixture["leaves"]
    if fixture["kind"] == "dict":
        return {"kind": "dict", "keys": [leaf["name"] for leaf in leaves],
                "children": {leaf["name"]: {"kind": "leaf", "path": leaf["path"]}
                             for leaf in leaves}}
    if fixture["kind"] == "list":
        return {"kind": "list",
                "children": [{"kind": "leaf", "path": leaf["path"]} for leaf in leaves]}
    return {"kind": "leaf", "path": leaves[0]["path"]}


def build_operands(fixture: dict, convert=None):
    """按 fixture 的 kind 还原 operand pytree（单叶子/ dict / list）。"""
    leaves = {leaf["path"]: fixture["arrays"][leaf["path"]] for leaf in fixture["leaves"]}
    return tree_from_spec(fixture["input_tree"], leaves, convert=convert)


def compute_arrays_sha256(arrays: dict[str, np.ndarray]) -> str:
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
    """写候选/golden 共用的 case.json（原用例 + ``_resolved``），不改源用例文件。"""
    resolved = {
        "op": OP_NAME,
        "case_id": fixture["case_id"],
        "variant": fixture["variant"],
        "mode": fixture["mode"],
        "base_mode": fixture["base_mode"],
        "family": fixture["family"],
        "kind": fixture["kind"],
        "leaves": fixture["leaves"],
        "leaf_order": fixture["leaf_order"],
        "input_tree": fixture["input_tree"],
        "pred": fixture["pred_descriptor"],
        "pred_kind": fixture["pred_kind"],
        "selected": fixture["selected"],
        "true_name": fixture["true_name"],
        "false_name": fixture["false_name"],
        "branch_ids": [fixture["true_id"], fixture["false_id"]],
        "branch_exprs": [fixture["true_name"], fixture["false_name"]],
        "output_paths": fixture["output_paths"],
        "output_shapes": fixture["output_shapes"],
        "seed": fixture["seed"],
        "dtype": fixture["dtype"],
        "expected": fixture["expected"],
        "expected_error": fixture["expected_error"],
        "inputs_sha256": fixture["inputs_sha256"],
        "numel": fixture["numel"],
    }
    payload = dict(case)
    payload.update({
        "case_id": fixture["case_id"],
        "variant": fixture["variant"],
        "op": OP_NAME,
        "mode": fixture["base_mode"],
        "pred": fixture["pred_descriptor"],
        "pred_kind": fixture["pred_kind"],
        "branch_ids": [fixture["true_id"], fixture["false_id"]],
        "branch_exprs": [fixture["true_name"], fixture["false_name"]],
        "operands": [{"path": leaf["path"], "shape": leaf["shape"],
                      "dtype": leaf["dtype"]} for leaf in fixture["leaves"]],
        "expected": fixture["expected"],
        "expected_error": fixture["expected_error"],
        "seed": fixture["seed"],
        "_resolved": resolved,
    })
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return payload


def fixture_from_case(case_obj: dict, arrays: dict[str, np.ndarray]) -> dict:
    """golden / 候选用：从 case.json（含 ``_resolved``）+ inputs.npz 还原 fixture。"""
    resolved = case_obj.get("_resolved")
    if not isinstance(resolved, dict):
        raise ValueError("case.json 缺少 _resolved（应由 runner / pytest 写入）")
    fixture = {
        "schema": SCHEMA,
        "op": OP_NAME,
        "case_id": case_obj.get("case_id"),
        "variant": case_obj.get("variant", ""),
        "mode": resolved["mode"],
        "base_mode": resolved.get("base_mode") or base_mode_of(resolved["mode"]),
        "group": case_obj.get("group"),
        "required": bool(case_obj.get("required", True)),
        "expected": resolved.get("expected", "match_cpu"),
        "expected_error": resolved.get("expected_error"),
        "dtype": resolved.get("dtype", "float32"),
        "family": resolved.get("family", "numpy"),
        "seed": int(resolved.get("seed", DEFAULT_SEED)),
        "kind": resolved["kind"],
        "leaves": [dict(item) for item in resolved["leaves"]],
        "leaf_order": list(resolved["leaf_order"]),
        "input_tree": resolved["input_tree"],
        "arrays": {key: np.asarray(value) for key, value in arrays.items()},
        "pred_kind": resolved.get("pred_kind"),
        "selected": bool(resolved.get("selected")),
        "true_name": resolved["true_name"],
        "false_name": resolved["false_name"],
        "true_id": int(resolved["branch_ids"][0]),
        "false_id": int(resolved["branch_ids"][1]),
    }
    fixture["pred_descriptor"] = resolved.get("pred")
    fixture["pred"] = pred_from_descriptor(fixture["pred_descriptor"])
    fixture["output_paths"] = list(resolved.get("output_paths") or [])
    fixture["output_shapes"] = {key: list(value) for key, value in
                                (resolved.get("output_shapes") or {}).items()}
    fixture["inputs_sha256"] = compute_arrays_sha256(fixture["arrays"])
    expected = resolved.get("inputs_sha256")
    if expected and expected != fixture["inputs_sha256"]:
        raise ValueError(
            f"inputs.npz 指纹与 case.json 不一致：{fixture['inputs_sha256']} != {expected}")
    missing = [path for path in fixture["leaf_order"] if path not in fixture["arrays"]]
    if missing:
        raise ValueError(f"inputs.npz 缺叶子：{missing}")
    fixture["numel"] = int(sum(int(np.prod(leaf["shape"], dtype=np.int64))
                               for leaf in fixture["leaves"]))
    return fixture


def leaf_bytes(shape: list[int], dtype: str = "float32") -> dict:
    """按 ``bytes = elements * dtype_size`` 实算，返回元素数/字宽/总字节/%32。"""
    size = {"float32": 4, "int32": 4, "bool": 1}.get(str(dtype))
    if size is None:
        raise ValueError(f"未知 dtype：{dtype!r}")
    elements = int(np.prod(shape, dtype=np.int64)) if shape else 1
    total = elements * size
    return {"elements": elements, "dtype": dtype, "dtype_size": size,
            "bytes": total, "bytes_mod_32": total % 32}


if __name__ == "__main__":  # 自检：同一 (case, variant, mode) 必须字节级相同
    demo = {"case_id": "cond_true_eager", "pred": True, "operand": "float32[8,16]",
            "branches": ["x + 1", "x - 1"], "mode": ["eager"], "expected": "match_cpu"}
    f1, f2 = make_fixture(demo, "", "eager"), make_fixture(demo, "", "eager")
    print("deterministic:", f1["inputs_sha256"] == f2["inputs_sha256"],
          f1["inputs_sha256"][:16])
    print("outputs:", f1["output_shapes"])
    print("bytes w17:", leaf_bytes([17]))
