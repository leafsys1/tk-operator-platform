# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""jax_lax_scan 测试夹具：确定性输入 + 规范循环体 + pytree 编解码。

本模块是**测试输入**的唯一来源，golden（JAX CPU 子进程）与候选（NPU 可执行文件 / Python 接口）
都从同一份 ``inputs.npz`` 读取数组，不各自生成随机数。设计依据：
``.cannbot/jax_lax_scan/workflow1/0-黑盒测试设计.md`` §2.1/§2.1.1/§5.5。

约定
----
* 随机源：``numpy.random.default_rng(seed)``，抽样顺序在 ``make_fixture`` 中写死；
  同一 ``(case_id, seed)`` 每次得到字节级相同的输入（``input_sha256`` 可复核）。
* 叶子路径（也是 ``inputs.npz`` / ``outputs.npz`` / ``golden.npz`` 的键）：
  ``init``（单叶子 carry）、``init/0``、``init/1``（tuple carry）、
  ``xs``（单 tensor）、``xs/add``、``xs/scale``（dict pytree xs）、
  ``final_carry``、``final_carry/0``、``final_carry/1``、``ys``、
  ``jvp_primal``、``jvp_tangent``、``vjp_primal``、``vjp_cotangent``。
* 树描述（``carry_tree`` / ``xs_tree``）是纯 JSON：
  ``{"kind": "leaf", "path": "init"}``、``{"kind": "tuple", "children": [...]}``、
  ``{"kind": "list", "children": [...]}``、
  ``{"kind": "dict", "keys": [...], "children": {key: spec}}``。
  候选/测试都用 ``tree_from_spec`` 还原 pytree，不靠猜。
* 循环体用数组 API 中立的写法（``+``、``*``、``reshape``、``astype``/``.to``），
  同一函数可在 numpy / JAX / torch 上执行；``torch`` 只在 ``_cast`` 内惰性导入，
  保证 golden 子进程不会因为 import 本模块而拉起 ``torch_npu``。

明确边界：本模块只负责**测试输入**，不是 oracle；逐叶子判定见 ``criteria/check.py``。
"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

import numpy as np

DEFAULT_SEED = 20261026
DEFAULT_WIDTH = 4

SCHEMA = "opx-scan-fixture/1"
Y_PATH = "ys"


# ---------------------------------------------------------------------------
# 规范循环体（fixture 冻结；节点 1 定义，节点 2 的 .so 注册表按 body_id 实现同一语义）
# ---------------------------------------------------------------------------


def b1_add(carry, x):
    """new = carry + x；y = new（L0 加法 carry）。"""
    new = carry + x
    return new, new


def b3_pytree(carry, x):
    """carry = (int32 counter, f32[W] acc)；new = (counter + 1, acc + x)；y = acc + x。"""
    counter, acc = carry
    new_acc = acc + x
    return (counter + 1, new_acc), new_acc


def b4_quad(carry, x):
    """new = 0.5 * carry + x；y = new * new（仅 JVP/VJP 使用，避免超越函数）。"""
    new = 0.5 * carry + x
    return new, new * new


def b5_bad_shape(carry, x):
    """故意把 carry 从 [4] reshape 成 [2, 2]：破坏 carry shape 不变式。"""
    return carry.reshape(2, 2), carry


def b6_bad_dtype(carry, x):
    """故意把 carry 从 int32 转成 float32：破坏 carry dtype 不变式。"""
    return _cast(carry, "float32"), carry


def b7_bad_struct(carry, x):
    """故意把单叶子 carry 变成二元组：破坏 carry pytree 结构不变式。"""
    return (carry, carry), carry


def pytree_xs_add_scale(carry, x):
    """xs 为 dict(add, scale)：new = carry + x["add"] * x["scale"]。"""
    new = carry + x["add"] * x["scale"]
    return new, new


def none_input_increment(carry, x):
    """xs=None：body 必须收到 x=None，new = carry + 1.0。"""
    assert x is None, "none_input_increment 要求 body 收到 x=None"
    new = carry + 1.0
    return new, new


def _cast(x, dtype: str):
    """数组 API 中立的 dtype 转换（只做转换，不改数值）。"""
    module = type(x).__module__.split(".")[0]
    if module == "torch":
        import torch  # noqa: PLC0415 —— 只在 torch 张量路径惰性导入

        return x.to(getattr(torch, dtype))
    return x.astype(dtype)


BODY_NAMES = {
    1: "B1_add",
    2: "B3_pytree",
    3: "B4_quad",
    4: "B5_bad_shape",
    5: "B6_bad_dtype",
    6: "B7_bad_struct",
    7: "pytree_xs_add_scale",
    8: "none_input_increment",
}

BODIES = {
    1: b1_add,
    2: b3_pytree,
    3: b4_quad,
    4: b5_bad_shape,
    5: b6_bad_dtype,
    6: b7_bad_struct,
    7: pytree_xs_add_scale,
    8: none_input_increment,
}

for _body_id, _fn in BODIES.items():
    # 候选按属性识别 body，不要求与本模块是同一个 module 实例（pytest 与 runner 的导入名可能不同）。
    _fn.body_id = _body_id
    _fn.body_name = BODY_NAMES[_body_id]

BODY_NAME_TO_ID = {name: body_id for body_id, name in BODY_NAMES.items()}


def get_body(body_id) -> callable:
    """按 body_id / body 名取循环体。"""
    if isinstance(body_id, str):
        body_id = BODY_NAME_TO_ID[body_id]
    return BODIES[int(body_id)]


def body_id_of(case: dict) -> int:
    """用例 → body_id（冻结映射；未知用例显式失败，不猜）。"""
    fixture = resolve_fixture_name(case)
    return FIXTURE_BODY_ID[fixture]


# ---------------------------------------------------------------------------
# 用例 → fixture / 几何参数（设计 §5.5 的映射表）
# ---------------------------------------------------------------------------

CASE_FIXTURE = {
    "scan_sum_len8_f32": "B1_add",
    "scan_reverse_len7_f32": "B1_add",
    "scan_pytree_carry": "B3_pytree",
    "scan_pytree_xs": "pytree_xs_add_scale",
    "scan_xs_none_length5": "none_input_increment",
    "scan_jvp_vjp": "B4_quad",
    "scan_bad_carry_shape": "B5_bad_shape",
    "scan_sup_rev_pytree_len6": "B3_pytree",
    "scan_sup_len1_f32": "B1_add",
    "scan_sup_len1_reverse_f32": "B1_add",
    "scan_sup_naninf_len8": "B1_add",
    "scan_sup_bad_carry_dtype": "B6_bad_dtype",
    "scan_sup_bad_carry_struct": "B7_bad_struct",
    "scan_len128_width256_f32": "B1_add",
    "scan_len1024_width256_f32": "B1_add",
}

FIXTURE_BODY_ID = {
    "B1_add": 1,
    "B3_pytree": 2,
    "B4_quad": 3,
    "B5_bad_shape": 4,
    "B6_bad_dtype": 5,
    "B7_bad_struct": 6,
    "pytree_xs_add_scale": 7,
    "none_input_increment": 8,
}

#: 用例里的 ``body`` 字段 → fixture（冻结输入用的是 fixture 名）
BODY_FIELD_FIXTURE = {
    "pytree_xs_add_scale": "pytree_xs_add_scale",
    "none_input_increment": "none_input_increment",
}

FIXTURE_XS_KIND = {
    "B1_add": "tensor",
    "B3_pytree": "tensor",
    "B4_quad": "tensor",
    "B5_bad_shape": "tensor",
    "B6_bad_dtype": "tensor",
    "B7_bad_struct": "tensor",
    "pytree_xs_add_scale": "pytree",
    "none_input_increment": "none",
}


def resolve_fixture_name(case: dict) -> str:
    """解析用例对应的 fixture 名；未知用例直接报错（避免静默用错输入）。"""
    explicit = case.get("fixture")
    if explicit:
        if explicit not in FIXTURE_BODY_ID:
            raise ValueError(f"未知 fixture：{explicit}（支持：{sorted(FIXTURE_BODY_ID)}）")
        return explicit
    body_field = case.get("body")
    if body_field in BODY_FIELD_FIXTURE:
        return BODY_FIELD_FIXTURE[body_field]
    case_id = case.get("case_id")
    if case_id in CASE_FIXTURE:
        return CASE_FIXTURE[case_id]
    raise ValueError(
        f"用例 {case_id!r} 没有 fixture 映射；请补 case.fixture 或 body 字段"
        f"（已冻结：{sorted(CASE_FIXTURE)}）"
    )


def _first_int_after_bracket(text: str) -> int | None:
    m = re.search(r"\[\s*(\d+)\s*(?:,\s*(\d+)\s*)?\]", text or "")
    if not m:
        return None
    return int(m.group(2) or m.group(1))


def case_geometry(case: dict) -> dict:
    """从用例字段解析 (length, width, reverse, seed, dtype)。

    优先 ``shape``；否则 ``length``/``width``；宽度缺失时从 ``carry``/``xs`` 描述解析，
    仍缺失按冻结规则用 4（设计 §5.5 的宽度来源说明）。不生成数组，可单独用于 smoke 选例。
    """
    seed = int(case.get("seed", DEFAULT_SEED))
    dtype = str(case.get("dtype", "float32"))
    reverse = bool(case.get("reverse", False))

    shape = case.get("shape")
    length = width = None
    if shape:
        if len(shape) != 2:
            raise ValueError(f"{case.get('case_id')}: shape 必须是 [length, width]，收到 {shape}")
        length, width = int(shape[0]), int(shape[1])
    else:
        if case.get("length") is not None:
            length = int(case["length"])
        if case.get("width") is not None:
            width = int(case["width"])
    if length is None:
        raise ValueError(f"{case.get('case_id')}: 缺少 length/shape，无法生成静态循环输入")
    if width is None:
        for key in ("carry", "xs"):
            width = _first_int_after_bracket(str(case.get(key) or ""))
            if width is not None:
                break
    if width is None:
        width = DEFAULT_WIDTH
    return {"length": int(length), "width": int(width), "reverse": reverse,
            "seed": seed, "dtype": dtype}


# ---------------------------------------------------------------------------
# pytree 编解码
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
    """递归展开 tuple/list/dict，返回 [(路径, 叶子)]；叶子路径与 npz 键一致。"""
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


def _join(prefix: str, key: str) -> str:
    return f"{prefix}/{key}" if prefix else key


# ---------------------------------------------------------------------------
# 输入生成
# ---------------------------------------------------------------------------


def make_fixture(case: dict) -> dict:
    """生成一条用例的确定性输入与元数据（不落盘）。"""
    fixture_name = resolve_fixture_name(case)
    body_id = FIXTURE_BODY_ID[fixture_name]
    geom = case_geometry(case)
    length, width, seed = geom["length"], geom["width"], geom["seed"]
    dtype = geom["dtype"]
    reverse = geom["reverse"]
    xs_kind = FIXTURE_XS_KIND[fixture_name]

    if dtype not in ("float32", "int32"):
        raise ValueError(f"{case.get('case_id')}: 本批只支持 float32/int32，收到 {dtype}")

    rng = np.random.default_rng(seed)
    arrays: dict[str, np.ndarray] = {}

    def uniform_xs():
        return rng.uniform(-1.0, 1.0, size=(length, width)).astype(np.float32)

    if fixture_name == "B3_pytree":
        arrays["init/0"] = np.asarray(0, dtype=np.int32)
        arrays["init/1"] = np.zeros(width, dtype=np.float32)
        arrays["xs"] = uniform_xs()
        carry_value = (arrays["init/0"], arrays["init/1"])
        xs_value = arrays["xs"]
    elif fixture_name == "B6_bad_dtype":
        arrays["init"] = np.zeros(width, dtype=np.int32)
        arrays["xs"] = uniform_xs()
        carry_value = arrays["init"]
        xs_value = arrays["xs"]
    elif fixture_name == "pytree_xs_add_scale":
        # 冻结抽样顺序：先 add，再 scale（同一 rng 连续取）。
        arrays["init"] = np.zeros(width, dtype=np.float32)
        arrays["xs/add"] = uniform_xs()
        arrays["xs/scale"] = uniform_xs()
        carry_value = arrays["init"]
        xs_value = {"add": arrays["xs/add"], "scale": arrays["xs/scale"]}
    elif fixture_name == "none_input_increment":
        arrays["init"] = np.zeros(width, dtype=np.float32)
        carry_value = arrays["init"]
        xs_value = None
    else:
        # B1_add / B4_quad / B5_bad_shape / B7_bad_struct
        arrays["init"] = np.zeros(width, dtype=np.float32)
        arrays["xs"] = uniform_xs()
        carry_value = arrays["init"]
        xs_value = arrays["xs"]

    if case.get("case_id") == "scan_sup_naninf_len8" or case.get("special") == "naninf":
        if fixture_name != "B1_add":
            raise ValueError("naninf 补充用例固定使用 B1_add fixture")
        if length < 6 or width < 4:
            raise ValueError("naninf 补充用例需要 length>=6 且 width>=4")
        arrays["xs"][2, 0] = np.nan
        arrays["xs"][3, 1] = np.inf
        arrays["xs"][4, 2] = -np.inf
        arrays["xs"][5, 3] = np.float32(-0.0)

    # 前缀固定为 init / xs：npz 键、树描述与候选 ABI 的叶子顺序必须一致。
    carry_tree = tree_spec(carry_value, "init")
    xs_tree = None if xs_value is None else tree_spec(xs_value, "xs")

    fixture = {
        "schema": SCHEMA,
        "case_id": case.get("case_id"),
        "fixture": fixture_name,
        "body_id": body_id,
        "body_name": BODY_NAMES[body_id],
        "length": length,
        "width": width,
        "reverse": reverse,
        "seed": seed,
        "dtype": dtype,
        "family": "torch",
        "xs_kind": xs_kind,
        "carry_tree": carry_tree,
        "xs_tree": xs_tree,
        "carry_paths": tree_leaf_paths(carry_tree),
        "xs_paths": [] if xs_tree is None else tree_leaf_paths(xs_tree),
        "y_paths": [Y_PATH],
        "arrays": arrays,
    }
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
    """写候选/ golden 共用的 case.json（冻结用例 + `_resolved` 元数据），不改 cases.json。"""
    resolved = {
        "fixture": fixture["fixture"],
        "body_id": fixture["body_id"],
        "body_name": fixture["body_name"],
        "length": fixture["length"],
        "width": fixture["width"],
        "reverse": fixture["reverse"],
        "seed": fixture["seed"],
        "dtype": fixture["dtype"],
        "family": fixture["family"],
        "xs_kind": fixture["xs_kind"],
        "carry_tree": fixture["carry_tree"],
        "xs_tree": fixture["xs_tree"],
        "carry_paths": fixture["carry_paths"],
        "xs_paths": fixture["xs_paths"],
        "y_paths": fixture["y_paths"],
        "inputs_sha256": fixture["inputs_sha256"],
    }
    payload = dict(case)
    payload["_resolved"] = resolved
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return payload


def fixture_from_case(case_obj: dict, arrays: dict[str, np.ndarray]) -> dict:
    """golden 子进程用：从 case.json（含 `_resolved`）+ inputs.npz 还原 fixture。"""
    resolved = case_obj.get("_resolved")
    if not isinstance(resolved, dict):
        raise ValueError("case.json 缺少 _resolved（应由 run_cases.py / pytest 写入）")
    fixture = {
        "schema": SCHEMA,
        "case_id": case_obj.get("case_id"),
        "fixture": resolved["fixture"],
        "body_id": int(resolved["body_id"]),
        "body_name": resolved["body_name"],
        "length": int(resolved["length"]),
        "width": int(resolved["width"]),
        "reverse": bool(resolved["reverse"]),
        "seed": int(resolved["seed"]),
        "dtype": resolved["dtype"],
        "family": resolved.get("family", "torch"),
        "xs_kind": resolved["xs_kind"],
        "carry_tree": resolved["carry_tree"],
        "xs_tree": resolved["xs_tree"],
        "carry_paths": list(resolved["carry_paths"]),
        "xs_paths": list(resolved["xs_paths"]),
        "y_paths": list(resolved.get("y_paths") or [Y_PATH]),
        "arrays": {key: np.asarray(val) for key, val in arrays.items()},
    }
    fixture["inputs_sha256"] = compute_arrays_sha256(fixture["arrays"])
    expected = resolved.get("inputs_sha256")
    if expected and expected != fixture["inputs_sha256"]:
        raise ValueError(
            f"inputs.npz 指纹与 case.json 不一致：{fixture['inputs_sha256']} != {expected}"
        )
    return fixture


if __name__ == "__main__":  # 自检：同一 (case, seed) 必须字节级相同
    demo = {"case_id": "scan_sum_len8_f32", "shape": [8, 16], "dtype": "float32", "seed": DEFAULT_SEED}
    f1, f2 = make_fixture(demo), make_fixture(demo)
    print("deterministic:", f1["inputs_sha256"] == f2["inputs_sha256"], f1["inputs_sha256"][:16])
    print("tree:", json.dumps(f1["carry_tree"], ensure_ascii=False), json.dumps(f1["xs_tree"]))
