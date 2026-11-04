# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""jax_lax_while_loop 测试夹具：确定性输入、变体展开、pytree 描述与字节实算。

本模块是**测试输入的唯一来源**：golden（JAX 0.8.1 CPU 子进程）与候选（Ascend C 直调
可执行文件 / Python 接口）都从同一份 ``inputs.npz`` 与同一份 ``case.json`` 读取，不各自
生成随机数，避免 RNG 差异冒充精度差异。设计依据
``$WORK_DIR/0.1-黑盒测试设计.md`` §2.1 / §2.2 / §3.2 / §5.4–§5.5。

约定
----
* 基础种子 ``20261029``；单变体 ``default_rng(seed)``，多变体 ``default_rng(seed + offset)``：
  c03 ``l2=0/l7=1/l13=2``、c07 ``rt0=0/rt1=1/rt8=2``、c10 ``w17=0/w33=1``、
  c12 ``rt0=0/rt1=1/rt7=2``、c13 四个 cond 变体 ``0..3``、c15 ``shape=0/dtype=1``、
  S1 ``weak=0/strong=1``。
* 输入 npz 键与稳定叶子路径一一对应：``carry/0``、``carry/step``、``carry/state/1/aux``、
  裸 carry 为 ``carry``；JVP 切向量为 ``tangent/0``、``tangent/1``。
* 输出键：普通 carry 为 ``out/...``；JVP 为 ``primal/...`` 与 ``tangent/...``。
* 本模块只负责**测试输入与 fixture 语义**，不是 oracle；逐叶数值判定见
  ``criteria/check.py``，CPU JAX 参考见 ``golden.py``。
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

OP_NAME = "jax_lax_while_loop"
SCHEMA = "opx-jax-while-loop-fixture/1"
DEFAULT_SEED = 20261029
CRITERIA_REF = "ops-precision-standard@1"
ELEM_KEY = "carry"
TANGENT_KEY = "tangent"

#: 设计 §5.5 的 16 条冻结用例（按 `cases.json` 顺序即 c01–c16）+ 2 条基准 + 3 条补充。
GATE_IDS = (
    "while_zero_iterations",
    "while_one_iteration",
    "while_dynamic_multi_iterations",
    "while_scalar_carry",
    "while_tuple_carry",
    "while_nested_dict_carry",
    "while_jit_runtime_limit",
    "while_jvp_multi_iteration",
    "while_jvp_zero_iteration",
    "while_unaligned_tail_widths",
    "while_large_tensor",
    "while_iteration_token_and_hits",
    "while_invalid_condition",
    "while_carry_structure_mismatch",
    "while_carry_shape_or_dtype_mismatch",
    "while_iteration_limit_exceeded",
)
BENCH_IDS = ("while_short_small_carry", "while_long_large_tensor")
SUPPLEMENT_IDS = (
    "while_sup_weak_type",
    "while_sup_nonfinite",
    "while_sup_tensor_carry_data_cond",
)

#: 变体偏移（设计 §2.2；看到结果后不得调整）
VARIANT_OFFSETS = {
    "": 0,
    "l2": 0, "l7": 1, "l13": 2,
    "rt0": 0, "rt1": 1, "rt7": 2, "rt8": 2,
    "w17": 0, "w33": 1,
    "bool1": 0, "bool2": 1, "int_scalar": 2, "float_scalar": 3,
    "shape": 0, "dtype": 1,
    "weak": 0, "strong": 1,
}

#: 五类稳定拒绝码（设计 §4.3）
REJECT_CODES = (
    "INVALID_CONDITION",
    "CARRY_STRUCTURE_MISMATCH",
    "CARRY_SHAPE_MISMATCH",
    "CARRY_DTYPE_MISMATCH",
    "ITERATION_LIMIT_EXCEEDED",
)


# ---------------------------------------------------------------------------
# 用例加载与展开
# ---------------------------------------------------------------------------


def _read_json(path: Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def load_gate(path: Path) -> dict:
    suite = _read_json(path)
    if suite.get("op") != OP_NAME:
        raise ValueError(f"cases.op={suite.get('op')!r} != {OP_NAME!r}（{path}）")
    if suite.get("criteria_ref") != CRITERIA_REF:
        raise ValueError(f"cases.criteria_ref={suite.get('criteria_ref')!r} != {CRITERIA_REF!r}")
    cases = list(suite.get("cases") or [])
    if len(cases) != 16:
        raise ValueError(f"冻结 cases 应为 16 条，实际 {len(cases)} 条")
    if tuple(str(case.get("case_id")) for case in cases) != GATE_IDS:
        raise ValueError("冻结 cases 的 case_id/顺序与设计 §5.0 不一致")
    return suite


def load_bench(path: Path) -> dict:
    bench = _read_json(path)
    if bench.get("op") != OP_NAME:
        raise ValueError(f"benchmark.op={bench.get('op')!r} != {OP_NAME!r}（{path}）")
    cases = list(bench.get("cases") or [])
    if len(cases) != 2:
        raise ValueError(f"冻结 benchmark 应为 2 条，实际 {len(cases)} 条")
    if tuple(str(case.get("case_id")) for case in cases) != BENCH_IDS:
        raise ValueError("冻结 benchmark 的 case_id/顺序与设计 §5.0 不一致")
    out = []
    for case in cases:
        item = dict(case)
        item.setdefault("mode", ["jit"])
        item.setdefault("expected", "match_cpu")
        out.append(item)
    return {**bench, "cases": out}


def load_supplement(path: Path) -> dict:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"补充用例文件不存在：{path}")
    suite = _read_json(path)
    if suite.get("op") != OP_NAME:
        raise ValueError(f"supplement.op={suite.get('op')!r} != {OP_NAME!r}")
    cases = list(suite.get("cases") or [])
    if tuple(str(case.get("case_id")) for case in cases) != SUPPLEMENT_IDS:
        raise ValueError("补充用例的 case_id/顺序与设计 §5.4 不一致")
    return suite


def load_whitebox(path: Path) -> dict:
    """白盒用例（节点 3 创建）：文件不存在是正常状态，返回空表。"""
    path = Path(path)
    if not path.is_file():
        return {"schema": "opx-cases/1", "op": OP_NAME, "cases": []}
    suite = _read_json(path)
    if suite.get("op") != OP_NAME:
        raise ValueError(f"whitebox.op={suite.get('op')!r} != {OP_NAME!r}")
    return suite


def variants_for(case: dict) -> list[str]:
    """一条用例的变体列表；单变体返回 ``[""]``。展开口径见设计 §5.0。"""
    case_id = str(case.get("case_id"))
    if case_id == "while_dynamic_multi_iterations":
        return ["l2", "l7", "l13"]
    if case_id == "while_jit_runtime_limit":
        return ["rt0", "rt1", "rt8"]
    if case_id == "while_iteration_token_and_hits":
        return ["rt0", "rt1", "rt7"]
    if case_id == "while_unaligned_tail_widths":
        return ["w17", "w33"]
    if case_id == "while_invalid_condition":
        return ["bool1", "bool2", "int_scalar", "float_scalar"]
    if case_id == "while_carry_shape_or_dtype_mismatch":
        return ["shape", "dtype"]
    if case_id == "while_sup_weak_type":
        return ["weak", "strong"]
    variants = case.get("variants")
    if variants:
        return [str(item) for item in variants]
    return [""]


def modes_for(case: dict) -> list[str]:
    modes = [str(mode) for mode in (case.get("mode") or ["eager"])]
    if not modes:
        raise ValueError(f"{case.get('case_id')}: mode 不能为空")
    for mode in modes:
        if mode not in ("eager", "jit", "jvp_eager", "jvp_jit"):
            raise ValueError(f"{case.get('case_id')}: 未知 mode {mode!r}")
    return modes


def is_reject(case: dict) -> bool:
    if case.get("expected_error"):
        return True
    if str(case.get("case_id")) in (
            "while_invalid_condition",
            "while_carry_structure_mismatch",
            "while_carry_shape_or_dtype_mismatch",
            "while_iteration_limit_exceeded"):
        return True
    return any(isinstance(item, dict) and item.get("expected_error")
               for item in (case.get("variants") or []))


def expand_items(cases: list[dict]) -> list[dict]:
    """按 ``(case, variant, mode)`` 展开；16 条冻结用例应得 51 项。"""
    items: list[dict] = []
    for case in cases:
        for variant in variants_for(case):
            for mode in modes_for(case):
                items.append({
                    "case": case,
                    "case_id": str(case.get("case_id")),
                    "variant": variant,
                    "mode": mode,
                    "display_mode": "reject" if is_reject(case) else mode,
                })
    return items


def count_summary(cases: list[dict]) -> dict:
    items = expand_items(cases)
    return {
        "identities": len(cases),
        "variants": len({(item["case_id"], item["variant"]) for item in items}),
        "mode_executions": len(items),
    }


def variant_offset(variant: str) -> int:
    try:
        return VARIANT_OFFSETS[variant]
    except KeyError as exc:
        raise ValueError(f"未知 variant：{variant!r}（支持 {sorted(VARIANT_OFFSETS)}）") from exc


def limit_value(case: dict, variant: str) -> int | None:
    case_id = str(case.get("case_id"))
    if case_id == "while_dynamic_multi_iterations":
        return {"l2": 2, "l7": 7, "l13": 13}[variant]
    if case_id == "while_jit_runtime_limit":
        return {"rt0": 0, "rt1": 1, "rt8": 8}[variant]
    if case_id == "while_iteration_token_and_hits":
        return {"rt0": 0, "rt1": 1, "rt7": 7}[variant]
    return None


def tail_width(case: dict, variant: str) -> int | None:
    if str(case.get("case_id")) == "while_unaligned_tail_widths":
        return {"w17": 17, "w33": 33}[variant]
    return None


# ---------------------------------------------------------------------------
# 树描述（case.json / golden 共用）
# ---------------------------------------------------------------------------


def _leaf(path: str, shape: list[int], dtype: str) -> dict:
    return {"kind": "leaf", "path": path, "shape": [int(x) for x in shape], "dtype": dtype}


def _tuple(children: list[dict]) -> dict:
    return {"kind": "tuple", "children": children}


def _dict(children: list[tuple[str, dict]]) -> dict:
    return {"kind": "dict", "keys": [key for key, _ in children],
            "children": {key: child for key, child in children}}


def tree_spec(case: dict, variant: str) -> dict:
    """init carry 的稳定树描述（输入 npz 键、shape、dtype）。"""
    case_id = str(case.get("case_id"))
    if case_id in ("while_zero_iterations", "while_one_iteration"):
        return _tuple([_leaf("carry/0", [], "int32"), _leaf("carry/1", [8], "float32")])
    if case_id == "while_dynamic_multi_iterations":
        return _tuple([_leaf("carry/0", [], "int32"), _leaf("carry/1", [], "int32"),
                       _leaf("carry/2", [4, 17], "float32")])
    if case_id == "while_scalar_carry":
        return _leaf("carry", [], "int32")
    if case_id == "while_tuple_carry":
        return _tuple([_leaf("carry/0", [], "int32"), _leaf("carry/1", [6, 4], "float32"),
                       _leaf("carry/2", [6, 2], "float32")])
    if case_id == "while_nested_dict_carry":
        return _dict([
            ("step", _leaf("carry/step", [], "int32")),
            ("state", _tuple([
                _leaf("carry/state/0", [5, 3], "float32"),
                _dict([("aux", _leaf("carry/state/1/aux", [5], "float32"))]),
            ])),
        ])
    if case_id in ("while_jit_runtime_limit", "while_iteration_token_and_hits"):
        return _tuple([_leaf("carry/0", [], "int32"), _leaf("carry/1", [], "int32"),
                       _leaf("carry/2", [4, 33] if case_id == "while_jit_runtime_limit" else [16],
                             "float32")])
    if case_id in ("while_jvp_multi_iteration", "while_jvp_zero_iteration"):
        return _tuple([_leaf("carry/0", [], "int32"), _leaf("carry/1", [5, 4], "float32")])
    if case_id == "while_unaligned_tail_widths":
        width = tail_width(case, variant)
        return _tuple([_leaf("carry/0", [], "int32"), _leaf("carry/1", [width], "float32")])
    if case_id == "while_large_tensor":
        return _tuple([_leaf("carry/0", [], "int32"), _leaf("carry/1", [1024, 256], "float32")])
    if case_id == "while_invalid_condition":
        return _leaf("carry", [8], "float32")
    if case_id == "while_carry_structure_mismatch":
        return _tuple([_leaf("carry/0", [4], "float32"), _leaf("carry/1", [4], "float32")])
    if case_id == "while_carry_shape_or_dtype_mismatch":
        return _leaf("carry", [4], "float32")
    if case_id == "while_iteration_limit_exceeded":
        return _tuple([_leaf("carry/0", [], "int32"), _leaf("carry/1", [8], "float32")])
    if case_id == "while_short_small_carry":
        return _tuple([_leaf("carry/0", [], "int32"), _leaf("carry/1", [16], "float32")])
    if case_id == "while_long_large_tensor":
        return _tuple([_leaf("carry/0", [], "int32"), _leaf("carry/1", [1024, 256], "float32")])
    if case_id == "while_sup_weak_type":
        return _leaf("carry", [], "float32")
    if case_id == "while_sup_nonfinite":
        return _tuple([_leaf("carry/0", [], "int32"), _leaf("carry/1", [8], "float32")])
    if case_id == "while_sup_tensor_carry_data_cond":
        return _leaf("carry", [4], "float32")
    raise ValueError(f"{case_id}: 没有树描述（未知用例）")


def tangent_spec(case: dict, variant: str) -> dict | None:
    case_id = str(case.get("case_id"))
    if case_id not in ("while_jvp_multi_iteration", "while_jvp_zero_iteration"):
        return None
    return _tuple([
        {"kind": "leaf", "path": "tangent/0", "shape": [], "dtype": "float0"},
        _leaf("tangent/1", [5, 4], "float32"),
    ])


def leaf_paths(spec: dict) -> list[dict]:
    """按稳定顺序展开树描述的叶子表。"""
    if spec["kind"] == "leaf":
        return [spec]
    if spec["kind"] == "tuple":
        out: list[dict] = []
        for child in spec["children"]:
            out.extend(leaf_paths(child))
        return out
    if spec["kind"] == "dict":
        out = []
        for key in spec["keys"]:
            out.extend(leaf_paths(spec["children"][key]))
        return out
    raise ValueError(f"未知树节点：{spec['kind']!r}")


def output_tree_spec(case: dict, variant: str) -> dict:
    """输出 carry 的树描述（路径前缀 ``out`` 在 output_paths 中拼接）。"""
    return tree_spec(case, variant)


def _output_key(prefix: str, path: str) -> str:
    """``carry``/``carry/0``/``carry/state/0`` → ``out``/``out/0``/``out/state/0``。"""
    if "/" not in path:
        return prefix
    return f"{prefix}/{path.split('/', 1)[1]}"


def output_paths(case: dict, variant: str, mode: str) -> list[str]:
    """候选 outputs.npz 必须具备的键（含 JVP 的 primal/tangent）。"""
    case_id = str(case.get("case_id"))
    if is_reject(case):
        return []
    leaves = leaf_paths(output_tree_spec(case, variant))
    if mode in ("jvp_eager", "jvp_jit"):
        tan = leaf_paths(tangent_spec(case, variant) or {})
        return ([_output_key("primal", leaf["path"]) for leaf in leaves]
                + [_output_key("tangent", leaf["path"]) for leaf in tan])
    return [_output_key("out", leaf["path"]) for leaf in leaves]


def output_leaf_specs(case: dict, variant: str, mode: str) -> dict[str, dict]:
    """输出键 → 叶子描述（dtype/shape）。"""
    specs: dict[str, dict] = {}
    if is_reject(case):
        return specs
    leaves = leaf_paths(output_tree_spec(case, variant))
    if mode in ("jvp_eager", "jvp_jit"):
        for leaf in leaves:
            specs[_output_key("primal", leaf["path"])] = dict(leaf)
        for leaf in leaf_paths(tangent_spec(case, variant) or {}):
            specs[_output_key("tangent", leaf["path"])] = dict(leaf)
    else:
        for leaf in leaves:
            specs[_output_key("out", leaf["path"])] = dict(leaf)
    return specs


# ---------------------------------------------------------------------------
# 树容器指纹（设计 §2.3 第 139 行：稳定叶子路径集合 + 容器类型精确相等）
# ---------------------------------------------------------------------------


def tree_signature(value) -> list | str:
    """把实际 pytree 归一化为可 JSON 的容器指纹。

    ``tuple`` 与 ``list`` 必须可区分（两者叶子路径相同，只有容器类型不同），
    ``dict`` 保留插入顺序。叶子统一记为 ``"leaf"``：

    * tuple ``(a, b)`` → ``["tuple", ["leaf", "leaf"]]``
    * list  ``[a, b]`` → ``["list", ["leaf", "leaf"]]``
    * dict  ``{"x": a}`` → ``["dict", [["x", "leaf"]]]``
    """
    from collections.abc import Mapping

    if isinstance(value, Mapping):
        # dict 键无语义顺序；JAX pytree 会按键排序，这里统一排序后比较容器类型。
        return ["dict", [[str(key), tree_signature(value[key])]
                         for key in sorted(value.keys(), key=str)]]
    if isinstance(value, tuple):
        return ["tuple", [tree_signature(item) for item in value]]
    if isinstance(value, list):
        return ["list", [tree_signature(item) for item in value]]
    return "leaf"


def spec_tree(spec: dict) -> list | str:
    """把 ``tree_spec``/``tangent_spec`` 归一化为与 :func:`tree_signature` 同构的指纹。"""
    kind = spec["kind"]
    if kind == "leaf":
        return "leaf"
    if kind == "tuple":
        return ["tuple", [spec_tree(child) for child in spec["children"]]]
    if kind == "dict":
        return ["dict", [[str(key), spec_tree(spec["children"][key])]
                         for key in sorted(spec["keys"], key=str)]]
    raise ValueError(f"未知树节点：{kind!r}")


def expected_output_tree(case: dict, variant: str, mode: str) -> list:
    """候选/golden 返回值在 flatten **之前**的期望容器指纹。

    普通 mode 就是 carry 树；JVP 的候选接口返回 ``(primal, tangent)`` 二元组。
    """
    if mode in ("jvp_eager", "jvp_jit"):
        tan = tangent_spec(case, variant)
        if tan is None:
            raise ValueError(f"{case.get('case_id')}: {mode} 缺少 tangent 树描述")
        return ["tuple", [spec_tree(tree_spec(case, variant)), spec_tree(tan)]]
    return spec_tree(tree_spec(case, variant))


# ---------------------------------------------------------------------------
# 确定性输入生成
# ---------------------------------------------------------------------------


def _gen_float(rng: np.random.Generator, shape: list[int]) -> np.ndarray:
    return rng.uniform(-1.0, 1.0, size=shape).astype(np.float32)


def input_arrays(case: dict, variant: str, seed: int = DEFAULT_SEED) -> dict[str, np.ndarray]:
    """按设计 §2.2/§5.5 生成确定性输入；同一 case/variant 的 eager/jit 共用。"""
    case_id = str(case.get("case_id"))
    offset = variant_offset(variant)
    rng = np.random.default_rng(int(seed) + offset)
    if case_id in ("while_zero_iterations", "while_one_iteration"):
        return {"carry/0": np.asarray(0, dtype=np.int32), "carry/1": _gen_float(rng, [8])}
    if case_id == "while_dynamic_multi_iterations":
        return {"carry/0": np.asarray(0, dtype=np.int32),
                "carry/1": np.asarray(limit_value(case, variant), dtype=np.int32),
                "carry/2": _gen_float(rng, [4, 17])}
    if case_id == "while_scalar_carry":
        return {"carry": np.asarray(0, dtype=np.int32)}
    if case_id == "while_tuple_carry":
        return {"carry/0": np.asarray(0, dtype=np.int32),
                "carry/1": _gen_float(rng, [6, 4]),
                "carry/2": _gen_float(rng, [6, 2])}
    if case_id == "while_nested_dict_carry":
        return {"carry/step": np.asarray(0, dtype=np.int32),
                "carry/state/0": _gen_float(rng, [5, 3]),
                "carry/state/1/aux": _gen_float(rng, [5])}
    if case_id == "while_jit_runtime_limit":
        return {"carry/0": np.asarray(0, dtype=np.int32),
                "carry/1": np.asarray(limit_value(case, variant), dtype=np.int32),
                "carry/2": _gen_float(rng, [4, 33])}
    if case_id in ("while_jvp_multi_iteration", "while_jvp_zero_iteration"):
        return {"carry/0": np.asarray(0, dtype=np.int32),
                "carry/1": _gen_float(rng, [5, 4]),
                "tangent/0": np.zeros((), dtype=np.dtype([("float0", "V")])),
                "tangent/1": _gen_float(rng, [5, 4])}
    if case_id == "while_unaligned_tail_widths":
        width = tail_width(case, variant)
        return {"carry/0": np.asarray(0, dtype=np.int32), "carry/1": _gen_float(rng, [width])}
    if case_id == "while_large_tensor":
        return {"carry/0": np.asarray(0, dtype=np.int32),
                "carry/1": _gen_float(rng, [1024, 256])}
    if case_id == "while_iteration_token_and_hits":
        return {"carry/0": np.asarray(0, dtype=np.int32),
                "carry/1": np.asarray(limit_value(case, variant), dtype=np.int32),
                "carry/2": _gen_float(rng, [16])}
    if case_id == "while_invalid_condition":
        return {"carry": _gen_float(rng, [8])}
    if case_id == "while_carry_structure_mismatch":
        return {"carry/0": _gen_float(rng, [4]), "carry/1": _gen_float(rng, [4])}
    if case_id == "while_carry_shape_or_dtype_mismatch":
        return {"carry": _gen_float(rng, [4])}
    if case_id == "while_iteration_limit_exceeded":
        return {"carry/0": np.asarray(0, dtype=np.int32), "carry/1": _gen_float(rng, [8])}
    if case_id == "while_short_small_carry":
        return {"carry/0": np.asarray(0, dtype=np.int32), "carry/1": _gen_float(rng, [16])}
    if case_id == "while_long_large_tensor":
        return {"carry/0": np.asarray(0, dtype=np.int32),
                "carry/1": _gen_float(rng, [1024, 256])}
    if case_id == "while_sup_weak_type":
        return {"carry": np.asarray(0.0, dtype=np.float32)}
    if case_id == "while_sup_nonfinite":
        special = np.asarray([np.nan, np.inf, -np.inf, -0.0, 0.0, 1.0, -1.0, 3.4028235e38],
                             dtype=np.float32)
        return {"carry/0": np.asarray(0, dtype=np.int32), "carry/1": special}
    if case_id == "while_sup_tensor_carry_data_cond":
        return {"carry": np.asarray([4.0, 2.0, 1.0, 0.5], dtype=np.float32)}
    raise ValueError(f"{case_id}: 没有输入生成规则（未知用例）")


def write_inputs_npz(arrays: dict[str, np.ndarray], path: Path) -> str:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, **arrays)
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


def load_inputs(path: Path) -> dict[str, np.ndarray]:
    with np.load(path) as data:
        return {key: data[key] for key in data.files}


def case_json_payload(case: dict, variant: str, mode: str) -> dict:
    """写进每次候选调用的 case.json（候选/证据共用；不改冻结 cases）。"""
    payload = dict(case)
    payload.update({
        "schema": SCHEMA,
        "op": OP_NAME,
        "case_id": str(case.get("case_id")),
        "variant": variant,
        "mode": mode,
        "display_mode": "reject" if is_reject(case) else mode,
        "tree": tree_spec(case, variant),
        "tangent": tangent_spec(case, variant),
        "expected_iterations": expected_iterations(case, variant),
        "expected_counts": expected_counts(case, variant),
        "bytes": bytes_info(case, variant),
    })
    return payload


def write_case_json(case: dict, variant: str, mode: str, path: Path) -> dict:
    payload = case_json_payload(case, variant, mode)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return payload


# ---------------------------------------------------------------------------
# 语义绑定（golden 与候选接口按同一份 fixture 重建）
# ---------------------------------------------------------------------------


def expected_iterations(case: dict, variant: str) -> int | None:
    case_id = str(case.get("case_id"))
    if is_reject(case):
        return 64 if case_id == "while_iteration_limit_exceeded" else None
    table = {
        "while_zero_iterations": 0,
        "while_one_iteration": 1,
        "while_scalar_carry": 5,
        "while_tuple_carry": 5,
        "while_nested_dict_carry": 4,
        "while_jvp_multi_iteration": 6,
        "while_jvp_zero_iteration": 0,
        "while_unaligned_tail_widths": 5,
        "while_large_tensor": 12,
        "while_short_small_carry": 4,
        "while_long_large_tensor": 64,
        "while_sup_weak_type": 3,
        "while_sup_nonfinite": 2,
        "while_sup_tensor_carry_data_cond": 2,
    }
    if case_id in ("while_dynamic_multi_iterations", "while_jit_runtime_limit",
                   "while_iteration_token_and_hits"):
        return limit_value(case, variant)
    if case_id not in table:
        raise ValueError(f"{case_id}: 没有迭代次数期望")
    return table[case_id]


def expected_counts(case: dict, variant: str) -> dict:
    """正常终止：body_hits=iterations=n, cond_hits=n+1；c16 安全阀：三者都是 64。"""
    case_id = str(case.get("case_id"))
    n = expected_iterations(case, variant)
    if case_id == "while_iteration_limit_exceeded":
        return {"iterations": 64, "body_hits": 64, "cond_hits": 64, "limit_flag": 1}
    if n is None:
        return {"iterations": None, "body_hits": None, "cond_hits": None, "limit_flag": 0}
    return {"iterations": n, "body_hits": n, "cond_hits": n + 1, "limit_flag": 0}


def condition_output_kind(case: dict, variant: str) -> str | None:
    if str(case.get("case_id")) != "while_invalid_condition":
        return None
    return {"bool1": "bool[1]", "bool2": "bool[2]",
            "int_scalar": "int32 scalar", "float_scalar": "float32 scalar"}[variant]


def expected_reject_code(case: dict, variant: str) -> str:
    case_id = str(case.get("case_id"))
    if case_id == "while_carry_shape_or_dtype_mismatch":
        return {"shape": "CARRY_SHAPE_MISMATCH", "dtype": "CARRY_DTYPE_MISMATCH"}[variant]
    code = case.get("expected_error")
    if code not in REJECT_CODES:
        raise ValueError(f"{case_id}: expected_error 非法：{code!r}")
    return str(code)


def body_id(case: dict, variant: str) -> int:
    """fixture 的 body 词表 ID（写入 case.json，便于候选分派与证据复核）。"""
    table = {
        "while_zero_iterations": 1,
        "while_one_iteration": 2,
        "while_dynamic_multi_iterations": 3,
        "while_scalar_carry": 4,
        "while_tuple_carry": 5,
        "while_nested_dict_carry": 6,
        "while_jit_runtime_limit": 7,
        "while_jvp_multi_iteration": 8,
        "while_jvp_zero_iteration": 8,
        "while_unaligned_tail_widths": 9,
        "while_large_tensor": 10,
        "while_iteration_token_and_hits": 7,
        "while_short_small_carry": 11,
        "while_long_large_tensor": 12,
        "while_sup_weak_type": 13,
        "while_sup_nonfinite": 14,
        "while_sup_tensor_carry_data_cond": 15,
        "while_iteration_limit_exceeded": 16,
    }
    case_id = str(case.get("case_id"))
    if case_id in ("while_invalid_condition", "while_carry_structure_mismatch",
                   "while_carry_shape_or_dtype_mismatch"):
        return 0
    return table[case_id]


def cond_id(case: dict, variant: str) -> int:
    table = {
        "while_zero_iterations": 1,
        "while_one_iteration": 2,
        "while_dynamic_multi_iterations": 3,
        "while_scalar_carry": 4,
        "while_tuple_carry": 5,
        "while_nested_dict_carry": 6,
        "while_jit_runtime_limit": 3,
        "while_jvp_multi_iteration": 7,
        "while_jvp_zero_iteration": 1,
        "while_unaligned_tail_widths": 8,
        "while_large_tensor": 9,
        "while_iteration_token_and_hits": 3,
        "while_short_small_carry": 10,
        "while_long_large_tensor": 11,
        "while_sup_weak_type": 12,
        "while_sup_nonfinite": 2,
        "while_sup_tensor_carry_data_cond": 13,
        "while_iteration_limit_exceeded": 14,
    }
    case_id = str(case.get("case_id"))
    if case_id in ("while_invalid_condition", "while_carry_structure_mismatch",
                   "while_carry_shape_or_dtype_mismatch"):
        return 0
    return table[case_id]


# ---------------------------------------------------------------------------
# 字节 / 对齐实算（设计 §3.2）
# ---------------------------------------------------------------------------


def dtype_size(dtype: str) -> int:
    text = str(dtype)
    if "float0" in text:
        return 0
    if "64" in text:
        return 8
    if "16" in text:
        return 2
    if text in ("bool", "int8", "uint8"):
        return 1
    return 4


def bytes_info(case: dict, variant: str) -> dict:
    """逐叶 ``elements × dtype_size = bytes``、``bytes % 32``；0-d 为 1 个元素。"""
    tables = {}
    specs = list(leaf_paths(tree_spec(case, variant)))
    tan = tangent_spec(case, variant)
    if tan:
        specs += leaf_paths(tan)
    for leaf in specs:
        elements = 1
        for dim in leaf["shape"]:
            elements *= int(dim)
        size = dtype_size(leaf["dtype"])
        total = elements * size
        tables[leaf["path"]] = {
            "elements": elements,
            "dtype": leaf["dtype"],
            "dtype_size": size,
            "bytes": total,
            "bytes_mod_32": total % 32,
        }
    return tables


# ---------------------------------------------------------------------------
# 汇总
# ---------------------------------------------------------------------------


def summarize(ws: Path) -> dict:
    """读取目标仓的三类用例并返回计数；供 runner / pytest / 自检共用。"""
    ws = Path(ws)
    gate = load_gate(ws / "cases" / OP_NAME / "cases.json")
    bench = load_bench(ws / "cases" / OP_NAME / "benchmark.json")
    supplement = load_supplement(ws / "tests" / f"{OP_NAME}_supplement_cases.json")
    whitebox = load_whitebox(ws / "tests" / f"{OP_NAME}_whitebox_cases.json")
    return {
        "gate": count_summary(gate["cases"]),
        "benchmark": count_summary(bench["cases"]),
        "supplement": count_summary(supplement["cases"]),
        "whitebox": count_summary(whitebox["cases"]),
        "pytest": {
            "identities": (len(gate["cases"]) + len(bench["cases"])
                           + len(supplement["cases"]) + len(whitebox["cases"])),
            "mode_executions": (count_summary(gate["cases"])["mode_executions"]
                                + count_summary(bench["cases"])["mode_executions"]
                                + count_summary(supplement["cases"])["mode_executions"]
                                + count_summary(whitebox["cases"])["mode_executions"]),
        },
    }


__all__ = [
    "OP_NAME", "SCHEMA", "DEFAULT_SEED", "CRITERIA_REF",
    "GATE_IDS", "BENCH_IDS", "SUPPLEMENT_IDS", "REJECT_CODES",
    "load_gate", "load_bench", "load_supplement", "load_whitebox",
    "variants_for", "modes_for", "is_reject", "expand_items", "count_summary",
    "variant_offset", "limit_value", "tail_width",
    "tree_spec", "tangent_spec", "leaf_paths", "output_tree_spec", "output_paths",
    "output_leaf_specs", "tree_signature", "spec_tree", "expected_output_tree",
    "input_arrays", "write_inputs_npz", "load_inputs",
    "case_json_payload", "write_case_json",
    "expected_iterations", "expected_counts", "condition_output_kind",
    "expected_reject_code", "body_id", "cond_id",
    "dtype_size", "bytes_info", "summarize",
]
