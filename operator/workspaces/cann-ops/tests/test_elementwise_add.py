"""elementwise_add · pytest 交付件（真机运行，读 cases/<op>/cases.json）。"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

WS = Path(__file__).resolve().parents[1]
if str(WS) not in sys.path:
    sys.path.insert(0, str(WS))

np = pytest.importorskip("numpy")
torch = pytest.importorskip("torch")
pytest.importorskip("torch_npu")

from criteria.check import check as criteria_check  # noqa: E402
from ops.elementwise_add.python.elementwise_add import add  # noqa: E402
from ops.elementwise_add.scripts.golden import compute_golden  # noqa: E402
from tools import gen_inputs  # noqa: E402

CASES = json.loads((WS / "cases/elementwise_add/cases.json").read_text(encoding="utf-8"))
MATCH = [c for c in CASES["cases"] if c.get("required", True) and c.get("expected") == "match"]
REJECT = [c for c in CASES["cases"] if c.get("required", True) and c.get("expected") == "reject"]


@pytest.mark.parametrize("case", MATCH, ids=[c["case_id"] for c in MATCH])
def test_match(case: dict) -> None:
    n = int(case["n_a"])
    a = gen_inputs.generate(case.get("gen_a", {}), n, int(case["seed"]), case.get("dtype", "float32"))
    b = gen_inputs.generate(case.get("gen_b", {}), n, int(case["seed"]) + 1, case.get("dtype", "float32"))
    want = compute_golden(a, b)
    got = add(torch.from_numpy(a).npu(), torch.from_numpy(b).npu()).cpu().numpy()
    verdict = criteria_check(got, want)
    assert verdict["is_pass"], f"{case['case_id']}: {verdict.get('failure_reasons')}"


@pytest.mark.parametrize("case", REJECT, ids=[c["case_id"] for c in REJECT])
def test_reject(case: dict) -> None:
    code = case.get("expect_reject_code")
    with pytest.raises((ValueError, TypeError, RuntimeError)) as ei:
        na, nb = int(case["n_a"]), int(case["n_b"])
        dtype = torch.float16 if case.get("dtype") == "float16" else torch.float32
        add(torch.zeros(na, dtype=dtype).npu(), torch.zeros(nb, dtype=dtype).npu())
    assert code in str(ei.value), f"{case['case_id']} 期望 {code}，实际：{ei.value}"
