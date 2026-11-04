"""自动优化闭环对比与门禁（M3-a）：`services/perf_diff.py` 的 fail/warn code 行为。

对应方案 v1.1 §13-A5（performance-diff.json + 成对 run 关联）与需求 C6（自动优化闭环）。
不连远端、不跑容器：用合成 run 目录 + manifest 造出「优化前 / 优化后」两组执行。
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from app.models import Loop, Run
from app.services import perf_diff

SNAP_A = "a" * 64
SNAP_B = "b" * 64
CRITERIA = "ops-precision-standard@1"
SUITE = "cases@deadbeef"
DIGEST = "sha256:cafe"


def _sha(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def _run_dir(root: Path, name: str, *, snapshot: str, criteria: str = CRITERIA, suite: str = SUITE,
             perf_cases: list[dict] | None = None, corr_cases: list[dict] | None = None) -> Path:
    d = root / name
    (d / "out").mkdir(parents=True)
    arts = []
    common = {"spec_version_id": 1, "code_snapshot_sha256": snapshot, "criteria_ref": criteria,
              "suite_ref": suite}
    if perf_cases is not None:
        perf = {**common, "cases": perf_cases,
                "summary": {"primary_case": perf_cases[0]["case_id"] if perf_cases else None}}
        p = d / "out" / "performance.json"
        body = json.dumps(perf, ensure_ascii=False).encode()
        p.write_bytes(body)
        arts.append({"type": "performance", "path": "out/performance.json", "sha256": _sha(body)})
    if corr_cases is not None:
        corr = {**common, "cases": corr_cases}
        p = d / "out" / "correctness.json"
        body = json.dumps(corr, ensure_ascii=False).encode()
        p.write_bytes(body)
        arts.append({"type": "correctness", "path": "out/correctness.json", "sha256": _sha(body)})
        rp = d / "out" / "run.log"
        rp.write_bytes(b"ok\n")
        arts.append({"type": "log", "path": "out/run.log", "sha256": _sha(b"ok\n")})
    man = {"schema": "opx-manifest/1", "run_id": name, "exit_code": 0, "image_digest": DIGEST,
           "code_snapshot_sha256": snapshot, "spec_version_id": 1, "criteria_ref": criteria,
           "suite_ref": suite, "device": "0", "artifacts": arts}
    mp = d / "manifest.json"
    mp.write_text(json.dumps(man, ensure_ascii=False, indent=2), encoding="utf-8")
    return mp


def _perf_case(cid: str, med: float, *, std: float = 0.02, n: int = 1048576) -> dict:
    return {"case_id": cid, "n": n,
            "candidate": {"device_us": {"median_us": med, "std_us": std}},
            "baseline": {"device_us": {"median_us": med + 0.3}}}


def _perf_case_median(cid: str, med: float | None, *, std: float = 0.02,
                      n: int = 1048576) -> dict:
    """构造候选侧 `median_us` 缺失 / 为 0 的用例（模拟 benchmark 漏采，M12 口径）。"""
    device = {} if med is None else {"median_us": med, "std_us": std}
    return {"case_id": cid, "n": n,
            "candidate": {"device_us": device},
            "baseline": {"device_us": {"median_us": (med or 0.0) + 0.3}}}


def _corr(cid: str, status: str = "PASS") -> dict:
    return {"case_id": cid, "status": status, "required": True}


def _add(db, task, *, action, manifest: Path, snapshot: str, loop, criteria: str = CRITERIA,
         suite: str = SUITE, device: str = "0") -> Run:
    row = Run(task_id=task.id, loop_id=loop.id, action=action, status="ok", exit_code=0,
              manifest_path=str(manifest), code_snapshot_sha256=snapshot, criteria_ref=criteria,
              suite_ref=suite, image_digest=DIGEST, device_id=device, spec_version_id=1)
    db.add(row)
    db.commit()
    return row


def _loop(db, task, kind: str = "perf") -> Loop:
    row = Loop(task_id=task.id, kind=kind, scope="增大分块", hypothesis="MTE2 搬运受限",
               judge_condition="精度不退化 + 同条件 A/B")
    db.add(row)
    db.commit()
    return row


def _pair(db, seeded, tmp_path: Path, *, before_med: float, after_med: float,
          before_corr: list[dict] | None = None, after_corr: list[dict] | None = None,
          after_criteria: str = CRITERIA) -> tuple[Loop, list[Run]]:
    task = seeded["task"]
    loop = _loop(db, task)
    perf_b = [_perf_case("bench_1m", before_med)]
    perf_a = [_perf_case("bench_1m", after_med)]
    cb = before_corr if before_corr is not None else [_corr("add_norm_001")]
    ca = after_corr if after_corr is not None else [_corr("add_norm_001")]
    mb_test = _run_dir(tmp_path, "b_test", snapshot=SNAP_A, corr_cases=cb)
    mb_bench = _run_dir(tmp_path, "b_bench", snapshot=SNAP_A, perf_cases=perf_b)
    ma_test = _run_dir(tmp_path, "a_test", snapshot=SNAP_B, criteria=after_criteria, corr_cases=ca)
    ma_bench = _run_dir(tmp_path, "a_bench", snapshot=SNAP_B, criteria=after_criteria, perf_cases=perf_a)
    runs = [
        _add(db, task, action="test", manifest=mb_test, snapshot=SNAP_A, loop=loop),
        _add(db, task, action="benchmark", manifest=mb_bench, snapshot=SNAP_A, loop=loop),
        _add(db, task, action="test", manifest=ma_test, snapshot=SNAP_B, loop=loop, criteria=after_criteria),
        _add(db, task, action="benchmark", manifest=ma_bench, snapshot=SNAP_B, loop=loop, criteria=after_criteria),
    ]
    return loop, runs


def _codes(report: dict) -> set[str]:
    return {f["code"] for f in report["findings"]}


def _loop_objective(db, task, objective: dict | None) -> Loop:
    """建一个带 objective 的 perf Loop（None = 模拟没有 objective 的旧 Loop）。"""
    loop = _loop(db, task)
    if objective is not None:
        loop.objective_json = json.dumps(objective, ensure_ascii=False)
        db.commit()
    return loop


def _runs_abc(db, task, tmp_path: Path, *, loop, perf_before: list[dict],
              perf_after: list[dict], before_corr: list[dict] | None = None,
              after_corr: list[dict] | None = None, tag: str = "p") -> list[Run]:
    """造 before/after 各 test+benchmark 四条 run（合成产物，不连远端）。"""
    cb = before_corr if before_corr is not None else [_corr("add_norm_001")]
    ca = after_corr if after_corr is not None else [_corr("add_norm_001")]
    mb_test = _run_dir(tmp_path, f"{tag}_b_test", snapshot=SNAP_A, corr_cases=cb)
    mb_bench = _run_dir(tmp_path, f"{tag}_b_bench", snapshot=SNAP_A, perf_cases=perf_before)
    ma_test = _run_dir(tmp_path, f"{tag}_a_test", snapshot=SNAP_B, corr_cases=ca)
    ma_bench = _run_dir(tmp_path, f"{tag}_a_bench", snapshot=SNAP_B, perf_cases=perf_after)
    return [
        _add(db, task, action="test", manifest=mb_test, snapshot=SNAP_A, loop=loop),
        _add(db, task, action="benchmark", manifest=mb_bench, snapshot=SNAP_A, loop=loop),
        _add(db, task, action="test", manifest=ma_test, snapshot=SNAP_B, loop=loop),
        _add(db, task, action="benchmark", manifest=ma_bench, snapshot=SNAP_B, loop=loop),
    ]


def test_improved_passes(db_session, seeded, tmp_path: Path):
    loop, runs = _pair(db_session, seeded, tmp_path, before_med=10.0, after_med=8.0)
    report = perf_diff.evaluate_loop(loop, runs)
    assert report["verdict"] == "pass", report["findings"]
    assert report["conclusion"] == "improved"
    assert report["cases"][0]["delta_pct"] == pytest.approx(-20.0)
    assert report["precision"]["non_regression"] is True


def test_no_significant_change(db_session, seeded, tmp_path: Path):
    loop, runs = _pair(db_session, seeded, tmp_path, before_med=10.0, after_med=10.1)
    report = perf_diff.evaluate_loop(loop, runs)
    assert report["verdict"] == "pass"
    assert report["conclusion"] == "no_significant_change"
    assert "W_PERF_NO_SIGNIFICANT_CHANGE" in _codes(report)


def test_perf_regression_warns_but_does_not_fail(db_session, seeded, tmp_path: Path):
    """不设加速比硬线：变慢只 warn，不判 fail（需求调整已定）。"""
    loop, runs = _pair(db_session, seeded, tmp_path, before_med=10.0, after_med=12.0)
    report = perf_diff.evaluate_loop(loop, runs)
    assert report["verdict"] == "pass"
    assert report["conclusion"] == "regressed"
    assert "W_PERF_REGRESSED" in _codes(report)


def test_selected_target_regression_drives_conclusion_and_hard_gate(db_session, seeded, tmp_path: Path):
    """benchmark 主用例变快也不能掩盖本轮 target_cases 的显著回退。"""
    task = seeded["task"]
    loop = _loop(db_session, task)
    loop.objective_json = json.dumps({
        "type": "perf", "target_dimension": "cases", "target_cases": ["bench_64k"],
        "constraints": {"perf_non_regression": True},
    })
    db_session.commit()
    perf_b = [_perf_case("bench_1m", 10.0), _perf_case("bench_64k", 5.0, n=65536)]
    perf_a = [_perf_case("bench_1m", 8.0), _perf_case("bench_64k", 8.0, n=65536)]
    mb_test = _run_dir(tmp_path, "target_b_test", snapshot=SNAP_A, corr_cases=[_corr("add_norm_001")])
    mb_bench = _run_dir(tmp_path, "target_b_bench", snapshot=SNAP_A, perf_cases=perf_b)
    ma_test = _run_dir(tmp_path, "target_a_test", snapshot=SNAP_B, corr_cases=[_corr("add_norm_001")])
    ma_bench = _run_dir(tmp_path, "target_a_bench", snapshot=SNAP_B, perf_cases=perf_a)
    runs = [
        _add(db_session, task, action="test", manifest=mb_test, snapshot=SNAP_A, loop=loop),
        _add(db_session, task, action="benchmark", manifest=mb_bench, snapshot=SNAP_A, loop=loop),
        _add(db_session, task, action="test", manifest=ma_test, snapshot=SNAP_B, loop=loop),
        _add(db_session, task, action="benchmark", manifest=ma_bench, snapshot=SNAP_B, loop=loop),
    ]
    report = perf_diff.evaluate_loop(loop, runs)
    assert report["conclusion"] == "regressed"
    assert report["verdict"] == "fail"
    assert report["summary"]["decision_basis"] == "target_cases"
    assert report["summary"]["target_case_ids"] == ["bench_64k"]
    assert "E_TARGET_PERF_REGRESSION" in _codes(report)


def test_precision_round_target_cases_do_not_gate_benchmark(db_session, seeded, tmp_path: Path):
    """H2 回归：precision 轮的 target_cases 是 correctness 用例 id，不是 benchmark case_id。

    旧实现把它们当 benchmark case_id 去配对 → 永远 E_TARGET_CASE_MISSING / verdict=fail
    （而 conclusion 又可能是 improved，自相矛盾）。修复后 perf-diff 回到 benchmark primary
    的兼容口径，精度目标匹配仍由 optimization.report 的 precision 段负责。
    """
    task = seeded["task"]
    loop = _loop_objective(db_session, task, {
        "type": "precision", "target_dimension": "cases", "target_cases": ["add_norm_001"],
        "constraints": {"precision_non_regression": True, "perf_non_regression": True},
    })
    runs = _runs_abc(db_session, task, tmp_path, loop=loop,
                     perf_before=[_perf_case("bench_1m", 10.0)],
                     perf_after=[_perf_case("bench_1m", 9.5)])
    report = perf_diff.evaluate_loop(loop, runs)
    assert report["verdict"] == "pass", report["findings"]
    assert "E_TARGET_CASE_MISSING" not in _codes(report)
    assert "E_TARGET_PERF_REGRESSION" not in _codes(report)
    assert report["summary"]["decision_basis"] == "benchmark_primary"
    assert report["summary"]["target_case_ids"] == []
    assert report["conclusion"] == "improved"
    # 精度侧仍按全局「不退化」硬拦，与 target_cases 无关
    assert report["precision"]["non_regression"] is True


def test_dimension_all_non_primary_regression_drives_gate(db_session, seeded, tmp_path: Path):
    """H3 回归：dimension=all 时任一可配对用例显著回退都必须驱动 regressed / 门禁 fail。

    旧实现只看 benchmark primary（bench_1m 变快）→ verdict=pass / improved，bench_64k 的
    +85% 回退被放行（fail-open）；报告层却按全部用例聚合判回退，两层结论矛盾。
    """
    task = seeded["task"]
    objective = {"type": "perf", "target_dimension": "all",
                 "constraints": {"perf_non_regression": True}}
    loop = _loop_objective(db_session, task, objective)
    perf_b = [_perf_case("bench_1m", 10.0), _perf_case("bench_64k", 4.0, n=65536)]
    perf_a = [_perf_case("bench_1m", 9.0), _perf_case("bench_64k", 7.4, n=65536)]
    report = perf_diff.evaluate_loop(
        loop, _runs_abc(db_session, task, tmp_path, loop=loop, tag="all-hard",
                        perf_before=perf_b, perf_after=perf_a))
    assert report["conclusion"] == "regressed"
    assert report["verdict"] == "fail"
    assert report["summary"]["decision_basis"] == "all_cases"
    assert "E_TARGET_PERF_REGRESSION" in _codes(report)
    finding = next(f for f in report["findings"] if f["code"] == "E_TARGET_PERF_REGRESSION")
    assert finding["evidence"]["case_ids"] == ["bench_64k"]
    # 主用例（bench_1m）的「提升」不能抹掉整体结论
    assert next(r for r in report["cases"] if r["case_id"] == "bench_1m")["verdict"] == "improved"

    # perf_non_regression=false：不硬拦，但也如实记 regressed/warn，不自报 improved/pass
    soft = _loop_objective(db_session, task, {"type": "perf", "target_dimension": "all"})
    soft_report = perf_diff.evaluate_loop(
        soft, _runs_abc(db_session, task, tmp_path, loop=soft, tag="all-soft",
                        perf_before=perf_b, perf_after=perf_a))
    assert soft_report["conclusion"] == "regressed"
    assert soft_report["verdict"] == "pass"
    assert "W_PERF_REGRESSED" in _codes(soft_report)
    assert "E_TARGET_PERF_REGRESSION" not in _codes(soft_report)


def test_legacy_loop_without_objective_keeps_benchmark_primary(db_session, seeded, tmp_path: Path):
    """兼容口径：没有 objective 的旧 Loop 仍只看 benchmark primary，不因非主用例回退改结论。"""
    task = seeded["task"]
    loop = _loop_objective(db_session, task, None)
    perf_b = [_perf_case("bench_1m", 10.0), _perf_case("bench_64k", 4.0, n=65536)]
    perf_a = [_perf_case("bench_1m", 9.0), _perf_case("bench_64k", 7.4, n=65536)]
    report = perf_diff.evaluate_loop(
        loop, _runs_abc(db_session, task, tmp_path, loop=loop, tag="legacy",
                        perf_before=perf_b, perf_after=perf_a))
    assert report["conclusion"] == "improved"
    assert report["verdict"] == "pass"
    assert report["summary"]["decision_basis"] == "benchmark_primary"
    assert "E_TARGET_PERF_REGRESSION" not in _codes(report)


@pytest.mark.parametrize("before_med,after_med", [
    (10.0, None),   # after 缺 median_us
    (10.0, 0.0),    # after median_us=0（漏采常见形态）
    (None, 10.0),   # before 缺
    (0.0, 10.0),    # before 为 0
])
def test_missing_measurement_is_inconclusive_not_improved(
        db_session, seeded, tmp_path: Path, before_med: float | None, after_med: float | None):
    """M12 回归：任一侧缺有效 median → inconclusive/missing_data，不得按 −100% 判 improved。"""
    task = seeded["task"]
    loop = _loop_objective(db_session, task, {"type": "perf", "target_dimension": "all"})
    report = perf_diff.evaluate_loop(
        loop, _runs_abc(db_session, task, tmp_path, loop=loop, tag="missing",
                        perf_before=[_perf_case_median("bench_1m", before_med)],
                        perf_after=[_perf_case_median("bench_1m", after_med)]))
    assert report["verdict"] == "pass", report["findings"]  # 缺测量 ≠ fail（判不了如实 inconclusive）
    assert report["conclusion"] == "inconclusive"
    row = report["cases"][0]
    assert row["verdict"] == "inconclusive"
    assert row["missing_data"] is True
    assert row["delta_pct"] is None          # 不伪造 0% / −100%
    assert "W_PERF_MISSING_DATA" in _codes(report)
    assert "W_PERF_INCONCLUSIVE" in _codes(report)


def test_target_case_present_but_unmeasured_is_inconclusive_not_case_missing(
        db_session, seeded, tmp_path: Path):
    """目标用例在前后报告里都存在、只是缺有效 median → inconclusive；E_TARGET_CASE_MISSING 不成立。"""
    task = seeded["task"]
    loop = _loop_objective(db_session, task, {
        "type": "perf", "target_dimension": "cases", "target_cases": ["bench_64k"],
        "constraints": {"perf_non_regression": True},
    })
    report = perf_diff.evaluate_loop(
        loop, _runs_abc(db_session, task, tmp_path, loop=loop, tag="target-unmeasured",
                        perf_before=[_perf_case("bench_1m", 10.0),
                                     _perf_case("bench_64k", 5.0, n=65536)],
                        perf_after=[_perf_case("bench_1m", 8.0),
                                    _perf_case_median("bench_64k", None, n=65536)]))
    assert report["verdict"] == "pass", report["findings"]
    assert "E_TARGET_CASE_MISSING" not in _codes(report)
    assert "E_TARGET_PERF_REGRESSION" not in _codes(report)
    assert report["conclusion"] == "inconclusive"
    assert report["summary"]["decision_basis"] == "target_cases"
    row = next(r for r in report["cases"] if r["case_id"] == "bench_64k")
    assert row["verdict"] == "inconclusive" and row["missing_data"] is True


def test_target_case_absent_from_reports_still_fails(db_session, seeded, tmp_path: Path):
    """E_TARGET_CASE_MISSING 保留：目标用例在报告里根本不存在时仍然 fail，且结论不自相矛盾。"""
    task = seeded["task"]
    loop = _loop_objective(db_session, task, {
        "type": "perf", "target_dimension": "cases", "target_cases": ["bench_nope"],
        "constraints": {"perf_non_regression": True},
    })
    report = perf_diff.evaluate_loop(
        loop, _runs_abc(db_session, task, tmp_path, loop=loop, tag="target-absent",
                        perf_before=[_perf_case("bench_1m", 10.0)],
                        perf_after=[_perf_case("bench_1m", 8.0)]))
    assert report["verdict"] == "fail"
    assert "E_TARGET_CASE_MISSING" in _codes(report)
    assert report["conclusion"] == "inconclusive"   # 不能一边 fail 一边 improved


def _set_oracle(manifest: Path, *, golden: str, criteria: str = "criteria-a") -> None:
    """给合成 correctness.json 注入 oracle 指纹（模拟 tools/run_cases.py 新产物）。"""
    path = manifest.parent / "out" / "correctness.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    data["oracle"] = {"golden_sha256": golden, "criteria_sha256": criteria}
    path.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")


def test_oracle_changed_is_hard_fail(db_session, seeded, tmp_path: Path):
    """参考实现/判据内容变了：即使两侧用例都 PASS，精度对比也不成立。"""
    loop, runs = _pair(db_session, seeded, tmp_path, before_med=10.0, after_med=8.0)
    _set_oracle(Path(runs[0].manifest_path), golden="a" * 64)
    _set_oracle(Path(runs[2].manifest_path), golden="b" * 64)
    report = perf_diff.evaluate_loop(loop, runs)
    assert report["verdict"] == "fail"
    assert "E_ORACLE_CHANGED" in _codes(report)
    assert report["oracle"]["before"]["golden_sha256"] == "a" * 64
    assert report["oracle"]["after"]["golden_sha256"] == "b" * 64


def test_precision_regression_fails(db_session, seeded, tmp_path: Path):
    """精度退化是硬拦：优化不得让正确性变差。"""
    loop, runs = _pair(db_session, seeded, tmp_path, before_med=10.0, after_med=7.0,
                       before_corr=[_corr("add_norm_001", "PASS")],
                       after_corr=[_corr("add_norm_001", "FAIL")])
    report = perf_diff.evaluate_loop(loop, runs)
    assert report["verdict"] == "fail"
    assert "E_PRECISION_REGRESSION" in _codes(report)
    assert report["conclusion"] == "regressed"


def test_not_comparable_when_criteria_differs(db_session, seeded, tmp_path: Path):
    loop, runs = _pair(db_session, seeded, tmp_path, before_med=10.0, after_med=8.0,
                       after_criteria="other-standard@1")
    report = perf_diff.evaluate_loop(loop, runs)
    assert "E_PERF_DIFF_NOT_COMPARABLE" in _codes(report)


def test_device_change_recorded_but_not_a_hard_block(db_session, seeded, tmp_path: Path):
    """2026-10-28：去掉「必须同卡」的要求——前后落在不同设备不再判不可比，只记录。"""
    task = seeded["task"]
    loop = _loop(db_session, task)
    mb_test = _run_dir(tmp_path, "dev_b_test", snapshot=SNAP_A, corr_cases=[_corr("add_norm_001")])
    mb_bench = _run_dir(tmp_path, "dev_b_bench", snapshot=SNAP_A,
                        perf_cases=[_perf_case("bench_1m", 10.0)])
    ma_test = _run_dir(tmp_path, "dev_a_test", snapshot=SNAP_B, corr_cases=[_corr("add_norm_001")])
    ma_bench = _run_dir(tmp_path, "dev_a_bench", snapshot=SNAP_B,
                        perf_cases=[_perf_case("bench_1m", 8.0)])
    runs = [
        _add(db_session, task, action="test", manifest=mb_test, snapshot=SNAP_A,
             loop=loop, device="0"),
        _add(db_session, task, action="benchmark", manifest=mb_bench, snapshot=SNAP_A,
             loop=loop, device="0"),
        _add(db_session, task, action="test", manifest=ma_test, snapshot=SNAP_B,
             loop=loop, device="3"),
        _add(db_session, task, action="benchmark", manifest=ma_bench, snapshot=SNAP_B,
             loop=loop, device="3"),
    ]
    report = perf_diff.evaluate_loop(loop, runs)
    assert "E_PERF_DIFF_NOT_COMPARABLE" not in _codes(report)
    assert report["comparability"]["device_id"] == {
        "before": "0", "after": "3", "same": False, "hard": False,
        "note": "设备只记录不硬判（允许自动切卡）；跨卡绝对耗时可比性交由使用方判断",
    }
    assert report["conclusion"] == "improved"


def test_incomplete_when_only_one_snapshot(db_session, seeded, tmp_path: Path):
    task = seeded["task"]
    loop = _loop(db_session, task)
    m = _run_dir(tmp_path, "only", snapshot=SNAP_A, perf_cases=[_perf_case("bench_1m", 10.0)],
                 corr_cases=[_corr("add_norm_001")])
    runs = [
        _add(db_session, task, action="test", manifest=m, snapshot=SNAP_A, loop=loop),
        _add(db_session, task, action="benchmark", manifest=m, snapshot=SNAP_A, loop=loop),
    ]
    report = perf_diff.evaluate_loop(loop, runs)
    assert "E_PERF_LOOP_INCOMPLETE" in _codes(report)
    assert report["conclusion"] == "inconclusive"


def test_non_perf_loop_is_fail(db_session, seeded, tmp_path: Path):
    task = seeded["task"]
    loop = _loop(db_session, task, kind="design")
    report = perf_diff.evaluate_loop(loop, [])
    assert "E_NOT_PERF_LOOP" in _codes(report)


def test_write_lands_performance_diff_json(db_session, seeded, tmp_path: Path):
    loop, runs = _pair(db_session, seeded, tmp_path, before_med=10.0, after_med=8.0)
    report = perf_diff.evaluate_loop(loop, runs, write=True)
    out = Path(report["written_to"])
    assert out.is_file() and out.name == "performance-diff.json"
    assert json.loads(out.read_text(encoding="utf-8"))["schema"] == "opx-perf-diff/1"


def test_suite_ref_diff_alone_is_not_a_hard_fail(db_session, seeded, tmp_path: Path):
    """suite_ref 会随 cases/ 里的非正确性文件（如基准配置）一起变 → 不能拿它当硬条件。"""
    task = seeded["task"]
    loop = _loop(db_session, task)
    mb_test = _run_dir(tmp_path, "b_test", snapshot=SNAP_A, suite="cases@old",
                       corr_cases=[_corr("add_norm_001")])
    mb_bench = _run_dir(tmp_path, "b_bench", snapshot=SNAP_A, suite="cases@old",
                        perf_cases=[_perf_case("bench_1m", 10.0)])
    ma_test = _run_dir(tmp_path, "a_test", snapshot=SNAP_B, suite="cases@new",
                       corr_cases=[_corr("add_norm_001")])
    ma_bench = _run_dir(tmp_path, "a_bench", snapshot=SNAP_B, suite="cases@new",
                        perf_cases=[_perf_case("bench_1m", 8.0)])
    runs = [
        _add(db_session, task, action="test", manifest=mb_test, snapshot=SNAP_A, loop=loop, suite="cases@old"),
        _add(db_session, task, action="benchmark", manifest=mb_bench, snapshot=SNAP_A, loop=loop, suite="cases@old"),
        _add(db_session, task, action="test", manifest=ma_test, snapshot=SNAP_B, loop=loop, suite="cases@new"),
        _add(db_session, task, action="benchmark", manifest=ma_bench, snapshot=SNAP_B, loop=loop, suite="cases@new"),
    ]
    report = perf_diff.evaluate_loop(loop, runs)
    assert report["verdict"] == "pass", report["findings"]
    assert report["comparability"]["suite_ref"]["same"] is False
    assert report["comparability"]["suite_ref"]["hard"] is False
    assert report["comparability"]["bench_cases"]["same"] is True


def test_recommendations_structured_with_evidence(db_session, seeded, tmp_path: Path):
    """结构化建议（§4.6 R1）：本轮改动 + 面向下一步；证据引用 run/指标，不空口。"""
    task = seeded["task"]
    loop = _loop(db_session, task)
    perf_b = [_perf_case("bench_1m", 10.0), _perf_case("bench_64k", 5.0, n=65536)]
    perf_a = [_perf_case("bench_1m", 8.0), _perf_case("bench_64k", 7.0, n=65536)]  # 主用例变快、小 shape 回退
    mb_test = _run_dir(tmp_path, "b_test", snapshot=SNAP_A, corr_cases=[_corr("add_norm_001")])
    mb_bench = _run_dir(tmp_path, "b_bench", snapshot=SNAP_A, perf_cases=perf_b)
    ma_test = _run_dir(tmp_path, "a_test", snapshot=SNAP_B, corr_cases=[_corr("add_norm_001")])
    ma_bench = _run_dir(tmp_path, "a_bench", snapshot=SNAP_B, perf_cases=perf_a)
    runs = [
        _add(db_session, task, action="test", manifest=mb_test, snapshot=SNAP_A, loop=loop),
        _add(db_session, task, action="benchmark", manifest=mb_bench, snapshot=SNAP_A, loop=loop),
        _add(db_session, task, action="test", manifest=ma_test, snapshot=SNAP_B, loop=loop),
        _add(db_session, task, action="benchmark", manifest=ma_bench, snapshot=SNAP_B, loop=loop),
    ]
    report = perf_diff.evaluate_loop(loop, runs)
    assert report["conclusion"] == "improved"
    recs = report["recommendations"]
    assert recs and recs[0]["status"] == "validated"
    assert recs[0]["observed_evidence"] and recs[0]["validation_method"]
    nxt = [r for r in recs if r["status"] == "proposed"]
    assert nxt and "bench_64k" in nxt[0]["affected_cases"]
    assert all({"id", "bottleneck_hypothesis", "proposed_change", "validation_method", "status"} <= set(r)
               for r in recs)


def test_chip_profile_is_wellformed():
    """硬件画像（方案 §13-A11）必须是合法 YAML，且带来源与不确定项。"""
    yaml = pytest.importorskip("yaml")
    path = Path(__file__).resolve().parents[2] / "workspaces" / "cann-ops" / "ops" / "chips" / "dav-2201.yaml"
    if not path.is_file():
        pytest.skip("硬件画像不在这台机上（运行时数据）")
    doc = yaml.safe_load(path.read_text(encoding="utf-8"))
    assert doc["schema"] == "opx-chip-profile/1"
    assert doc["npu_arch_flag"] == "dav-2201"
    keys = {f["key"] for f in doc["facts"]}
    assert {"ub_size_bytes", "l1_size_bytes", "l0c_size_bytes"} <= keys
    assert doc["runtime_authoritative"] and doc["uncertain"]
    assert doc["optimization_levers"], "优化杠杆清单不能为空（agent 的指路输入）"


def test_text_target_dimension_uses_benchmark_primary(db_session, seeded, tmp_path: Path):
    """文字优化点轮（target_dimension=text）不拿 case_id 配对：按 benchmark primary 兜底。"""
    task = seeded["task"]
    loop = _loop_objective(db_session, task, {
        "type": "perf", "target_dimension": "text",
        "focus": "优化 64K 小 shape 的搬运路径",
        "target_cases": ["case_001"],  # correctness 风格 id：text 维度必须忽略
    })
    report = perf_diff.evaluate_loop(
        loop, _runs_abc(db_session, task, tmp_path, loop=loop, tag="text-target",
                        perf_before=[_perf_case("bench_1m", 10.0)],
                        perf_after=[_perf_case("bench_1m", 8.0)]))
    assert report["summary"]["decision_basis"] == "benchmark_primary"
    assert "E_TARGET_CASE_MISSING" not in _codes(report)
    assert report["conclusion"] == "improved"
