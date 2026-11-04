"""基准/采集执行器的纯逻辑自测（M1-b）。

只测**不碰设备的部分**：样本聚合、派生量、报告结构、msopprof CSV 解析。
真正跑容器（torch 基线 / msopprof 采集）由 `scripts/nm_test.sh` 实测，
结果记在 `docs/PROGRESS.md`；这里保证「报告写法 + 判据 + 状态标注」这条链不会被改坏。

对应方案 v1.1：§4.6（性能采集与指标状态）、§5（门禁）。
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
WS = REPO / "workspaces" / "cann-ops"
TOOLS = WS / "tools"

pytestmark = pytest.mark.skipif(not WS.is_dir(), reason="任务工作区不在这台机上（运行时数据）")


def _load(name: str):
    spec = importlib.util.spec_from_file_location(f"opx_{name}", TOOLS / f"{name}.py")
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def bench():
    return _load("run_benchmark")


@pytest.fixture(scope="module")
def prof():
    return _load("run_profile")


# ---------------------------------------------------------------- 聚合与派生


def test_aggregate_uses_median_as_primary(bench):
    agg = bench.aggregate([10.0, 12.0, 14.0, 100.0])  # 一个离群点（共享机被邻居打扰）
    assert agg["count"] == 4
    assert agg["median_us"] == 13.0            # 中位数不被离群点拖偏
    assert agg["mean_us"] == 34.0
    assert agg["min_us"] == 10.0
    assert agg["max_us"] == 100.0
    assert agg["std_us"] > 0


def test_aggregate_even_count_and_empty(bench):
    assert bench.aggregate([1.0, 3.0])["median_us"] == 2.0
    with pytest.raises(ValueError):
        bench.aggregate([])


def test_derive_speedup_and_bandwidth(bench):
    d = bench.derive(baseline_median_us=20.0, candidate_median_us=10.0, n=1024)
    assert d["moved_bytes"] == (2 * 1024 + 1024) * 4
    assert d["speedup_baseline_over_candidate"] == pytest.approx(2.0)
    # 10us 搬 12KB → 1.2288 GB/s
    assert d["candidate_achieved_bw_gbps"] == pytest.approx(12 * 1024 / 10e-6 / 1e9, rel=1e-6)
    # 默认画像仍是 elementwise_add（1 flop / 12B）
    assert d["arithmetic_intensity"] == pytest.approx(1024 / (3 * 1024 * 4))


def test_derive_honors_memory_and_compute_profile(bench):
    """relu 是 1 读 1 写，不能沿用 add 的 2 读 1 写；AI 按画像算。"""
    d = bench.derive(20.0, 10.0, 1024, bytes_per_elem=4, reads_per_elem=1,
                     writes_per_elem=1, flops_per_elem=2.0)
    assert d["moved_bytes"] == 2 * 1024 * 4
    assert d["arithmetic_intensity"] == pytest.approx(2.0 / (2 * 4))
    assert d["reads_per_elem"] == 1 and d["writes_per_elem"] == 1


def test_derive_explicit_traffic_for_matrix_like_ops(bench):
    """矩阵类 AI 随形状变：用例可直接给 traffic_bytes / flops。"""
    d = bench.derive(20.0, 10.0, 1024, moved_bytes=3 * 1024 * 4, flops=2 * 1024 ** 3)
    assert d["moved_bytes"] == 3 * 1024 * 4
    assert d["arithmetic_intensity"] == pytest.approx((2 * 1024 ** 3) / (3 * 1024 * 4))


def test_case_profile_defaults_and_overrides(bench):
    base = {"memory": {"reads_per_elem": 2, "writes_per_elem": 1},
            "compute": {"flops_per_elem": 1}}
    assert bench.case_profile(base, {}) == {
        "reads": 2, "writes": 1, "flops_per_elem": 1.0,
        "moved_bytes": None, "flops": None,
    }
    over = bench.case_profile(base, {"reads_per_elem": 1, "writes_per_elem": 1,
                                     "flops_per_elem": 2, "traffic_bytes": 4096, "flops": 8192})
    assert over["reads"] == 1 and over["flops_per_elem"] == 2.0
    assert over["moved_bytes"] == 4096 and over["flops"] == 8192


def test_collect_device_samples_retries_flaky_side(bench, tmp_path: Path):
    """msopprof 偶发采不到（目录在、缺 OpBasicInfo.csv）时，重试后仍能拼出样本，不整轮 abort。"""
    calls = {"n": 0}

    def flaky_run(application, out_dir, *, aic_metrics, warm_up, timeout):
        calls["n"] += 1
        # 第 2 次调用失败（模拟偶发没匹配到 kernel），其余成功
        if calls["n"] == 2:
            d = out_dir
            d.mkdir(parents=True, exist_ok=True)
            (d / "OPPROF_1").mkdir(exist_ok=True)
            return d / "OPPROF_1"          # 缺 OpBasicInfo.csv
        (out_dir / "OPPROF_1").mkdir(parents=True, exist_ok=True)
        (out_dir / "OPPROF_1" / "OpBasicInfo.csv").write_text(
            "Op Name,Op Type,Task Duration(us),Block Dim,Mix Block Dim,Device Id,Pid,Current Freq,Rated Freq,\n"
            f"elementwise_add_kernel,vector,{8.0 + calls['n'] * 0.1},43,NA,0,1,1850,1850,\n",
            encoding="utf-8")
        return out_dir / "OPPROF_1"

    samples, info, failures = bench.collect_device_samples(
        bench._profile_module().parse_op_basic_info, "app", tmp_path,
        invocations=3, retries=2, run_fn=flaky_run)
    assert len(samples) == 3, (samples, failures)
    assert failures, "失败应当被记下来而不是吞掉"
    assert info["op_name"] == "elementwise_add_kernel"


def test_collect_device_samples_all_fail_returns_empty(bench, tmp_path: Path):
    def bad_run(application, out_dir, *, aic_metrics, warm_up, timeout):
        raise RuntimeError("msopprof 没匹配到 kernel")

    samples, _info, failures = bench.collect_device_samples(
        bench._profile_module().parse_op_basic_info, "app", tmp_path,
        invocations=2, retries=2, run_fn=bad_run)
    assert samples == []
    assert len(failures) == 4


# ---------------------------------------------------------------- 报告结构


def _case(bench, *, primary=True):
    case = {
        "case_id": "bench_1m", "n": 1048576, "dtype": "float32", "primary": primary,
        "baseline": {"impl": "torch.add", "op_name": "Add_abc", "device_us": bench.aggregate([8.3, 8.4, 8.35])},
        "candidate": {"impl": "ascendc-direct-invoke", "op_name": "elementwise_add_kernel",
                      "device_us": bench.aggregate([8.0, 8.1, 8.05])},
        "derived": {},
    }
    case["derived"] = bench.derive(case["baseline"]["device_us"]["median_us"],
                                   case["candidate"]["device_us"]["median_us"], n=case["n"])
    return case


def test_performance_report_has_aligned_versions_and_measured_metrics(bench):
    report = bench.build_report(op="elementwise_add", run_id="r1", spec_version_id=1,
                                criteria_ref="ops-precision-standard@1", code_snapshot_sha256="a" * 64,
                                suite_ref="cases@deadbeef", conditions={"invocations": 5, "warm_up": 2},
                                cases=[_case(bench)])
    assert report["schema"] == "opx-performance/1"
    # 门禁逐字比对这四个字段
    for key in ("spec_version_id", "criteria_ref", "code_snapshot_sha256", "suite_ref"):
        assert key in report
    ids = {m["id"]: m for m in report["metrics"]}
    assert ids["op_time_us"]["status"] == "measured"
    assert ids["op_time_us"]["required"] is True
    assert all(m["status"] in ("measured", "estimated", "unsupported", "failed") for m in report["metrics"])
    # 主值 = 设备侧中位数（两侧同一把尺子）
    assert ids["op_time_us"]["value"] == pytest.approx(8.05)
    assert ids["baseline_op_time_us"]["value"] == pytest.approx(8.35)


# ---------------------------------------------------------------- msopprof 解析


def test_col_mean_skips_na(prof):
    rows = [{"x": "1"}, {"x": "NA"}, {"x": ""}, {"x": "3"}, {"x": "abc"}]
    assert prof.col_mean(rows, "x") == 2.0
    assert prof.col_mean(rows, "nope") is None


def test_parse_op_basic_info(tmp_path: Path, prof):
    p = tmp_path / "OpBasicInfo.csv"
    p.write_text(
        "Op Name,Op Type,Task Duration(us),Block Dim,Mix Block Dim,Device Id,Pid,Current Freq,Rated Freq,\n"
        "elementwise_add_kernel,vector,8.080162,43,NA,0,724,1850,1850,\n", encoding="utf-8")
    info = prof.parse_op_basic_info(p)
    assert info["op_name"] == "elementwise_add_kernel"
    assert info["op_type"] == "vector"
    assert info["task_duration_us"] == pytest.approx(8.080162)
    assert info["block_dim"] == 43
    assert info["device_id"] == 0


def test_build_metrics_marks_four_categories_measured_and_cube_unsupported(prof):
    op_basic = {"op_name": "elementwise_add_kernel", "op_type": "vector", "task_duration_us": 8.08}
    per_block = {"gm_to_ub_bw": 30.9, "main_mem_read_bw": 30.9, "main_mem_write_bw": 15.5,
                 "mte2_ratio": 0.84, "mte3_ratio": 0.38, "ub_read_bw": 30.9, "ub_write_bw": 15.4,
                 "gm_to_ub_datas_kb": 192.0, "vec_ratio": 0.09, "vec_fp32_fops": 25472.0,
                 "l2_hit_rate": 0.17}
    metrics = {m["id"]: m for m in prof.build_metrics(op_basic, per_block)}
    kinds = {m["kind"] for m in metrics.values() if m["status"] == "measured"}
    assert {"duration", "data_movement", "storage", "compute"} <= kinds
    assert metrics["op_time_us"]["required"] is True
    assert metrics["cube_ratio"]["status"] == "unsupported"   # vector 算子没有 cube 单元
    for m in metrics.values():
        assert m["status"] in ("measured", "estimated", "unsupported", "failed")
        assert m["source"], f"{m['id']} 没记来源"


def test_parse_per_block_aggregates_real_shape(tmp_path: Path, prof):
    (tmp_path / "Memory.csv").write_text(
        "aiv_time(us),aiv_gm_to_ub_bw(GB/s),aiv_main_mem_read_bw(GB/s),GM_to_UB_datas(KB),\n"
        "5.9,30.0,30.0,192.0,\n6.1,32.0,32.0,192.0,\n", encoding="utf-8")
    (tmp_path / "MemoryUB.csv").write_text(
        "aiv_ub_read_bw_vector(GB/s),aiv_ub_write_bw_vector(GB/s),\n30.0,15.0,\n32.0,16.0,\n",
        encoding="utf-8")
    (tmp_path / "PipeUtilization.csv").write_text(
        "aiv_vec_ratio,aiv_mte2_ratio,aiv_mte3_ratio,\n0.08,0.80,0.30,\n0.10,0.88,0.40,\n",
        encoding="utf-8")
    out = prof.parse_per_block(tmp_path)
    assert out["gm_to_ub_bw"] == pytest.approx(31.0)
    assert out["ub_read_bw"] == pytest.approx(31.0)
    assert out["mte2_ratio"] == pytest.approx(0.84)
    assert out["blocks_sampled"] == 2


@pytest.mark.skipif(
    not (WS / "cases" / "elementwise_add" / "benchmark.json").is_file(),
    reason="工作区已清空存量算子产物（task-lifecycle-v2 迁移）；等新任务产出基准用例后再验",
)
def test_benchmark_suite_and_tools_contract():
    """工作区契约：基准用例定义 + 两个执行器都在，且条件字段齐备。"""
    b = json.loads((WS / "cases" / "elementwise_add" / "benchmark.json").read_text(encoding="utf-8"))
    assert b["primary_case"] in {c["case_id"] for c in b["cases"]}
    for key in ("scope", "invocations", "warm_up", "aggregate"):
        assert key in b["measure"], f"基准条件缺 {key}"
    cases = json.loads((WS / "cases" / "elementwise_add" / "cases.json").read_text(encoding="utf-8"))
    cons = cases["constraints"]
    for c in b["cases"]:
        assert cons["n_min"] <= c["n"] <= cons["n_max"], f"{c['case_id']} N 越出规格范围"
    for name in ("tools/run_benchmark.py", "tools/run_profile.py", "tools/torch_add_baseline.py"):
        assert (WS / name).is_file(), f"缺执行器 {name}"
