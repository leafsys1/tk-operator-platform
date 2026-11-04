"""四动作契约的前置检查（build_plan）—— 缺项分支必须有具体原因，不能"齐备了但其实跑不了"。"""

from __future__ import annotations

from pathlib import Path

import pytest

from app.models import DEFAULT_CRITERIA_REF
from app.services import exec_actions


@pytest.fixture(autouse=True)
def _existing_smoke_for_unrelated_plan_tests(monkeypatch):
    """本文件多数用例只测其它缺项；统一假定当前快照已有 smoke，专门分支另测。"""
    monkeypatch.setattr(exec_actions, "smoke_gate_reason", lambda *_a, **_kw: None)


def test_unknown_action(db_session, seeded):
    with pytest.raises(ValueError):
        exec_actions.build_plan(db_session, seeded["task"], "nope")


def test_ready_plan(db_session, seeded):
    plan = exec_actions.build_plan(db_session, seeded["task"], "test")
    assert plan.ready, plan.missing
    assert plan.criteria_ref == DEFAULT_CRITERIA_REF  # 项目默认档
    assert plan.criteria_source == "项目"
    assert plan.code_snapshot_sha256 and plan.snapshot_files >= 4
    assert "scripts/nm_test.sh" in plan.command and "--action test" in plan.command
    assert plan.image_digest == "sha256:deadbeef"


def test_test_and_benchmark_require_current_smoke(db_session, seeded, monkeypatch):
    reason = "当前条件没有门禁通过的 smoke"
    monkeypatch.setattr(exec_actions, "smoke_gate_reason", lambda *_a, **_kw: reason)
    for action in ("test", "benchmark"):
        plan = exec_actions.build_plan(db_session, seeded["task"], action)
        assert plan.ready is False
        assert reason in plan.missing
    assert exec_actions.build_plan(db_session, seeded["task"], "build").ready is True


def test_missing_workspace(db_session, seeded):
    seeded["project"].workspace_path = "/nowhere/nope"
    db_session.commit()
    plan = exec_actions.build_plan(db_session, seeded["task"], "build")
    assert any("工作区不存在" in m for m in plan.missing)


def test_missing_entry_script(db_session, seeded, workspace: Path):
    (workspace / "test.sh").unlink()
    plan = exec_actions.build_plan(db_session, seeded["task"], "test")
    assert any("缺入口脚本 test.sh" in m for m in plan.missing)


def test_toolchain_only_image_blocked_for_test_but_not_build(db_session, seeded):
    """P1.5-a 的教训：只有 CANN 工具链、不带 torch 的镜像，test 一定跑不起来。"""
    seeded["image"].torch_version = None
    db_session.commit()
    test_plan = exec_actions.build_plan(db_session, seeded["task"], "test")
    assert any("torch/torch_npu 版本" in m for m in test_plan.missing)
    build_plan = exec_actions.build_plan(db_session, seeded["task"], "build")
    assert not any("torch/torch_npu 版本" in m for m in build_plan.missing)


def test_jax_task_uses_jax_runtime_record_instead_of_torch_metadata(db_session, seeded):
    """JAX 原语走专用 runner：已验收 JAX 镜像不应被 torch 门禁误伤。"""
    seeded["task"].op_name = "jax_lax_scan"
    seeded["image"].torch_version = None
    seeded["image"].note = "JAX/JAXLIB 0.8.1; backend=npu probe PASS"
    db_session.commit()

    plan = exec_actions.build_plan(db_session, seeded["task"], "test")
    assert not any("torch/torch_npu 版本" in m for m in plan.missing)
    assert not any("JAX/JAXLIB 验收记录" in m for m in plan.missing)


def test_jax_task_requires_explicit_jax_runtime_record(db_session, seeded):
    """仅凭 jax_* 名字不能放行任意工具链镜像。"""
    seeded["task"].op_name = "jax_lax_scan"
    seeded["image"].torch_version = None
    seeded["image"].note = "CANN toolchain only"
    db_session.commit()

    plan = exec_actions.build_plan(db_session, seeded["task"], "test")
    assert any("JAX/JAXLIB 验收记录" in m for m in plan.missing)


def test_not_op_dev_image_is_missing(db_session, seeded):
    seeded["image"].is_op_dev = False
    db_session.commit()
    plan = exec_actions.build_plan(db_session, seeded["task"], "test")
    assert any("不是「算子开发镜像」" in m for m in plan.missing)


def test_no_spec_version(db_session, seeded):
    seeded["task"].current_spec_version_id = None
    db_session.commit()
    plan = exec_actions.build_plan(db_session, seeded["task"], "test")
    assert any("还没有规格版本" in m for m in plan.missing)


def test_cases_confirmation_without_hash_is_blocked(db_session, seeded):
    """历史脏数据只有确认时间、没有哈希时，不能跳过用例组硬门禁。"""
    seeded["task"].cases_confirmed_sha256 = None
    db_session.commit()
    plan = exec_actions.build_plan(db_session, seeded["task"], "test")
    assert any("确认记录缺少哈希" in m for m in plan.missing)


def test_missing_case_suite_dir(db_session, seeded, workspace: Path):
    for f in (workspace / "cases").iterdir():
        f.unlink()
    (workspace / "cases").rmdir()
    plan = exec_actions.build_plan(db_session, seeded["task"], "test")
    assert any("用例集目录 cases/" in m for m in plan.missing)


def test_criteria_precedence(db_session, seeded):
    seeded["task"].criteria_ref = "项目内自定义档@3"
    db_session.commit()
    plan = exec_actions.build_plan(db_session, seeded["task"], "test")
    assert (plan.criteria_ref, plan.criteria_source) == ("项目内自定义档@3", "任务")

    # 判据档没版本号 → 只 warn（不拦），但要明确说出来
    seeded["task"].criteria_ref = "没有版本号的档"
    db_session.commit()
    plan = exec_actions.build_plan(db_session, seeded["task"], "test")
    assert plan.ready
    assert any("没有版本号" in w for w in plan.warnings)


def test_execute_refuses_but_says_why(db_session, seeded):
    """同步入口不再承载真执行：必须明确指向异步作业（别静默什么都不做）。"""
    plan = exec_actions.build_plan(db_session, seeded["task"], "test")
    with pytest.raises(exec_actions.RemoteExecutionNotReady) as e:
        exec_actions.execute(plan, confirm=True)
    msg = str(e.value)
    assert "异步作业" in msg and "dry_run=false" in msg and "action-jobs" in msg


def test_task_device_config_wired_into_plan(db_session, seeded):
    """任务配置里的 device 必须真的进 plan（页面保存 ≠ 生效，审查 #6）。"""
    task = seeded["task"]
    task.device_id = "3"
    db_session.commit()

    plan = exec_actions.build_plan(db_session, task, "build")
    assert plan.device_id == "3"
    assert "OPX_DEVICES=3" in plan.command
    # 没给 timeout 时保持默认 1800（设备检查关闭，只有显式设备照样生效）
    assert plan.timeout_s == 1800

    # 请求体/plan 覆盖任务配置
    override = exec_actions.build_plan(db_session, task, "build", device_id="5")
    assert override.device_id == "5"
    assert "OPX_DEVICES=5" in override.command


def test_explicit_device_still_goes_through_busy_check(db_session, seeded, monkeypatch):
    """显式指定（任务配置或请求体）也要走占用检查：指定的卡被占就报缺项，不硬挤。"""
    from app.services import devices as dev

    task = seeded["task"]
    task.device_id = "dev3"
    db_session.commit()

    def fake_check(*_a, **_kw):
        return dev.DeviceReport(
            ok=True, devices=[dev.DeviceState(i, i != 3) for i in range(8)]
        )

    monkeypatch.setattr(exec_actions.dev, "check_devices", fake_check)
    busy = exec_actions.build_plan(db_session, task, "build", check_device=True)
    assert busy.device_id is None
    assert any("设备 3 被占" in m for m in busy.missing)
    assert busy.device_state["busy"] == [3]

    task.device_id = "4"
    db_session.commit()
    free = exec_actions.build_plan(db_session, task, "build", check_device=True)
    assert free.device_id == "4" and free.ready


def test_timeout_precedence(db_session, seeded, monkeypatch):
    """超时优先级：payload > task.timeout_s > env OPX_ACTION_TIMEOUT > 1800。"""
    task = seeded["task"]

    monkeypatch.setenv("OPX_ACTION_TIMEOUT", "600")
    assert exec_actions.build_plan(db_session, task, "build").timeout_s == 600

    task.timeout_s = 2400
    db_session.commit()
    assert exec_actions.build_plan(db_session, task, "build").timeout_s == 2400
    assert exec_actions.build_plan(db_session, task, "build", timeout_s=120).timeout_s == 120

    monkeypatch.delenv("OPX_ACTION_TIMEOUT")
    task.timeout_s = None
    db_session.commit()
    assert exec_actions.build_plan(db_session, task, "build").timeout_s == 1800


# ---------------------------------------------------------------- 任务生命周期 v2：用例组确认门禁


def test_actions_blocked_when_cases_unconfirmed(db_session, seeded):
    """未确认用例组 → 四动作 plan 都报「用例组未确认」（硬门禁，等价于「未确认不允许开始开发」）。"""
    seeded["task"].cases_confirmed_at = None
    db_session.commit()
    for action in ("build", "test", "benchmark", "profile"):
        plan = exec_actions.build_plan(db_session, seeded["task"], action)
        assert plan.ready is False
        assert any("用例组未确认" in m for m in plan.missing), (action, plan.missing)


def test_actions_blocked_when_cases_changed(db_session, seeded):
    """确认后用例集被改动（哈希漂移）→ plan 要求重新确认，不能拿旧确认跑动作。"""
    seeded["task"].cases_confirmed_sha256 = "f" * 64  # 与工作区 cases/ 现算哈希不一致
    db_session.commit()
    plan = exec_actions.build_plan(db_session, seeded["task"], "test")
    assert plan.ready is False
    assert any("用例集已变更" in m for m in plan.missing), plan.missing
