"""门禁校验器 v1（方案 v1.1 §5 + §4.5）：把「产物齐备」升级成「结果有效 + 业务达标 + 版本一致」。

判定口径（方案 §5 硬拦的四件事）：
1. **产物完整**：manifest 列出的产物都在、sha256 对得上；按动作必需的那几类也在。
2. **结果有效**：退出码、报告能不能解析、逐用例状态是否合法、必需用例有没有真跑（NOT_RUN 不算过）。
3. **业务达标**：correctness 报告里没有 FAIL/ERROR；必需指标没有 failed。
4. **版本一致**：代码快照 / 规格版本 / 用例集 ref / 判据档版本 / 镜像 digest —— run 行、manifest、
   报告三者必须一致（三硬规则第 3 条「可复现的最低要求」）。

输出 = `gate_report`（JSON），字段与 fail code 清单见 `docs/gate-report-v1.md`。
**退出码不当结论**：`exit_code == 0` 只是必要条件（方案 §3.1 硬规则 1）。
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

from ..models import Run

#: 每个动作**必需**的产物类型（manifest 的 artifacts[].type）
REQUIRED_ARTIFACTS: dict[str, tuple[str, ...]] = {
    "build": ("log",),
    # 开发内冒烟（2026-10-23）：编译 + 一条最小用例跑通；报告 smoke.json 由 tools/smoke_case.py 产出
    "smoke": ("log", "smoke"),
    "test": ("log", "correctness"),
    "benchmark": ("log", "performance"),
    # D9 已定（2026-10-16 真机实测）：msopprof 能采到四类指标 → 从「只要 log」收紧为「必须 metrics」
    "profile": ("log", "metrics"),
}

#: manifest 里必须出现的版本字段（缺一个就没有「版本一致」可言）
MANIFEST_VERSION_KEYS = ("code_snapshot_sha256", "spec_version_id", "criteria_ref", "image_digest")

CASE_STATUSES = ("PASS", "FAIL", "ERROR", "INVALID", "NOT_RUN", "UNSUPPORTED")
METRIC_STATUSES = ("measured", "estimated", "unsupported", "failed")

#: 报告状态枚举 → 该用例是否算「过」
CASE_PASSING = ("PASS",)


def _sha256_file(path: Path, limit: int | None = None) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _load_json(path: Path) -> tuple[dict | None, str | None]:
    try:
        return json.loads(path.read_text(encoding="utf-8")), None
    except FileNotFoundError:
        return None, "文件不存在"
    except (json.JSONDecodeError, UnicodeDecodeError) as e:
        return None, f"不是合法 JSON：{e}"


def _criteria_version_ok(ref: str | None) -> bool:
    """判据档必须有版本号（`档名@版本`），否则「报告与本判据一致」这句判不了。"""
    return bool(ref) and "@" in str(ref) and str(ref).split("@", 1)[1].strip() != ""


def _norm(path: str) -> str:
    return str(Path(path)).rstrip("/")


class _Report:
    def __init__(self, run: Run, run_dir: Path | None) -> None:
        self.run = run
        self.run_dir = run_dir
        self.findings: list[dict] = []

    def fail(self, code: str, message: str, **evidence) -> None:
        self.findings.append({"code": code, "level": "fail", "message": message, "evidence": evidence})

    def warn(self, code: str, message: str, **evidence) -> None:
        self.findings.append({"code": code, "level": "warn", "message": message, "evidence": evidence})

    @property
    def failed(self) -> list[dict]:
        return [f for f in self.findings if f["level"] == "fail"]


# ---------------------------------------------------------------- 各段检查


def _check_manifest(rep: _Report) -> tuple[dict | None, Path | None]:
    run = rep.run
    if not run.manifest_path:
        rep.fail("E_NO_MANIFEST", "这次 run 没登记 manifest_path —— 没有证据，门禁不成立")
        return None, None
    man_path = Path(str(run.manifest_path)).expanduser()
    if not man_path.is_file():
        rep.fail("E_NO_MANIFEST", "manifest 文件不存在", path=str(man_path))
        return None, None
    man, err = _load_json(man_path)
    if man is None:
        rep.fail("E_MANIFEST_UNREADABLE", f"manifest 读不了：{err}", path=str(man_path))
        return None, None
    missing = [k for k in ("run_id", "exit_code", "artifacts", *MANIFEST_VERSION_KEYS) if k not in man]
    if missing:
        rep.fail("E_MANIFEST_INCOMPLETE", f"manifest 缺关键字段：{', '.join(missing)}", keys=missing)
    return man, man_path.parent


def _check_exit(rep: _Report, man: dict) -> None:
    rc = rep.run.exit_code if rep.run.exit_code is not None else man.get("exit_code")
    if rc is None:
        rep.fail("E_EXIT_UNKNOWN", "退出码缺失（既不在 run 行也不在 manifest）")
    elif int(rc) != 0:
        rep.fail("E_EXIT_NONZERO", "执行退出码非 0 → 本次结果无效（退出码 0 也不等于业务通过，见 §3.1 硬规则 1）",
                 exit_code=int(rc))


def _check_artifacts(rep: _Report, man: dict) -> None:
    run_dir = rep.run_dir
    arts = man.get("artifacts") or []
    if not arts:
        rep.fail("E_ARTIFACT_EMPTY", "manifest 里没有任何产物")
    for a in arts:
        p = Path(str(a.get("path", "")))
        full = (run_dir / p) if run_dir else p
        if not full.is_file():
            rep.fail("E_ARTIFACT_MISSING", f"产物文件不存在：{a.get('path')}", path=str(full))
            continue
        if a.get("sha256"):
            actual = _sha256_file(full)
            if actual != a["sha256"]:
                rep.fail("E_ARTIFACT_SHA_MISMATCH", f"产物 sha256 对不上：{a.get('path')}",
                         path=str(full), expected=a["sha256"], actual=actual)
        else:
            rep.warn("W_ARTIFACT_NO_SHA", f"产物没记 sha256：{a.get('path')}")
    have = {a.get("type") for a in arts}
    for need in REQUIRED_ARTIFACTS.get(rep.run.action, ()):
        if need not in have:
            rep.fail("E_REQUIRED_ARTIFACT_MISSING",
                     f"动作 {rep.run.action} 必需产物类型缺失：{need}（manifest 里只有 {sorted(x for x in have if x)}）",
                     need=need)


def _check_versions(rep: _Report, man: dict) -> None:
    run = rep.run
    pairs = [
        ("E_SNAPSHOT", "code_snapshot_sha256", "代码快照哈希"),
        ("E_SPEC_VERSION", "spec_version_id", "规格版本"),
        ("E_CRITERIA", "criteria_ref", "判据档"),
        ("E_IMAGE_DIGEST", "image_digest", "镜像 digest"),
    ]
    for prefix, key, label in pairs:
        row_val = getattr(run, key, None)
        man_val = man.get(key)
        if row_val in (None, "") and man_val in (None, ""):
            rep.fail(f"{prefix}_MISSING", f"{label} 在 run 行与 manifest 里都没有 —— 不满足「版本可对齐」")
        elif row_val in (None, ""):
            rep.fail(f"{prefix}_MISSING", f"{label} 没记在 run 行上", manifest=man_val)
        elif man_val in (None, ""):
            rep.fail(f"{prefix}_MISSING", f"{label} 没记在 manifest 里（无法证明跑的哪一版）", run=row_val)
        elif str(row_val) != str(man_val):
            rep.fail(f"{prefix}_MISMATCH", f"{label} 不一致：run 行 {row_val} ≠ manifest {man_val}",
                     run=row_val, manifest=man_val)

    if not _criteria_version_ok(run.criteria_ref):
        rep.fail("E_CRITERIA_VERSION_MISSING",
                 f"判据档没有版本号（应写成 档名@档版本，当前 {run.criteria_ref!r}）—— 方案 §11.1 要求可自定义档位")
    if not run.suite_ref:
        if run.action == "test":
            rep.fail("E_SUITE_REF_MISSING", "test 动作必须记录用例集 ref（否则「跑了哪些用例」不可复现）")
        else:
            rep.warn("W_SUITE_REF_MISSING", f"{run.action} 动作没有记用例集 ref")


def _pick_artifact(man: dict, *types: str) -> dict | None:
    """按产物类型取文件，**同类多文件时优先 .json**。

    为什么要这么写：correctness.json 与 correctness.md 的 stem 都是 "correctness"，
    一拍脑袋用 `{a["type"]: a}` 建字典就会留下最后一个（.md），于是「报告根本没被解析」
    却什么都不报 —— 2026-10-15 实测踩到（run#4 的用例 FAIL 被判漏，只有 E_EXIT_NONZERO 兜住）。
    这里只认机读文件；万一某类型只有非 .json，由调用方明确报错，不静默跳过。
    """
    arts = [a for a in (man.get("artifacts") or []) if a.get("type") in types]
    if not arts:
        return None
    for a in arts:
        if str(a.get("path", "")).endswith(".json"):
            return a
    return arts[0]


def artifact_paths(run: Run) -> dict[str, Path]:
    """run 的产物：**类型 → 绝对文件路径**（同类多文件优先 `.json`）。

    门禁判定与结果视图共用（结果视图要把 `correctness.json` / `performance.json` /
    `metrics.json` 的内容读出来画图）。缺 manifest / 读不了就返回空 dict。
    """
    if not run.manifest_path:
        return {}
    man_path = Path(str(run.manifest_path)).expanduser()
    man, _err = _load_json(man_path)
    if not man:
        return {}
    try:
        root = man_path.parent.resolve()
    except OSError:
        return {}
    out: dict[str, Path] = {}
    for a in man.get("artifacts") or []:
        rel = str(a.get("path", ""))
        atype = a.get("type")
        if not rel or not atype:
            continue
        # manifest 是执行产物，不是可信输入：绝对路径、`..` 和指向目录外的
        # 符号链接都不能进入门禁/结果视图/交付包（审查 H8）。
        candidate = Path(rel)
        if candidate.is_absolute():
            continue
        try:
            full = (root / candidate).resolve()
            full.relative_to(root)
        except (OSError, ValueError):
            continue
        cur = out.get(atype)
        if cur is None or (full.suffix == ".json" and cur.suffix != ".json"):
            out[atype] = full
    return out

def _check_smoke_report(rep: _Report, man: dict) -> None:
    """冒烟报告（smoke.json）：解析 + verdict=PASS + 与 run 版本一致。

    smoke 的定位是「开发完成后先证明编得出、跑得通、最小用例过」——它不是正式正确性测试，
    但失败必须硬拦（否则后面 test/benchmark 的失败就是可预见的返工）。
    """
    art = _pick_artifact(man, "smoke")
    if art is None:
        if rep.run.action == "smoke":
            rep.fail("E_SMOKE_REPORT_MISSING", "smoke 动作没有 smoke 产物类型（应产出 smoke.json）")
        return
    full = (rep.run_dir / str(art.get("path"))) if rep.run_dir else Path(str(art.get("path")))
    if full.suffix != ".json":
        rep.fail("E_REPORT_UNREADABLE",
                 f"smoke 产物不是机读文件：{art.get('path')}（门禁只认 smoke.json）", path=str(full))
        return
    data, err = _load_json(full)
    if data is None:
        rep.fail("E_REPORT_UNREADABLE", f"smoke.json 读不了：{err}", path=str(full))
        return
    for key, label in (("spec_version_id", "规格版本"), ("code_snapshot_sha256", "代码快照"),
                       ("criteria_ref", "判据档"), ("suite_ref", "用例集")):
        if key in data and str(data[key]) != str(getattr(rep.run, key, None)):
            rep.fail("E_REPORT_VERSION_MISMATCH",
                     f"smoke.json 里的{label} {data[key]} 与 run 行 {getattr(rep.run, key, None)} 不一致",
                     key=key, report=data[key], run=getattr(rep.run, key, None))
    verdict = data.get("verdict")
    if verdict != "PASS":
        case = data.get("case") or {}
        rep.fail("E_SMOKE_FAILED",
                 f"最小用例冒烟未通过（verdict={verdict!r}）：{case.get('detail') or case.get('status')}",
                 case_id=case.get("case_id"), detail=case.get("detail"))
        return
    case = data.get("case") or {}
    if not case.get("case_id"):
        rep.fail("E_SMOKE_CASE_MISSING", "smoke.json 里没有 case.case_id —— 不知道跑的是哪条最小用例")
    elif case.get("status") != "PASS":
        rep.fail("E_SMOKE_FAILED", f"最小用例 {case.get('case_id')} 状态是 {case.get('status')!r}")


def _check_case_report(rep: _Report, man: dict) -> None:
    """正确性报告：解析 + 逐用例状态 + 报告与候选版本是否一致（§4.5）。"""
    art = _pick_artifact(man, "correctness")
    if art is None:
        if rep.run.action == "test":
            rep.fail("E_CORRECTNESS_REPORT_MISSING", "test 动作没有 correctness 产物类型（应产出 correctness.{json,md}）")
        return
    full = (rep.run_dir / str(art.get("path"))) if rep.run_dir else Path(str(art.get("path")))
    if full.suffix != ".json":
        rep.fail("E_REPORT_UNREADABLE",
                 f"correctness 产物不是机读文件：{art.get('path')}（门禁只认 correctness.json）",
                 path=str(full))
        return
    data, err = _load_json(full)
    if data is None:
        rep.fail("E_REPORT_UNREADABLE", f"correctness.json 读不了：{err}", path=str(full))
        return

    # 报告来源必须与候选版本一致（§4.5：报告缺失/解析失败/证据版本不匹配 → 不满足门禁）
    for key, label in (("spec_version_id", "规格版本"), ("code_snapshot_sha256", "代码快照"),
                       ("criteria_ref", "判据档"), ("suite_ref", "用例集")):
        if key in data and str(data[key]) != str(getattr(rep.run, key, None)):
            rep.fail("E_REPORT_VERSION_MISMATCH",
                     f"报告里的{label} {data[key]} 与 run 行 {getattr(rep.run, key, None)} 不一致 —— 报告不是这一版跑出来的",
                     key=key, report=data[key], run=getattr(rep.run, key, None))

    cases = data.get("cases")
    if not isinstance(cases, list) or not cases:
        rep.fail("E_CASE_LIST_EMPTY", "correctness.json 里没有 cases 列表")
        return
    not_run = [c.get("case_id") for c in cases if c.get("status") == "NOT_RUN" and c.get("required", True)]
    if not_run:
        rep.fail("E_CASE_NOT_RUN", f"必需用例未执行：{not_run[:10]}（未执行不计通过，§4.5）", case_ids=not_run[:50])
    failed = [c.get("case_id") for c in cases if c.get("status") in ("FAIL", "ERROR")]
    if failed:
        rep.fail("E_CASE_FAILED", f"用例未通过（FAIL/ERROR）：{failed[:10]}", case_ids=failed[:50])
    bad = sorted({str(c.get("status")) for c in cases if c.get("status") not in CASE_STATUSES})
    if bad:
        rep.fail("E_CASE_STATUS_INVALID", f"用例状态不在枚举 {CASE_STATUSES} 里：{bad}", statuses=bad)
    unsupported = [c for c in cases if c.get("status") == "UNSUPPORTED"]
    unexpected_unsupported = [
        c.get("case_id") for c in unsupported
        if c.get("expected") != "match_cpu_or_explicit_unsupported"
    ]
    if unexpected_unsupported:
        rep.fail(
            "E_CASE_UNSUPPORTED_UNEXPECTED",
            "用例未声明允许显式 unsupported，不能用 UNSUPPORTED 代替通过",
            case_ids=unexpected_unsupported[:50],
        )
    allowed_unsupported = [
        c.get("case_id") for c in unsupported
        if c.get("expected") == "match_cpu_or_explicit_unsupported"
    ]
    if allowed_unsupported:
        rep.warn(
            "W_CASE_EXPLICIT_UNSUPPORTED",
            "用例按已确认契约显式分类为 UNSUPPORTED（不记为数值 PASS）",
            case_ids=allowed_unsupported[:50],
        )


def _check_metrics(rep: _Report, man: dict) -> None:
    """指标状态与来源（§4.6）：estimated 必须附假设，failed 的必需指标 → 性能分析未完成。"""
    art = _pick_artifact(man, "metrics", "performance")
    if art is None:
        if rep.run.action == "profile":
            rep.fail("E_METRICS_REPORT_MISSING",
                     "profile 动作没有 metrics 产物类型（应产出 metrics.json）—— D9 已定，四类指标可采")
        return
    full = (rep.run_dir / str(art.get("path"))) if rep.run_dir else Path(str(art.get("path")))
    if full.suffix != ".json":
        rep.fail("E_REPORT_UNREADABLE",
                 f"指标产物不是机读文件：{art.get('path')}（门禁只认 metrics.json / performance.json）",
                 path=str(full))
        return
    data, err = _load_json(full)
    if data is None:
        rep.fail("E_REPORT_UNREADABLE", f"指标文件读不了：{err}", path=str(full))
        return

    # 指标报告同样要能对齐版本（否则「报告不是这一版跑的」在性能侧没有防线）
    for key, label in (("spec_version_id", "规格版本"), ("code_snapshot_sha256", "代码快照"),
                       ("criteria_ref", "判据档"), ("suite_ref", "用例集")):
        if key in data and str(data[key]) != str(getattr(rep.run, key, None)):
            rep.fail("E_REPORT_VERSION_MISMATCH",
                     f"指标报告里的{label} {data[key]} 与 run 行 {getattr(rep.run, key, None)} 不一致",
                     key=key, report=data[key], run=getattr(rep.run, key, None))

    for m in (data.get("metrics") or []):
        status = m.get("status")
        if status not in METRIC_STATUSES:
            rep.fail("E_METRIC_STATUS_INVALID",
                     f"指标 {m.get('id')} 的状态 {status!r} 不在枚举 {METRIC_STATUSES} 里")
            continue
        if not m.get("source"):
            rep.warn("W_METRIC_NO_SOURCE", f"指标 {m.get('id')} 没记来源")
        if status == "estimated" and not m.get("assumption"):
            rep.warn("W_METRIC_ESTIMATED_NO_ASSUMPTION", f"指标 {m.get('id')} 是估算值却没写假设")
        if status == "failed" and m.get("required", False):
            rep.fail("E_METRIC_REQUIRED_FAILED", f"必需指标采集失败：{m.get('id')} → 性能分析未完成")


def _check_device(rep: _Report, man: dict) -> None:
    """测量条件可追溯（§4.6）：基准/采集必须记设备与是否独占。"""
    if rep.run.action not in ("benchmark", "profile"):
        return
    device = rep.run.device_id or man.get("device")
    lock = man.get("lock_mode")
    if not device or not lock:
        rep.warn("W_DEVICE_UNRECORDED", "基准/采集没记设备与锁模式 → 邻居是否在同一张卡上不可知")
    elif lock != "exclusive":
        rep.warn("W_DEVICE_SHARED", f"设备 {device} 不是独占（lock_mode={lock}）→ 耗时可能被邻居污染，同条件对比不成立",
                 device=device, lock_mode=lock)


# ---------------------------------------------------------------- 交付（任务生命周期 v2）


def _evaluate_delivery(run: Run) -> dict:
    """交付是**本地动作**（无远端 manifest/容器），按 run.artifacts 直接判：

    - run 必须 ok；
    - 必须同时有 `delivery`（交付包 zip）与 `manifest` 两个产物，文件在盘、sha256 对得上；
    - 版本三键（spec / 代码快照 / 用例集）与判据档必须记录（交付门禁的实质校验在
      `services/delivery.check` 里做，这里只保证这条 run 自身的产物与版本可对齐）。
    """
    findings: list[dict] = []

    def fail(code: str, message: str, **evidence) -> None:
        findings.append({"code": code, "level": "fail", "message": message, "evidence": evidence})

    def warn(code: str, message: str, **evidence) -> None:
        findings.append({"code": code, "level": "warn", "message": message, "evidence": evidence})

    if run.status != "ok":
        fail("E_RUN_NOT_FINISHED" if run.status not in ("ok", "failed") else "E_DELIVERY_FAILED",
             f"交付 run 状态是 {run.status!r}")
    try:
        arts = list(run.artifacts)
    except Exception:  # noqa: BLE001 —— detached run：产物没加载出来就按缺失判
        arts = []
    by_type = {a.type: a for a in arts}
    for need in ("delivery", "manifest"):
        if need not in by_type:
            fail("E_REQUIRED_ARTIFACT_MISSING", f"交付必需产物类型缺失：{need}")
    for a in arts:
        if a.type not in ("delivery", "manifest"):
            continue
        path = Path(str(a.path)).expanduser()
        if not path.is_file():
            fail("E_ARTIFACT_MISSING", f"交付产物文件不存在：{path}", path=str(path))
            continue
        if a.sha256:
            actual = _sha256_file(path)
            if actual != a.sha256:
                fail("E_ARTIFACT_SHA_MISMATCH", "交付产物 sha256 与登记不一致", path=str(path),
                     expected=a.sha256, actual=actual)
    if run.spec_version_id in (None, ""):
        fail("E_SPEC_VERSION_MISSING", "交付 run 没记录规格版本")
    if not run.code_snapshot_sha256:
        fail("E_CODE_SNAPSHOT_MISSING", "交付 run 没记录代码快照")
    if not run.suite_ref:
        fail("E_SUITE_REF_MISSING", "交付 run 没记录用例集 ref")
    if not _criteria_version_ok(run.criteria_ref):
        fail("E_CRITERIA_VERSION_MISSING", f"判据档没有版本号（当前 {run.criteria_ref!r}）")
    if not run.device_id:
        warn("W_DEVICE_UNRECORDED", "交付动作不绑设备（正常：本地打包动作）")

    failed = [f for f in findings if f["level"] == "fail"]
    return {
        "run_id": run.id,
        "task_id": run.task_id,
        "action": run.action,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "verdict": "pass" if not failed else "fail",
        "summary": {
            "checks": len({f["code"] for f in findings}) or 0,
            "fail": len(failed),
            "warn": len([f for f in findings if f["level"] == "warn"]),
        },
        "version_alignment": {
            "spec_version_id": run.spec_version_id,
            "code_snapshot_sha256": run.code_snapshot_sha256,
            "suite_ref": run.suite_ref,
            "criteria_ref": run.criteria_ref,
            "image_digest": run.image_digest,
            "device_id": run.device_id,
        },
        "findings": findings,
        "notes": [
            "交付动作 = 本地打包：不连远端容器、不需要设备。",
            "交付门禁的实质校验（版本证据 / 用例确认 / 产物齐备）在 services/delivery.check，判不过不会产生本 run。",
        ],
    }


# ---------------------------------------------------------------- 对外入口


def evaluate(run: Run, *, run_dir: Path | None = None, write: bool = False) -> dict:
    """对一次真实执行做门禁判定，返回 gate_report（不抛异常；判不了本身就是结论）。"""
    if run.action == "delivery":
        report = _evaluate_delivery(run)
        if write:
            # 交付 run 的 manifest 目录 = manifest 文件所在目录
            base = Path(str(run.manifest_path)).expanduser().parent if run.manifest_path else None
            if base is not None and base.is_dir():
                out = base / "gate_report.json"
                out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
                report["written_to"] = str(out)
        return report

    rep = _Report(run, run_dir)

    if run.status not in ("ok", "failed"):
        rep.fail("E_RUN_NOT_FINISHED", f"run 状态是 {run.status!r}，还没跑完就没有可判的结果")

    man, man_dir = _check_manifest(rep)
    if man_dir is not None:
        rep.run_dir = man_dir
    if man is not None:
        _check_exit(rep, man)
        _check_artifacts(rep, man)
        _check_versions(rep, man)
        _check_smoke_report(rep, man)
        _check_case_report(rep, man)
        _check_metrics(rep, man)
        _check_device(rep, man)

    failed = rep.failed
    summary = {
        "checks": 0,  # 下面按 code 去重计数，便于一眼看出判了几类
        "fail": len(failed),
        "warn": len([f for f in rep.findings if f["level"] == "warn"]),
    }
    summary["checks"] = len({f["code"] for f in rep.findings}) or 0
    report = {
        "run_id": run.id,
        "task_id": run.task_id,
        "action": run.action,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "verdict": "pass" if not failed else "fail",
        "summary": summary,
        "version_alignment": {
            "spec_version_id": run.spec_version_id,
            "code_snapshot_sha256": run.code_snapshot_sha256,
            "suite_ref": run.suite_ref,
            "criteria_ref": run.criteria_ref,
            "image_digest": run.image_digest,
            "device_id": run.device_id,
        },
        "findings": rep.findings,
        "notes": [
            "退出码 0 不等于业务通过（§3.1 硬规则 1）：判定看报告 + 版本一致性。",
            "必需用例未执行（NOT_RUN）不计通过（§4.5）。",
        ],
    }
    if write and rep.run_dir is not None:
        out = Path(rep.run_dir) / "gate_report.json"
        out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        report["written_to"] = str(out)
    return report
