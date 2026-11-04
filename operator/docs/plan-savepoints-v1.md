# 版本存档（savepoints）v1 · 设计与契约

> 状态：2026-10-21 起实现。背景：现在 `runs.code_snapshot_sha256` 只是**指纹**，不存源码，
> 「跑过哪版、证据对不对得上」可查，「回到那一版」做不到。v1 补上 **源码内容存档 + 存盘点 + 回滚** 的最小闭环。
>
> 非目标（放 v2）：git 后端 / 任意两存档 diff / 跨任务共享存档 / 自动 GC 归档 / 把存档挂进交付包。

## 1. 一句话模型

**按内容哈希寻址的源码归档（tar.gz）+ 指向它的存盘点（savepoint）行。**
存盘点只是「标签 + 环境引用 + 时间 + 类型」，不重复存代码；同一 `sha256` 只归档一份。

## 2. 存储布局

- 归档根：`settings.savepoint_root`（环境变量 `OPX_SAVEPOINT_ROOT`；空 = `<repo_root>/archives`）。
- 源码归档：`<归档根>/sources/<code_snapshot_sha256>.tar.gz`（sha 是 64 位 hex，天然安全）。
- `.gitignore` 增加 `archives/`。
- 归档规则必须与 `exec_actions.snapshot()` **完全相同**（同一组 skip 目录、同一文件集合）：
  `runs, .git, node_modules, __pycache__, .venv, dist, .dsh, .cannbot, .skills-src, .opx-chat`。
  解包后重算 `snapshot(解包目录) == 存档 sha` 才算有效（损坏/缺文件 → 拒绝回滚）。
- 归档写成 `.tmp` 再 `os.replace`，原子落盘；单进程内用锁避免并发重复打包。
- 安全：解包逐成员校验（拒绝绝对路径、`..`、符号链接/硬链接、非普通文件）；回滚写文件前校验目标在
  workspace 内。

## 3. 数据模型（新表 `savepoints`）

| 列 | 类型 | 说明 |
|---|---|---|
| `id` | int PK | |
| `project_id` | FK projects.id (CASCADE) | |
| `task_id` | FK tasks.id (CASCADE) | |
| `label` | str(200) | 人看的名字 |
| `note` | text | 备注 |
| `kind` | str(32) | `manual` / `auto` / `rescue`（`accept` 预留） |
| `code_snapshot_sha256` | str(80) | 与 `runs.code_snapshot_sha256` 同算法 |
| `archive_path` | str(600) | 归档文件绝对路径 |
| `archive_bytes` | bigint | |
| `spec_version_id` | FK spec_versions.id | 创建时取自快照对应 run，没有则取 task 当前 |
| `criteria_ref` / `suite_ref` / `image_digest` | str | 同上（可空） |
| `env_recipe_snapshot` | text | 同上（可空） |
| `parent_id` | FK savepoints.id (SET NULL) | `rescue` 行记录「为回滚哪个存档而做」 |
| `created_at` | datetime | UTC |
| `created_by` | str(80) | v1 固定 `page` |

索引：`ix_savepoints_task_id`、`ix_savepoints_code_snapshot_sha256`。

迁移：`revision = "a9b8c7d6e5f4"`，`down_revision = "e3f4a5b6c7d8"`（当前 head）。

**删除任务**：`services/task_deletion.py` 显式 `DELETE FROM savepoints WHERE task_id=...`（SQLite 未开
`PRAGMA foreign_keys`，不能只靠 cascade）；`counts()` 增加 `savepoints` 键。归档文件是内容寻址、可能被
其它任务引用，v1 **不随任务删除清理**（留待 GC）。

## 4. API（新 router，include 进 `app/main.py`）

所有时间为 UTC ISO。错误沿用项目风格：`HTTPException(4xx, "中文说明")`。

### 4.1 `GET /api/tasks/{task_id}/savepoints?limit=50`

返回 `list[SavepointOut]`，按 `created_at desc, id desc`。`limit` 1..200，默认 50。

`SavepointOut`：

```json
{
  "id": 1, "task_id": 11, "label": "M1 通过正确性+基准", "note": null,
  "kind": "manual", "code_snapshot_sha256": "d8997f...", "archive_bytes": 12345,
  "spec_version_id": 7, "spec_version": 7,
  "criteria_ref": "ops-precision-standard@1", "suite_ref": "cases@840be5ba...",
  "image_digest": "local-id:sha256:...",
  "runs": [
    {"id": 30, "action": "test", "status": "ok", "device_id": "1", "loop_id": null,
     "ended_at": "2026-10-21T08:48:00+00:00"}
  ],
  "in_sync": false,
  "parent_id": null,
  "created_at": "2026-10-21T09:00:00+00:00"
}
```

- `runs`：**按 `code_snapshot_sha256` 相等实时匹配**该任务的 run（最多 20 条，id 升序）；不是创建时快照，
  新跑同快照的 run 会自动出现。
- `in_sync`：全量 `snapshot(workspace)` 是否等于存档 sha（一次请求只算一次；工作区不存在 → false）。
  注意它是**全量**口径：工作区是项目共享的，别的任务改动也会让它 false。按任务范围的差异看 `/diff`。
- `spec_version`：`SpecVersion.version`（人看的 vN），不是 DB id。

### 4.2 `POST /api/tasks/{task_id}/savepoints`

```json
{"label": "可选，缺省「存档 YYYY-MM-DD HH:mm」", "note": "可选"}
```

- 对当前工作区打包（必需成功）→ 建 `kind=manual` 行 → 201 返回 `SavepointOut`。
- 环境引用取「同 sha 的最新 ok run」，没有则取 task 当前规格 / 任务或项目判据档。
- 错误：404 任务；400 工作区不存在 / 快照被截断（>20000 文件）/ 归档过大（>256MB）。

### 4.3 `GET /api/savepoints/{id}`

返回 `SavepointOut`（同上，单个）。

### 4.4 `GET /api/savepoints/{id}/diff`

对比「存档代码」与「当前工作区」：

```json
{
  "savepoint": { ...SavepointOut... },
  "saved_sha256": "d8997f...", "current_sha256": "abcd...",
  "in_sync": false,
  "changed": [
    {"path": "ops/elementwise_mul/op_host/elementwise_mul.asc", "status": "modified",
     "additions": 2, "deletions": 1, "scope": "task"}
  ],
  "outside_scope": ["tools/measure_curve.py"],
  "diff": "--- a/...\n+++ b/...\n...",
  "diff_truncated": false
}
```

- `scope`：`task` = 路径在 `ops/<op_name>/` 或 `cases/<op_name>/` 下；其余 `shared`。
- `op_name` 为空时所有路径都算 `shared`。
- `diff` 复用 `optimize.diff_captures`，超 `MAX_DIFF_CHARS`（200000）截断并置 `diff_truncated=true`。
- `in_sync` 同样是全量 sha 口径。
- 错误：404 存档；409 归档文件缺失或损坏。

### 4.5 `POST /api/savepoints/{id}/restore`

```json
{"scope": "task", "note": "可选"}
```

- `scope`：`task`（默认）只回滚任务范围路径；`full` 全工作区回滚（会还原别的任务/共享文件，UI 明示风险）。
- 流程：
  1. 404 存档/任务；400 任务无 `op_name` 且 `scope=task`；409 有运行中作业
     （`action_runner.running_for_task(task_id)` 或该任务 `AgentJob.status == "running"` 且
     `kind in ("对话","开发","优化")`）；409 归档缺失/损坏（解包后 sha 校验不过）。
  2. 若当前全量 sha == 存档 sha → 直接返回 `restored=false`（不产生 rescue）。
  3. **先做急救存档**：`kind=rescue`、`label="回滚前自动存档（目标 #<id>）"`、`parent_id=<目标 id>`；
     急救存档失败 → 中止回滚（409），不动工作区。
  4. 按 scope 应用差异：存档里有 → 写入；存档里没有 → 删除；`shared` 路径跳过并上报。
  5. 返回：

```json
{
  "savepoint": {...}, "rescue": {...} | null,
  "restored": true, "in_sync": true,
  "current_sha256": "d8997f...",
  "changed": [{"path": "...", "status": "modified", "additions": 2, "deletions": 1, "scope": "task"}],
  "skipped_outside_scope": ["tools/measure_curve.py"],
  "note": "可选回显"
}
```

- `in_sync` 在这里是**范围口径**：应用后「存档 vs 当前」在 scope 内是否还有差异。全量 sha 可能因
  shared 文件不同而不等，属预期。
- 回滚**不动** runs / 规格指针 / loops；旧 run 仍是旧快照的证据（A/B 配对不受影响）。

### 4.6 `GET /api/savepoints/{id}/archive`

`FileResponse` 下载 tar.gz，`filename="savepoint-<id>-<sha[:12]>.tar.gz"`。
路径必须位于归档根下（防 DB 被篡改后任意读文件）。

### 4.7 `DELETE /api/savepoints/{id}`

204。只删行，不删归档（内容寻址、可能被别的行/任务引用；GC 放 v2）。

## 5. 服务函数（`app/services/savepoints.py`）

```python
SNAPSHOT_SKIP = {...}  # 与 exec_actions.snapshot 的 skip 集合逐字一致
MAX_ARCHIVE_BYTES = 256 * 1024 * 1024
MAX_AUTO_KEEP = 50     # 每任务自动存档保留条数（超出按最旧删行，归档不删）

class SavepointError(ValueError): ...

def archive_root() -> Path
def task_scope_of(task, path: str) -> str          # "task" | "shared"
def current_sha(workspace: Path) -> str            # exec_actions.snapshot().sha256（截断则抛错）
def ensure_archive(workspace: Path) -> tuple[Path, int, str]   # 幂等、原子、加锁
def create(db, task, *, label, note=None, kind="manual", parent_id=None) -> Savepoint
def out(db, sp, *, current_sha=None, workspace=None) -> dict   # 拼 SavepointOut
def list_for_task(db, task, *, limit=50) -> list[dict]
def diff(db, sp, task) -> dict
def restore(db, sp, task, *, scope="task", note=None) -> dict
def auto_capture(db, task, workspace, *, label, kind="auto") -> Savepoint | None
def get_archive_path(sp) -> Path                  # 校验在归档根下
```

- `auto_capture`：**best-effort**（任何异常只 log，不抛）；最新一条 auto 存档同 sha 时跳过（归档仍幂等补齐）；
  建行后 `db.flush()` 并裁剪 auto 到 `MAX_AUTO_KEEP`；**不 commit**（由调用方提交）。
- `restore` 内部禁止并发回滚（模块级 `threading.Lock`）。

## 6. 自动归档挂点（3 处，均 best-effort）

1. `services/action_runner.py::start()`：建 job 前，对 `plan.workspace` 调
   `auto_capture(label=f"{plan.action} 执行前自动存档")` —— 每次真实执行都留下可回滚的源码。
2. `services/conversations.py::send()`：`before = optimize.capture(cwd)` 之后，对 `cwd` 调
   `auto_capture(label="对话改前自动存档")`。
3. `routers/runs.py::start_loop_optimize()`：起 agent 前，对 workspace 调
   `auto_capture(label="自动优化改前自动存档")`。

导入注意事项：`action_runner` 顶层不要 import `savepoints`（savepoints 的 `restore` 会懒加载
`action_runner` 查运行中作业，避免循环依赖）。`conversations` 顶层 import 可以。

## 7. 前端（`web/src`，与后端契约严格对齐）

- `types.ts`：加 `SavepointRun`、`Savepoint`、`SavepointChangedFile`、`SavepointDiff`、
  `SavepointRestoreResult`；`TaskDeletePreview.counts` 加 `savepoints: number`。
- `lib/api.ts`：`listSavepoints` / `createSavepoint` / `savepointDiff` / `restoreSavepoint` /
  `deleteSavepoint` / `savepointArchiveUrl`。
- 新组件 `components/SavepointPanel.tsx`：
  - 头部：标题「版本存档」+ 说明 +「存档当前版本」（展开 label/note 表单）+「刷新」。
  - 列表：存档（label + 类型 chip：手动/自动/回滚前 + note title）、时间、代码 sha（前 12 位）、
    规格 vN、证据 run chips（`#30 test ok`，最多 6 个 + `+N`）、操作：对比当前 / 回滚 / 下载 / 删除。
  - 空态：说明「每次真实执行/改代码前会自动归档；点存档当前版本可手动留一个可回滚的基线」。
  - 对比：调 `/diff`，展示 in_sync、变更文件表（路径/状态/+ -/范围）、统一 diff（`<details>`）。
  - 回滚：先取 diff → 内联确认区（label、变更数、范围 radio「任务范围（推荐）/ 全工作区」、
    shared 文件警告、note 输入、确认/取消）→ 调 restore → 展示 rescue 存档 id、skipped 列表、
    是否 in_sync；随后刷新列表并 `onChanged()`。
  - 删除：`window.confirm` 后调接口。
  - 自动行不隐藏，但用 chip 弱化；默认排序新→旧。
- `components/TaskWorkbench.tsx`：新增第 8 个分区：
  `{ key: "versions", label: "版本", hint: "源码存档与回滚：每次执行自动归档；手动存档可绑定当前测试/性能证据" }`，
  放在「性能与优化」之后、「记录与交付」之前，渲染 `<SavepointPanel task={task} onChanged={refresh} />`。
- `components/TaskDeleteDialog.tsx`：`COUNT_LABELS` 增加 `["savepoints", "版本存档"]`。

## 8. 验收（必须全过）

- `cd backend && .venv/bin/python -m pytest`（全量，当前 267 → 新增 savepoints 用例后全绿）。
- `cd web && npx tsc --noEmit && npm run build`。
- 新增 `backend/tests/test_savepoints.py`，至少覆盖：
  1. 归档往返：`snapshot(解包目录) == 原 sha`；同 sha 重复归档只有一份文件。
  2. 解包安全：构造含 `../` / 绝对路径的恶意 tar → 拒绝且不写工作区外。
  3. 手动创建：环境引用取同 sha run；`runs` 匹配、`in_sync`、`spec_version` 正确。
  4. diff：任务范围/共享范围分类、unified diff、全量 in_sync。
  5. 回滚（task 范围）：任务路径还原、shared 路径不动、rescue 行生成、返回 in_sync=true；
     再查 `/diff` 范围口径无差异。
  6. 回滚（full）：shared 也还原。
  7. 守卫：有 running ActionJob / AgentJob → 409；归档缺失 → 409；当前已一致 → `restored=false`。
  8. 删除任务：savepoints 行被显式清理，counts 有值。
  9. 迁移往返：`upgrade head → downgrade base → upgrade head` 含 savepoints 表。
- 端到端（Lead 做）：对真实库 `alembic upgrade head`，起服务后 `GET /api/tasks/11/savepoints` 200；
  页面「版本」分区可存档/对比/回滚。

## 9. 文件所有权（并行写，勿越界）

- 后端：`backend/app/models.py`、`config.py`、`main.py`、`schemas.py`（如需要）、
  `services/savepoints.py`（新）、`services/action_runner.py`、`services/conversations.py`、
  `services/task_deletion.py`、`routers/savepoints.py`（新）、`routers/runs.py`、
  `alembic/versions/a9b8c7d6e5f4_savepoints.py`（新）、`tests/conftest.py`、`tests/test_savepoints.py`（新）。
- 前端：`web/src/components/SavepointPanel.tsx`（新）、`TaskWorkbench.tsx`、`TaskDeleteDialog.tsx`、
  `lib/api.ts`、`types.ts`。
- Lead：`docs/plan-savepoints-v1.md`、`.gitignore`、真实库迁移、全量回归、端到端。
