# 开发进展（进展台账）

> 只记「已达成的状态 + 证据 + commit」，不写过程流水。
> 真相源顺序：方案 = `PLAN-web-console-v1.1.md`（**v1.2，2026-10-20 落版**）；进展 = 本文件；代码 = `backend/` `web/` `ops/` `scripts/`。
> ⚠️ 交接状态（2026-10-29）：JAX 任务 #12 `lax.scan`、#11 `lax.associative_scan`、#10 `lax.cond`、#9 `lax.while_loop` 均已完整交付，并采用 `operator-v1` 证据作用域，不会被其它算子变更误伤。#9 最终有效门禁为 build #251 / smoke #252 / test #253 / benchmark #254，delivery #255；共享黑盒 51/51、pytest 138/138、白盒 120/120 全部通过，旧失败、crash 与 402 记录完整保留审计。
> 最近更新：**2026-10-29（#9 `lax.while_loop` 完整交付）**：修复 `ComputeIntLeaf` 仅写回 4 字节导致的 int32 张量叶截断；最终 operator-v1 为 `cb954eb1d713e80c4070a9d8a45066c85e5162a9581cf5f2d09010e7857dd077`，固定用例集 `cases@288876910a5e2843`。镜像 #145、宿主设备 5 上四动作门禁全部 PASS，delivery #255 已生成并将 Web 阶段同步为交付。
> 最近更新：**2026-10-28（本地合并 Gitee `origin/main` 到自动优化门禁分支）**：合入 `ac31700` JAX `lax.associative_scan` 交付 + 证据质量门禁、`755667d` 忽略本地浏览器验收缓存；本地自动优化 Step 0–3（证据门禁 / 自动验证链 / 报告归档）保留；迁移链 head `f2b3c4d5e6a7`（真实库升级前先备份）。本次合并仅 `docs/PROGRESS.md` 顶部交接块一处文本冲突，已人工收敛；上一轮合并的 5 处文本冲突（runs.py / ChatPage.tsx / api.ts / 本文件 / test_savepoints.py）审查记录仍在 `runs/merge-review-2026-10-27/`（不入库）。
> 最近更新：**2026-10-27（#11 交付 + 证据质量复盘）**：六节点 workflow 全部 PASS；白盒阶段两次失败暴露并修正执行模式作用域混淆、分支命中自报、字节对齐误判。防错规则已固化到工作区安装策略，页面执行历史新增“当前有效 / 历史失败”区分。
> 最近更新：**2026-10-27（自动优化门禁 v1：Step 3 报告归档与复现）**：报告改为 append-only 仓库
> `reports/loops/<loop_id>/<seq>-<report_sha12>/`（`report.json` 规范化 JSON + `manifest.json`
> files/raw sha 清单 + `raw/` 两侧 run 关键产物副本 + `repro.md`），服务 `services/optimization_reports.py`；
> `optimization.report(write=True)` 委托归档，`accept_evidence` 的 G7 报告门禁改为「最新归档存在且
> 深度校验通过（已有归档但被改动 → 409，不偷偷重生成）」；API `GET/POST /loops/{id}/optimization-reports`、
> `GET .../{sha}`、`POST .../{sha}/verify`；报告 payload 增 `evidence`（run 路径/环境/可比性/oracle/
> raw/复现命令）；新增独立目录只读校验脚本 `scripts/repro_report.py`（`--json/--show-commands`，不重跑真机）；
> 前端轮次卡加「报告归档」块（生成/刷新/最近 5 条/校验/下载/manifest/就地错误）。
> `pytest` **472 → 485 passed**、`tsc --noEmit` / `npm run build` 通过；改动未提交。
> 两路只读审查（后端/前端）后修复：采纳门禁绑定当前候选（旧有效归档不再被当证据）、
> `.tmp-*` 残留不参与列表且自动清理、manifest 损坏按「归档损坏」409 而非静默重生成、raw 产物路径
> 越界/绝对路径拒收、写盘后实测深度校验（不再写死 valid）、raw 变更触发新归档、每侧 raw 预算
> 避免 after 侧饿死、归档前 best-effort 落 gate 报告、非 JSON report 的 verify 返回 invalid
> 而 404、脏 seq 兜底、repro.md 用相对路径、`reports/` 进 .gitignore；前端补原字节下载端点、
> 类型对齐平铺返回、快速/深度校验语义分开、下载/manifest 防重与防御性兜底。
> 8077 已用新代码重启（http://127.0.0.1:8077）；真实库 Loop #4 已归档一份报告并完成
> 深度校验/原字节下载/独立目录复现脚本演示（reports/ 已 gitignore）。上次：**2026-10-27（自动优化门禁 v1：Step 2 自动验证链）**：`services/loop_orchestrator.py`
> 把「agent 改完 → build → smoke → test → benchmark → report」串成服务端可恢复/可取消的链：
> 持久化 `loop_auto_runs`（迁移 `a3b4c5d6e7f8`）+ 进程内 asyncio 驱动；每步走「预算 allowed
> 审计 → 起动作 → 轮询终态 → status=ok 且 gate_verdict=pass → 远端清理确认」；失败先落
> `failed_candidate` 存点再持租约回退、phase=failed；benchmark regressed 保留现场；report 落盘
> 失败停在 awaiting_adoption；取消/重启 → blocked。API `POST/GET /loops/{id}/auto-run` +
> `/auto-run/cancel`（loop 级取消会一并停链）；前置一次性预检（G1/候选 sha/四动作预算/plan）
> 不满足 409 `E_AUTO_*`，并支持从 blocked/failed 重试。前端轮次卡加「自动跑完验证」按钮 +
> 五步进度块（门禁/失败原因/清理结果/预算）+ 取消 + running 轮询；期间禁用抢工作区按钮。
> `pytest` **459 → 472 passed**、`tsc --noEmit` / `npm run build` 通过；改动已提交（Step 2 一次提交）。
> 两路独立只读审查后修复：取消/起作业竞态（起后复查+补杀+回收预算）、`_fail` 取消守卫（不静默回退/
> 不覆盖 blocked）、blocked 粘性（动作收尾不再翻转）、资源类失败→blocked 不回退、超时先清理再回退、
> report 前校验 candidate 存点（E_AUTO_CANDIDATE）、regressed 结论落 step/view、preflight/report
> 移出事件循环、跨入口互斥（手动动作/一键回退/restore/对话/采纳/否决）；前端修状态读取失败、
> done 文案与 regressed 渲染、刷新恢复误接手等。
> 真实库已升级到 `a3b4c5d6e7f8`（升级前备份 `backend/opx.db.bak-20261027-pre-step2`）；8077 已用新代码重启
> （http://127.0.0.1:8077）。
> 上次：**2026-10-27（自动优化门禁 v1：Step 1b）**：确认/启动两步落到接口 ——
> `POST /loops/{id}/confirm` 冻结草案（objective/baseline/workspace sha + instruction）并发
> 10 分钟一次性 token（库里只存 sha256）；`POST /loops/{id}/start` 同事务 CAS 消费 token 后启动，
> 目标/基线/工作区被改、重放/过期/跨轮 token 一律 409（`E_CONFIRM_*`）；`/optimize/chat` 保留兼容。
> 复用 run 改 `loop_run_links`（不再改写 `runs.loop_id`，证据读取按并集；预算仍按 loop_id 计）；
> 新增 `loop_attempts`（start 开、退出 running phase 统一收尾，取消/重启对账）+ `GET /loops/{id}/attempts`；
> `savepoints.restore` 持项目级租约。迁移 `e9f0a1b2c3d4`（`loop_confirmations`/`loop_attempts`/`loop_run_links`）；
> 前端对话页改 confirm→start（失败回滚同一套）、轮次卡加「尝试历史」。
> `pytest` **443 → 459 passed**、`tsc --noEmit` / `npm run build` 通过；改动已提交（Step 0/1a/1b 同一提交）。
> 两路独立只读审查（后端并发/事务、前端契约/交互）已完成并修复：`/start` 同步→async（否则起 agent 必 500）、
> link-only 基线的候选存点、启动失败 PENDING 泄漏、transition CAS 失败不再回滚调用方事务、终态守卫、
> cancel 按 job 精确释放租约、revert 持租约、复用 run 排他、`/start` 缺 token 409；前端修切会话状态污染、
> banner 身份校验、长 detail 截断、`E_CONFIRM_BUSY` 可行动提示等。
> 真实库已升级到 `e9f0a1b2c3d4`（升级前备份 `backend/opx.db.bak-20261027-pre-step1b`）；8077 已用新代码重启
> （http://127.0.0.1:8077）。Step 2 自动链未开始。
> 上次：**2026-10-26（自动优化门禁 v1：Step 0 + Step 1a）**：设计契约 `docs/plan-optimize-auto-v1.md`。
> Step 0 四项硬门禁：① oracle 隔离与指纹（`E_PROTECTED_PATH` + 默认 allowed_paths 收窄 +
> correctness `oracle.*_sha256` + `E_ORACLE_CHANGED`）；② baseline/accept 证据门禁（未就绪禁止改码；
> 采纳要求四条 run gate pass、perf-diff pass、精度不退化、报告落盘、工作区=候选）；③ gate verdict
> 替代 exit code；④ smoke 纳入 loop 预算/动作枚举。Step 1a：`loops.phase` 状态机 + `loop_events` +
> `workspace_leases`（迁移 `c9d0e1f2a3b4`）、建轮幂等键、启动 CAS（重复/并发 409）、取消接口、
> 重启 running→blocked、agent/action/对话共用工作区租约、轮次卡 phase chip。
> `pytest` **443 passed**、`tsc --noEmit` / `npm run build` 通过；改动未提交。
> 真实库已升级到 `c9d0e1f2a3b4`（升级前备份 `backend/opx.db.bak-20261026-pre-step1`）；8077 进程仍是旧代码，
> 重启后加载新代码。Step 1b 余项：draft/confirm token、loop_run_links、restore 租约；Step 2 自动链未开始。
> 上次：**2026-10-23（`1cf4408` 审查修复）**：HIGH 4（面板重置/精度轮误判/`dimension=all` fail-open/旧 Loop 越界门禁缺失）+ 主要 MED（预算回收/并发/绕过、诊断只读仅 prompt、verifier 自报、symlink 写穿、搜索运行态、证据限额、去重竞态、缺测量 fail-safe）已修；`pytest` 320 → **374**；新增迁移 `a7b8c9d0e1f2`（`loop_budget_audits.claimed_at` + `optimization_ideas` 唯一索引）。见下节「2026-10-23 审查修复」。
> 上次：**2026-10-26（origin/main 同步后复查修复）**：已同步到 `472a35d`；此前 `1cf4408` 审查修复已提交为 `edc41d3`，任务生命周期 v2 / 默认镜像为 `de17cae`，`elementwise_sub` 与精简性能集为 `7d9f5c0` / `472a35d`。本轮复查又修复版本证据、用例确认、交付/文档路径安全、重复交付覆盖、项目执行机/镜像错配等 fail-open 问题；本地 `pytest` **434 passed**、`tsc --noEmit` / `npm run build`、全量 shell 语法、空库迁移至 `b8c9d0e1f2a3` 均通过。真实库已停服务备份为 `backend/opx.db.bak-20261026-pre-b8c9-review-fixes` 后升级到 head（`PRAGMA integrity_check=ok`），8077/5173 已重启并通过健康、任务列表、阶段看板与版本接口冒烟。
> 最近更新：**2026-10-21 收工（版本存档 savepoints v1 + 优化轮次 v1（阶段一）+ PerfDiffPanel 合并删除 + 快照口径修复）**：`pytest` 206 → **309**、`tsc --noEmit` / `npm run build` 通过；迁移 `a9b8c7d6e5f4`（savepoints）/ `b1c2d3e4f5a6`（optimization rounds）。
> 同日早些时候：**2026-10-21（任务外页面审查修复 + 端点 key 迁移到本机密钥库 + 端点驱动 dsh 模型路由：56 条修复收口；迁移 `d2e3f4a5b6c7`（端点软删）/ `e3f4a5b6c7d8`（secret_ref）；key 不再落 SQLite，改 macOS Keychain → Linux Secret Service → 0600 文件；dsh 模型路由改为按「API 管理」当前端点动态生成 patch（修重启后 401）；对话页并入任务详情「对话」分区（独立对话页与 agent run 入口移除，开发全走任务会话）；`pytest` 206 → **276**、`tsc --noEmit` / `npm run build` 通过，详见下「任务外页面审查修复实测记录」「端点 key 迁移到本机密钥库」「端点驱动 dsh 模型路由」「任务外页面审查收口」）**
> 上次：**2026-10-21（任务页审查修复 + 端点定价收尾：HIGH 2 类 / MED 9 类 / LOW 若干；随后补修 4 个残项（定版后 chat 只写日志、create_spec 冻结 op_name、回填 dirty 字段、侧栏切走 flush）；`pytest` 206、`tsc --noEmit` / `npm run build` 通过，详见下「任务页审查修复实测记录」）**
> 上次：**2026-10-21（隐藏「流式原型」页：侧栏入口与视图分支移除，组件 `web/src/components/StreamProto.tsx` 与后端 `/api/proto/*` 保留）** · 2026-10-21（文档口径校正：过期状态 / 测试数 / DB 快照；无代码改动） · 2026-10-21（任务删除 + 规格版本显示修正：id 与 vN 分离、旧结果/当前规格提示） · 2026-10-21（任务删除：预览影响面 + 硬删除 + 可选清理本机文件） · 2026-10-21（任务页复查微调：失败作业与成功 run 并列、记录页折叠） · 2026-10-20（任务详情布局重构：作业台六分区） · 2026-10-20（复查修复：Roofline 去硬编码 + 优化 Loop 证据刷新 + 正式 agent run 逐 token/过程区重设计） · 2026-10-20（任务详情补齐三块） · 2026-10-20（前端补齐：规格落版等） · 2026-10-20（前端与功能对齐） · 2026-10-20（M2 交付形态调整） · 2026-10-20（P0-2 对话页：自研 B 上线） · 2026-10-20（Loop #4 自适应 tile：三轮 A/B 首次采纳） · 2026-10-20（方案 v1.2 落版 + 三项决策） · 2026-10-20（四动作真执行） · 2026-10-20（Loop #3 + 设备映射修复） · 2026-10-19（四页占位全补齐）

## 一句话现状

方案已落版 **v1.2**（控制台主线 + 算子业务契约层 + 2026-10-15 需求调整）。**P0 骨架 / P1 执行链 / P1.5-a 容器闭环 / P1.5-b 契约最小集 / P1.5-c 门禁与环境 / M1-a 首个算子**全部跑通并实测 —— 容器里能看到 **8 张卡**、`torch_npu` 可用；门禁校验器能机判，设备互斥（远端 flock + 可见设备对齐）与版本对齐（内容哈希快照 / 判据档 / 用例集 ref / 镜像 digest）已落地，CANNBot **40 skills + 3 agents** 已装进任务工作区并带 lock 指纹。
**M1-a（2026-10-15）**：首个算子 `elementwise_add` 的 **Ascend C 直调**实现真机跑通 —— 19 条用例（常规/边界/数值边界/异常/回归载体）**全过**，第一份 `correctness.json` 落盘，`GET /api/runs/5/gate` **判 pass**；`runs`/`artifacts` 从此有真实数据。过程中抓出三个真问题（模板分块不回推 GM 偏移、容器每次干净导致 test 依赖 build 产物、门禁同类产物撞车造成假阴性），都已修 + 加回归。
**需求调整（2026-10-15）已并入方案 v1.2（2026-10-20 落版）**：需求文本按**分层**处理 —— `REQUIREMENT.md` 现行需求置顶、原始文本降为附录（`docs/REQUIREMENT.md` §一 / 附录 A）；方案正文直接采用 A1–A13（§13 保留为变更记录）。**三项决策（用户 2026-10-20 定）**：① 案例库 = **10 个案例全部真机全链路（硬性）**，算子清单可在同场景内替换；② 自动优化 = 必需能力 + 至少一份有效 A/B 报告、不设加速比硬线；③ 需求文本分层（现行置顶）。**D9 已定并解除前置**；**M2 仍放最后做**（2026-10-19 定）。**当前缺口**：M2 案例内容 **0/10**（框架已完成）、交付 / 独立目录复现（验收 #10/#12）未验、P3 写入侧与 P5 能力路由未做；详见「待办 / 未决与风险」。

**P0-3「任务开始」页（2026-10-15，用户要求「优先把页面功能规划好」）**：功能规划 `docs/UI-task-start-v1.md`；布局样机 `docs/mockups/task-start-v1.html`（A/B/C 同页切换 + 亮/暗）→ **用户选定变体 B（对话优先 + 抽屉）**；**静态骨架已落码**（commit `21f707c`：`web/src/components/TaskStart.tsx` + `App.tsx` 的 `task-new` view + `index.css` 组件类 —— 假数据、`tsc --noEmit` 通过、浏览器逐项点通，**未接后端**）。

**P0-3 第一步「草稿生命周期」已落码（2026-10-15，详见下「P0-3 第一步实测记录」）**：四问已拍（草稿首次输入即建 / 上下文空间确认时落盘 / 本页是入口非首页 / 先草稿后对话解析，见 `UI-task-start-v1.md` §2.3）；`tasks` 加草稿与创建期字段（迁移 `c7d8e9f0a1b2`）；`POST /api/tasks` 扩字段 + 新增 `PATCH /api/tasks/{id}`（存草稿/回填/转正）+ 列表 `?draft=` 过滤；前端去假数据、首次输入即建草稿、改动自动保存、`?id=` 重新加载。

**P0-3 第二、三步「对话式澄清（dsh 会话）+ 定版 / 上下文空间」已落码（2026-10-15，详见下「P0-3 第二步 + 收尾实测记录」）**：`POST /api/tasks/{id}/chat` **走 dsh 会话**（`--session-id` 续会话），逐轮产出「回复 + 规格草稿 + 待确认项」；`POST /api/tasks/{id}/finalize` 是**定版门禁**（待澄清清零 → 落 `spec_versions` v1 → 生成上下文空间 `tasks/<slug>/` → 草稿转正）。**P0-3 至此闭环。**

**流式能力（2026-10-15 调研 + 原型，详见下「流式能力调研 + dsh 插件 spike」）**：实测 `dsh headless` / `dsh acp` **只给提交态**（`text` / `agent_message_chunk` 整段一次性）；但 dsh 的**官方扩展点** `ctx.on("agent/assistant-stream", …)` 能拿到**真 token 级**增量 —— 已用几行 Cordis 插件打通（`ops/dsh/opx-typewriter/`），**挂载不需要 pnpm**（`--patch` + 相对 `name`）。**不动 harness、不换 opencode**。原型旁路跑通（后端 SSE + 前端「流式原型」页，dsh / opencode 可切）；**2026-10-21 起该页已从侧栏隐藏**（组件与后端接口保留，流式主链路走「任务开始 / 对话」，原型只作对照）。

**流式已并入「任务开始」真实对话（2026-10-15，详见下「流式并入实测记录」）**：定下**两段式输出契约**（先流式说人话 → `<<<SPEC>>>` → 末尾 JSON），插件挂进 `/chat/stream` 的每轮 dsh；**prose 逐 token 上屏、JSON 尾由后端吞掉**（`services/prose_relay.py` 三态状态机），复用已有的 agent-job SSE 通道（新增 `{"type":"delta"}` 事件）。实测单轮 88~99 段增量、prose 449~471 字、**零 JSON 泄漏**、落库正文与上屏**逐字一致**、两轮续会话 session 不变；插件静默时有兜底（整段补发）。旧格式（整段一个 JSON）继续兼容。`pytest` 56 → **73 项**。

**M1-b 已闭环（2026-10-16，详见下「M1-b 实测记录」）**：`benchmark`/`profile` 两个动作**实装并真机跑通**（run #6/#7 门禁 **pass**）；**D9 定档**——`msopprof` 四类指标（耗时/数据搬运/存储使用/计算单元负载）**都能采到**，门禁对 `profile` 从「只要 log」收紧为**必需 `log`+`metrics`**。基准有一处关键纠偏：**两侧都改走 msopprof 量设备侧 `Task Duration`**（用 host 墙钟会把 PyTorch 派发开销当成算子差异，算出 5–6× 的假加速比），真实结果 **torch.add 8.46us vs 候选 7.98us ≈ 1.06×**（小尺寸 64k 如实报出候选反而更慢）。`pytest` 73 → **85 项**。

**M3-a 已闭环（2026-10-16，详见下「M3-a 实测记录」）**：自动优化闭环跑通首个真机 A/B —— **硬件画像**（`ops/chips/dav-2201.yaml`）+ **`perf` Loop**（成对 run 关联）+ **`performance-diff.json`**（前/后耗时对比 + 精度不退化 + 不确定性 + 结论枚举）+ **门禁规则**（精度退化硬拦、性能回退只 warn）+ **P2 对比视图**。首个 A/B **如实**：分块 4096→8192 后主用例 **-1.2%**（落在不确定性内 → `no_significant_change`），64k 反而 **+19.6% 回退**，精度 19/19 不退化 → 该优化**不予采纳**（说明当前实现已接近该 shape 搬运上限，小 shape 应走 core-shrink 方向）。`pytest` 85 → **97 项**。

**M3-b（自动优化增强 · agent 自动改代码）已闭环（2026-10-19，详见下「M3-b 实测记录」）**：把 perf Loop 里原本手工的「按优化假设改算子代码」交给 agent —— `POST /api/loops/{id}/optimize` 起一次 dsh job（工作区 cwd、挂 skills），跑完算**改前/改后代码快照 + 逐文件 unified diff + 结构化摘要**落 `loops.optimize_json`，并提供 `POST /api/loops/{id}/optimize/revert` **一键回退**；前端 PerfDiffPanel 加「让 agent 自动优化」+ 进度 + diff + 回退，并可从 `performance-diff` 的**待验证建议「开新 Loop 并让 agent 执行」**。真机端到端冒烟通过（dsh 15 次工具调用、590 段 token 流、证据 `changed=false` 如实记录）。**边界**：agent 只改代码，**不跑** build/test/benchmark（四动作远端执行仍照 `plan.command` 单独触发，属另一件事）。`pytest` 102 → **111 项**。

**页面补齐（2026-10-19，详见下「页面补齐实测记录」）**：任务详情新增「**轮次与关联**」（Loop / AgentJob / Run 归属 + 每轮消耗）与「**交付**」（验收状态 + 复现命令 + 产物清单与下载）两个分区，补齐方案 §7 的执行记录/交付；**P4「消耗情况」页**上线（`GET /api/usage` 只算真实调用、排除 `is_demo` 示例行；成本未定价如实标注）；新增 `GET /api/artifacts/{id}/download`。`pytest` 111 → **113 项**。

**测试管理页（2026-10-19，详见下「测试管理实测记录」）**：P3 读侧上线 —— `GET /api/tasks/{id}/cases` 把**用例矩阵**（工作区 `cases/<op>/cases.json`）与**逐例结果**（各次 `test` run 的 `correctness.json`）合并，给「用例矩阵 + 逐例历史 + 失败点 + 执行记录（含报告下载）」。前端 `TestPanel`。`pytest` 113 → **116 项**。

**阶段看板（2026-10-19，详见下「阶段看板实测记录」）**：P3 读侧上线 —— 阶段由**真实证据推导**（各动作最近一次 run 的 `gate.evaluate` verdict + 是否有规格），不信手填的 `tasks.stage`；`GET /api/stages` 按推导阶段分组，`POST /api/tasks/{id}/stage/sync` 按证据把 stage 落正并写 `stage_events`（推进留痕）。**「交付」段当前永不为真**（四动作不含 delivery，如实反映）。`pytest` 116 → **119 项**。

**API 管理（2026-10-19，详见下「API 管理实测记录」）**：P4 配置态上线 —— 建 `endpoints` 表（迁移 `b2c3d4e5f6a7`，含一条真实默认路由）+ `GET/POST/PATCH/DELETE /api/endpoints` + 连通性探针（`GET {base_url}/models`）。**凭据只存引用名**（环境变量名），不进库/不回显；探针只验连通性、不产生费用。前端 `ApiPanel`。**至此四页占位（阶段看板 / 测试管理 / API 管理 / 消耗情况）全部补齐。** `pytest` 119 → **123 项**。

**Loop #3 真机 A/B（2026-10-20，详见下「Loop #3 实测记录」）**：把「agent 自动改代码 → 同卡 A/B → `perf-diff` 判定 → 一键回退」整条闭环在真机上完整跑了一遍。假设 = 满块（16KB、32B 对齐）从 `DataCopyPad` 改走 `DataCopy`（agent 自动改，1 文件 +13/−6）。**结果如实：1M 主用例 8.00 → 8.18us（+2.25%，刚过 ±2.17% 不确定性，判 `regressed`）；64k 7.34 → 7.34us（0.00%）；精度 19/19 不退化 → 不予采纳，已回退。** 过程中抓到一个真 bug：**设备锁与实际用卡不是同一张**（容器固定暴露 0–7、算子写死 `aclrtSetDevice(0)`，`--devices 1` 只锁了 dev1）→ 已修 `scripts/nm_test.sh`：容器内可见设备 = 锁定的卡（`device_id` 从此是真的）。`pytest` 123 项不变。

**四动作「点一下就真执行」已接通（2026-10-20，详见下「四动作真执行实测记录」）**：`dry_run=false` 不再是 501 —— 后端照 plan 起**异步作业**（新表 `action_jobs`，迁移 `c3d4e5f6a7b8`），子进程跑 `scripts/nm_test.sh`，脚本自己登记 run，跑完按 `run_ref`（= `runs.remote_workdir`）找回 run 并跑门禁，verdict 回填作业行。前端 `ActionPlanPanel` 加「真执行」+ 1.5s 轮询 + 实时日志 + 取消。**真机实测 build：job #1 → run #19 → 门禁 pass。** `pytest` 123 → **128 项**。

**P0-2 对话页已上线（2026-10-20，详见下「P0-2 对话页实测记录」）**：**自研（方案 B）** —— 新建 `conversations` / `messages` 表（迁移 `d4e5f6a7b8c9`）+ `services/conversations.py`（复用 `agent_runner`，不新写引擎）+ 前端 `ChatPage`。**任务会话默认挂 skills**（cwd = 任务工作区，40 个 CANNBot skill、可读写代码）；轻对话用空目录（不挂 skills、不碰工作区）。真机两轮续会话：16 段逐 token 增量、session 不变、消息与会话 token 正确回填。**不反代 `dsh web`**（WS/cookie/版本耦合成本高）。`pytest` 130 → **136 项**。

**Loop #4 自适应 tile（2026-10-20，详见下「Loop #4 实测记录」）**：先补了两组真机测量（核数–耗时曲线 + tile 扫描），坐实「64k 的瓶颈是 tile 太大、只起了 16/48 核」，然后按 N 自适应（`N≤65536` 用 512，否则 4096）。**三轮 A/B 首次采纳**：1M 主用例 **8.10→8.10us（0.0%）**、64k **7.44→5.18us（−30.4%，improved）**、精度 19/19 不退化。`pytest` 130 不变（算子侧改动，未加控制台测试）。

**下一步**：① 自动优化已跑通三轮，**Loop #4 首次采纳**（64k −30.4%、1M 零退化）—— 下一轮可把「按形状补测 + 自适应阈值」做成常规动作（新案例按各自 shape 扫一遍）；② 流式增强余项（P0-2 已上线；正式 agent run 流式、过程区视觉重设计仍待做）；③ `perf_diff` 的「设备一致」检查从「比台账字符串」升级为「比实际可见设备/物理卡」；④ ~~四动作取消传到远端~~ ✅ 已实装（2026-10-20：本地进程组 + 远端 `docker rm -f`，见「页面级增强」）；⑤ **任务详情六分区（2026-10-20）已按复查反馈微调（2026-10-21：失败作业/成功 run 并列、记录页折叠与窄屏检查，见下「任务页复查微调」）；余下分区命名、URL 持久化待定**。**M2 案例库已延后到最后做（2026-10-19 用户定，见下「待办」）。**

**P2 结果视图已完成（2026-10-16）**：新增 `GET /api/runs/{id}/report?type=…`（读回机读报告）+ `GET /api/chips`（硬件画像）后端接口；前端「结果视图」（ECharts 按需引入）：**正确性**逐用例误差、**性能**候选 vs 基线 + **极限对比/Roofline 判定**（达成带宽 vs HBM 峰值）、**指标**四类状态与来源；**结构化优化建议**（§4.6 R1：证据/假设/改动/预期/风险/验证法）随 `perf-diff` 出；**导出 JSON**（报告与对比）。`pytest` 97 → **102 项**。

**端点定价 + 消耗金额（2026-10-21，解 D2，详见下「端点定价实测记录」）**：用户复查时问「费用合计 未定价 应该是 API 管理加的费用吧，没找到配置的地方」—— 确认属实：此前 `endpoints` 没有单价字段、`agent_jobs.cost` 从未回填，消耗页只能如实标「未定价」。本轮补上：**API 管理页可配单价**（人民币元 / 每百万 token：输入 / 输出 / 缓存读 / 缓存写，in+out 都配才算已定价）+ 调用创建时快照 `agent_jobs.endpoint_id/model` + `services/pricing.py` 按 token 实时算金额（费用不落库，改单价历史实时重算；早期调用回退当前端点并标「估」）。`pytest` 170 → **176 项**。

## 2026-10-27：`lax.scan` 首轮交付 + 证据按算子隔离

- **任务 #12 `jax_lax_scan` 已完成首轮交付**：恢复并核验同规格、同设备、同镜像、同用例集的
  build #73、smoke #74、test #76、benchmark #58，有效门禁全部 PASS；delivery run #77 已生成。
- **证据身份改为按任务/算子隔离**：代码快照只覆盖 `ops/<op>/`，用例哈希只覆盖
  `cases/<op>/`。其它算子的开发不再使已通过任务误报“代码快照不一致/用例集已变更”。
- **历史证据不改写**：旧 run 的原始 manifest、全工作区哈希与失败记录继续保留；核验后的
  operator-v1 别名作为附加元数据，交付选择当前规格下最近一组匹配且 PASS 的证据，不再被后续失败覆盖。
- **回归结果**：后端全量 pytest 通过；前端 `tsc --noEmit` 与生产构建通过；#12 交付预览
  `missing=[]`，交付包内 correctness 来自 test #76、performance 来自 benchmark #58。

## 2026-10-27：`lax.associative_scan` 完整交付 + 复盘固化

- **任务 #11 完整交付**：workflow1 六个节点全部 PASS；最终算子快照
  `cf080296b95fd8d96ce08d57173403fbec22fdb0ed6ca3fee42c1a13e78a53f1`，固定用例集
  `cases@6d6bc47fa93738dc`。同镜像、设备 0、同规格/用例集下 build #99、smoke #100、
  test #101、benchmark #102 门禁全部 PASS；基准中位数 80.212us / 308.806us。
- **交付**：delivery run #103 已生成；包内包含算子、Python API、冻结用例、pytest、设计文档与
  真实门禁证据。交付包 sha256 为
  `b39993c012c6ddf1bd13c3b82811e9a68a66178c83bb2e22a8d1cc34dddf579f`。
- **失败证据不抹除**：节点 3 前两次白盒验收失败继续保留。第一次缺 9 叶子、ACL 首次初始化与
  CLI 分支证据；第二次把 whole-tree 的 9 叶子限制误套到 per-leaf，并把 16B 误写成 32B 对齐。
  第三次修正测试范围、计算口径和记录后通过，节点 4 完成缺陷修复与全量回归。
- **流程防错固化**：工作区证据质量门禁新增六项硬规则——权威身份源、operator-v1 隔离、分支命中
  必须有原始证据、执行模式分列、对齐字节实算、PASS/xfail/unsupported 分列；交付前重新核对
  当前快照与四动作。执行历史页面把红色失败明确标为“历史失败”，当前交付证据标为“当前有效”。
- **#10 `jax_lax_cond` 已于 2026-10-28 完成交付**：正式修复 workflow 六节点全部 PASS，最终有效
  build #226、smoke #227、test #232、benchmark #229、delivery #233；固定用例集
  `cases@233778f456b4eb9b`，operator-v1 为
  `55b69ca6a7dbc10b22be8976091cdac6d00931f68eee552fd29f9f8ee121c426`。AFIR 对
  outer-`jax.jit` tracer pred 的后端限制按原始证据保留，CPU/模拟/unsupported 均未计作 PASS。
- **#9 `jax_lax_while_loop` 已于 2026-10-29 完成交付**：spec v1、16+2 用例与
  `cases@288876910a5e2843` 全程冻结；最终 operator-v1 为
  `cb954eb1d713e80c4070a9d8a45066c85e5162a9581cf5f2d09010e7857dd077`。镜像 #145、
  宿主设备 5 上 build #251、smoke #252、test #253、benchmark #254 全部通过，delivery #255
  已生成；共享黑盒 51/51、pytest 138/138、白盒 120/120 全部 PASS，0 skipped/xfail/unsupported。
  白盒阶段发现的 `ComputeIntLeaf` 仅写 4 字节问题已改为按 `outNumel` 完整分块读写；旧 crash、
  DSH 402、白盒失败与修复前真机记录继续保留，不覆盖当前有效证据。

## 2026-10-26：JAX 无卡准备 + 对话发起自动优化

- **JAX 无卡准备**：第一批 10 项已按公共环境前置 → 控制流 → 微分/SPMD → attention →
  Pallas 的顺序建立 Web 任务；固定官方 `cann/xla-npu` commit、Python/JAX/CANN/AFIR 组合，
  可复现镜像 `opx-xla-npu:jax081-cann920-910b` 在远端构建（构建不占 NPU）。任务、卡数、
  用例草案和有卡后的串行门禁见 `docs/plan-jax-primitive-cases.md`；该条记录的是无卡准备阶段，
  后续真机结论见下一条。
- **JAX 公共前置 #13 已真机通过（2026-10-26）**：镜像 id `b0a1aaec9133…`，固定 xla-npu
  revision `2a55287…`、Python 3.11.17、JAX/JAXLIB 0.8.1、CANN 9.2.0-beta.2。无卡检查通过；
  设备 0 独占探针识别 `backend=npu / devices=[npu:0]`，AFIR 编译成功，阻塞式 `jax.jit`
  得到 `[3,5,7,9]`。首轮因缺 `ASCEND_MLIR_PYTHON_PATH` 真实失败，补齐官方环境配置后重跑通过；
  证据在远端 `/data/lfx/op-agent/runs/task13-xla-npu-probe/`。下一项 #12 `lax.scan` 已落用例草案，
  停在人工整组确认门禁前。
- **优化交互改为“对话在前、记录在后”**：性能与优化页的新方向入口只跳到任务对话；
  用户发送自然语言方向后，才同时落为会话消息和 `optimization_ideas(source=conversation)`，再生成文字目标 objective、
  基线存档与独立 Loop；已有同快照 test+benchmark 时立即起 `kind=优化` Agent，没有基线则明确
  停在 `waiting_baseline`，已有上一条开放 Loop 时保留新思路为 `waiting_previous`，不绕过串行规则。
- **安全边界不变**：只自动产出候选代码；同条件 A/B、允许路径、测量预算和 verifier 门禁继续生效；
  自动采纳仍关闭，候选 test+benchmark 之后必须人工采纳或否决。
- **审查收口**：复核 2026-10-23 流程审查报告，已修 H8 产物路径越界/无界读取、H7 普通任务
  对话结束不刷新、H4 创建期死锁（合法 op_name、草稿禁写规格、非空对象规格、模型回填校验），
  以及 H5 smoke 硬前置（test/benchmark 与交付只认当前条件下通过的 smoke）；
  其余 High 项的当前状态与顺序见 `docs/review-2026-10-23-fix-checklist.md`。
- **验收**：后端临时库 **441 passed**；`tsc --noEmit` / `npm run build` / `git diff --check` 通过；
  Playwright 实际点通「性能与优化 → 去对话提出优化方向 → 对话模式横幅 → 输入方向 → 按钮启用」，
  并在窄宽页面确认任务列表操作按钮不再换行；未向真实任务提交测试方向、未污染真实 Loop/run。

## 2026-10-26：同步 `472a35d` 后的生命周期 v2 复查修复

- **版本/交付证据 fail-closed**：小版本固化快照后若工作区再变更，交付必须先建新小版本并重跑；run 缺 `suite_ref` 不再被当作匹配。未知动作不能写入空豁免。
- **用例确认硬门禁**：只有 `cases_confirmed_at`、缺少确认哈希，或当前哈希无法计算时均拒绝 build/test/benchmark/profile，避免历史脏数据绕过整组确认。
- **路径安全**：文档上传拒绝 `task-docs/task-N` 符号链接写穿；交付预检与打包拒绝工作区外解析和 `ops/<op>/` 内符号链接，避免把本机任意文件带入 zip。
- **交付证据不可变**：重复交付的 zip 文件名加入 run id，旧 Artifact 不再指向被后一次交付覆盖的内容。
- **项目配置一致性**：创建/更新项目时强制默认镜像属于默认执行机；只换执行机但遗留旧机镜像会返回 400，成对切换可正常保存。
- **验收**：临时库后端全量 **434 passed**；前端类型检查与生产构建通过；tracked shell 全量 `bash -n` 通过；新空 SQLite 从 base 升级到 Alembic head `b8c9d0e1f2a3`；`git diff --check` 通过。真实库停服务备份后由 `c2d3e4f5a6b7` 升级到 `b8c9d0e1f2a3`，完整性检查通过，任务/run 数量为 2/21；8077/5173 重启后接口与页面均为 200。

**任务页审查修复（2026-10-21，详见下「任务页审查修复实测记录」）**：对任务列表 / 任务开始 / 任务详情作业台 / 测试管理 / 阶段看板做了三路只读 code review（27 条，报告在 `runs/review/`，不入库），本轮修掉 **HIGH 2 类**（流式 AgentJob 起进程失败/卡住会永久锁死任务；`finalize` 非幂等且阶段静默回退）与**全部中危**（任务配置 device/timeout 未接线、阶段/规格刷新链路、SSE 断流误判、上下文空间与 run 目录归属、无 run 台账仍标 ok、删除结果反馈、时区、表头错位、测试管理默认错任务等）+ 多数低危；随后补修 4 个残项（定版后 chat 只写 `chat_log`、`create_spec` 冻结 `op_name`、对话结束回填不覆盖 dirty 字段、侧栏切走前 flush）。`pytest` 176 → **206 项**。

**任务外页面审查修复（2026-10-21，详见下「任务外页面审查修复实测记录」）**：对服务器与镜像 / API 管理 / 消耗情况 / 对话 / 案例库做三路只读 review（56 条，报告在 `runs/review/`，不入库），修掉 **HIGH 4**（导入脚本 `manifest.name` 路径穿越/`--force` 删库外、端点主键复用导致历史费用被改写、`Infinity`/超大单价让消耗页白屏、对话回退旧轮覆盖后续改动）与主要中危：SSH 目标选项注入、对话并发/SSE 断流/取消/接回、会话删除附件与幽灵 agent、案例一键发起非原子、key→env 回退口径、端点删除语义（改软删）、镜像元数据不可编辑/探针串旧机/镜像清单只增不删、`op_name` 路径校验、案例报告畸形 JSON、资源页执行机口径与镜像竞态；新增迁移 `d2e3f4a5b6c7`（`endpoints.deleted_at`）。`pytest` 206 → **274 项**，`tsc --noEmit` / `npm run build` 通过。

## 2026-10-23 审查修复（commit `1cf4408` 后续，未提交）

**背景**：对 `1cf4408` 做三路只读审查（后端服务层 / API+前端 / 独立验证；报告 `runs/review/2026-10-23-pull-1cf4408.md`，不入库），发现 4 HIGH + 多 MED；本节记录修复后的状态。

**核心修复**：

- **前端面板**：重置 effect 只依赖 `task.id`，父级 `onChanged → reloadTask` 不再清空通知/验证结果/证据/输入，也不再掐断刚连的 SSE；拆出「预算审计数据刷新」effect；诊断搜索刷新后可恢复轮询、可取消；证据区展示 verifier / pipeline / 自动回退原因（`guard_reverted`）。
- **perf_diff 门禁口径**：精度轮的 `target_cases`（correctness 用例 id）不再被当 benchmark 目标 → 不再误判 `E_TARGET_CASE_MISSING/fail`；`dimension=all` 对全部可配对用例保守聚合（任一 regressed 即 regressed），与报告层复用同一聚合函数；缺有效中位数 → `delta_pct=null` / `missing_data=true` / inconclusive，不再伪造 −100% improved；`E_TARGET_CASE_MISSING` 仅在 case_id 真正不存在时 fail。
- **自动优化安全边界**：旧 Loop（无 objective）改为只读（API 409）；`pipeline.verifier` 为空时即使 `verification=pass` 也自动回退；`allowed_paths` 为空时 fail-closed（有改动一律回退）；`ops/<op>/` 下指向工作区外的符号链接按越界处理并回滚；证据新增 `attempted_after_sha256` / `guard_reverted` / `guard_reason`。
- **预算硬门禁**：`check-then-insert` 用进程内锁串行化（单 uvicorn worker 前提，注释写明多 worker 需 DB 原子写）；`plan_action` busy 预检先于审计、启动失败 `void_audit` 回收额度；删除可被调用方设置的 `OPX_BUDGET_RESERVED`，改为服务端一次性预留（页面写 allowed 审计 → `OPX_BUDGET_AUDIT_ID` → 脚本 claim，`claimed_at` 只能消费一次）；`nm_test.sh` 带 `--loop` 一律 fail-closed 校验；409 文案改为「新建下一轮」；运行中 Run 不再被 ActionJob/Run 双计。
- **P2 优化点搜索**：诊断 job 前后工作区快照比对 + 自动回滚（只读不再只靠 prompt）；`_finish` 先解析后入库，异常不再半截提交；去重靠新唯一索引 `(task_id, dedupe_key)` + `IntegrityError` 兜底；「一思路一 Loop」以「存在未关闭 Loop」判定 + 进程锁，PATCH 不再能写系统态；僵尸 running 搜索自愈、启动失败收尾、`build_prompt` 移线程池；证据包/`extra` 分项限长 + 总长硬上限；解析严格校验 `ideas` 数组。
- **数据层**：迁移 `a7b8c9d0e1f2`（down_revision `c2d3e4f5a6b7`）新增 `loop_budget_audits.claimed_at` 与 `optimization_ideas` 唯一索引（升级前清理历史重复行）。本机真实库已升级到 head，升级前备份 `backend/opx.db.bak-20261023-pre-fix`、`backend/opx.db.bak-20261023-pre-a7b8`。

**验收**：`pytest` 320 → **374 passed**（新增 `test_idea_search.py`、`test_loop_budget.py` 与 perf_diff/前端契约回归）；`tsc --noEmit` / `npm run build` 通过；`bash -n` 全部 tracked shell 通过；临时库迁移 `upgrade → downgrade -1 → upgrade` 往返通过；真实库副本启动冒烟 + 新增端点（ideas / idea-searches / budget-audits）200、旧 Loop optimize 409。**改动尚未提交**；当前 8077 进程仍是旧代码，重启后生效（库已迁移，可安全重启）。

**未做/后续（本次明确不做，避免扩大范围）**：多 worker/多实例下的预算原子性（当前单 worker）、预算门禁对「不传 `--loop` 的直跑」仍不计费（协议式边界）、未消费预留无超时回收、真实 dsh 端到端与浏览器 e2e 未跑（需真机/起服务）、`/runs` 手工改挂 objective Loop 的审计绑定未强制。

**2026-10-23 二次修订（按用户规划对齐 P2 流程）**：

- **文字优化点目标**：`objective` 新增第三维 `target_dimension="text"` + `focus`（文字描述「需要优化的点」）。由「思路池 → 按此思路建轮次」时默认使用文字目标，不再默认「全部用例」：不指定 benchmark 目标用例，perf-diff/报告按 benchmark primary 兜底；focus 进 scope/judge 与自动优化 Agent prompt。普通「新建优化轮次」表单仍是用例/全部两维。
- **建轮次弹窗**：「按此思路建轮次」不再单击直建，改为弹窗：只读思路摘要 + 可编辑必填「需要优化的点」textarea + 精度不退化（固定）/性能不退化 + 每动作次数 + 复用基线 run；提交 text objective。
- **否决/标记无效**：「否决」按钮对**所有未关闭轮次**可见（含基线未跑、agent 失败、候选验证失败、预算耗尽），解决「失败后轮次卡 open、思路卡 validating、串行规则挡住下一条思路」的问题。后端 `reject` 本就支持无候选存档；关联思路一律置 `rejected`（负样本）。
- **否决可回退**：`POST /api/loops/{id}/reject` 新增 `revert`。`revert=true` 时优先用 `loops.optimize` 的改前快照回退，否则回退到建轮次时的 baseline 存点；回退失败不影响否决，结果写在 `reverted`（`ok/mode/reason`）。
- **预算默认 2→3 次/动作**（`DEFAULT_BUDGET`）：基线+候选 2 次后留 1 次失败/环境类重试；已有轮次保留已存的 `runs_per_action`。前端建轮次表单默认同步 3。
- **展示**：轮次卡「目标」对 text 维度显示「优化点：focus」（长文可折叠），自动优化确认框同步替换「目标用例」行；旧数据（cases/all 维度）展示不变。
- **验收**：`pytest` 374 → **378 passed**；`tsc --noEmit` / `npm run build` 通过；迁移无变更（沿用 `a7b8c9d0e1f2`）。
- **未做**：真实 dsh/浏览器 e2e 仍未跑（需真机/起服务）；text 维度暂只在思路入口开放，普通新建轮次表单若也要文字目标可后续小改。

**2026-10-23 三次修订（自动优化对话化，按用户交互要求）**：

- **入口**：「性能与优化」轮次卡的「自动优化」不再弹确认框，直接跳转到任务详情「对话」分区；顶部横幅展示本轮 `Loop #N / 优化点（focus）或目标 / 允许改动路径 / 每动作预算`，说明「只产出候选，仍需人工验证采纳」。
- **交互**：输入框输入「参考指令」（可留空），回车/点按钮即启动本轮受控自动优化（不是普通聊天）；过程复用现有 agent job SSE 在对话里逐 token 显示；完成后后端自动落一条助手消息（摘要/改动文件/回退原因/verifier/下一步），会话 session 续用，页面刷新/切回可按会话 `running_job` 自动接回。
- **完成之后**：同一会话输入就是普通对话，可追问或让 agent 继续改代码（改完提示回轮次卡重跑候选验证）；「在对话中查看」打开历史轮次会话时默认只看历史，需点「重新运行自动优化」才 armed，避免误触重跑。
- **后端**：新增 `POST /api/loops/{id}/optimize/chat`（建/复用会话 + 写 user 消息 + 起 `kind=优化` job，`AgentJob.conversation_id` 已有字段、**无迁移**）、`GET /api/loops/{id}/conversation`（「在对话中查看」用）；旧 `POST /api/loops/{id}/optimize` 保留兼容。`conversation.running_job` 查询不再按 kind 过滤（优化 job 也能接回/显示运行中）。
- **前端**：`ChatPage` 增加自动优化上下文横幅与启动分支；`TaskWorkbench` 增加 `openOptimizeChat` 跨分区带入；`OptimizationRoundPanel` 删除旧确认框/SSE 启动与重连逻辑，保留证据/验证/采纳/否决/回退/预算，`agentRunning` 改由 `task.agent_jobs` 推导。
- **验收**：`pytest` 378 → **380 passed**；`tsc --noEmit` / `npm run build` 通过；后端 `/openapi.json` 86 path（新增两条）；服务已重启（8077）。
- **未做**：浏览器 e2e 与真实 dsh 全链路（需真机）；「会话中断后恢复优化」仍按 agent_runner 的普通中断语义（不自动重跑）。

## 2026-10-22 自动优化链路修正 + 真机复验

**链路修正**：`POST /api/loops/{id}/optimize` 的提示契约改为必须读取 `ops-direct-invoke` / `workflow-orchestrator` / 性能 Skill，并按 `ops-direct-invoke-architect`（只读诊断）→ `ops-direct-invoke-developer`（限 `ops/<op>/`）→ `ops-direct-invoke-verifier`（独立只读复核）串行、同步等待；缺少 `verification=pass` 或改到 objective 的 `allowed_paths` 外，后端按改前快照自动回滚。当前 Loop 已验证过的 A/B 与上一轮结论会注入下一次自动优化，防止重复失败方向或把“恢复基线”冒充新优化。第一次实跑抓到后台 subagent 提前结束（job #3），已补“不得提前完成”硬门禁与回归测试。

**候选门槛再修正**：Loop #4 的首次 job #6 暴露 architect 把“有证据的可证伪假设”误解为“必须先有该候选自身 A/B”，导致实验永远无法开始。提示契约现明确：只要收益机制有本机证据/源码定量支持、安全边界可证、单变量可隔离且可被后续 test+benchmark 证伪，就应实施一个最小待验证候选，并把“收益未证实”写入风险，不得把它当作拒绝理由。回归测试已锁住该文案。

**目标门禁修正**：`perf_diff` 总结论优先用 `loops.objective.target_cases`；多目标保守聚合（任一目标回退即 `regressed`）。objective 声明 `constraints.perf_non_regression=true` 时，目标用例显著回退记 `E_TARGET_PERF_REGRESSION` / `verdict=fail`；旧非 objective Loop 仍保持性能回退只 warn 的兼容口径。此前 `bench_1m` 为报告 primary 会遮住 `bench_64k` 回退，现已用单测锁住。

**任务 #1 / Loop #2 同条件真机 A/B（910B1，设备 5，镜像 digest `56c16f…`，suite `cases@840be5ba462c7592`）**：

- before：test run **#8**（19/19 PASS）+ benchmark run **#9**，代码 `4e880b…`；`bench_1m=8.180us`，`bench_64k=5.240us`，64K `block_dim=43`。
- 自动优化：agent job **#4**（DeepSeek Flash）完整留下 architect / developer / verifier 三段证据；候选把 64K 从 `512×3/核、43 核` 改为 `1408×1/核、47 核`，静态分片复核 67,404 条断言 0 失败。
- after：test run **#10**（编译成功，19/19 PASS）+ benchmark run **#11**，代码 `cc3cbcea…`；`bench_1m=8.160us`（−0.244%，无显著变化），目标 `bench_64k=8.300us`（**+58.397%**，不确定性 ±7.617%，显著回退），`block_dim=47`。
- 结论：`perf_diff verdict=fail / conclusion=regressed`；Loop #2 已人工否决，candidate 存档 #11 标记 `rejected`，报告落在 `runs/r1-20261022-113650/optimization-report.json`；随后恢复 baseline 存档 #8，当前工作区源码 SHA 回到 `4e880b…`。该结果证伪“把 64K 从 43 核增到 47 核会提速”，后续不得重复此候选。

**Loop #3（48 核非均匀分配，只读诊断）**：基线 test run **#12**（19/19 PASS）+ benchmark run **#13**（`bench_1m=8.160us`、`bench_64k=5.220us`）。CANNBot job **#5** 复算后正确拒绝改码：128 个 512 tile 在 48 核上虽可分为 `32×3 + 16×2`，但最长核仍为 3 tile / 1536 元素，关键路径不变。随后同设备5/同镜像/固定 512 tile 的只读核数曲线（`runs/diag-corecurve-20261022-1220/out/core-curve.json`）：43/32/26/16 核中位数分别为 **5.600/7.220/6.580/7.020us**，证明缩核方向不成立。Loop #3 已否决，源码从未改动。

**Loop #4（同 43 核的单次搬运）同条件真机 A/B**：

- before：test run **#14**（19/19 PASS）+ benchmark run **#15**，代码 `4e880b…`；`bench_1m=8.160us`，目标 `bench_64k=5.680us`，均为 43 核。
- 自动优化：首次 job #6 因上述过度保守门槛不改码；修正契约后 job **#7** 完整跑通 architect → developer → verifier，`verification=pass`。候选只改 `SMALL_TILE_LENGTH 512→1536` 及注释；64K 保持 43 核/每核最大 1536 元素/尾核 1024，唯一变量是每核循环 3→1。
- after：test run **#16**（19/19 PASS）+ benchmark run **#17**，代码 `03a1daa…`；`bench_1m=7.900us`（路径未改，不归因于候选），目标 `bench_64k=5.760us`（**+1.408%**，不确定性 ±15.495%，`no_significant_change`），均为 43 核。
- 结论：目标用例无显著提升，不采纳；Loop #4 已否决，candidate 存档 #17 标记 `rejected`，报告落在 `runs/r1-20261022-124720/optimization-report.json`；已恢复 baseline 存档 #15，当前工作区 SHA 回到 `4e880b…`。结果表明在当前噪声下，减少 512 tile 循环没有可确认的 64K 收益。

**任务 #2 / Loop #5（`elementwise_sub` 小 shape 自适应 tile Demo）同条件真机 A/B**：

- 显式演示基线：小/大 shape 暂均用 4096 tile；baseline 存档 **#21**，test run **#18**（13/13 PASS）+ benchmark run **#19**，代码 `dddaeb67…`；`bench_64k=7.440us`（16 核），`bench_1m=8.220us`（43 核）。原始 512 自适应版另有手动存档 **#19**，演示失败可独立恢复。
- 自动优化：agent job **#8**（DeepSeek Flash）完整执行 architect → developer → verifier，`verification=pass`；唯一语义变量是 `SMALL_TILE_LENGTH 4096→512`，并同步修正注释；64K 的理论参与核数 16→43，1M 仍走 4096 tile 路径。
- 候选：test run **#20**（13/13 PASS）+ benchmark run **#21**，代码 `ce867357…`；目标 `bench_64k=5.580us`，相对基线 **−25.0% / improved**；`bench_1m=8.140us`（−0.973%，不确定性 ±2.062%，`no_significant_change`）；正确性无退化。规格、判据、镜像 digest、设备 5、benchmark 用例全部同条件。
- 结论：`perf_diff verdict=pass / conclusion=improved`；用户已在页面人工采纳，Loop #5 `status=closed / conclusion=accepted`，版本 **c1** / accept 存点 **#23** 与当前工作区同步。可复现报告已落在 `runs/r2-20261022-143607/optimization-report.json`。**自动采纳仍关闭**：本次为真实用户审批。

**验收**：后端全量 `pytest` **315 passed**；前端 `npx tsc --noEmit`、`npm run build` 通过；8077 / 5173 已启动供页面操作。自动采纳仍未开启，否决与回滚均保留 run / Loop / savepoint 证据。

## 2026-10-22：优化点搜索 + 测量预算与收尾（功能阶段二/三）

**优化点搜索 MVP**：迁移 `c2d3e4f5a6b7` 新增 `optimization_ideas` / `idea_searches`；页面支持只读诊断搜索、用户补充思路入池、服务端去重、已有验证/否决结论回喂和“一思路一 Loop”串行验证。诊断 job 明确禁止改文件及调用 build/test/benchmark/profile；采纳/否决会把结论回写思路池，自动采纳仍永不开启。

**测量预算硬门禁**：objective Loop 的 `runs_per_action` 已从“只记录”升级为后端强制；页面动作与 `nm.sh --loop <id>` 直调都在产生远端 run 前校验，`OPX_LOOP_ID` 随执行环境透传，允许/拒绝均写 `loop_budget_audits`。旧非 objective Loop 保持兼容并增加只读入口。新增自动优化确认对话框，启动前明确展示 Loop、目标用例、允许改动路径、每动作预算、参考 Loop 和“只产出候选、必须人工采纳”。

**验收与数据边界**：迁移真实库前备份 `backend/opx.db.backup-20261022-p2p3`，当前 schema head 为 `c2d3e4f5a6b7`。全量 `pytest` **320 passed**，`tsc --noEmit` / `npm run build` / shell 语法检查通过。浏览器 e2e 使用真实库的临时副本（未污染任务 #2）：点通优化点入池与去重、预算超限 409、旧 Loop 展开和自动优化确认框；真实任务 #2 / Loop #5 仍为已采纳 c1。

**后续可选项**：跨芯片 port、归档 GC、报告 Markdown 渲染仍未做；它们不属于本次阶段二/三的核心闭环。

## 2026-10-21 收工：版本存档 + 优化轮次 v1（阶段一）（提交 `bc49126`）

**状态**：已提交 `bc49126`（同时纳入此前工作区未提交的页面/端点/资源/对话/案例库改动）；`pytest` **309 passed**、`tsc --noEmit` / `npm run build` 通过；真实库迁移 head `b1c2d3e4f5a6`（迁移前备份 `backend/opx.db.bak-20261021-opt-rounds`）；8077 后端 / 5173 前端在跑。

**版本存档 savepoints v1（迁移 `a9b8c7d6e5f4`）**：源码按 `code_snapshot_sha256` 内容寻址归档（`archives/sources/<sha>.tar.gz`，同一版一份；解包后重算哈希校验）；`savepoints` 表（manual / auto / rescue / baseline / candidate / accept / rejected）；任务详情第 8 分区「版本」：手动存档、对比当前（任务/共享范围 + unified diff）、一键回滚（task/full，回滚前自动急救存档，运行中作业 409）、下载、删除。四动作执行 / 对话发消息 / 自动优化起 agent 前自动归档（best-effort，同 sha 去重，自动行保留 50 条）。设计/契约：`docs/plan-savepoints-v1.md`。

**优化轮次 v1（阶段一；口头简称 P1，只指本功能分期，不是历史里程碑 P1 执行链；迁移 `b1c2d3e4f5a6`）**：任务详情「性能与优化」新增「优化轮次」——新建轮次（目标类型 性能 / 精度（不达标→达标）/ 硬件适配；目标维度 用例 / 全部；**目标用例数据驱动**：从最近 ok benchmark/test 报告动态读；约束 / 预算 / **用户补充思路**）→ 建 Loop（`loops.objective_json`）+ baseline 存档 + 自动复用同 sha 未挂 run → agent 按 objective 改代码（prompt 注入目标/约束/用户文本；硬件画像按 `target_chip` 选）→ 「跑基线 / 跑候选」一键 test+benchmark（自动挂 `loop_id`、设备 pin、顺序执行、进度显示已运行时长、刷新可恢复）→ benchmark 成功自动落 candidate 存档 → **人工采纳**为版本 `cN`（candidate 原地升级）或否决 → `optimization-report.json`（objective / 目标快照 / 代码 / before-after run / perf-diff / 逐例精度 / repro 命令，`write=true` 落优化后 benchmark 目录）。设计/契约：`docs/plan-optimize-rounds-v1.md`。

**前端合并**：按 2026-10-21 实测反馈，`PerfDiffPanel.tsx` **已删除**；其「让 agent 改代码 + 优化前后对比」并入 `OptimizationRoundPanel.tsx` 的轮次卡片。旧面板的「开 Loop / 关联 run / 按结论关闭 / 落盘」由轮次流程替代；非 objective 的旧 Loop 仅保留数量提示（无操作面板）。

**本日修复（重要）**：`scripts/nm_test.sh` 的快照实现少跳过 `.opx-chat`，与后端 `snapshot()` / 存档 archiver 不一致 → 同一次执行「baseline 存档 sha（`7d965a…`）≠ run 快照 sha（`d8997f…`）」。已补 skip + `backend/tests/test_snapshot.py` 回归（新增 `.opx-chat` 夹具），并把轮次 #6 的旧 run **#34(test) / #35(benchmark) 摘挂**（`runs.loop_id=NULL`，run 未删，仍在任务台账）。新跑的 test/benchmark 快照会与 baseline 存档一致。

**当前实机状态（Lead 本机，仅本机；新拉代码不会自动拥有，见 `docs/NEXT-2026-10-22.md` §1/§2）**：任务 #11 / Loop #6（perf，目标 `bench_1m`，baseline 存档 `7d965a…`）当前 **0 条 run**。需要先在页面刷新后「跑基线（test + benchmark）」，确认两条 run 快照都是 `7d965a…`；再让 agent 改代码 → 跑候选验证（自动 candidate）→ 采纳/否决 → 导出报告。**不要删除 Loop #6 或历史 run**（目前 Loop 无删除接口；run #34/#35 是历史证据）。

**已知限制 / 交接注意（本节为 2026-10-21 快照；阶段二/三完成状态见上一节）**：
- ~~P2 优化点搜索 / P3 测量预算硬兜底、旧 Loop 入口、浏览器 e2e 未做~~ → 2026-10-22 已完成；跨芯片 port、报告 Markdown 仍为可选后续。自动采纳按设计**永不开启**。
- 归档文件不随存点删除/任务删除清理（内容寻址、可能被引用），GC 留 v2；回滚是逐文件写入（非事务），急救存档保证可再滚回。
- ~~非 objective 的旧 perf Loop（如 Loop #5）在页面上无操作面板~~ → 已补只读入口。
- ~~前端尚未做真实浏览器点击 e2e~~ → 已用临时 DB 副本完成，不改真实运行证据。


## 任务外页面审查修复实测记录（2026-10-21）

**范围**：`runs/review/SUMMARY-other-pages.md` 的 5 页 56 条（服务器与镜像 / API 管理 / 消耗情况 / 对话 / 案例库；任务相关页上一轮已审）。Lead + 4 路并行修复 + 独立复核；源码改动均未提交（工作树）。

### 修掉的高危

| 问题 | 修法 | 证据 |
|---|---|---|
| **H1** 导入脚本 `manifest.name` 可穿越到库外；`--force` 会 `rmtree` 任意目录（`name="."` 删整个库） | `_safe_bundle_name`（单段/禁 `..`/`.`/`\`/控制字符/≤120）+ target 必须在 dest 内才允许 rmtree；zip 条目/单文件/总解压/压缩比四项上限；index.json 原子写 | `test_case_import.py` + `test_chat_library_fixes.py`（穿越/`--force`/zip 限制） |
| **H2** 端点物理删除后主键复用 → 历史 `agent_jobs.endpoint_id` 指向新端点、费用静默重算 | 端点改**软删**（迁移 `d2e3f4a5b6c7` 加 `deleted_at`）：保留行/id/价目供历史计价；列表/当前端点/重名检查过滤；删除时清 key、释放唯一名、active 顺延；`pricing.endpoints_by_id` 故意含软删行 | 临时库：删端点→新建 id 不复用、历史 ¥1000 不变（`test_endpoints.py`） |
| **H3** `Infinity`/超大单价入库 → 金额溢出 inf → `/api/usage` 返回 `cost:null` → 消耗页 `null.toFixed` 白屏 | `_check_prices` 用 `math.isfinite` + 上限 1e6；`cost_of` 非有限返回 None（标未定价）；前端 `money()` 对 null/非有限兜底；`excluded_demo_rows` 改真实计数 | 端点 `1e999`/`1e308` → 400；溢出历史数据 `cost_known=false`、cost 不为 null（`test_usage.py`/`test_pricing.py`） |
| **H4** 对话「回退本轮」不校验文件现状，回退旧轮覆盖后续轮改动 | `changes_json` 每文件存 `after_sha256`；回退先全量校验（旧记录无 hash 跳过），任一冲突 → 409 且**一个文件都不改** | `test_chat_library_fixes.py` 两轮同文件回退冲突 → 409、内容保持 v2 |

### 主要中危 / 置信修复

- **SSH 目标选项注入（安全）**：`services/ssh.validate_ssh_target`（拒绝 `-` 开头/空白/控制字符/超长）；hosts create/update 400；`ssh_exec`、`devices.check_devices` 兜底；`sync_image_meta.py` 改走 `ssh_target/ssh_port/ssh_key` 同源解析。附带修复 `devices.check_devices` 缺失 `port` 参数的潜在 TypeError。
- **对话页**：发送加进程内互斥锁 + running 范围扩到 `对话/开发/优化`；`plain_stream=True`（代码围栏/反引号开头不再整段不上屏）；`_finish` 中断原因可见、tokens 计 cache；删除会话清附件行/文件/目录并拦 running job（409）；附件分块读（>20MB → 413）+ 每会话 20 个上限 + `DELETE …/attachments/{aid}`；light 模式附件相对路径修正；`_view` 暴露 `running_job`。前端 `ChatPage` 移植 `/live` + 退避重连 + 重放去重 + `error` 事件 + 停止按钮 + 刷新/切页自动接回 + 发送失败回滚 + 请求序号 + 附件删除 + 回退文案。
- **案例库**：一键发起改「全量冲突预检 → 再写 → 失败回滚文件/目录/Task」；`op` 单段校验；`target_chip` 从 bundle 透传到 Task；案例报告只接受 dict、非 list/dict 兜底、`_safe_child` 防越界读；`CasePanel` 补 index 错误卡/空态/刷新/目标芯片列/「发起后需定版」提示。
- **API / 消耗**：key→env 口径统一（启动基线在 lifespan 记录；清空/activate 无 key 端点回退基线而不是 `pop`；改**非当前**端点 key 不再误清当前 key）；端点字段校验（provider 白名单、base_url scheme/query、max_model_len>0）；探针错误体对 key 脱敏；删除当前端点显式顺延 active；`UsagePanel` 金额走 `money()`、按天含 cache、补 cache_write/金额列；`ApiPanel` 半配置单价可见、删除/清空二次确认。
- **服务器与镜像**：探针失败清 `cards/cann_version`、写 `probe_error`、旧镜像标 `stale`（换目标不再串 A 机信息）；`docker_ok` 按命令成功而非镜像数；镜像按 `image_id`（悬空镜像）优先对账，远端消失且无标记/未被引用则删、有标记/被引用标 `missing`；`PATCH /api/host-images` 支持 digest/CANN/Python/torch 元数据（截断）、`is_op_dev=true` 必须有 recipe、悬空镜像禁止标记；前端镜像请求竞态守卫 + 加载错误与空态分离 + 元数据行内编辑 + 执行机口径改 `projectHost` + unknown/null 显示语义 + address-only 可编辑。
- **通用**：`op_name` 增加单段校验（TaskCreate/TaskUpdate/SpecVersionCreate，挡 `../..`/绝对路径）；`case_report` 不读工作区外；`App` 侧栏「执行机」、`docs` 口径同步。

### 验证

- 后端全量（临时库）：`pytest -q` → **274 passed**（新增 68：hosts 探针/SSH + 对话/案例/回退/导入安全 + 端点软删/单价/usage）。
- 前端：`npx tsc --noEmit` 0 错误；`npm run build` 通过（`web/dist` 已更新）。
- 迁移：真实 `backend/opx.db` 先备份（`opx.db.bak-20261021-endpoint-softdelete`）再 `alembic upgrade head` 到 `d2e3f4a5b6c7`；DB 权限收紧为 `0600`。
- 新代码以真实库副本在 8079 冒烟：`/api/health`、`/api/conversations`（含 `running_job`）、`/api/hosts`（含 `probe_error`）、`/api/usage`（demo 计数）均 200。
- 复核报告：`runs/review/verify-fixes.md`（独立复核员）。

### 遗留（如实记录）

1. ~~页面 key 明文存在 SQLite~~ 已修（2026-10-21 晚）：key 迁到本机密钥库（macOS Keychain → Linux Secret Service → 0600 文件），DB 只存 `secret_ref`（迁移 `e3f4a5b6c7d8`）；启动时 `migrate_legacy_secrets()` 把旧 `api_key` 搬走并清空。
2. ~~模型路由仍是 dsh patch 写死~~ 已修（2026-10-21 晚）：按「API 管理」当前端点动态生成 dsh patch，key 与路由同源（见下「端点驱动 dsh 模型路由」）；静态 patch 仅作无端点兜底。
3. 对话发送锁是**单进程** `threading.Lock`；多进程部署需 DB 唯一约束。
4. 会话删除对「DB 标 running 但进程已崩溃」的僵尸 job 会 409（未做 task_deletion 的 stale 兜底）。
5. 旧 `changes_json`（无 `after_sha256`）回退不做冲突校验（兼容口径）。
6. 探针「部分成功」语义：docker 命令失败时保留旧镜像标 `stale`，不做删除对账（宁可不清错）。
7. 低危未做：消耗页按端点/时间维度、TaskRecords 逐条金额精度、`is_active=false` 显式取消、案例筛选、M2 案例内容 0/10（原待办）。

## 端点 key 迁移到本机密钥库（2026-10-21 晚）

**触发**：复查发现页面配置的模型 API key 明文存 `endpoints.api_key`（DB 虽已 0600，但备份/拷库会带出 key）。按同类 agent 的常见做法（Claude Code macOS Keychain / Linux 0600 credentials；Cline 用 IDE SecretStorage；Electron 用 safeStorage）改成「key 本体放本机密钥库，DB 只存引用」。

| 项 | 落点 |
|---|---|
| 密钥存储抽象 | `backend/app/services/secret_store.py`：auto = macOS Keychain（`security`，key 走 stdin 不进 argv）→ Linux Secret Service（`secret-tool`）→ **0600 文件回退**；`index.json`（0600）记录 ref→backend，避免「写时回退、读时找不到」；文件目录 0700/文件 0600、原子替换 |
| DB | `endpoints.secret_ref`（迁移 `e3f4a5b6c7d8`）；旧 `api_key` 列保留兼容但新数据不再写 |
| 启动迁移 | `llm_config.migrate_legacy_secrets()`：把旧明文列搬到密钥库并清空；失败只告警、保留旧列兜底（不阻塞启动） |
| 运行链路 | `llm_config.endpoint_key()` 优先读密钥库（旧列兜底）→ 同步进程 `OPX_LLM_KEY` 给 dsh；探针/创建/更新/清空/删除端点全部改走密钥库；删除端点同时删密钥 |
| 页面 | `GET /api/endpoints` 增加 `key_backend`（keychain/secret-service/file/legacy），API 管理页显示「已配置 · Keychain/本机文件」，仍不回显明文 |
| 配置 | `OPX_SECRET_BACKEND=auto|keychain|secret-service|file`、`OPX_SECRET_DIR`、`OPX_KEYCHAIN_SERVICE`；headless/SSH 建议显式 `file` |

**验证**：临时库 `pytest -q` → **269 passed**（+2：key 不落库/删除清密钥、旧明文列启动迁移；conftest 强制 file 后端 + 临时 secret 目录）；`tsc --noEmit` / `npm run build` 通过；真实库副本先 `alembic upgrade head` 到 `e3f4a5b6c7d8`，再以 file 后端启动验证：旧 35 字符 key 自动搬到 `values/endpoint-1.key`（0600）、DB `api_key` 清空、`secret_ref=endpoint-1`、`GET /api/endpoints` 返回 `has_key=true / key_backend=file`。
**真实库**：已备份（`opx.db.bak-20261021-secret-store`）并升级迁移；**旧 key 的搬仓发生在下一次后端重启时**（lifespan 迁移），重启前旧列仍在、功能不受影响。

**边界/遗留**：未在 SSH/无 GUI 环境实测 Keychain（自动回退到 0600 文件）；key 仍在进程环境/内存里有明文（dsh 需要）；多机部署时密钥库不跟随，需要重新配置或在目标机用 file 后端；Linux Secret Service 需 D-Bus 会话。


## 端点驱动 dsh 模型路由（2026-10-21 晚）

**触发**：重启后端后，「任务开始」页第一轮对话报「本轮没有得到回答（dsh 会话可能失败）」。查该轮 dsh 会话：`turn_end error: 401 {"message":"Upstream request failed: Invalid credential"}`。

**根因**：dsh 的模型路由写死在静态 patch（`ops/dsh/llm-opencode-go.patch.yml` → opencode go 网关），而「API 管理」当前端点是 **DeepSeek 官方**（`https://api.deepseek.com`）。启动逻辑按「页面 key 优先」把 DeepSeek key 同步进 `OPX_LLM_KEY`（dsh 的 `apiKeyEnv`）→ dsh 拿 DeepSeek key 请求 opencode 网关，必然 401。旧进程只是**启动得早**（页面 key 是后配的）才没暴露；重启使问题显形。

**修法**：模型路由改为**按当前端点动态生成 dsh patch**，key 与路由同源 ——

| 项 | 落点 |
|---|---|
| 生成 | `llm_config.build_dsh_patch()` / `render_dsh_patch()`：`llm-pi-ai` 的 provider 取端点 `provider/base_url/model`（`api: openai-completions`；opencode 网关补 `x-opencode-session`），`agent-default-model` 同步；`contextWindow` 取 `max_model_len`（异常值回落 131072） |
| 落盘 | `settings.dsh_generated_patch`（默认 `backend/.runtime/dsh-llm.patch.yml`，gitignored）；原子写；端点缺 `base_url/model` 或没有端点 → 删除生成文件、回退静态 patch |
| 使用 | `agent_runner.build_command()` 与 `/api/proto/dsh` 都走 `llm_config.dsh_patch_for_run()`（有生成用生成，否则静态兜底） |
| 同步时机 | `apply_active_key()` 内完成（启动 / 建端点 / 改 key / 切换 active / 删除端点都会走到），保证 key 与路由一致 |

**验证**（真机 + 真网关）：
- dsh 直跑 DeepSeek 官方：`dsh --profile headless --patch <生成的 patch> --json "只回两个字：收到"` → exit 0、`{"type":"text","text":"收到"}`。
- 重启 8077 后：`GET /api/endpoints` → `key_backend="keychain"`；`POST /api/endpoints/1/probe` → 200，列出 `deepseek-flash` / `deepseek-v4-pro`。
- 真实对话（临时草稿任务，跑完即删）：`POST /api/tasks/{id}/chat` → 200，`reply="收到"`，usage `inputTokens=7781 / outputTokens=110`。
- 生成 patch 内容：`baseURL=https://api.deepseek.com`、`models[0].id=deepseek-flash`、`agent-default-model=deepseek/deepseek-flash`。
- 全量 `pytest` → **276 passed**（新增：动态 patch 字段与切端点、缺字段回退；测试改用临时 `OPX_DB_URL`，lifespan 不再碰真实 `opx.db`）。

**边界**：`~/.config/op-agent/env` 里的 opencode key 是旧值（opencode CLI 已轮换，新 key 在其 `auth.json`），现在只在「没有页面端点 / 端点没 key」时兜底；主链路由页面端点 key 驱动。旧静态 patch 仍指向 opencode 网关，未配置匹配端点的部署会 401（属配置问题，页面可显式切换端点）。

## 对话并入任务详情；agent run 入口移除（2026-10-21 晚）

**触发**：用户用「发起 agent run」做 elementwise_mul 开发，CANNBot PM 按协议给出「启动前四项确认」后本轮结束；页面上没有地方回答。先加过「续聊」（续同一 dsh 会话），但用户可能同时有多条会话，**自动接回的那条不一定是想续的**，入口仍然歧义；而且 agent run 与对话本来就是同一件事（都是 dsh 会话 + workspace），只是没落会话/消息。

**最终判断**：**开发只保留一个入口 —— 任务「对话」**。对话页并进任务详情（默认只列本任务会话、新建自动绑任务），**移除侧栏独立「对话」页，并直接移除 agent run（前端面板 + `POST /api/tasks/{id}/agent-runs` + `AgentRunCreate`）**；开发、多轮确认、追问、附件、逐轮回退全部走任务会话。概览页只保留「四动作真机执行 + 门禁」。

| 项 | 落点 |
|---|---|
| 前端（任务页） | `TaskWorkbench` 新增「对话」分区（概览之后）；`ChatPage` 改为任务内嵌：props `{taskId, taskTitle}`，只列 `task_id` 匹配的会话、新建默认绑定本任务；面板保持挂载，切走不掐断 SSE |
| 前端（导航 / 概览） | `App` 移除侧栏「对话」入口与 `view=chat` 分支；概览移除 `AgentRunPanel`（组件文件删除），脚注指向「对话」分区 |
| 后端 | 删除 `POST /api/tasks/{id}/agent-runs` 与 `AgentRunCreate`；`agent_runner` / `/api/agent-jobs/*` 保留（对话与自动优化仍用），`session_id` 参数保留给对话续会话 |
| 数据 | 无表结构改动；历史 `agent_jobs.kind=开发` 行保留在台账里；新开发调用 = `conversations`（`kind=对话`） |

**验证**：`pytest` **276 passed**（删除 agent-runs 用例后全量仍全绿）、`tsc --noEmit` 0 错误、`npm run build` 通过（dist 已更新）。真人验证待用户：任务 #11 →「对话」→ 新建会话（挂 skills）→ 对齐 PM 四项确认 → 开发。

**设计边界**：会话仍绑定单任务（`conversations.task_id` NOT NULL），不做跨任务通用对话；轻对话（空目录、不挂 skills）作为同一分区的可选模式保留。

**修复（同日）**：agent 答复是 Markdown 但按纯文本显示 —— 前端新增 `Markdown` 组件（`react-markdown` + `remark-gfm`：标题 / 列表 / 表格 / 代码块 / 引用 / 链接；**不开 raw HTML** 防注入，链接新窗口），对话页 assistant 消息与「任务开始」页 agent 答复都改用它；用户消息仍纯文本；流式过程中仍是纯文本实时上屏，结束后落库并按 Markdown 渲染。依赖已入 `web/package.json` / lock。

## 执行五步 + AI 审查入口（2026-10-21 晚）

**触发**：用户提出「真实执行四步窗口前面加个开发，变成 5 步；四步跑完也加 AI 审查，并可带信息跳到对话」。

**落点**：

| 项 | 落点 |
|---|---|
| 五步 | 概览「执行五步（开发 + 四动作）」：① 开发（→「对话」，预填开发 kickoff：按 `tasks/<op>/TASK.md` + `spec/spec.json` 实现）② 编译 ③ 正确性 ④ 基准 ⑤ 采集；步骤按钮显示该动作最近 run #；四动作仍是「先算 plan → 真执行」 |
| AI 审查 | 每个动作只要已有 run，步骤条下方出现「审查 <动作>」：把 run/状态/exit_code/门禁 verdict/fail_codes/warn_codes/报告接口（`/api/runs/{id}/report?type=…`）/run 目录写进对话输入框（**预填不自动发**）；当前作业跑完另有「AI 审查本次」 |
| 对话带入 | `ChatPage` 新增 `seed`：跳转时确保有本任务会话（无则新建、有则用当前/最近）→ 打开 → 预填并聚焦输入框，提示「可编辑后发送」 |
| 文案 | 概览 tab 提示 / 脚注改为五步 + AI 审查；`ActionPlanPanel` 标题改「执行五步（开发 + 四动作）」 |

**验证**：`tsc --noEmit` 0 错误、`npm run build` 通过（dist 已更新）；**无后端改动**（AI 审查复用现有 run 报告接口与会话 API，不需要重启）。

**边界**：AI 审查只负责「带上下文」——真正跑四动作仍是用户点「真执行」；审查结论/改动落在对话里（可继续多轮）。

**修复（同日）**：从概览「开发 / AI 审查」跳进对话时，左侧会话列表一开始为空、重进页面才显示 —— `seed` 带入只 `open()` 了新会话，没有重新拉列表；改成先 `loadConvs()` 再 `open()`。`tsc` / `build` 通过。

## 工作区上机入口：任务感知（2026-10-21 晚）

**背景**：对话 agent 原先无法上机 —— `nm_test.sh`/`ops/nm.conf` 在控制台仓库根，不在任务工作区；补了工作区
`scripts/nm.sh` 后，又发现它只读 `ops/nm.conf`（默认 tk36），**不跟当前任务配置的执行机/镜像走**，会出现
「页面配的是 A、agent 跑的是 B」。

**修法**：`scripts/nm.sh` 改为**任务感知**——
1. 任务 id：`--task` > 环境变量 `OPX_TASK_ID` > 按 `--op` 找最近的未定版任务；
2. `POST /api/tasks/{id}/actions`（dry_run）取**当前任务的 plan**（执行机 SSH 目标/端口/私钥、镜像、设备、规格、判据）；
3. 把 plan 注入与控制台四动作同一套 `OPX_*` 环境变量，再 `exec nm_test.sh --task <id>`（跑完照常回填 run 台账）；
4. 本地产物落 `<workspace>/runs/`（`OPX_LOCAL_RUNS`，因为 agent 的 bash 工具在 workspace-write 沙箱里，仓库根 runs/ 不可写）；
5. 顺带把 `.uvcache/.workflow-venv/.opx-chat` 加进 rsync 排除，避免一次 sync 拖几百 MB 私有缓存。

**边界**：容器运行参数（`OPX_DOCKER_ARGS` 驱动挂载/`--privileged`、`OPX_ENTRYPOINT`、`OPX_ENV_RECIPE`）仍是
`ops/nm.conf` 的本机全局配置，不随任务走；当前只有一台执行机（tk36）没问题，接入第二台/不同镜像 recipe 时需要
把这些字段挪进 host/image 配置（后续项）。

**验证**：控制台里新建临时会话让 agent 跑 `scripts/nm.sh --dry-run --op elementwise_mul --action build` → 输出
`[nm.sh] 任务 #11 · 动作 build · 执行机 tk36 · 镜像 lfx-op-dev-910b:torch210-cann910 · 设备 1`，exit 0，
rsync 到 tk36 成功（`~/.ssh/known_hosts` 的写保护只是非致命警告）；临时会话已删除。四动作路径未改动（默认 `runs/`）。

## 默认「不静默」+ 上机入口写进默认提示（2026-10-21 晚）

**背景**：CANNBot 模板要求「启动后尽量静默执行」「你可以离开了，即将开始静默开发」，用户明确不要静默；
且工作区 AGENTS.md 的追加段会在 `scripts/install_skills.sh` 重装 skills 时被模板覆盖。

**做法（两层默认，不依赖单次手工改文件）**：

| 层 | 落点 | 说明 |
|---|---|---|
| 每轮对话 prompt | `services/conversations.build_prompt()` | 任务模式下固定追加两条：①「不要静默」——workflow/节点开始、完成、失败、等决策都简短汇报，长轮询也报一行；② 真机验证入口 `scripts/nm.sh`（先 `--dry-run`，按当前任务执行机/镜像跑并回填台账，勿与四动作同时触发） |
| 工作区 AGENTS.md | `scripts/install_skills.sh` 后置步骤（③.5） | 每次装完 skills 后，幂等地把「真机验证入口 + 用户偏好（不要静默）」追加到 `$TARGET/AGENTS.md`；已存在则跳过 —— 重装不会再被上游模板覆盖 |

**验证**：重启 8077 后新建临时会话实测 `agent_jobs.prompt` 含「不要静默」与「scripts/nm.sh」；临时会话已删（job #39 cancelled）；
`pytest` **277 passed**（在 `test_task_mode_uses_workspace_and_skills_prompt` 增加两条断言）。

**边界**：上游 cannbot 模板正文仍写「静默」；覆盖靠控制台 prompt + AGENTS 追加段。若直接绕过 `scripts/install_skills.sh`
用 cannbot CLI 装 skills，则没有 AGENTS 追加段（prompt 默认仍生效）。

## 任务外页面审查收口（2026-10-21 晚）

**结论**：本轮「任务之外 5 个页面」审查**关闭**。56 条中高危 4/4 已修；主要中危已修；剩余按处置分为「接受为已知限制」与「转后续里程碑」，不再作为本轮阻塞项。

### 收口批次（最后 6 条 + 前端提示）

| 项 | 处置 | 落点 |
|---|---|---|
| 探针无 per-host 并发锁（hosts #7） | ✅ 修 | `routers/hosts.py`：per-host `threading.Lock`，并发探针 409；单进程足够 |
| 探针部分成功原因不落库（hosts #9） | ✅ 修 | 成功但有 `res.errors` 时写 `[探针告警]` 到 `npu_state`，`_host_out.probe_error` 回读；页面按 fail/告警分别展示 |
| 会话僵尸 running 阻止删除（chat #5 遗留） | ✅ 修 | `conversations.delete`：DB running 但 REGISTRY 无、且超 `agent_timeout` → 视为僵尸，标 `interrupted` 后允许删除 |
| `is_active=false` 被静默忽略（api #14） | ✅ 修 | PATCH 显式 false → 400「请用另一个端点设为当前」，避免页面/运行态不一致 |
| 端点名大小写重名（api #15） | ✅ 修 | create/update 用 `lower(name)` 查重 |
| 私钥上传弱校验/全量读内存（hosts #11） | ✅ 修 | 8KB 分块读 + 64KB 上限；仅接受 PEM/OpenSSH 私钥头（拒 `PRIVATE KEY-----` 子串文本） |
| 模型路由提示缺失（api #8） | ✅ 提示 → 已接线（2026-10-21 晚） | 先加提示；随后按当前端点动态生成 dsh patch，页面 provider/base_url/model 真正决定 agent 调用（见「端点驱动 dsh 模型路由」） |

### 接受为已知限制（不修，记录理由）

- ~~**模型路由写死 patch**~~ ✅ 已修（2026-10-21 晚）：`llm_config` 按当前端点生成 `backend/.runtime/dsh-llm.patch.yml`，`agent_runner`/原型链路统一使用；旧静态 patch 只做无端点兜底。
- **对话发送单进程锁**：当前单进程部署充分；多进程再上 DB 唯一约束。
- **旧 `changes_json` 回退无冲突校验**：仅影响历史数据，新数据都有 `after_sha256`。
- **探针重定向/SSRF、删除服务器 ActionJob 引用/解绑、私钥路径回显、消耗页端点/时间维度、TaskRecords 逐条金额精度、前端自动化测试、真实代理 SSE 压测**：低危/体验/测试建设项，转后续批次。
- **Keychain 在 headless/SSH 未实测**：自动回退 0600 文件，已记录；多机不跟随密钥库。
- **安全后续（用户侧）**：迁移前的 DB 备份仍含旧明文 key，确认后删除或轮换；`~/.config/op-agent/env` 的 `OPX_LLM_KEY` 按需轮换。

### 转后续里程碑

- **M2 案例库内容 0/10**（最大验收缺口，工作量 15–25 天）；一键发起落规格/上下文空间；案例筛选。
- 多项目假设、`agent_jobs.error` 列/D8 预算等上一轮遗留。

### 验证

- 后端全量（临时库）：`pytest -q` → **274 passed**（含收口批次新增 5 条回归）。
- 前端：`tsc --noEmit` 0 错误；`npm run build` 通过（`web/dist` 已更新）。
- 迁移 head：`d2e3f4a5b6c7`（软删）→ `e3f4a5b6c7d8`（secret_ref）；真实库已备份并升级，旧 key 在下次重启时搬入本机密钥库。
- 真实 8077 仍是旧进程，**需重启**后新代码/密钥迁移才生效。

## 接手须知（2026-10-16 会话末快照）

> 注：本节是 **2026-10-16 快照**，用于回看当时的服务/自检方式；其中的测试数、DB 状态与待办已被后续里程碑刷新 —— **最新状态以顶部「一句话现状」+ 文末「待办 / 未决与风险」为准**。

**新会话第一步**：读本文件顶部「一句话现状」+ 最近的实测记录（「P2 结果视图」/「M3-a 实测记录」/「M1-b 实测记录」）；要动门禁/报告格式就看 `docs/gate-report-v1.md` 与 `docs/perf-diff-v1.md`。

**本会话（2026-10-16）做了什么**（详见各实测记录）：**M1-b**（`benchmark`/`profile` 实装 + D9 定档）→ **M3-a**（自动优化闭环：硬件画像 + `perf` Loop + `performance-diff.json` + 门禁 + 对比视图，首个真机 A/B 结论如实、优化不予采纳并回退）→ **P2 结果视图**（`GET /api/runs/{id}/report` + `GET /api/chips` + ECharts 正确性/性能/指标 + 极限/Roofline + 结构化建议 + 导出）。`pytest` 85 → **102**；两次上机共登记 4+2 条 run（#6/#7 与 Loop #2 的 #8–#11）。

**服务与自检（本机跑，D4）**：

- 一键起：`scripts/dev.sh`（后端 8077 + 前端 5173，Ctrl-C 一起退）。**opencode go 的 key** 由 `scripts/dev.sh` 自动 `source ~/.config/op-agent/env`（600 权限、仓库外；路径可用 `OPX_ENV_FILE` 覆盖），也可临时 `export OPX_LLM_KEY=…`；**只从环境变量取，不进库/不进仓库**，不设也能开页面（只有发起 agent 调用需要）。会话末 8077 上有一个 uvicorn 在跑（直接在 `backend/` 用 `.venv/bin/python -m uvicorn app.main:app` 起的，**带/不带 `--reload` 取决于怎么起**，pid 会变，别记 pid；**改了 `gate.py` / `models.py` 这类被 import 的模块必须重启后端**，否则新代码不生效 —— 本会话踩过一次：门禁修复没重启，以为没修好）。
- **改了 `models.py` / 加了迁移之后必须重启后端**：老进程持有的 SQLAlchemy 模型与库结构不一致时，所有任务相关请求会 500（本会话就踩了：一个跑了 14 小时的老进程因此 500，已换掉）。
- 自检：`cd backend && .venv/bin/python -m pytest`（**当时 136 项**；当前 206，2026-10-21）；`bash scripts/install_skills.sh --verify`（skills 指纹复核）；`curl -s http://127.0.0.1:8077/api/health`。
- **流式（逐 token）要求控制台自己的地址可达**：dsh 里挂的 `opx-typewriter` 插件会把增量 POST 回 `OPX_DELTA_BASE_URL`（默认 `http://127.0.0.1:8077`）→ 端口不是 8077 时要 `export OPX_DELTA_BASE_URL=http://127.0.0.1:<端口>`；`OPX_CHAT_STREAM=0` 可整体关掉（退回「过程流式 + 兜底补发」）。判断插件是否生效：`GET /api/agent-jobs/{id}/live` 的 `text_deltas`。
- 真实上机一次：`OPX_DEVICES=0 OPX_TASK_ID=2 OPX_SPEC_VERSION_ID=1 OPX_CRITERIA_REF=ops-precision-standard@1 bash scripts/nm_test.sh --action test`（控制台算出的 plan.command 会带全这些参数）。M1-a 起这条能真跑完：容器内自动补编 → 19 条用例 → `correctness.json` → 回填 run → 门禁。**M1-b 起** `--action benchmark` / `--action profile` 同样能真跑完（benchmark：两侧 msopprof 设备侧耗时对比；profile：四类指标）；跑 benchmark/profile 时记得带 `OPX_IMAGE_ID=52 OPX_HOST_ID=1`（回填 run 用），设备由脚本按 `OPX_DEVICES` 取独占锁。**M3-a 起**再加 `OPX_LOOP_ID=<perf loop>` 就把 run 挂进优化 Loop（成对 A/B）；跑完用 `GET /api/loops/{id}/perf-diff` 看对比。
- 任务工作区的**交付件已入仓**（`cases/` `criteria/` `ops/` `tools/` `*.sh`）：只有 `.dsh/` `.cannbot/` `runs/` `**/build/` 在 `.gitignore` 里。本机改完 agent 代码/用例，直接 `git commit` 就能留下版本。

**运行时资产（不在 git 里，换了会话/机器要重建）**：

| 位置 | 是什么 | 怎么重建 |
|---|---|---|
| `workspaces/cann-ops/` | 任务工作区：**交付件已入仓** —— `cases/`（用例矩阵 + `benchmark.json`）、`criteria/`、`ops/<op>/`、`ops/chips/dav-2201.yaml`（硬件画像）、`tools/`（用例/基准/采集执行器）、四个入口脚本、`spike/`；运行时大件不入仓（`.dsh/`(40 skills/3 agents) + `.cannbot/dependencies` ~450M + `runs/` + `**/build/`） | `scripts/install_skills.sh`（skills 重建）；交付件 `git checkout` 即可 |
| `.skills-src/cannbot/` | CANNBot 来源仓（177M，git 忽略） | 脚本会自动从 `~/.cache/cannbot` 拷或 `git clone https://gitcode.com/cann/cannbot.git` |
| `backend/opx.db` | SQLite（迁移到 head `a1b2c3d4e5f6`）；**4 个任务 / 26 条 run**（task #2 #1–#23，task #10 #24–#26 = build/test/benchmark）、Loop #1–#4（#2/#3/#4 已关闭）、`action_jobs` 6 条（4 ok / 2 failed）、4 个对话（10 条消息）；**`runs.started_at/ended_at` 目前全为 NULL**，排序/时间以 id 与 manifest `finished_at` 为准 | `alembic upgrade head` + `scripts/seed.py --probe` |
| `runs/<run_id>/` | 每次真实执行的 manifest + 产物（本地留存）；**Loop #2 的 `performance-diff.json` 落在 `runs/r20261016-112226/`** | 重跑一次即可（debug 型 run 目录可删，如 `runs/m1b-*`/`runs/m3a-*`） |
| `ops/nm.conf` | 上机测试配置（模板 `nm.conf.example` 入仓） | `cp ops/nm.conf.example ops/nm.conf` |
| `~/.hermes/cache/scratch/opx.db.bak-20261015` | 迁移前的手工备份（临时） | 不需要就删 |
| `~/.config/op-agent/env` | opencode go 的 `OPX_LLM_KEY`（600 权限、仓库外）；`scripts/dev.sh` 自动 `source` | 重建：`mkdir -p ~/.config/op-agent && (umask 077; echo "export OPX_LLM_KEY=<key>" > ~/.config/op-agent/env) && chmod 600 ~/.config/op-agent/env` |

**本会话明确没做的 / 待接**：① ~~四动作"点一下就真执行"~~ ✅ 已接通（2026-10-20，异步作业）；② **M2 案例库**（10 个案例内容 **0/10**，最大验收缺口；交付形态已改为外置 bundle）；③ ~~P0-2 对话页~~ ✅ 已完成（2026-10-20）；④ ~~自动优化第三轮~~ ✅ 已完成（Loop #4 采纳）；⑤ 证据链备份（用户指定不做，D3 保留策略待定）；⑥ launchd 自启（D10）；⑦ 凭据轮换（opencode go key 曾出现在聊天记录）。

**本机跑用例工具（M1-a 起）**：工作区的用例执行器要 numpy，本机默认 python 没有 →
`~/venvs/opx-tools-3.12`（`uv venv --python 3.12` + numpy）专门干本机预检与自测，例如：
`~/venvs/opx-tools-3.12/bin/python workspaces/cann-ops/tools/run_cases.py --op elementwise_add --workspace workspaces/cann-ops --exe <假实现> …`
（把 `--exe` 指向一个假实现就能在不花容器 run 的前提下验「报告写法 + 判据 + 拒绝码匹配」；
**真正跑用例必须在容器里**，本机预检只是接线自测。）

**正式 run 与调试会话**：调试型容器会话走 `nm_test.sh` 跑临时命令、**不设 `OPX_API`/`OPX_TASK_ID`** → 天然不进台账、不计入 D8 预算；正式四动作才登记 run。

## 里程碑

| 里程碑 | 状态 | 证据 |
|---|---|---|
| 立项与同类调研 | ✅ | `c8254be`：`docs/REQUIREMENT.md` + `docs/prior-art.md` |
| 方案定稿 v1.0 | ✅ | `eaeb84b`：`docs/PLAN-web-console.md`（保留为溯源记录） |
| **方案合并 v1.1** | ✅ | `docs/PLAN-web-console-v1.1.md`（吸收外部建议稿 v2.0，逐字归档于 `docs/proposals/`）；取舍见其附录 A |
| 设计样机（皮肤 B 深色玻璃 + 亮） | ✅ | `docs/mockups/console-ia-v1.html`、`console-pages-v1.html` |
| 前置验证 V1–V5 | ✅ V1/V2/V3 · 🟡 V4/V5 | V1 真 JSONL 事件流；V2 SDK 免系统 Node；V3 cannbot skills 装进 `.dsh/skills`（40 skill + 3 agent）；V4 `dsh web` token→cookie 通（**WebSocket 过反代未验**）；V5 tk36 只读探针（8×910B1、98 镜像） |
| P0 骨架 | ✅ | `6b9c3e7`：FastAPI + SQLAlchemy/Alembic + SQLite；React 19 + Vite + Tailwind v4；真实 SSH 探针接 tk36 |
| **P1 执行链** | ✅ | `83d087e`（实测见下） |
| **P1.5-b 契约最小集** | ✅ | `d71771b`：迁移 `b7f2c9a4e1d3` + spec v1 + 四动作契约 + 前端分列 |
| **P1.5-a 容器闭环** | ✅ | 见下「P1.5-a 实测记录」：`npu devices 8` / `VERDICT npu: OK` / `VERDICT torch_npu: OK` |
| **P1.5-c 门禁与环境** | ✅ | `3cc37b8`（见下「P1.5-c 实测记录」）：迁移 `c9a4e2f7b6d1`、`pytest` 38 项通过、真实 run（`r20261015-1040-test`）走通「取锁 → 容器 → manifest → 登记 → 门禁」 |
| **M1-a 首个算子：Ascend C 直调实现 + 用例矩阵 + `build`/`test` 真机跑通** | ✅ | commit `fee3900`（见下「M1-a 实测记录」）：19 条用例全过、第一份 `correctness.json` 落盘、run #5 门禁 **pass** |
| **需求调整归档（2026-10-15）** | ✅ 已落（方案落版 v1.2 待定） | `REQUIREMENT.md` §四（新原文逐字 + D1–D4 差异 + C5/C6 拆解 + 三条已定口径）+ 方案 §13（A1–A13 / 10 案例建议清单 / 排期建议）+ 正文 6 处 `⚠️ §13-A#` 交叉引用 |
| **P0-3「任务开始」页** | ✅ 规划 + 样机 + 骨架（`21f707c`）+ 草稿生命周期（`c7d8e9f0a1b2`）+ 对话式澄清（dsh 会话，`e9f0a1b2c3d4`/`e2f3a4b5c6d7`）+ 定版 / 上下文空间 + **逐 token 流式** | `docs/UI-task-start-v1.md`（§3.1 输出契约）；`POST /api/tasks/{id}/chat`（dsh 会话，**流式**）+ `/finalize`（定版门禁）+ `/context`；前端打字机；**P0-3 闭环** |
| **M1-b `benchmark.sh` / `profile.sh` + D9 指标摸底** | ✅ 已实装并实测（见下「M1-b 实测记录」） | 两侧都走 `msopprof` 设备侧 `Task Duration`（同卡同条件）；run #6 benchmark / #7 profile **门禁 pass**；**D9 已定**：四类指标均可采，门禁 profile 收紧为 `log`+`metrics` |
| **M3-a 自动优化闭环（首个真机 A/B）** | ✅ 已跑通（见下「M3-a 实测记录」） | 硬件画像 `ops/chips/dav-2201.yaml` + `perf` Loop #2 + `performance-diff.json` + 门禁规则（精度退化硬拦）+ P2 对比视图；A/B = run #8/#9 vs #10/#11：主用例 **no_significant_change**、64k 回退 +19.6%、精度 19/19 不退化 |
| **M3-b 自动优化增强（agent 自动改代码）** | ✅ 机制闭环 + 真机冒烟（见下「M3-b 实测记录」） | 迁移 `f1a2b3c4d5e6`（`loops.optimize_json`）；`POST/GET /api/loops/{id}/optimize` + `/optimize/revert`；前端 PerfDiffPanel 入口；`pytest` 111 |
| **页面补齐：任务详情分区 + P4 消耗情况** | ✅ 完成（2026-10-19，见下「页面补齐实测记录」） | `GET /api/usage` + `GET /api/artifacts/{id}/download`；前端 `LoopPanel` / `DeliveryPanel` / `UsagePanel`；`pytest` 113 |
| **测试管理页（P3 读侧）** | ✅ 完成（2026-10-19，见下「测试管理实测记录」） | `GET /api/tasks/{id}/cases`（用例矩阵 + 逐例历史 + 失败点 + 执行记录）；前端 `TestPanel`；`pytest` 116 |
| **阶段看板（P3 读侧 + 按证据同步）** | ✅ 完成（2026-10-19，见下「阶段看板实测记录」） | `GET /api/stages` + `POST /api/tasks/{id}/stage/sync`；前端 `StagePanel`；`pytest` 119 |
| **API 管理（P4 配置态）** | ✅ 完成（2026-10-19，见下「API 管理实测记录」） | 迁移 `b2c3d4e5f6a7`（`endpoints`）+ CRUD + 探针；前端 `ApiPanel`；`pytest` 123 |
| **Loop #3 真机 A/B（满块 DataCopy）+ 设备映射修复** | ✅ 完成（2026-10-20，见下「Loop #3 实测记录」） | run #15–#18（同卡 device 2）：19/19 PASS、门禁全 pass、`perf-diff` conclusion=`regressed`（1M +2.25%）→ 不采纳、已回退；`scripts/nm_test.sh` 修「锁哪张卡就跑哪张卡」；`pytest` 123（未变） |
| **四动作真执行（异步作业）** | ✅ 完成（2026-10-20，见下「四动作真执行实测记录」） | 迁移 `c3d4e5f6a7b8`（`action_jobs`）+ `services/action_runner.py` + 3 个端点 + 前端「真执行」；真机 build → job #1 / run #19 / 门禁 pass；`pytest` 128 |
| **方案 v1.2 落版 + 需求分层** | ✅ 完成（2026-10-20，见下「方案 v1.2 落版 + 三项决策」） | A1–A13 并入正文；`REQUIREMENT.md` 现行置顶 / 原文附录；M2 定为 **10/10 真机硬性**、算子可替换；A3 阶段三门禁 + A11 prompt 内联；`pytest` 130 |
| **Loop #4 自适应 tile** | ✅ 完成（2026-10-20，见下「Loop #4 实测记录」） | 先测量（核数曲线 + tile 扫描）再改；run #20–#23：1M 8.10→8.10us、64k **7.44→5.18us（−30.4%）**、精度 19/19 → **采纳**；`pytest` 130 |
| **P0-2 对话页（自研 B）** | ✅ 完成（2026-10-20，见下「P0-2 对话页实测记录」） | 迁移 `d4e5f6a7b8c9`（conversations/messages）+ 服务/路由 + `ChatPage`；任务会话默认挂 skills；两轮续会话实测；`pytest` 136 |
| **M2 交付形态调整 + 案例库框架** | ✅ 完成（2026-10-20，见下「M2 交付形态调整」） | `scripts/import_cases.py`（zip 校验/导入/索引）+ `CasePanel`（说明页；同日后续补列表/一键发起，见「页面级增强」）+ 侧栏入口；`pytest` 141 |
| **前端与功能对齐 + 补齐** | ✅ 完成（2026-10-20，见下「前端与功能对齐」） | `/tasks/{id}/stage` + `/gates`；`GatePanel` / `SpecPanel` / `ContextPanel` / `TaskConfigPanel` / `ImplementationPanel` / `TaskCasePanel`；规格落版、执行历史、草稿筛选、工作区浏览/diff/构建日志、任务信息编辑；`pytest` 145 |
| **页面级增强（对话/案例库/取消）** | ✅ 完成（2026-10-20，见下「页面级增强」） | 附件上传 + 命令 tab + 每轮改动审阅/回退；案例库列表/一键发起；run/action job 取消下发 `docker rm -f`；迁移 `e5f6a7b8c9d0`；`pytest` 151 |
| **流式能力调研 + dsh 插件原型** | ✅ 调研 + 原型（`e473e79`，旁路） | `ops/dsh/opx-typewriter/`（Cordis 插件）+ `backend/app/routers/prototype.py` + 前端「流式原型」页（**2026-10-21 起从侧栏隐藏，组件与接口保留**）；实测 dsh 插件**逐 token**（text 50 段 / 工具 78 段）；opencode SSE 作对照 |
| **流式并入「任务开始」真实对话** | ✅ 已并入（见下「流式并入实测记录」） | 两段式契约（`<<<SPEC>>>`）+ `services/prose_relay.py` 切分 + `POST /api/agent-jobs/{id}/delta` + SSE `delta` 事件 + 前端打字机；真机实测 88~99 段/轮、零泄漏、落库逐字一致 |
| **P2 结果视图** | ✅ 完成（2026-10-16） | `GET /api/runs/{id}/report` + `GET /api/chips` + 前端「结果视图」（ECharts：正确性/性能/指标 + 极限对比/Roofline + 状态来源）+ 结构化优化建议 + 导出 |
| P3 阶段与测试 · P4 API + 消耗 | 🟡 **P3/P4 读侧完成（2026-10-19）**；P3 写侧（手动推进/打回、判据档管理）未做 | D5（200608 基线导入）仍未定，但不阻塞已完成的读侧 |

## 需求调整与「任务开始」页（2026-10-15 会话记录）

**一、需求调整（新文本给出两块能力）**

- 归档：`REQUIREMENT.md` §四（新原文逐字 + 与（一）的逐条差异 D1–D4 + 新增能力块 C5/C6 + 三条已定口径 + 未确认项）；方案 `PLAN-web-console-v1.1.md` §13 差额修订清单（**A1–A13** + 10 案例建议清单 + 排期建议）；正文 6 处加 `⚠️ §13-A#` 交叉引用（§1.2 首版范围 / §8 P6 / §11.1 已定 / §11.2 的 D8·D9 / §12 验收 #8 / 页首状态行）。
- 差异：**D1** 「辅助生成」→「自动生成」（措辞升格，主流程本已自动）；**D2 新增「内置不少于 10 个典型算子开发案例，覆盖矩阵计算 / 归一化 / 激活函数 / 注意力机制」**（现工作区只有 `elementwise_add`，**四个场景覆盖率为 0**）；**D3 自动优化从「可选扩展（P6）」升为必需能力**（基于目标硬件特性生成优化代码 + 优化前后精度与性能对比 + 可复现报告）；**D4** 四类指标与"结合形状/dtype/硬件信息"未逐字要求（我方保留，只多不少）。
- 已定口径（用户 2026-10-15 拍）：① 案例库 = **10 个案例齐备**（规格/用例/参考实现/实现代码/基线），覆盖四场景的 **≥3 个真机跑通全链路**（**2026-10-20 已改：10/10 全部真机，本条仅作历史记录**），其余给可复现的离线或模拟器记录；② 自动优化 = **完成闭环 + 对比报告，不设加速比硬线**（提升不显著如实报告）—— 与原文「不以性能提升为验收」拆成两句：**自动优化是必需能力，提升幅度不作硬线**；③ **D8 改按案例 / 按 Loop 计**（每案例四动作各 1–2 次，受控修复仍 ≤3 轮，环境/契约类失败仍不计入）；④ **D9 升级为自动优化（M3）的硬前置**。
- 排期建议（当时未开工）：M1-b（benchmark/profile 实装 + D9 摸底，1–2 天）→ **M3-a 自动优化闭环拿已完成的 `elementwise_add` 当首个 A/B 样张**（3–4 天）→ M2 案例库（框架 1–2 天 + 10 个案例内容 6–9 天，优先产出四场景各 1 个真机）→ 方案落版 v1.2（0.5 天）。顺序理由：闭环风险最高且卡 D9，案例库工作量最大可分批，**M2 依赖本页可用**。
- ~~**未确认**：① 新文本与已归档（一）是**替换**还是**追加**；② 「≥3 个真机全链路」的口径~~ → **已定（2026-10-20）**：① **分层**（现行置顶 / 原文附录）；② **10 个全部真机全链路（硬性）**，算子可替换。

**二、P0-3「任务开始」页（新建任务 · 对话式定义算子）**

- 功能规划 `docs/UI-task-start-v1.md`：**单页三栏不分步**（分步会把对话打断）；两个产出物 —— **算子需求文档**（`spec.json` 十段 + `spec.md` 由它渲染禁手改 + `spec_versions` 一行）与 **任务上下文空间**（`tasks/<slug>/`：`TASK.md` / `AGENTS.md` / `context.yaml` / `spec/` / `references/` / `notes/`，**只指向**既定的 `cases/` `criteria/` `ops/`，不复制真相源）；**精度方案三块**（oracle / 容差档 `tasks.criteria_ref`（列已存在）/ 一致性口径 —— 用户那句"跟 torch CPU 一一对应"= oracle=torch CPU + shape·dtype 完全一致 + 特殊值逐元素比对，页面上做成一个 chip 一次定三块）；**就绪检查 7 项**（复用 `exec_actions.build_plan()` 的 preflight，不新写判定）；**上下文装载契约五层**（① 每轮固定 = `TASK.md` 紧凑摘要 + `spec` 语义/输入输出/正确性三段，**由后端拼进 prompt**——DSH 每轮独立进程，"模型记得读过"不能当假设；② 按阶段选 skills 子集；③ 按动作装判据/用例摘要；④ 模型按需自取；⑤ 禁止默认装全量历史 / 40 个 skills / 原始大产物），每轮装载清单哈希记 `agent_jobs.context_ref`（待加列）→ "这轮 agent 看到了什么"可复核。
- 布局样机 `docs/mockups/task-start-v1.html`：A 三栏并列 / B 对话优先 + 抽屉 / C 文档优先 + 底条，**同页切换 + 亮暗 + 可点交互**（待确认项就地闭合驱动主按钮可用性、chip 联动精度方案、抽屉开关、装载清单展开）→ **用户选定 B**。
- 其余已定（按建议落）：对话面板**自研轻面板**（复用 headless SSE 事件流；不在创建向导里 iframe 挂 `dsh web`）；草稿用 `tasks.is_draft` 新列（不混进阶段统计）；**「沉淀为案例」机制进 v1.2**（跑顺的任务可打包成 `cases/<op>/` 案例 → **10 个案例不必手写**）；本页落点 **P0-3**（紧跟 P0-2，它是对话能力的第一个消费场景）。
- 静态骨架（本次提交）：`web/src/components/TaskStart.tsx` + `App.tsx`（`task-new` view + 任务列表「+ 新建任务」入口）+ `index.css` 组件类；假数据、未接后端。验证：`npx tsc --noEmit` 通过；浏览器逐项点通（页头/tabs/3 条待确认项 → 闭合后计数 0 且主按钮才可点 / chip 联动精度方案 / 抽屉开关），无接口错误。
- 下一步：`POST /api/tasks/{id}/parse`（对话式解析 → spec 草稿 + 待澄清项）→ 规格版本闭合 → `POST /api/tasks/{id}/context`（生成上下文空间 + `context.yaml`）。
- 本会话运行时：后端 8077（`backend/.venv/bin/python -m uvicorn app.main:app`）与 vite 5173（`npm run dev -- --port 5173`）都在跑，未接 `OPX_LLM_KEY`（只有 agent 调用需要它）。

**三、验证样机时抓到的两个自身问题**（落码注意事项，都已修）：① 类名撞车 —— 布局变体容器用 `.v{display:none}`，把配置行的值 `<span class="v">` 一起隐藏（第一版整列空值）→ 组件类名要带前缀；② 待确认项的"答案"若嵌在"问题"元素内部，闭合态隐藏问题会把答案一起隐藏。

## 流式能力调研 + dsh 插件 spike（2026-10-15）

**问题**：dsh 的程序化面能不能拿到**真 token 级增量**（打字机）？

**实测结论**：

- `dsh headless --json` / `dsh acp`：**只给提交态**（`text` / `agent_message_chunk` 整段一次性；ACP 的 thought 与 message 同一时刻各一整块）。dsh 内部 `ctx.llm.stream` 是 token 级，但这两个对外接口故意不外发。
- **官方扩展点存在**：`ctx.on("agent/assistant-stream", ({ agent, frame }) => …)` —— agent 作用域，`dsh-headless` 自己就用它把 reasoning 实时写 stderr；`frame.chunk.type ∈ {text-delta, reasoning-delta, …}`，文本在 `chunk.text`，会话标识 = `agent.id`。
- **spike 跑通**：几行 Cordis 插件 hook 该事件 → 拿到**逐 token** 增量（实测一次答复 **50 段 `text-delta` + 47 段 `reasoning-delta`**）；挂载**不需要 pnpm** —— `--patch` 里 `insert` + 相对 `name`（锚定 patch 文件旁）。
- 对照：**opencode** 服务端原生有 SSE `/api/event` 的 `session.text.delta`（也逐 token，鉴权 `Basic opencode:<password>`）；但那是**另一个 harness**。

**原型（旁路，未进正式链路）**：`ops/dsh/opx-typewriter/`（插件）+ `ops/dsh/opx-typewriter.patch.yml` + `backend/app/routers/prototype.py`（`/api/proto/dsh/{delta,stream}`、`/api/proto/opencode/stream`）+ 前端「流式原型」页（可切 dsh / opencode；**2026-10-21 起从侧栏隐藏**，组件与接口保留）。实测 `/api/proto/dsh/stream` **端到端逐 token**（meta → reasoning… → text 逐段 → done）。

**工具场景实测（2026-10-15）**：让 agent「用 bash 列目录并数文件」→ 插件同时收到 `tool-call-delta` **78 段**（工具参数逐 token）+ `text-delta` **51 段**（正文逐 token）+ `reasoning-delta` 39 段，**工具照常执行**（回答里给出结果）。即插件是 `agent/assistant-stream` 上的**只读观察者**，不干扰 agent loop 与工具调用。

**与 `/chat` 的关系**：`/chat` 是**多轮**（`--session-id` 续 dsh 会话 + 每轮后端重拼「当前规格/待确认项」）；原型 `/api/proto/dsh/stream` 是**单轮**（每次新会话，只为验证流式）。并入时多轮逻辑不动，**每轮一条流**。

**结论**：**dsh 插件路线可行**（不动 harness、不换 opencode）。要并进「任务开始」真实对话还差一个**契约决策**：现 `/chat` 让模型只输出 JSON，流出来会是 JSON 字符；需改成「**先流式说人话 + 末尾附 JSON**」前端只流式显示 prose、末尾再解析更新规格。并入时每轮仍走 `--json`（取末尾结构化结果），插件只负责把 token 增量送到前端。（**2026-10-15 已完成并入，见下节**）

## 流式并入实测记录（2026-10-15，两段式契约 + prose 切分 + 真机打字机）

**范围**：把逐 token 流式从旁路原型**并进「任务开始」页的真实澄清对话**（用户 2026-10-15 指定）。正式 agent run（任务详情「发起」）**本次未开**（最小改动，先验通路）。

**契约（两段式）**：模型先写**给用户看的自然语言回复**（= `reply`，逐 token 上屏），另起一行输出 `<<<SPEC>>>`，再另起一行输出 JSON `{spec, questions, ready}`。理由：只输出 JSON 会把 `{"spec": ...` 喷进气泡；只输出散文又丢结构。**旧格式（整段一个 JSON、`reply` 在 JSON 里）继续兼容** —— 历史对话、模型不听话、老测试都不受影响。

**后端**：

| 位置 | 做了什么 |
|---|---|
| `services/chat.py` | 新 prompt（两段式 + 明确「不要出现 JSON/围栏/字段名」）；新解析入口 **`parse_output(text) -> (reply, parsed)`**（分隔符切分优先，无分隔符退回 `extract_json`）；`prose_only()` 供兜底用；`TurnResult.reply` |
| `services/prose_relay.py` | **新模块**：`ProseRelay` 三态状态机（`prose` / `tail` / `suppress`）—— 只放行可上屏的 prose，吞掉 JSON 尾；**防抖**：分隔符被切成两半时扣住前缀不发、末尾换行先留着（避免把 `<<<SPE` 或多余空行喷到气泡里）；起手就是 `{` / ` ``` ` 时判定「整段是 JSON」→ 一个字都不上屏 |
| `services/agent_runner.py` | `JobState.push_delta()`（text 增量 → 切分 → SSE `{"type":"delta"}`；reason 增量**只记数不上屏**，stderr 已经带了推理）；`build_command(..., stream=True)` 追加 `--patch ops/dsh/opx-typewriter.patch.yml`；`start(..., stream=..., prose_of=...)` 注入 `OPX_DELTA_URL`/`OPX_RUN_ID`；`emit_fallback_prose()` 兜底 |
| `routers/runs.py` | **新端点 `POST /api/agent-jobs/{job_id}/delta`**（插件回流专用）；`/live` 增 `prose` 与 `text_deltas`（判断插件是否生效） |
| `routers/tasks.py` | `/chat/stream` 起 job 时带 `stream=settings.chat_stream` + `prose_of=chat.prose_only`；`_stream_on_done`/`chat_turn` 走 `parse_output`，`chat_log` 里 assistant 的 text **= 流中那段 prose**；顺手抽出 `_draft_parts()`（4 处重复的草稿取值收敛成一个） |
| `config.py` | `dsh_stream_patch` / `chat_stream`（总开关 `OPX_CHAT_STREAM`）/ `delta_base_url`（`OPX_DELTA_BASE_URL`） |

**前端**（`web/src/components/TaskStart.tsx`）：SSE 里 `{"type":"delta"}` 累加进**助手气泡同款样式**的流式气泡（带 `▋` 光标）—— 就是打字机；过程事件（thinking / 步骤 / 工具）收进默认折叠的「过程（n）」小按钮；跑完重拉任务，气泡换成落库的正式消息（**逐字一致，无跳变**）。**过程区未做视觉重设计**（复用现有 class/变量，一次只改一处）。

**自测**：`pytest` **73 项**（56 → 73，新增 `tests/test_chat_stream_relay.py` 17 项：两段式解析 / 旧格式兼容 / 尾段无 `reply` 键 / 分隔符在但 JSON 坏了要报错 / 增量切分 / 分隔符跨增量切碎 / JSON-first 抑制 / flush / push_delta 只发 prose / 兜底补发且不重复 / `build_command` 只在 stream 时挂插件 / delta 端点归属与迟到增量 / 404）；`tsc --noEmit` + `npm run build` 通过。

**真机端到端（三步，全用真 dsh 0.2.0-rc.2 + 真模型）**：

| # | 场景 | 结果 |
|---|---|---|
| 1 | 单轮 `/chat/stream`（httpx 订阅 SSE 到 `done`） | 17.9s；事件 `{_meta:1, session:1, status:4, delta:88, thinking:1, text:1, final:1}`；prose **449 字**、首 token 13.1s；**prose 里没有 `{`、没有 `<<<SPEC>>>`**（JSON 零泄漏）；`chat_log` 最后一条 assistant **与流中 prose 逐字一致**；spec_draft 拿到 4 字段 + 3 待确认项 |
| 2 | 两轮续会话 | 第 1 轮 15.2s / 99 段 / 462 字；第 2 轮 9.9s / 80 段 / 471 字；**两轮 `dsh_session_id` 相同**（续会话成立）；两轮都零泄漏、落库都逐字一致；第 2 轮 cacheRead 8064（会话缓存命中） |
| 3 | 真浏览器（Chrome，页面 http://127.0.0.1:8077） | 逐秒采样对话区文本长度 85 → 297 → 539 → 589，可见 `▋` 光标递增（**真打字机**）；全过程中**页面没出现过 `{"spec"` 或 `<<<SPEC>>>`**；完成后气泡是通顺中文（视觉复核）、右栏规格草稿长出 8 项、下方「待确认 3 项」 |

**成本**：每轮 prompt/token **与并入前持平**（固定层没变，流式不额外花 token；只有本机回环 HTTP 开销）。实测两轮 in 2229/2175、out 1246/744。

**边界与未做**：① 过程区仍是「折叠的等宽小字」，视觉重设计留待出样机；② 正式 agent run 未开流式；③ `reason` 增量只记数不上屏（要显示推理需先把 `_stderr` 那条路收敛掉，避免重复）；④ 插件是 dsh 开发者预览期扩展点 —— 失效时靠兜底补发，`/live` 的 `text_deltas` 是判断依据。

## M1-b 实测记录（2026-10-16，`benchmark` / `profile` 实装 + D9 定档）

**范围**：四动作里剩下的两个 —— `benchmark`（基准对比）与 `profile`（性能采集）实装，并把 **D9**（指标体系支持范围）在真机上一次定死。**不改算子源码**：候选 exe sha256 `c4039046…` 与 M1-a run#5 `test` 验证过的**逐字节相同**（只新增了采集/基准工具与用例定义）。

**D9 定档（真机实测，镜像 `lfx-op-dev-910b:torch210-cann910`）**：`msopprof`（= `msprof op`，CANN 9.1 自带；命令面 `--application/--output/--aic-metrics/--warm-up/--launch-count`）采单算子，**四类指标都能采到**：

| 需求要求的指标类 | msopprof 来源 | 状态 |
|---|---|---|
| 算子执行耗时 | `OpBasicInfo.csv` → `Task Duration(us)`（算子级） | measured |
| 数据搬运 | `Memory.csv`（GM→UB / 主存读写带宽与字节数）+ `PipeUtilization.csv`（mte2/mte3 占比） | measured |
| 存储使用 | `MemoryUB.csv`（UB 读/写带宽）+ `Memory.csv` 的 `GM_to_UB_datas(KB)` | measured |
| 计算单元负载 | `ArithmeticUtilization.csv`（vec ratio / fp32 ratio / fops）+ `PipeUtilization.csv` | measured |

cube 相关列对 vector 算子为 NA → 如实标 `unsupported`（「该算子没有这类单元」而非「采不到」）。据此把 `gate.REQUIRED_ARTIFACTS["profile"]` 从 `("log",)` 收紧为 `("log","metrics")`，`W_METRICS_MISSING` → `E_METRICS_REPORT_MISSING`（fail）。

**基准的一处关键纠偏**：第一版用 host 墙钟（启动 + 同步）对比，得到「候选比 `torch.add` 快 5–6×」—— 那是把 **PyTorch 的 host 派发开销**当成了算子差异（实测 `torch.add` host 端到端约 80us，设备侧只有约 8us）。改成 **两侧都走 msopprof 量设备侧 `Task Duration(us)`**：候选跑可执行文件里的那一次 kernel，基线跑 `tools/torch_add_baseline.py` 的 `torch.add`（msopprof 能采到它，Op Name `Add_<hash>_high_performance_*`）。同卡同条件后真实结果：

| 用例 | torch.add（设备） | 候选（设备） | 加速比 | 说明 |
|---|---|---|---|---|
| `bench_1m`（1M，主用例） | 8.46us | 7.98us | **≈1.06×** | 带宽受限主场景；达成带宽候选 ~1542 GB/s |
| `bench_64k`（64k） | 3.74us | 7.42us | 0.50× | **候选反而更慢**（固定开销占比高）——如实记录，不粉饰 |

**交付件（都在任务工作区，已入仓）**：

| 位置 | 是什么 |
|---|---|
| `cases/elementwise_add/benchmark.json` | 基准用例定义（用例/dtype/种子 + 固定条件：`invocations`/`warm-up`/`aic-metrics`/聚合/测量范围） |
| `tools/run_benchmark.py` | 基准执行器：两侧都调 msopprof 采 `Task Duration`、重复 N 次取中位数、写 `performance.json` + 证据 |
| `tools/torch_add_baseline.py` | 基线小程序：只做 `torch.add(a,b,out=c)`，供 msopprof 采集 |
| `tools/run_profile.py` | 采集执行器：跑 `msopprof` → 解析 `OpBasicInfo.csv` + 各能力组 CSV → 四类指标 `metrics.json` + 证据 |
| `benchmark.sh` / `profile.sh` | 四动作入口**实装**（容器干净时先补编；产物类型 `performance` / `metrics`） |

**门禁与自测**：run #6（benchmark）/ #7（profile）台账登记成功、**门禁 pass、零 finding**（`run.log`+`performance.json` / `run.log`+`metrics.json`，版本字段与 run 行逐字一致）。`pytest` 73 → **85 项**（新增 `tests/test_benchmark_profile.py` 8 项 + `test_gate.py` 4 项：聚合/派生/报告结构/msopprof CSV 解析、profile 缺 metrics 判 fail、指标报告版本不一致判 fail、benchmark 报告可 pass）；`test_workspace_contract.py` 的「未实装必须明确失败」改为「必须已实装且接上执行器」。

**边界与未做**：① msopprof 一次调用只给一个 `Task Duration`（`--launch-count>1` 未产出多行，且对 torch 应用不稳定）→ 分布靠**多次调用**取中位数（`invocations`，默认 5）；② 基线绑定 `torch.add`（首条算子语义），其他算子要按 §4.6「基线按算子与用例选」各自定；③ 结果视图（P2）还没把 `performance.json`/`metrics.json` 画出来 —— 页面暂时只能用 API/文件看（M3-a 已加「优化前后对比」分区）；④ `performance-diff.json` 已在 M3-a 落地（见下「M3-a 实测记录」）。

## API 管理实测记录（2026-10-19，P4 配置态）

**范围**：方案 §7 边界②「API 管理（配置态）≠ 消耗（运行态）」的配置态一半 —— 登记模型端点 + 验连通性。

**新增交付件**：

| 位置 | 是什么 |
|---|---|
| `backend/alembic/versions/b2c3d4e5f6a7_endpoints.py` | 建 `endpoints` 表（模型 `models.Endpoint` 早有、一直没建表）+ **落一条真实默认路由**（opencode go 网关） |
| `backend/app/services/endpoint_probe.py` | 探针：`GET {base_url}/models`（OpenAI 兼容、带 Bearer）；opencode go 补 `x-opencode-session`；**key 只从环境变量取**，不进库不回显 |
| `backend/app/routers/endpoints.py` | `GET/POST/PATCH/DELETE /api/endpoints` + `POST /api/endpoints/{id}/probe` + `POST /api/endpoints/probe`（全部） |
| `web/src/components/ApiPanel.tsx` | 「API 管理」页：列表 + 登记 + 内联编辑 + 删除 + 单点/全部探针（模型清单 + 延迟） |

**实测（真库 + 真网关）**：迁移打到 head；列表返回默认路由；`POST /api/endpoints/1/probe` 对 `https://opencode.ai/inference/go/v1/models` **真返回 200**（~3.9s），列出 20+ 模型（`deepseek-v4.1-flash` / `deepseek-v4-pro` / `glm-5.x` / `kimi-k3` …），`probe_state` 落 `ok`。`pytest` 119 → **123 项**（`tests/test_endpoints.py` 4 项：CRUD+重名拒绝 / 探针落 ok / 探针失败落 fail / 缺 key 明确报错；`test_migration.py` 加 `endpoints` 断言）；`tsc --noEmit` + `npm run build` 通过。

**边界**：① 只做**连通性探针**（通不通 + 有哪些模型），不做「按端点跑一次生成」的计费探针；② ~~没有消耗金额~~ → **2026-10-21 已补端点定价与消耗金额，见下节**；③ 一条端点与 dsh 的 `--patch` 路由仍是**手工对齐**（页面改了 base_url/model，dsh patch 不会自动跟着改）。

## 端点定价实测记录（2026-10-21，解 D2 费用口径）

**范围**：补 P4 欠的「费用」半条链。**问题现场**（用户复查）：消耗页「费用合计」一直显示「未定价」，而「API 管理」页也没有任何单价可配 —— 因为这块从来没实现（方案 §11.2 D2 一直待定）。

**口径（本轮定）**：

| 项 | 结论 |
|---|---|
| 单价单位 | **人民币元 / 每百万 token**；`price_in` / `price_out` / `price_cache_read` / `price_cache_write` 四项 |
| 什么算已定价 | `price_in` + `price_out` 都配了；只配一半按未定价处理（别拿半个价算钱） |
| cache 两项 | 留空 = 不单独计价（按 0 计）；DeepSeek 类「cache 命中折扣」可单独填低值 |
| 汇总口径 | 按调用（job）算、再按任务 / 类型 / 天聚合；不需要「按 run 还是按 task」二选一 |
| 落库 or 实时 | **实时算、不落库**（`agent_jobs.cost` 只留给 P0 示例/手工行）—— 改单价后历史消耗立刻更新 |
| 早期调用 | 没有 `endpoint_id` 快照 → 回退「当前端点」价目估算，标 `cost_estimated` + 页面「估」 |

**新增交付件**：

| 位置 | 是什么 |
|---|---|
| `backend/alembic/versions/c1d2e3f4a5b6_endpoint_pricing.py` | `endpoints` 加 4 个单价列；`agent_jobs` 加 `endpoint_id` / `model`（计价归属，早期行为 NULL） |
| `backend/app/services/pricing.py` | 价目 → 费用：`cost_of`（token × ¥/M）、`job_cost`（已存金额 > 端点快照 > 当前端点估算）、`attach_endpoint`（调用创建时快照） |
| `backend/app/routers/usage.py` | 聚合时按价目算金额；新增 `priced_jobs` / `unpriced_jobs` / `cost_estimated` + `pricing` 元信息 + 未配价提示 |
| `backend/app/routers/tasks.py` | 任务详情给每条 `agent_jobs` 回填金额与 `cost_estimated`（与消耗页同口径） |
| `backend/app/routers/endpoints.py` | CRUD 接受单价（负数/NaN 400）；`priced` 由 in+out 推导 |
| 调用创建处（runs / tasks / conversations） | 起 job 时 `pricing.attach_endpoint(db, job)` 快照端点与模型 |
| `web/src/components/ApiPanel.tsx` | 新增「单价（¥/M）」列（编辑态 4 个输入）+ 登记表单 4 项 + 口径提示 |
| `web/src/components/UsagePanel.tsx` / `TaskWorkbench` / `TaskRecordsPanel` | 费用合计 / 按任务金额 / 任务页金额与「估」标记；未配价时直接提示去 API 管理 |

**自测**：`pytest` 170 → **176 项**（新增 `tests/test_pricing.py` 3 项：每百万换算 + cache 可选 / in+out 才算已定价 / 已存金额优先+估算；`test_usage.py` +2：按端点价目算金额与估算、未定价如实标注；`test_endpoints.py` +1：单价 CRUD 往返 + 负数拒绝；`test_migration.py` 加新列断言）；迁移打到本地库 head（`c1d2e3f4a5b6`）；`tsc --noEmit` + `npm run build` 通过。

**边界**：① 金额按**当前**价目实时算，不是账单对账 —— 改价会改历史金额（个人控制台可接受，换来「配完立刻有数」）；② 早期调用没有端点快照，只能按当前端点估算（已显式标注）；③ 端点改了 model/base_url 与 dsh `--patch` 路由仍需手工对齐（同 API 管理边界③）。

## 任务页审查修复实测记录（2026-10-21，HIGH 2 + MED 9 + LOW 若干）

**触发**：用户要求「看下任务相关的页面，有没有什么代码或者逻辑问题」。三路只读审查（前端列表/新建、前端详情/面板、后端接口/契约）+ Lead 交叉复核与临时库复现，共 **27 条**（高 2 / 中 9 / 低 16），完整证据在 `runs/review/`（gitignore 的运行时目录，不入库）：`SUMMARY.md`、`fe-task-list.md`、`fe-task-detail.md`、`be-task-api.md`。

**HIGH（2 类，均已实测复现并修复）**

| 问题 | 根因 | 修复 |
|---|---|---|
| 流式 AgentJob 起进程失败/卡住 → 任务永久「对话中」、`/live` 永久 running、`DELETE` 永久 409 | `agent_runner._consume` 无 try/finally、无超时、无取消；`chat_stream` 不校验 dsh 是否存在 | `_consume` 全路径兜底（异常/超时/取消 → job failed/interrupted + ended_at + error 摘要，`killpg` 杀进程组）；新增 `settings.agent_timeout`（`OPX_AGENT_TIMEOUT`，1800）；新增 `agent_runner.cancel()` + `POST /api/agent-jobs/{id}/cancel`；`chat_stream` 前置 `available()` 503、同任务在跑 409；`task_deletion.running_jobs` 僵尸兜底；REGISTRY 改有界保留最近 32 个结束状态（`/live`/`/events` 仍可读，不泄漏）；前端 TaskStart/AgentRunPanel 加「取消」按钮 |
| `finalize` 非幂等 + 阶段静默回退 | 无 `is_draft`/已定版门禁，每次调用 `version+1` 且写回「需求与设计」 | 已定版 → 409（改规格走 `POST /tasks/{id}/specs`）；定版后 stage 按 stageboard 推导写「开发」并写 `StageEvent(console)`；前端 `confirmCreate` 在 `await persist()` 之前置 in-flight，PATCH 不再携带 `is_draft` |

**MED（9 类）**

| 问题 | 修复 |
|---|---|
| 任务配置 `device_id` / `timeout_s` 未接线（页面保存不生效） | `exec_actions.build_plan` 接线：device = 请求体 > 任务配置 > 自动选空卡（显式设备也走占用检查）；timeout = 请求体 > 任务配置 > env > 1800；`fix_rounds_max` / `budget_per_action`（D8）UI 标注「未实装」 |
| 阶段/规格刷新链路断（详情跑完动作、SpecPanel 落版后头部不动） | App 抽 `refreshStages` 并在 `onChanged` 调；`SpecPanel` 加 `onChanged`；TaskWorkbench gates 依赖补 loops/run.loop_id 并展示错误/重试 |
| SSE 断流被当「完成」 | TaskStart / AgentRunPanel / PerfDiffPanel 的 `onerror` 改查 `/live`：running 则提示「连接中断，仍在运行」并退避重连（≤5 次），确认结束才 finish；加 `finished` 去重 |
| 上下文空间目录按可变 `op_name` 现算，改名/同 slug 删除后 `/context` 不存在、`delete_files` 漏删 | `context_space.find_space_dir`（marker 归属扫描兜底）；`get_context` / 删除接入；定版后 op_name 冻结（PATCH / 对话双重禁止） |
| `run_ref` 秒级不含 task id + 删除 run 目录不校验归属（跨任务误删风险） | `run_ref = r{task_id}-YYYYmmdd-HHMMSS`；`nm_test.sh` manifest 增 `task_id`；删除时读 manifest 校验归属（旧 manifest 兼容） |
| 动作作业 exit 0 但台账回填失败仍标 `ok`（页面绿色通过但无 run/门禁） | `_finalize` 在「无 run 且无 error」时置 `failed` 并写原因 |
| 任务详情费用把 `is_demo` 算进去 / 未定价无提示 | 端点定价链路（见上节）：调用创建快照端点、按 token 实时算；`is_demo` 过滤 + `cost_estimated`「估」标记 |
| 测试管理默认错任务（取最旧；从详情进入不带当前任务） | TestPanel 默认 `real[0]`（最新）+ `initialTaskId`；App 从详情「打开测试管理」传当前任务 id |
| 删除清理结果被丢弃 | TaskDeleteDialog 消费 `TaskDeleteResult`：展示删除/跳过/失败明细，点「关闭」再收尾 |

**LOW（已修）**：列表表头 7 th / 6 td 错位；执行历史时间早 8 小时（改走 `when()`）；任务详情请求 / 文件预览 / 镜像列表请求竞态；门禁请求失败静默（改错误+重试）；父级 `_check_task_refs` host/image 交叉校验；PATCH 显式 `title/is_draft=null` 由 500 改 422；`stage` 枚举；任务页 `task_id` 归属校验（register_run/patch_run）；工作区文件遍历剪枝 + git 超时兜底；`/api/stages` N+1 批量化；TaskStart 防重/防丢（载入失败锁表单与重试、离开前 flush、判据档留空 = 项目默认、超时正整数校验）；阶段看板草稿入口；任务开始页镜像显示与后端解析口径对齐；任务页取消按钮。

**残项补修（同日第二轮，用户确认「按建议来」）**

| 残项 | 修法 |
|---|---|
| 定版后 `/chat` 仍会改写 `spec_draft`/`raw_input`（旁路真相源） | `_apply_parsed` 在 `current_spec_version_id` 非空时**只追加 `chat_log`**，`spec_draft`/`raw_input`/`op_name`/`target_chip` 全冻结；返回体给当前已定版规格（不是本轮模型草稿）。定版后仍允许追问实现/历史 |
| `create_spec` 换 `op_name` 导致上下文目录名漂移 | 已定版任务的新规格版本若 `op_name` 与当前不同 → 409（算子名定版即冻结；改名需新建任务/先迁目录） |
| TaskStart 对话结束 `finish()` 回填覆盖未保存编辑 | 增加 per-field dirty 集合：只回填用户没改过的字段；保存成功后清 dirty |
| 侧栏切走丢掉 700ms 防抖编辑 | TaskStart 暴露 `flushSave` 给 App（`flushRef`），侧栏导航 `leaveView` 先 flush 再切；另加 `beforeunload` 未保存提示 |

**自测**：临时库（`OPX_DB_URL` 指向 mktemp）全量 `pytest` **206 passed**（`test_agent_runner` / `test_runs_ownership` 新增，`test_task_chat/draft/delete`、`test_action_runner/plan/stageboard/workspace_view` 扩断言；第二轮新增「定版后 chat 只写日志」「create_spec 改名 409」两条）；`tsc --noEmit` 0 错误、`npm run build` 通过；迁移仍为 `c1d2e3f4a5b6`（本轮未加迁移）；`backend/opx.db` 未被测试改写。

**遗留（如实记录）**：① `AgentJob` 无 `error` 列且本轮不加迁移 → 失败原因只在 `state`/SSE/`/live`，不落库；② `/api/agent-jobs/{id}/events` 仅重放最近 32 个结束 job；③ 若线上 project 默认 host/image 已错配，`_check_task_refs` 会让该任务的无关 PATCH 也 400（需先修正项目配置）；④ SSE 断流重连未在真实代理/浏览器下压测（仅代码路径 + tsc/build/pytest）；⑤ `fix_rounds_max` / `budget_per_action`（D8）仍只落库；⑥ 多项目假设（App 固定 `projects[0]`）与 `stage` 手工 PATCH 语义仍按现状；⑦ 审查「待定项」未构造复现：`perf_diff.evaluate_loop` 异常安全、`delete-preview` 大目录耗时、`_delete_files` 非 OSError 异常。

## Loop #3 实测记录（2026-10-20，第二轮真机 A/B + 设备映射修复）

**范围**：验证「agent 自动改代码 → 真机 A/B → `perf-diff` 判定 → 一键回退」整条闭环**能实际跑通**，同时用一条真实、最小、可回退的优化假设当样张（不是空转）。

**假设与改动**：`CopyIn` 里满块（`count == TILE_LENGTH` = 4096 float = 16KB，32B 对齐）改走 `DataCopy`，尾块保留 `DataCopyPad`。agent 自己查了工作区 asc-devkit 头，确认三参 `DataCopy` 重载在 2201 未被条件编译屏蔽（底层 `copy_gm_to_ubuf`，而 `DataCopyPad` 走 `copy_gm_to_ubuf_align_b32`）。**只改 1 个文件，+13/−6**。

**流程与证据**（task #2，Loop #3，同卡 device 2；D8：test / benchmark 各 2 次）：

| 环节 | run / job | 结果 |
|---|---|---|
| 优化前 test | run #15 | 19/19 PASS，门禁 pass，快照 `8513d22a…` |
| 优化前 benchmark | run #16 | 1M torch 8.56 / 候选 **8.00us**；64k 3.82 / **7.34us** |
| agent 自动改代码 | job #16 | 44 次工具调用、246 段 `text_delta`、`changed=true`；diff +13/−6 |
| 一键回退 | — | `reverted_clean=true` → 回到 `8513d22a…`（用于重跑干净基线） |
| 重放改动 | — | 恢复后快照回到 `a0eef89a…`、候选 exe `38d879e3…`（与 agent 当时逐字节一致） |
| 优化后 test | run #17 | 19/19 PASS，门禁 pass |
| 优化后 benchmark | run #18 | 1M torch 8.56 / 候选 **8.18us**；64k 3.68 / **7.34us** |

**判定（如实）**：`GET /api/loops/3/perf-diff?write=true` → verdict=`pass`、conclusion=**`regressed`**、warn `W_PERF_REGRESSED`：

- `bench_1m` **8.00 → 8.18us（+2.25%）**，不确定性 ±2.166% → 刚过阈值，判 **regressed**
- `bench_64k` **7.34 → 7.34us（0.00%）**，不确定性 ±9.86% → `no_significant_change`
- 精度 **19/19 不退化**；可比性四项（规格 / 判据 / 镜像 / 设备=device 2）全 `same`
- 报告落盘 `runs/r20261020-101447/performance-diff.json`

**结论**：该改动**不予采纳**，已一键回退（工作区回到 `8513d22a…`）。两轮真机 A/B（TILE 4096→8192、满块 DataCopy）都没赢 —— 当前实现已经贴着该 shape 的搬运上限，继续在「搬运动词」上打转收益很低；**下一轮得先补测量（64k 的 `aiv_time`、核数–耗时曲线）再提假设**。

**过程中抓到的真问题（比 A/B 本身更重要）**：

1. **设备锁和实际用卡不是同一张（真 bug，已修）**：`ops/nm.conf` 固定给容器 `ASCEND_RT_VISIBLE_DEVICES=0,1,…,7`，而算子写死 `aclrtSetDevice(0)`；`--devices 1` 只做了 **dev1 的 flock** → 台账 `device_id=1`、实际跑**物理 0**（0 号卡上有别人的进程）→ 优化后 test 两个用例 `aclrtSetDevice failed, ret=507033`、单例卡 ~151s。修法（`scripts/nm_test.sh`）：容器内**可见设备 = 锁定的卡**（`-e` + runner 内 `export` 双保险，后者保证优先级），`device_id` 从此是真的。
2. **`perf_diff` 的「设备一致」只比台账字符串** → 上面的错位会把「实际不同卡」判成可比。修设备映射是治本，但字符串检查仍只是必要条件（列进「下一步」③）。
4. **不能在 bash 脚本执行期间编辑它**：一次收尾阶段报 `syntax error near unexpected token '('`、run 没登记（容器里其实 19/19 全过）—— 解释器边读边执行，长脚本要等结束再改。
5. **冒烟轮保留但不计判定**：run #12/#13（`device_id` 记 1 / 实际 0）与 run #14（设备争用失败，环境类）留在台账里当证据；`perf-diff` 只取了 #15–#18（同卡同条件）。

**边界**：① 本轮时四动作异步执行器还没接通（当时照 `plan.command` 手工触发）—— **2026-10-20 已接通**，见下「四动作真执行实测记录」；② `DataCopy` 快路径对 64k 完全无变化 → 该 shape 的瓶颈不在 pad 开销；③ 只改脚本，`pytest` 未加项（当时 123 项全过）。

## 四动作真执行实测记录（2026-10-20，501 → 异步作业）

**范围**：把 `POST /api/tasks/{id}/actions` 的 `dry_run=false` 从「501 拒绝」变成**真起一次执行**。这是后续 M2 案例库（每个案例都要跑四动作）和日常使用的地基。

**新增交付件**：

| 位置 | 是什么 |
|---|---|
| `backend/alembic/versions/c3d4e5f6a7b8_action_jobs.py` | `action_jobs` 表：一次点击 ↔ 一次 run 的桥（`run_ref` / `run_id` / 日志 / 退出码 / 门禁 verdict） |
| `backend/app/services/action_runner.py` | 核心：照 plan 组环境变量 → 子进程跑 `scripts/nm_test.sh` → 收日志 → 按 `run_ref` 找回 run → `gate.evaluate` → 回填；`cancel()` 杀本地进程组 |
| `POST /api/tasks/{id}/actions`（`dry_run=false`） | 建作业并**立刻**返回 `action_job_id` / `run_ref`；前置不齐 → 409，同任务已在跑 → 409 |
| `GET /api/action-jobs/{id}` / `GET /api/tasks/{id}/action-jobs` / `POST /api/action-jobs/{id}/cancel` | 进度 + 实时日志尾部 / 历史 / 取消 |
| `web/src/components/ActionPlanPanel.tsx` | 「真执行」按钮 + 1.5s 轮询 + 实时日志（自动滚底）+ 取消 + 门禁 chip；跑完刷新任务（runs 立刻可见） |
| `app/main.py` lifespan | 进程重启把残留 `running` 作业标 `interrupted`（别让页面永远显示执行中） |

**契约（避免两套真相源）**：

- 后端**不手拼 `docker run`**：配方 / 设备互斥 / 超时与兜底清理仍在 `scripts/nm_test.sh`；
- **run 仍由脚本登记**（它 POST `/api/tasks/{id}/runs`），后端只按 `run_ref == runs.remote_workdir` 找回并在其上跑门禁；
- **退出码不是结论**：作业 `status=ok` 只代表进程跑完，业务达标看 `gate_verdict`。

**真机实测（task #2，build）**：`{"action":"build","dry_run":false}` → plan `ready=true`（自动选 device 1）→ job #1 → 容器编译 `BUILD_OK … sha256=c4039046…` → run #19 登记 → 门禁 **pass** → 作业 `status=ok / exit=0 / gate=pass`。前端轮询能读到同一份日志尾部。

**自测**：`pytest` 123 → **128 项**（`tests/test_action_runner.py` 5 项：日志 + 退出码、`run_ref` 找回 run 并回填门禁、非零退出判 failed、同任务并发拒绝、端点返回 `action_job_id`；`test_api.py` 原来的 501 断言改成异步作业契约）。`tsc --noEmit` + `npm run build` 通过。

**边界**：① 取消只杀本地进程组，远端容器靠脚本自身 `timeout` + `docker rm -f`（传到远端是后续）；② 同一任务同时只允许一个动作作业（进程内 REGISTRY + lifespan 清孤儿）；③ 设备占用检查仍在 plan 阶段做，作业启动后不再复查（脚本的 flock 兜底）；④ 作业日志只留尾部 200KB，完整日志在 `runs/<run_ref>/`。

## 方案 v1.2 落版 + 三项决策（2026-10-20）

**范围**：把 2026-10-15 需求调整的差额清单（A1–A13）**并进方案正文**、升版 v1.2，并按用户决定重排需求文本与 M2 口径。**不改功能代码**（除 A3/A11 的少量同步）。

**三项决策（用户 2026-10-20 定）**：

| # | 事项 | 结论 |
|---|---|---|
| 1 | 方案落版 | A1–A13 逐条确认后并入正文；§13 保留为**变更记录**。A1 = 自动优化必需能力（至少一份有效报告、不设幅度硬线）；A3 = **口径 B**（有 perf Loop 才要求有效对比报告；「没提速」不算无效）；A5 = 按实际口径（含 `regressed`；四类指标成对对比降为后续）；A8 = Loop + `perf-diff`（**不引入 `--baseline-run`**）；A10 = 案例库不建表（`cases/index.json`）；A12 = 统一「自动」，并写明「自动 = agent 执行 + 工具判定，人保留触发/审批/回退」 |
| 2 | 需求文本 | **分层**：`REQUIREMENT.md` 现行需求置顶，原始文本（2026-10-14）降为**附录 A** 并标「已被取代」 |
| 3 | 案例库 | **10 个案例全部真机全链路，硬性（10/10）**；不设 ≥3 底线、不留离线降级出口；**算子清单不写死**（同场景内可替换，但保持 10 个 + 四场景覆盖 + 全部真机）；M2 仍**最后做** |

**文档变化**：

| 文件 | 改了什么 |
|---|---|
| `docs/PLAN-web-console-v1.1.md` | 升 **v1.2**（文件名保留为路径稳定）；A1–A13 全部并入正文（§1.2 / §3.1 / §4.6 / §5 / §6 / §7 / §8 / §11.1 / §11.2 / §12）；新增 案例库页（八页）、M2/M3 路线行、`action_jobs` 表、`target.hardware_profile`；§13 变更为「已并入 v1.2」的变更记录 |
| `docs/REQUIREMENT.md` | 重排为「现行需求 + 附录 A 原始需求」；1.3 已定口径按 10/10 更新 |
| `docs/gate-report-v1.md` | 注明**对比报告不在单 run 门禁**（跨 run，见 perf-diff；阶段三按 A3 引用） |
| `docs/perf-diff-v1.md` | 标注 v1.2 依据与 Loop #3 第二轮验证 |
| `README.md` | 方案版本指向 v1.2（文件名不变） |

**代码同步（A3 / A11）**：

| 位置 | 做了什么 |
|---|---|
| `services/stageboard.py` | A3 口径 B：`perf_evidence()` 取最近一轮 perf Loop 的 `perf-diff`；**有 perf Loop 时**阶段三要求 `verdict=pass`，否则卡住（证据里显示 `perf-diff loop #N verdict …`）；无 perf Loop 不强制。**只喂该 Loop 自己的 run**（`loop_id` 过滤）—— 真机数据首次验证时发现「把冒烟轮 #12/#13/#14 一起喂进去」会让 Loop #3 被误判 `inconclusive`，已修 |
| `services/chat.py` | 规格草稿字段加 `target_hardware_profile`，默认指向 `ops/chips/dav-2201.yaml` |
| `services/optimize.py` | `hardware_brief()` 把 UB/L1/L0C/BT、核数/频率/HBM、32B 对齐、fp32 repeat **直接内联进优化 prompt**，并注明「运行时以 `PlatformAscendC` 为准，low/medium 不得当定论」 |

**自测**：`pytest` 128 → **130 项**（新增 `test_stageboard.py::test_perf_loop_blocks_acceptance_until_diff_is_valid`、`test_optimize.py::test_build_prompt_inlines_hardware_facts`）。

**未做 / 边界**：① 案例库 `cases/index.json` 与「案例库页」属 M2；② 样机仍七页（M2 补）；③ D8 仍是人工核对口径（无自动计数）；④ A3 只对「有 perf Loop」的任务生效（普通任务按测量+分析验收）。

## Loop #4 实测记录（2026-10-20，自适应 tile：三轮 A/B 首次采纳）

**方法**：先测量、再提假设（前两轮都是先改再看，结果都没赢）。

### 测量 1 · 核数–耗时曲线（`--force-blocks K`，64k / 1M）

| N | K=16 | K=8 | K=4 | K=2 | K=1 |
|---|---|---|---|---|---|
| 64k | **7.58us** | 8.88 | 8.22 | 15.88 | 24.54 |
| 1M | 10.90 | 18.64 | 34.04 | 66.54 | 79.28（默认 43 核 ≈8.0） |

**结论**：核越少越慢 —— **core-shrink 方向彻底证伪**（M3-a 的线索到此关闭）。

### 测量 2 · tile 扫描（容器内临时改 `TILE_LENGTH` 重编）

| TILE | 64k 核数 / 耗时 | 1M 核数 / 耗时 |
|---|---|---|
| **4096（原值）** | 16 / 7.60us | 43 / **8.08us** |
| 2048 | 32 / 7.48us | 47 / 8.36us |
| 1024 | 32 / 7.64us | 47 / 9.08us |
| **512** | 43 / **5.20us** | 48 / 11.86us |
| 256 | 43 / 5.62us | — |
| 128 | 47 / 8.40us | — |

**结论**：64k 的瓶颈不是「核太多」，而是 **tile 太大 → 只有 16 个 tile → 只起 16/48 核**；tile=512 让 43 核参与，64k 降到 5.20us。1M 必须保持 4096（512 退化 47%）。

### 假设与实现

- **假设**：`N ≤ 65536` 用 tile 512，否则 4096 → 1M 不退化、64k 提速约 30%。
- **实现（agent 自动改，3 个文件）**：`TILE_LENGTH` 降级为运行时 `tiling.tileLength`（进 `AddTilingData`）；kernel 的 Init/Process 用它算 `tileNum/offset`；UB buffer 仍按 `MAX_TILE_LENGTH=4096` 分配；host 侧按 N 选 512/4096；保留 `--force-blocks` 调试开关。
- **工具**：`tools/measure_curve.py` + `--force-blocks`（调试用，已入仓）。

### A/B（同卡 device 3，Loop #4）

| 环节 | run | 结果 |
|---|---|---|
| 优化前 test | #20 | 19/19 PASS，门禁 pass |
| 优化前 benchmark | #21 | 1M 8.10us / 64k **7.44us** |
| agent 自动改代码 | job #17 | 30 次工具调用、254 段 token；3 文件 |
| 优化后 test | #22 | 19/19 PASS，门禁 pass |
| 优化后 benchmark | #23 | 1M **8.10us** / 64k **5.18us** |

**判定**：`perf-diff` verdict=**pass**；`bench_1m` **0.0%**（无退化）、`bench_64k` **−30.4%**（不确定性 ±12.2% → **improved**）；精度 19/19 不退化。

**一个要说清的口径**：`perf-diff` 的**总结论**按主用例（`bench_1m`）算，所以是 `no_significant_change`；逐用例表里 64k 明确 `improved`。这是预期 —— 本轮优化只针对小 shape，主用例本来就不该变。

**结论**：**采纳，不回退**。Loop #4 已关闭（conclusion=`accepted`）。D8 用量：test/benchmark 各 2 次（4 条 run）。

**边界**：① 阈值 `65536` 来自这两组实测，其它 shape 未扫（M2 新案例应按各自形状补测）；② 512 对 64k 最优，但 256/128 已开始退化（每核迭代变多）；③ `measure_curve.py` 与 `--force-blocks` 留在工作区，属调试工具，不参与正式调用。

## P0-2 对话页实测记录（2026-10-20，自研方案 B）

> 2026-10-21 晚更新：独立「对话」页已并入任务详情「对话」分区（只列本任务会话），侧栏入口移除；本节保留作历史证据。
**范围**：方案 §7「对话」页 —— 和 agent 说话、看历史。**引擎不新写**：一次发送 = 一个 `agent_jobs` 作业（`kind=对话`），走 `services/agent_runner`（`dsh --profile headless --json` + SSE + 逐 token 插件）。

**方案选择（用户 2026-10-20 定）**：**B 自研**，不反代 `dsh web`。实测过 `dsh web`：完整 SPA、token→cookie（HttpOnly/SameSite=Strict/**authority 绑定**）、`/api` 与 WebSocket 都要鉴权、`--trusted-host` 可配 —— 技术上可行，但 WS 桥接 + cookie 透传 + 与 `0.2.0-rc` 的版本耦合是长期成本，且会把会话/任务拆成两套真相源。

**两种模式**（会话的 `mode`）：

| 模式 | cwd | skills | 能改代码 | 用途 |
|---|---|---|---|---|
| `task`（默认） | 任务工作区 | ✅ 40 个 | ✅ | 正常开发对话 |
| `light` | `.opx-chat/conv-<id>` 空目录 | ❌ | ❌ | 轻问答/解释概念 |

**新增交付件**：

| 位置 | 是什么 |
|---|---|
| `backend/alembic/versions/d4e5f6a7b8c9_conversations.py` | `conversations`（模型早有、一直没建表）+ `messages` + `agent_jobs.conversation_id` |
| `backend/app/services/conversations.py` | 建会话 / 发消息（落用户消息 → 起 dsh 作业）/ 收尾（落助手消息、回填 session、tokens）/ 两种 cwd / 同任务并发保护 |
| `backend/app/routers/conversations.py` | 列表 / 新建 / 详情（含消息）/ 发消息 / 删除 |
| `web/src/components/ChatPage.tsx` | 会话列表 + 任务/模式选择 + 对话流 + 逐 token 打字机 + 过程提示 + IME 防误发 |

**每轮固定层**（后端拼接，不依赖模型记忆）：任务标题/算子/目标芯片 + 工作区路径 + skills 提示 + 规格要点（数学定义/输入输出/oracle/容差）；轻对话则明确「不读写工作区、不调用工具」。

**真机实测（task #2，task 模式）**：

| 轮 | 结果 |
|---|---|
| 第 1 轮 | SSE **16 段** text delta、126 字；助手消息落库；session `session-670eaa25…` 回填；tokens 7464 |
| 第 2 轮 | **session 不变**（续会话成立）；消息 4 条；tokens 累计 19773 |

**自测**：`pytest` 130 → **136 项**（`tests/test_conversations.py` 6 项：建/列/详情、task 模式 cwd+skills prompt、light 模式空目录、同任务并发拒绝、收尾回填、端点 CRUD+发送）；`tsc --noEmit` + `npm run build` 通过。

**边界**：① 首版要求会话**绑定任务**（`agent_jobs.task_id` 非空；通用无任务对话后续放开）；② 附件/终端/审批 UI 未做；③ 同一任务同时只允许一个对话作业（两个 agent 同时改一个工作区会互相踩）；④ `messages` 只存正文，dsh 完整事件仍在 JSONL；⑤ 轻对话是「空目录 + 提示」，不是硬性工具禁用。

## M2 交付形态调整（2026-10-20，用户定）

**决定**：案例库不预置进仓库 —— 10 个案例做成**独立 zip bundle**，由用户自己保存；需要时用导入脚本装进本机库目录。**M2 内容仍最后做**，现在只做说明页 + 导入脚本。

**理由**：案例体积大、更新节奏与代码不同；仓库只管代码与工具，案例作为「补充包」按需导入，避免仓库膨胀和案例内容与代码强耦合。

**本次已做**：

| 位置 | 是什么 |
|---|---|
| `scripts/import_cases.py` | 校验 + 解包 + 索引：`manifest.json`（schema/name/version/cases）+ `cases/<op>/cases.json`；拒绝路径穿越；默认导入到 `~/.local/share/op-agent/case-library/`；`--list` / `--force` |
| `web/src/components/CasePanel.tsx` | 「案例库」页（占位说明）：为什么外置、bundle 结构、导入命令、默认路径、M2 口径 |
| 侧栏 | 新增「案例库」入口（M2） |
| `backend/tests/test_case_import.py` | 5 项：正常导入 / 推断 cases / 拒绝路径穿越 / 缺 manifest 或 cases.json / `--force` 去重 |

**口径变化**：

- 验收 #11 从「内置 ≥10 案例」改为「**提供 ≥10 案例的可导入 bundle**（仓库不预置）；导入后清单可查、可一键发起」。
- 10/10 真机全链路、四场景覆盖**不变**；M2 仍最后做。

**未做**：案例内容本身（**0/10**）。~~案例库页的「已导入列表/一键发起」、控制台读取库目录的 API~~ ✅ 已补（2026-10-20：`GET /api/case-library` + `POST /api/case-library/start` + `CasePanel`，见下「页面级增强 · 案例库」）。

## 前端与功能对齐（2026-10-20）

**范围**：页面骨架早已齐备，但审计发现 5 处「会误导判断」的对齐问题 + 若干后端能力未暴露，先修前 5 项与过期文案。

**新增接口**：

| 位置 | 是什么 |
|---|---|
| `GET /api/tasks/{id}/stage` | 单任务推导阶段（与阶段看板同一套：门禁 + perf-diff） |
| `GET /api/tasks/{id}/gates` | 四个动作**最近一次** run 的门禁 verdict + fail/warn codes（页面直接展示，不用逐个 run 请求） |

**前端改动**：

| 问题 | 修法 |
|---|---|
| 门禁结论没上页面（原来只看产物齐不齐） | 新增 `GatePanel`：推导阶段 + 四动作 verdict + fail codes + perf-diff；交付面板也改用门禁 verdict |
| 阶段显示用 `tasks.stage`（task #2 显示「需求与设计」，推导是「交付」） | 任务列表/详情改用 `/api/stages` + `/api/tasks/{id}/stage` 的**推导值**；存值不一致时标「未同步」 |
| 缺「需求与规格」分区 | 新增 `SpecPanel`：当前规格字段 + 假设/未确定项 + 版本下拉（`GET /tasks/{id}/specs`） |
| 费用全为 null 时显示 `¥0.00` | 改为「未定价」（与消耗页 `cost_known=false` 一致） |
| 结果视图只取「最近一次成功」 | 增加历史 run 下拉（标注快照前缀 + loop），默认仍取最近一次成功 |
| 文案过期 | 「方案 v1.1 §5」→ v1.2；「照 steps 手工跑」→ 点「真执行」；产物说明指向结果视图 |

**自测**：`pytest` 141 → **142 项**（新增 `/stage` + `/gates` 端点测试）；`tsc --noEmit` + `npm run build` 通过。

**2026-10-20 后续补齐（UI-only，无后端改动）**：

| 项 | 落点 |
|---|---|
| 规格**新建版本** | `SpecPanel` 加「新建版本」表单（spec_json / 假设 / 未确定项 / 变更原因 → `POST /tasks/{id}/specs`，设为当前） |
| 任务上下文空间 | 新增 `ContextPanel`：文件清单 + 大小 + sha256 前缀（`GET /tasks/{id}/context`） |
| 动作作业历史 | `ActionPlanPanel` 加「执行历史」折叠表（job/动作/状态/run/门禁/时间） |
| 任务配置覆盖 | 新增 `TaskConfigPanel`：执行机 / 镜像 / 设备 / 超时 / 修复轮次 / 每动作预算（空 = 项目默认） |
| 草稿筛选 | 任务列表「全部 / 正式 / 草稿」三档 |

**2026-10-20 再补齐（任务详情三块）**：

| 项 | 落点 |
|---|---|
| **实现与接口** | `ImplementationPanel`：工作区文件浏览（过滤 + 文本预览）、Git diff（分支/未提交/已跟踪 diff）、最新构建日志；后端新增 `GET /tasks/{id}/workspace(/file|/diff)` + `GET /artifacts/{id}/text`（只读、路径越界拒绝、跳过运行时大件、大小上限） |
| **测试与正确性** | `TaskCasePanel`：用例矩阵最新状态 + 失败点 + 最近执行 + 报告下载（复用 `GET /tasks/{id}/cases`），可跳「测试管理」 |
| **任务基本信息编辑** | `TaskConfigPanel` 增加标题 / 算子名 / 目标芯片 / 判据档（`PATCH /tasks/{id}`） |

**仍未暴露（后续）**：对话页执行前审批（附件 / 命令 / 每轮审阅回退见下一条）；案例库内容导入后的端到端验证（框架已可用，内容 0/10）。

## 页面级增强（2026-10-20，对话 / 案例库 / 取消）

### 对话增强

| 能力 | 实现 |
|---|---|
| **附件** | `POST /api/conversations/{id}/attachments`（20MB 上限，文件名净化，落 `<工作区>/.opx-chat/conv-<id>/attachments/`）；发送时把附件相对路径拼进 prompt；`ChatPage` 加「附件」按钮 + chips |
| **命令（终端）** | `POST /api/tasks/{id}/shell`：在工作区执行**一条**命令（默认 30s / 上限 60s，输出截断 64KB，环境变量剥掉 KEY/TOKEN/SECRET/PASSWORD）；`ImplementationPanel` 加「命令」tab（非交互 PTY） |
| **审阅 / 回退** | 任务模式每轮对话前抓工作区快照，跑完算逐文件 diff + 回退数据，落 `messages.changes_json`；助手消息下可展开 diff 并「回退本轮」（`POST /api/conversations/{id}/messages/{mid}/revert`）。**边界**：这是**事后审阅**；真正的**执行前审批**需要 dsh 的 approval answerer 插件，暂不做 |

**迁移**：`e5f6a7b8c9d0`（`attachments` 表 + `messages.changes_json`）；`requirements` 加 `python-multipart`。
**边界**：命令是单条、非交互；附件只存本机工作区（不随 run 同步到远端）；审阅只覆盖工作区代码文件（跳过 `.dsh`/`.cannbot`/`runs`/`build`）。

### 案例库（M2 框架）

| 能力 | 实现 |
|---|---|
| 列表 | `GET /api/case-library` 读 `~/.local/share/op-agent/case-library/index.json`；`CasePanel` 展示 bundle / 案例 / 是否含 `workspace/`、`ops/` |
| 一键发起 | `POST /api/case-library/start`：复制 `cases.json` → `cases/<op>/`；可选 `workspace/` → 工作区根、`ops/` → `ops/<op>/`；建任务。**目标已存在则 409，不覆盖** |
| 边界 | 案例内容（10 个）仍未做；导入仍走 `scripts/import_cases.py`（仓库不预置） |

### 取消 run 传远端

- 新增 `services/remote_cancel.py`：`ssh <host> "docker rm -f opx-<run_ref>"`（25s 超时，失败如实返回）。
- `POST /api/runs/{id}/cancel` 与 `POST /api/action-jobs/{id}/cancel` 都先杀本地进程组、再清远端容器；本地 run 若在跑则标 `interrupted` 并记 note。
- 测试用 monkeypatch 断言真实下发的命令（不碰远端）。

**自测**：`pytest` 145 → **151 项**（附件/命令/回退/案例库/远端取消）；`tsc --noEmit` + `npm run build` 通过；真机外冒烟：shell 在工作区执行、附件落到 `.opx-chat`、案例库空索引正确返回。

## 复查修复（2026-10-20，Roofline / 优化 Loop / 正式 agent 流式）

**范围**：上一轮复查列出的页面逻辑问题 1–3 +「正式 agent run 逐 token 流式 / 过程区重设计」。

| 问题 | 修法 |
|---|---|
| 结果视图把 AI 写死成 elementwise_add（`1/12`） | 新增 `web/src/lib/opProfile.ts`：优先读报告显式 `arithmetic_intensity`，否则按算子画像估算；未知算子（矩阵/注意力）**明确不做 Roofline 判定**，不再猜 |
| 性能图图例写死「基线(torch.add)」 | 基线名从 `conditions.baseline_impl` / 用例 `baseline.impl` 读，老报告退成「基线」 |
| benchmark 报告本身也只按 2 读 1 写算带宽 | `tools/run_benchmark.py` 支持 `benchmark.json` 的 `memory.reads_per_elem / writes_per_elem`、`compute.flops_per_elem`（用例级可覆盖；矩阵类可给 `traffic_bytes` / `flops`）；`derived` 增加 `arithmetic_intensity`，条件里回写画像。`elementwise_add/benchmark.json` 已显式声明 2 读 1 写、1 flop |
| 新建 perf Loop 的假设写死 elementwise_add | `PerfDiffPanel` 用 `task.op_name` 生成范围；有结构化建议时用建议的瓶颈假设，否则给通用「先测量再提假设」 |
| 从建议「开新 Loop 并让 agent 执行」后证据停在旧 Loop | `listen(jobId, targetLoopId)` 显式按目标 Loop 拉证据，修掉 `setLoopId` 后闭包仍是旧值的问题 |
| 正式 agent run 没有逐 token；过程区是原始事件行 | 后端 `agent_runner` 增加 `plain_stream`（正文原样放行，不套两段式 JSON 切分）；`POST /api/tasks/{id}/agent-runs` 开 `stream=settings.chat_stream, plain_stream=True`。前端 `AgentRunPanel` 重做：上方「正文」打字机气泡、下方可折叠「过程」（思考/工具/步骤/错误，带计数与图标）；`text/final` 仅在没有 delta 时整段兜底，避免重复 |

**顺带修复（白屏）**：`GET /api/tasks/{id}/gates` 对「还没跑过的动作」只返回 `{"action", "run_id": null}`，而 `DeliveryPanel` 按 `fail_codes.length` 渲染 → 打开任何 **0 run** 的正式任务（addrelu 等）直接整页白屏。修法：后端补齐 `verdict/fail_codes/warn_codes` 空值；前端 `DeliveryPanel` / `GatePanel` 加空值保护；`test_stageboard.py` 加回归断言。已用无头 Chrome 逐个点开 4 个正式任务验证，控制台零异常。

**自测**：`pytest` 151 → **155 项**（新增画像/显式流量/plain stream 测试）；`tsc --noEmit` + `npm run build` 通过。

## 服务器 / 镜像 / API key 增强（2026-10-20）

| 需求 | 落点 |
|---|---|
| 新增服务器入口 | 「服务器与镜像」页 + `POST /api/hosts`；新增后可探针、设为当前测试服务器 |
| 当前测试服务器 | `hosts.is_test_server`（迁移 `a1b2c3d4e5f6`）；`POST /api/hosts/{id}/set-test`；`build_plan` 优先级：任务显式 > 当前测试服务器 > 项目默认 |
| 测试镜像 | `host_images.is_test`；页面「设为测试镜像」；任务选镜像只暴露测试镜像（已选中的保留），未标记的默认隐藏 |
| 任务选镜像真正生效 | `build_plan` 之前只看项目默认，现在按 **任务 `host_id`/`image_id` 覆盖** 解析；镜像不属于执行机时明确报缺项 |
| API key 页面配置 | `endpoints.api_key`（迁移同一条）；接口只回 `has_key` 不回显；`endpoints.is_active` + `POST /api/endpoints/{id}/activate`；当前端点 key 同步到进程环境，供 dsh 的 `apiKeyEnv` 读取；没配 key 仍回退环境变量 |
| 新算子不再退化成 elementwise_add | `build_plan` / `action_runner` 注入 `OPX_OP=<task.op_name>`；`nm_test.sh` 支持 `--op` 并把它 export 进容器 runner。以前新任务跑四动作会默认找 `ops/elementwise_add` |
| agent run 刷新后可恢复 | `AgentRunPanel` 挂载时查任务的最新 `kind=开发` 且 `status=running` 的 job，重新订阅 `/api/agent-jobs/{id}/events`；后端 `JobState.events` 会重放已缓冲的正文/过程事件 |
| 构建缓存（省重复编译） | `scripts/nm_test.sh` 增加受控缓存：key = `op + 源码快照 sha256 + 镜像 digest`；`build` 成功后把候选 exe 存到 `<REMOTE_ROOT>/.build-cache/`，`smoke` / `test` / `benchmark` / `profile` 命中时放回 `/ws/ops/<op>/build/` 并跳过补编（2026-10-28 补：`smoke` 原先漏在复用名单外、每次重编，已修）；未命中仍自动补编并回存。manifest 记 `build_cache.status/key`。`OPX_BUILD_CACHE=0` 关闭，`OPX_FORCE_BUILD=1` 强制重编。**首轮真机踩到**：`$BUILD_CACHE_KEY（` 的变量名后紧跟全角括号，被 bash 当成变量名的一部分 → `unbound variable`；已统一改成 `${VAR}`，并修掉 `build.sh`/`test.sh` 里同类中文紧邻变量的写法 |

**自测**：`pytest` 155 → **158 项**（host 当前测试服务器唯一、镜像 `is_test`、任务/当前测试服务器优先级、key 存储不回显 + env 同步）；`tsc --noEmit` + `npm run build` 通过；无头 Chrome 点开服务器页 / 任务配置 / API 页零控制台异常。

## 服务器配置收敛 + 执行机真接线（2026-10-21）

**触发**：用户反馈「服务器与镜像，新增服务器的配置项是不是不太合理？」复查后确认：问题不在框多，而在**配置项与生效链路脱节**。

| 问题 | 根因 | 修法 |
|---|---|---|
| `cred_ref` 填了不生效；`address` 与 `ssh_alias` 语义重叠 | 凭据/目标只存在表单里；探针、设备检查、取消一律取全局 `settings.ssh_key` | 新增 `services/ssh.py` 统一解析 target/key（`~` 展开、私钥空回退全局）；四处调用同源 |
| 页面切换执行机后，四动作实际仍跑在 `ops/nm.conf` 的 `OPX_SSH`（默认 tk36） | `action_runner.build_env` 只注入 `OPX_HOST_ID`，没注入 `OPX_SSH` / `OPX_SSH_KEY`；`plan.host_ref` 只写进 plan | `ExecutionPlan` 增 `host_ssh_key`；`build_env` 注入两项（环境变量优先于 `ops/nm.conf`），manifest 的 `host` 与 `host_id` 不再打架 |
| 名称是唯一必填项，可以登记一台永远连不上的机器；登记后不能改 | 后端只校验名称；hosts 无 PATCH | 创建时要求 SSH 目标（名称可省，默认取目标）；`PATCH /api/hosts/{id}` 改名称/目标/私钥/备注；页面加「编辑」，保存后自动探针 |
| 详情页以 `address` 为主展示，只填 alias 时显示成 `tk37 · ` | `address` 实际是探针回填的 hostname | 连接目标统一展示 `ssh_alias or address`；`address` 仅作「探针得到的地址」补充 |
| 私钥只能手填路径，容易写错/不知道去哪找 | 浏览器安全模型拿不到用户选择文件的绝对路径 | 私钥字段加「选择文件…」：`POST /api/hosts/ssh-key` 接收上传 → 复制到 `~/.local/share/op-agent/keys/`（目录 0700 / 文件 0600）→ 页面回填路径；**私钥内容不进库、不回显**，仍保留手填已有路径 |
| 云主机控制台给的 `root@1.2.3.4:50023` 直接当 ssh 目标会 25s 超时（rc=124） | ssh 不接受 `host:port` 写法，把整串当 hostname 了 | `services/ssh.py` 增加 `split_ssh_target()`：支持 `user@host:port` / `ssh://user@host:port` / `[::1]:port`，拆出的端口经 `OPX_SSH_PORT` 传给 `nm_test.sh`（`ssh -p`）；探针 / 设备检查 / 四动作 / 取消同源 |
| 服务器只能增改、不能删 | hosts 只有 GET/POST/PATCH/probe | `DELETE /api/hosts/{id}`：连带删镜像清单；**仍被项目/任务/执行引用时 409** 并列出计数；删当前测试服务器时自动顺延第一台剩余服务器；页面详情加「删除」+ 二次确认 |

**落点**：`backend/app/services/ssh.py`（新）、`routers/hosts.py`、`schemas.py`、`config.py`（`ssh_key_dir`）、`services/exec_actions.py`、`services/action_runner.py`、`services/remote_cancel.py`；前端 `web/src/App.tsx`、`lib/api.ts`、`types.ts`。**语义定档**：`cred_ref` = 私钥路径（可空 = 全局 `OPX_SSH_KEY`），不再有「只存引用但不生效」的假配置；上传的私钥落在本机 key 目录，库里只存路径。

**自测**：`pytest` 162 → **170 项**（新增：SSH 目标必填 / 名称默认目标 / PATCH / 探针用 host 私钥 + 端口拆分 / 私钥上传落 0600 且不回显 / 非私钥拒绝 / 连接目标写法解析 / 删除服务器（镜像级联 + 测试服务器顺延 + 有引用 409）；plan + env 注入 `OPX_SSH` / `OPX_SSH_PORT` / `OPX_SSH_KEY` 及回退全局）；`tsc --noEmit` + `npm run build` 通过。

**未处理（如实记录，不夹带）**：镜像的 `env_recipe` 仍未进执行链（`build_env` 不设 `OPX_ENV_RECIPE`，docker 参数实际来自 `ops/nm.conf`；页面自动写入的 `<工作区>` / `<pylibs>` 是给人看的模板）。这是独立设计问题，本轮不动。

## 任务详情布局重构（2026-10-20，作业台六分区）

**触发**：用户反馈「任务里面，算子详情布局不太美观合理，规划调整下」。原页面把 13+ 张卡片一路平铺，模块宽度不一致、重要信息淹没在长页面里。

**落点**：

| 文件 | 改动 |
|---|---|
| `web/src/components/TaskWorkbench.tsx`（新） | 作业台壳：顶部摘要（算子/芯片/判据/规格 + 阶段 stepper + 四动作门禁 + 执行/调用/轮次/费用）、吸顶六分区导航、分区内容保持挂载（切走不掐断 SSE）、切回时补 `resize` 让隐藏时初始化的 ECharts 重新量宽 |
| `web/src/components/TaskRecordsPanel.tsx`（新） | 「记录」分区：Loop / AgentJob / Run 归属 + 调用台账 + 执行台账；产物按类型归并（`report ×N`），不再刷一屏同色胶囊 |
| `web/src/components/SpecPanel.tsx` | 规格改为文档式分组（标识与环境 / 语义与接口 / 正确性口径 / 性能 / 交付 / 假设与未确定项 / 其他字段）；标量直接读、数组列表、对象 key-value；`_meta` 不占版面；原始 JSON 折叠；版本级假设去重 |
| `web/src/components/TaskConfigPanel.tsx` | 执行配置默认折叠成一行摘要（标题/算子/芯片/判据/执行机/镜像/设备/超时），点「编辑」才展开 |
| `web/src/components/ContextPanel.tsx` | 窄栏文件列表；大小按 B/KB/MB 显示（不再全部 `0.0 GB`） |
| `web/src/components/DeliveryPanel.tsx` | 产物清单限高滚动；大小改用 `size()` |
| `web/src/lib/api.ts` | 新增 `size()`（按量级选单位） |
| `web/src/index.css` | 新增 `.wtab` 分区导航样式 |
| `web/src/App.tsx` | 任务详情改接 `TaskWorkbench`；标题栏加「← 任务列表」 |

**分区（2026-10-21 晚起为七分区）**：概览（四动作真机执行 + 阶段/门禁）/ **对话（本任务的 agent 会话，开发唯一入口；原独立对话页并入）** / 需求与规格（规格 + 配置 + 上下文空间）/ 实现与调用（工作区/diff/构建日志，只看不改）/ 测试与结果（用例 + 三类机读报告）/ 性能与优化（报告入口 + perf Loop 对比）/ 记录与交付（轮次/调用/执行台账 + 验收/复现/产物）。

**顺手修**：清掉页面里未渲染的 `**markdown**` 字面量（PerfDiffPanel / ResultsPanel / AgentRunPanel / ApiPanel）。

**验证**：`tsc --noEmit` + `npm run build` 通过；用 Chrome for Testing 151 无头打开真实任务 `elementwise_sub`，六分区 + 亮/暗主题逐页截图检查；运行时零异常（仅 favicon 404）。验证用 dev server / 浏览器已关。

**明天继续（未做 / 待确认）**：

1. 按实际使用反馈再定分区命名与顺序（当前六分区如左）；如果「算子详情」原意只是「需求与规格」字段展示，继续微调字段分组与标签。
2. ~~「记录与交付」偏长：复现命令默认 3 行 + 展开，产物清单按 run/类型折叠~~（2026-10-21 已完成，见下「任务页复查微调」）。
3. 窄屏（<1280）：1100px 下记录页已复查无横向溢出；摘要三栏、spec/config 双栏及其他分区待逐档截图检查。
4. 分区选择未做 URL 持久化；刷新任务详情会回到「概览」。

## 任务页复查微调（2026-10-21）

**触发**：复查六分区作业台时发现「最近一次 build job 失败」与「run #24 门禁 pass」并排出现，容易误读成 build 现在坏了；「记录与交付」仍偏长。

| 改动 | 落点 |
|---|---|
| 失败作业不再覆盖成功 run 的结论 | `ActionPlanPanel` 接收 `runs/gates`：最近作业失败但同动作已有成功 run 时，并列显示「run #N 成功 · 门禁 pass」和「失败只影响本次执行」；执行历史行可点击回看当次日志 |
| 复现命令默认 3 行 | `DeliveryPanel` 命令按 3 行截断，逐行「展开 / 收起」，复制按钮保留 |
| 产物按 run/类型归并 | `DeliveryPanel` 一次 test 的 `report ×22` 只占一行；展开看单个文件（路径 / 大小 / sha256 / 下载） |
| 窄屏复查 | 1100px 下记录页无横向溢出，命令列自动压缩、manifest 换行 |

**验证**：`tsc --noEmit` + `npm run build` 通过；无头 Chrome 打开任务 `elementwise_sub`，概览与六个分区逐页检查，展开命令 / 产物组无运行时异常。同一数据下记录页高度 2416 → 1928px。

## 任务删除（2026-10-21）

**范围**：任务列表 / 任务详情增加删除入口；硬删除控制台记录，本机产物默认保留、勾选才清。

| 落点 | 说明 |
|---|---|
| `GET /api/tasks/{id}/delete-preview` | 只读预览：各表数量、运行中作业、可清理目录与大小 |
| `DELETE /api/tasks/{id}?delete_files=false` | 硬删除；运行中的动作/对话作业返回 409，先取消或等结束 |
| 级联清理 | 显式删 artifacts → action_jobs → runs → messages/attachments → conversations → agent_jobs → stage_events → spec_versions → loops → task；不依赖 SQLite 未开启的 FK cascade |
| 文件清理（可选） | 只删属于该任务的 `tasks/<slug>`（校验 `.opx-task`）、`.opx-chat/task-<id>`、会话附件目录、允许根目录下的本地 `runs/<run_ref>`；远端 run 目录与 dsh JSONL 不动 |
| `web/src/components/TaskDeleteDialog.tsx` | 影响面计数 + 文件勾选 + 有 run 的任务必须输入算子名；列表与详情共用 |
| 前后端 API / 类型 | `api.deletePreview` / `api.deleteTask`，`TaskDeletePreview` / `TaskDeleteResult` |

**验证**：`pytest` 158 → **162 项**（预览计数、级联清空、默认保留文件、运行中 409、越界路径跳过）；`tsc --noEmit` + `npm run build` 通过；无头 Chrome 走通「草稿列表删除」与「任务详情：输入算子名后确认」两条路径，删除后回列表且接口 404。

## 规格版本显示修正（2026-10-21）

**问题**：页面多处把 `spec_versions.id` 当成 `vN` 展示。`elementwise_sub` 当前规格是 **v2（id 6）**，阶段看板却显示 `spec v6`、执行 plan 显示 `规格 v6`、结果视图显示 `规格 v5`（那批 run 绑定的是 v1/id 5），看起来版本对不上。

| 落点 | 改动 |
|---|---|
| `services/stageboard.py` | 新增 `spec_version_of(db, task)`；阶段视图增加 `spec_version` 字段，证据文本改为 `spec v2`；`build_task_view` 无 db 时从 task 关系兜底，拿不到就显示“版本号缺失”，不再回退成 id |
| `services/exec_actions.py` | `ExecutionPlan` 增加 `spec_version`（人看的 vN），plan 步骤里的“规格 v…”改用版本号 |
| `web/src/components/ActionPlanPanel.tsx` | plan 显示 `规格 v2`；只有拿不到版本号才退回 `#id` |
| `web/src/components/ResultsPanel.tsx` | 用 `task.spec_versions` 把报告里的 `spec_version_id` 映射成 vN；当 run 绑定的版本 ≠ 当前规格时提示：`本结果基于规格 v1；当前规格是 v2（哈希相同，内容未变）`；哈希不同则标红提示“旧结果不能直接当作当前规格的验收证据” |
| `web/src/types.ts` | `ExecutionPlan.spec_version` / `StageTaskView.spec_version` |

**验证**：`pytest` **162 passed**（补充 plan / stageboard 版本号断言）；`tsc --noEmit` + `npm run build` 通过；无头 Chrome 确认阶段看板显示 `spec v2`、结果视图显示 `规格 v1` + 与当前 v2 的差异提示。

## 阶段看板实测记录（2026-10-19，P3 读侧 + 按证据同步）

**范围**：方案 §5 的四阶段看板 —— 每个任务走到哪一阶段、卡在哪道门禁。**阶段由真实证据推导**（不信 `tasks.stage` 那个手填字段：它会长期停在种子值上）。

**阶段门禁口径（落到四动作，复用 `gate.evaluate` 不另写判定）**：

| 阶段 | 达成条件 |
|---|---|
| 一 · 需求与设计 | 有规格版本（`current_spec_version_id`） |
| 二 · 开发 | 最近一次 `build` 门禁 pass（真实编译通过） |
| 三 · 验收 | 最近一次 `test` 与 `benchmark` 门禁都 pass（精度 + 性能报告达标） |
| 四 · 交付 | 有交付产物 —— **当前四动作不含 delivery，一律未达成**（如实反映） |

**新增交付件**：

| 位置 | 是什么 |
|---|---|
| `backend/app/services/stageboard.py` | 推导：各动作最近一次 run 跑 `gate.evaluate` → 逐段门禁 → 推导当前阶段 + `next_step`；与 `tasks.stage` 对比标不一致 |
| `GET /api/stages` | 整块看板：按推导阶段分组（`?include_drafts=` 控制草稿） |
| `POST /api/tasks/{id}/stage/sync` | 按证据把 `tasks.stage` 落正 + 写一条 `stage_events`（推进留痕，`actor=console`）；已一致则 `changed=false` 不写 |
| `web/src/components/StagePanel.tsx` | 四列看板：任务卡（门禁点 + 下一步 + 存值/推导不一致时「按证据同步」） |

**实测（真库）**：4 个正式任务分组正确 —— #2 `elementwise_add` 推导到**交付**段（build/test/benchmark 门禁都 pass），存值仍是种子的「需求与设计」→ 标 `out_of_sync`；#9 推导「开发」；#1 存值/推导一致；#4 无规格 → 「需求与设计」。`pytest` 116 → **119 项**（`tests/test_stageboard.py` 3 项：仅规格→开发 / 加 build→验收、加 test+benchmark→交付 / 同步端点写事件且幂等）；`tsc --noEmit` + `npm run build` 通过。

**边界**：① 只做**读侧 + 按证据同步**，不做**自动推进/打回**（谁在什么时候推进仍需人点一下）；② 「交付」段没有交付动作支撑，永远达不到；③ 阶段门禁是**四动作的聚合口径**，与单 run 的 `gate_report` 不是一回事（后者更细）。

## 测试管理实测记录（2026-10-19，P3 读侧）

**范围**：方案 §7「测试管理」页 —— 用例集与执行记录、逐例结果与失败点。**只读**（不改用例、不跑执行）。

**新增交付件**：

| 位置 | 是什么 |
|---|---|
| `backend/app/services/case_report.py` | 聚合核心：用例矩阵（工作区 `cases/<op>/cases.json`，退回最近报告）+ 逐例结果（各 `test` run 的 `correctness.json`，经 `gate.artifact_paths` 解路径）→ 统一结构 |
| `GET /api/tasks/{id}/cases` | 返回：`cases`（含 `results`={run_id→状态}、最近状态/误差）、`runs`（状态/退出码/用例集 ref/快照/通过失败/报告可下载）、`failures`、`case_source`+指纹、判据档（任务>项目>默认） |
| `web/src/components/TestPanel.tsx` | 「测试管理」页：任务选择 + 概览卡 + 执行记录表 + 失败点表 + 用例矩阵表（逐 run 历史 chip） |

**实测（真库 task #2）**：19 条用例、4 组（常规/边界/数值边界/异常）；6 次 test run 中 4 次有报告（run #1/#3 无报告如实标「无/缺」）；**失败点**正确抓出 run #4 的 `add_norm_003`（matched_ratio 0.168、max_abs_err 7.17）与 `add_edge_005`；`case_source` + `case_source_sha256` 逐 run 如实回显。`pytest` 113 → **116 项**（`tests/test_case_report.py` 3 项：仅矩阵无 run / 合并结果与失败点 / 缺报告不崩）；`tsc --noEmit` + `npm run build` 通过。

**边界**：① 只读，**不做**「推进/打回」与判据档编辑（那要连阶段状态机，属 P3 写侧）；② 用例矩阵以**工作区** `cases/<op>/cases.json` 为准，历史 run 的 `case_source_sha256` 如实展示、不做「用例集变没变」的告警（留给后续）；③ 失败点按 run 记录，不做跨 run 的失败归类。

## 页面补齐实测记录（2026-10-19，任务详情分区 + P4 消耗情况）

**范围**：把方案 §7 里「任务详情分区」缺的两块（执行记录 / 交付）与 P4 的「消耗情况」页做出来 —— 都是**不碰硬件、不调模型**的读写，数据现成。

**新增交付件**：

| 位置 | 是什么 |
|---|---|
| `backend/app/routers/usage.py` | `GET /api/usage`：从 `agent_jobs` 聚合调用数 / token（in/out/cache）/ 费用，**排除 `is_demo` 示例行**；`cost` 未定价时如实返回 `cost_known=false`（不编金额） |
| `backend/app/routers/runs.py` 的 `GET /api/artifacts/{id}/download` | 产物下载（交付/复现用）；相对路径按 manifest 目录解析，文件不在盘上明确 404 |
| `web/src/components/UsagePanel.tsx` | 「消耗情况」页：总计卡 + 按任务 / 按类型 / 按天（条形） |
| `web/src/components/LoopPanel.tsx` | 任务详情「轮次与关联」：Loop / AgentJob / Run 归属 + 每轮 token（补齐执行记录分区） |
| `web/src/components/DeliveryPanel.tsx` | 任务详情「交付」：四动作验收状态 + 复现命令（可复制）+ 产物清单与下载 |

**实测**：`/api/usage` 真库返回正确（14 条真实调用，demo 行排除；totals/by_task/by_kind/by_day 齐）；`/api/artifacts/1/download` 以 `attachment; filename="run.log"` 正确返回 160+ 字节；`tsc --noEmit` + `npm run build` 通过；`pytest` 111 → **113 项**（`tests/test_usage.py` 2 项：聚合与 demo 排除、空库）。

**边界**：① 消耗只有 token，**没有金额**（`endpoints` 未建，无定价）；② 「交付」的验收状态是**展示判定**（按运行状态 + 必需产物类型），与门禁 `gate.py` 同口径但不重复调用门禁接口；③ 阶段看板 / 测试管理（读侧）/ API 管理仍待做。

## M3-b 实测记录（2026-10-19，自动优化增强 · agent 自动改代码）



**范围**：把 perf Loop 里**原本手工**的「按优化假设改算子代码」交给 agent（方案 §13-A5「agent 提议、工具判定」；M3 余项）。**只改代码**，不跑 build/test/benchmark —— 真机执行仍照 `plan.command` 单独触发（四动作远端执行器是另一件事）。

**新交付件**：

| 位置 | 是什么 |
|---|---|
| `backend/app/services/optimize.py` | 核心：抓改前快照 → 起 dsh job（工作区 cwd、挂 skills）→ 跑完算改后快照 + 逐文件 unified diff + 结构化摘要 → 落证据；`revert()` 一键回退 |
| `POST /api/loops/{id}/optimize` | 起一轮自动优化（`kind=优化` 的 agent job 挂到 Loop）；prompt 固定层 = 规格 + 本轮 Loop 假设/判定 + **参考轮次的结构化建议** + 硬件画像 + 实现文件清单 |
| `GET /api/loops/{id}/optimize` | 读证据（改前/改后快照、逐文件 diff、摘要、回退数据）；无记录给 `status="none"` 空态 |
| `POST /api/loops/{id}/optimize/revert` | 按证据里的改前内容把工作区还原，并复核哈希（`reverted_clean`） |
| `backend/alembic/versions/f1a2b3c4d5e6_loop_optimize.py` | 迁移：`loops.optimize_json`（一列，JSON 证据） |
| `web/src/components/PerfDiffPanel.tsx` | 「让 agent 自动优化」+ 进度（复用 agent-job SSE 打字机）+ diff + 回退；建议卡上「开新 Loop 并让 agent 执行」 |

**真机端到端冒烟（只读勘察，不改代码）**：在 task #2 上开临时 Loop → `POST /optimize`（`extra` 要求只读、`reference_loop_id=2`）→ 真 dsh 在**工作区**跑起来（15 次工具调用、590 段 `text_delta` 逐 token、单轮 1 turn、exit 0），证据落库：`status=done`、`changed=false`、`before==after`（**没被改**，指令遵守）、`agent_output` 按两段式契约解析出 `{summary, hypothesis_check, expected, risks}`、`reference_loop_id=2`。**证明**：工作区 cwd 能找到 `.dsh/skills`、插件逐 token 回流正常、两段式契约与证据/回退链路都在真实 dsh 上成立。冒烟 Loop/job 已从库中清掉。

**有价值的发现（只读勘察的结论，已手工复核 host 代码）**：M3-a 留下的「小 shape 走 **core-shrink**」线索**前提不成立** —— `ops/elementwise_add/op_host/elementwise_add.asc` 的 tiling 已经 **`blockNum = ceil(totalTiles / tilesPerCore)`**（`tilesPerCore = ceil(totalTiles / availableCoreNum)`），即**核数已随 tile 数隐式裁剪**：64k（`totalTiles=16`）本来就只起 **16 核**。要再「裁核」只能把 `min_rows_per_core` 抬到 >1（比 tile 数还少起核），那对 1M 主用例有**带宽吃不饱而退化**的风险，且证据不足。→ **下一轮应换线索**，不再在 core-shrink 上花真机轮次。（agent 还提示：需先采 64k 的 `aiv_time` 与「核数–耗时」曲线再决策。）

**门禁与自测**：`pytest` 102 → **111 项**（新增 `tests/test_optimize.py` 9 项：快照/diff/回退往返、超限文件不可回退、prompt 拼装、空态、端点起 job 与证据落库 + 回退、未改也如实记、无证据回退 409、缺 dsh/凭据 400、Loop 详情暴露 `optimize`）；`test_migration.py` 加 `optimize_json` 断言；`tsc --noEmit` + `npm run build` 通过。

**边界与未做**：① 只改代码，**不触发** test/benchmark（四动作异步执行器 2026-10-20 已接通，但 optimize 仍不自动串起四动作 —— 现在是两个独立动作）；② agent 可能不听话（照旧改/乱改）——靠 `changed_files` + diff + 一键回退兜底，不做「自动采纳」；③ 工作区跟主仓 git 无耦合（自己抓快照 + 存改前内容），回退不依赖 `git`；④ 大文件（>512KB）只记哈希、不做行级 diff 也不支持原样回退（会明确报 `revert_failed`）。

## M3-a 实测记录（2026-10-16，自动优化闭环 · 首个真机 A/B）


**范围**：把「自动优化」从需求文字做成**可验证的闭环** —— 硬件画像 + `perf` Loop + 同条件 A/B + `performance-diff.json` + 门禁规则 + P2 对比视图。设计口径：**AI（agent）提议、工具判定**；验收 = 闭环 + 对比报告，**不设加速比硬线**（需求 C6 / §13-A5）。

**新增交付件**：

| 位置 | 是什么 |
|---|---|
| `workspaces/cann-ops/ops/chips/dav-2201.yaml` | **目标硬件特性画像**（§13-A11）：UB 192KB / L1 512KB / L0C 128KB / BT 1KB / 核 48 向量核 / 频率 1850MHz / 32B DataCopy 对齐 / Vector 宽度 64(fp32)；带 source + confidence + 不确定项。**源自 `npu-arch` skill**，给 agent 读、给规格引，**不硬编码进算子**（运行时以 `PlatformAscendC` 为准） |
| `backend/app/services/perf_diff.py` | 对比与门禁核心：按 code_snapshot 配对前后、查可比性、逐用例比耗时与精度、出结论 |
| `GET /api/loops/{id}/perf-diff`（`?write=true`） | 入口；报告落到优化后那次 benchmark 的产物目录 |
| `PATCH /api/runs/{id}` | 只许把已有 run 挂到某个 Loop（改归属，不改执行数据） |
| `scripts/nm_test.sh` 的 `OPX_LOOP_ID` | 登记 run 时带上 loop_id（成对关联） |
| `web/src/components/PerfDiffPanel.tsx` | 任务详情「自动优化 · 优化前后对比」分区（假设/判定、前后 run 与快照、逐用例耗时表、精度不退化、findings、开/关 Loop、关联 run） |
| `docs/perf-diff-v1.md` | `performance-diff.json` 的机器可读契约 + code 清单 |

**判定口径（四段）**：配对完整 → 可比性（**硬**：规格/判据/镜像/设备 + **基准用例集指纹**；`suite_ref` 只记录不硬判，它会随 `cases/` 里非正确性文件一起变）→ 耗时（设备侧中位数，差异 > max(不确定性, 2%) 才算显著）→ 精度不退化（**硬拦**）。性能回退**只 warn**。

**首个 A/B（真机，Loop #2）**：

| 侧 | 快照 | test | benchmark | 候选主用例（1M，设备侧中位） |
|---|---|---|---|---|
| 优化前 | `6f91d609…` | run #8（19/19 PASS） | run #9 | 8.28us |
| 优化后 | `6fb63bff…` | run #10（19/19 PASS） | run #11 | 8.18us |

- **改动**：`elementwise_add_tiling.h` 的 `TILE_LENGTH` 4096 → 8192（假设：M1-b profiling 显示 `mte2_ratio≈0.75` 搬运受限，增大分块减少循环/搬运指令）。
- **结论（如实）**：`bench_1m` **-1.2%（±2.99% → 落在不确定性内）= `no_significant_change`**；`bench_64k` **+19.6% 回退**（大分块对小 shape 不利，块数变少、每块固定开销占比升高）；精度 **19/19 不退化**。门禁 **pass**、warn `W_PERF_NO_SIGNIFICANT_CHANGE`。**该优化不予采纳，工作区已回退**（`TILE_LENGTH` 回 4096；run #10/#11 的 8192 版本作为 Loop 证据保留）。
- **有价值的副作用**：拿到「小 shape 该走 core-shrink（裁核）而非加大分块」的**实测线索**（对应最佳实践 skill `common/core_shrink_design.md`），留作下一轮 Loop 的假设。
- 过程中还修了一个真 bug：benchmark 基线侧 **msopprof 偶发采不到 kernel**（一次失败就 abort 整轮）→ 给采集加**重试**（每样本最多 3 次，成功样本照用、失败记录留证），`run_benchmark.py` 的 `collect_device_samples` 加了单测。

**门禁与自测**：run #8/#9/#10/#11 门禁全 **pass**；`pytest` 85 → **97 项**（新增 `tests/test_perf_diff.py` 10 项：improved / no_significant_change / 性能回退只 warn / 精度退化 hard fail / 可比性 / 不完整 / 非 perf Loop / 落盘 / 硬件画像良构 / suite_ref 不硬判；`test_benchmark_profile.py` 加采集重试 2 项）；`tsc --noEmit` + `npm run build` 通过。

**边界与未做**：① A/B 的「优化前」quick 复用 M1-b 工具链版本（快照含 harness 变更，但**基准用例集与测量方法一致**，可比性检查已确认；`suite_ref` 差异如实记录不硬判）；② 本轮优化由我（扮演 agent 的工程动作）手工完成，**尚未接「agent 自动改代码」**——`perf` Loop 与对比报告已就绪，接 agent 循环是后续（M3 余项）；③ P2 对比视图只做了优化对比分区，精度的 ECharts 图仍属 P2。

## P0-3 第二步 + 收尾实测记录（2026-10-15，对话式需求澄清（**dsh 会话**）+ 定版 / 上下文空间 · 迁移 `e9f0a1b2c3d4` / `e2f3a4b5c6d7`）

**范围**：真对话（**走 dsh 会话**）+ 结构化产出（规格草稿 / 待确认项）+ **收尾**（定版 `spec_versions` v1 + 生成上下文空间 `tasks/<slug>/`）。P0-3 至此闭环。

**路线（2026-10-15 改定）**：**走 dsh 会话**（`services/chat.py` + `agent_runner.run_once`，用 `--session-id` 续会话），不再直连模型。理由：本页本就是「会话形式」，用 dsh 能让对话可续、事件/usage/会话 id 都落进 `agent_jobs`，与「对话页复用 DSH」一致。（曾短暂直连 opencode-go，用户要求换回 dsh；`services/llm.py` 已删。）每轮固定层仍由后端重拼（不依赖「模型记得」）。

**后端**：
- `POST /api/tasks/{id}/chat`：每轮重拼 prompt（固定层 + 当前规格草稿/待确认项 + 本轮）→ `agent_runner.run_once` 跑 dsh（会话目录 `workspace/.opx-chat/task-<id>`，**刻意空目录、不挂 skills**，让澄清轻快）→ 从 `final` 事件抠严格 JSON `{reply, spec, questions, ready}`；落 `tasks.spec_draft`（业务证据）与 `tasks.chat_log`（过程证据），镜像 `op_name`/`title`/`target_chip`，`raw_input`=用户原话；每轮记一条 `agent_jobs`（kind=需求澄清，含 usage/会话 id；失败也留痕）。
- `POST /api/tasks/{id}/chat/stream`（**流式**）：起 dsh 后台任务立刻返回 `job_id`，前端订阅 `GET /api/agent-jobs/{id}/events`（SSE）看过程；`agent_runner.start` 加了 `session_id` / `on_done` 回调，跑完由回调解析 `final` 落这一轮。⚠️ 端点必须 `async def`（同步 `def` 在线程池里没有事件循环，`asyncio.create_task` 直接 500 —— 又踩了一次台账记过的坑）。
- `POST /api/tasks/{id}/finalize`（**定版门禁**）：待澄清清零才放行 → 先落 `spec_versions` v1（先库）→ `services/context_space.py` 生成上下文空间（后盘）→ 草稿转正。`GET /api/tasks/{id}/context` 读文件清单。
- 迁移加列：`tasks.spec_draft` / `tasks.chat_log`（`e9f0a1b2c3d4`）、`tasks.dsh_session_id`（`e2f3a4b5c6d7`）。

**前端**：`TaskStart.tsx` 真实对话（消息流 / 发送 / 待确认项可点填入 / 右栏规格草稿逐轮同步 / 抽屉看完整草稿）；走 `/chat/stream` + SSE，生成期间显示过程（thinking / 步骤）；**输入框改 textarea 且挡 IME 组字回车**（中文输入法下「add→回车」不再误发），Enter 发送 / Shift+Enter 换行；`ready=false` 时「确认并定版」禁用；定版调 `/finalize`（落 spec v1 + 生成上下文空间）后跳任务详情。

**自测**：`pytest` **56 项**（新增 `tests/test_task_chat.py` 6 项——对话合并/持久化与会话 id、失败 502 且留痕、定版生 spec + 上下文空间文件、定版门禁 409、流式端点起 job 且完成回调落库、`extract_json` 容错）；迁移往返加 `spec_draft`/`chat_log`/`dsh_session_id` 断言；`tsc --noEmit` + `npm run build` 通过。
**真实 dsh 端到端冒烟**（起 8077，`OPX_LLM_KEY` 由 `dev.sh` 注入）：建草稿 → `POST /chat`（11.7s）→ 第二轮 `POST /chat`（9.3s）**会话 id 不变**（续会话成立）→ **流式** `POST /chat/stream`（15.1s，SSE 依次 `session → status → thinking → text → … → final`，完成后回填回复与规格、会话 id）→ 强制 `ready` 后 `POST /finalize` 得 spec v1 + 上下文空间 `workspaces/cann-ops/tasks/<slug>/`（7 个文件）。冒烟任务与生成目录已删（用户自己建的草稿未动）。

**未做 / 边界**：① 流式：headless 的 `text` 是**整段一次性**（实测）→ 当时做了「**过程流式**」（SSE 显 thinking / 步骤）；**现已由插件路线拿到真 token 并并入本页**（见「流式并入实测记录」）；② 澄清会话目录是**空目录**（不挂 skills）——若要澄清阶段也带领域知识，需另配；③ `/context` 装载清单「每轮拼进 prompt」与 `agent_jobs.context_ref` 仍待做；④ token 成本只在 `agent_jobs` 留摘要，页面聚合属 P4。

## P0-3 第一步实测记录（2026-10-15，任务开始页「草稿生命周期」· 迁移 `c7d8e9f0a1b2`）

**范围**：只做草稿生命周期（纯 CRUD）。对话式解析 `/parse`、上下文空间 `/context`、规格定版是**第二步**。

**规划口径**（`docs/UI-task-start-v1.md` §2.3，四问已拍）：① 草稿**首次输入即建**；② 上下文空间**确认时一次性落盘**（草稿期不建空壳目录）；③ 本页**是入口不是首页**；④ 实现顺序**先草稿后对话解析**。

**后端**：
- `tasks` 加字段：`is_draft` / `raw_input`（需求原文，草稿期承载；定版时转进 `spec_versions.raw_input`）/ `host_id` / `image_id` / `device_id` / `timeout_s` / `fix_rounds_max` / `budget_per_action`。**空 = 用项目默认**（与 `criteria_ref` 同一口径）。
  - `host_id`/`image_id` **不设 DB 级 FK**：SQLite 的 `ALTER TABLE ADD COLUMN` 加不了外键约束（Alembic 会退化成 `ADD CONSTRAINT` 报错）——存在性改由应用层校验（`_check_task_refs`）。
- `POST /api/tasks` 扩字段；**新增 `PATCH /api/tasks/{id}`**（`exclude_unset`：不传保持、显式 `null` 才清空 → 存草稿/回填/转正）；`GET /api/tasks?draft=true|false` 过滤（草稿不进阶段统计，但列表可见、可续编）。

**前端**（`web/src/components/TaskStart.tsx` 重写 + `App.tsx`）：去假数据；首次输入 / 改字段即建草稿（`POST`）；此后改动去抖 700ms 自动保存（`PATCH`）；`?id=` / 从列表点草稿进来即回填续编；「确认创建」= 转正（`is_draft=false`）→ 跳任务详情。任务列表给草稿打「草稿」chip + 「继续编辑」入口。

**自测**：`pytest` **50 项**通过（48 → 50：新增 `tests/test_task_draft.py` 2 项：建草稿→存草稿→重新加载→转正 + `?draft=` 过滤；`exclude_unset` 与未知 `host_id` 拒绝）。迁移往返加了新列断言。`npx tsc --noEmit` 与 `npm run build` 通过。
**真实库端到端冒烟**（迁移 `c7d8e9f0a1b2` 打到 `backend/opx.db`，起 8077）：建草稿 201 → PATCH 改标题/设备/超时（`raw_input` 保留）→ 重新加载一致 → `?draft=true` 命中、`?draft=false` 不含 → 转正后反过来 → 未知 `host_id` 400。冒烟后已删掉该临时任务、停掉后端。

**未做 / 边界**：① 对话式解析与结构化产物（第二步）；② 配置字段目前前端只暴露 `device_id`/`timeout_s`，`host_id`/`image_id` 仍按项目默认展示（后端已支持覆盖）；③ 预算 `fix_rounds_max`/`budget_per_action` 只落库、未见编辑 UI 与消费方（step 2 接 `/parse` 后由 plan/门禁消费）。

## M1-a 实测记录（2026-10-15，首个算子 `elementwise_add` · commit `fee3900`）

**路线**：**Ascend C 直调（KernelLaunch）**，由 CANNBot `ascendc-direct-invoke-template` 的 Vector add 样例（`references/add_custom/`）改造。
穿刺结论见 **`docs/M1-spike.md`**（容器内 CANN `9.1.0-beta.1` 完整装、`find_package(ASC)` 走 bisheng 编译链、`--npu-arch=dav-2201`＝910B、原样样例编译→NPU 运行→精度 PASSED）。

**交付件（都在任务工作区，已入仓）**：

| 位置 | 是什么 |
|---|---|
| `cases/elementwise_add/cases.json` | **19 条用例矩阵**：常规 4 / 边界 6 / 数值边界 5 / 异常 4 / 回归 0（回归留空——不造假用例）。每例带 shape、dtype、**输入生成方式与种子**、期望（`match`/`reject`）、必需标记 |
| `criteria/ops-precision-standard@1/` | **判据档的实体**：官方 `mixed_tolerance_check.py` 逐字节副本（sha256 `ccc41e7f…`，与 `ops/dsh/skills.lock` 一致）+ 档正文 + PROVENANCE（出处/指纹/「改判据必须换版本号」规则） |
| `criteria/check.py` | 机判入口：官方混合容差 + **非有限值口径**（NaN/Inf 逐元素精确比对，不混进容差通过率） |
| `ops/elementwise_add/` | 直调工程：kernel（`AscendC::Add` + UB 三段式 + `DataCopyPad` 尾块）+ **参数化调用接口** + `--npu-arch=dav-2201`；`scripts/golden.py`＝**独立**参考实现（CPU fp64 求和后按输出 dtype 舍入） |
| `tools/run_cases.py` + `gen_inputs.py` | 用例执行器：逐例生成输入 → 调候选 exe → 判据机判 → `correctness.{json,md}` + 最小证据 |
| `build.sh` / `test.sh` | 四动作里前两个的**实装**（`benchmark.sh`/`profile.sh` 仍是「NOT IMPLEMENTED + exit 2」的存根，故意不装成能跑） |

**真机实测（run 台账，task #2）**：

| run | 动作 | 结果 | 说明 |
|---|---|---|---|
| #2 `r20261015-1118-build` | build | ok / **门禁 pass** | 镜像内编译通过，`BUILD_OK … sha256=24a13ec7…`（另一次独立编译得到同一哈希 → 构建可复现） |
| #3 `r20261015-1125-test` | test | failed / 门禁 fail | **契约缺口**：容器每次都是干净的（`--rm` + 全新 rsync），build 产物不跨 run 存在 → `test.sh` 找不到候选 exe。门禁给出 `E_EXIT_NONZERO`/`E_REQUIRED_ARTIFACT_MISSING(correctness)`/`E_CORRECTNESS_REPORT_MISSING` |
| #4 `r20261015-1134-test` | test | failed / 门禁 fail | **真算子缺陷**：16/18 过，`add_norm_003`(N=1M) 与 `add_edge_005`(N=1M−1) 失败，`matched_ratio 0.168`、坏点从第 4096 个元素起全是 0 → 每核只有第一个 TILE 有结果 |
| #5 `r20261015-1250-test` | test | **ok / 门禁 pass** | 修后 19/19 全过；`suite_ref cases@09ca2e026c7d9cf9`、候选 exe sha256 `c4039046…`、快照 `a8ad486b…`、判据 `ops-precision-standard@1`、镜像 digest、设备 0 独占 —— 全部对齐 |

**过程中抓出的三个真问题（都已修 + 加了回归）**：

1. **CANNBot 模板的分块循环不回推 GM 偏移**（`op_kernel/*.asc` 的 `CopyIn/CopyOut` 一直用基址）：只有当「每核恰好 1 个 TILE」时才正确 —— 样例 host 的 tiling 刚好让它成立，所以一直没暴露；1M 用例（每核 6 块）把它照出来了。修法：`Process()` 里按 `i * TILE_LENGTH` 推进，`DataCopyPad(..., xGm[offset], ...)`。**建议反馈 CANNBot 上游**；我们这边补了 `add_edge_006`（N=262144）专门锁这条路。
2. **容器是干净的 → `test` 依赖 build 产物**：`test.sh` 在缺产物时按同一份源码快照**补编一次**（复用 `build.sh`，不复制编译逻辑）；「跑的是哪一版」由报告里的候选 exe sha256 兜着。
3. **门禁校验器假阴性（最危险的一个）**：`correctness.json` 与 `correctness.md` 的 stem 都是 `correctness`，`nm_test.sh` 打类型时不看扩展名 → manifest 里两条都是 `correctness` 类型，`gate.py` 用 `{type: art}` 取到的是 `.md`，于是 `.json` **整段没被解析**、用例 FAIL 被静默跳过（run#4 当时只报出 `E_EXIT_NONZERO`，靠退出码偶然兜住）。修了两头：生产者按扩展名定类型（只有 `.json` 才算机读类型，`.md` 一律 `report`），消费者同类优先取 `.json`、非机读时**明确报 `E_REPORT_UNREADABLE`**；回归测试 `test_correctness_md_same_type_must_not_hide_the_json`。修后复核 run#4：门禁已能报出 `E_CASE_FAILED: [add_norm_003, add_edge_005]`。

**自测与门禁**：`pytest` **48 项**通过（38 → 48：新增 `test_workspace_contract.py` 8 项（用例矩阵字段/规格范围/拒绝码/判据档指纹/四动作入口）+ `test_gate.py` 2 项同类产物回归）。

**本机侧新增/改动**：`.gitignore`（任务工作区**交付件入仓**，只忽略 `.dsh/` `.cannbot/` `runs/` `**/build/`）；`scripts/nm_test.sh`（动作产物类型按扩展名判定 + **把 spec_version_id / criteria_ref / suite_ref / 代码快照哈希注入容器**，让报告能回写与 run 对齐的版本字段）；`backend/app/services/gate.py`（同类产物取值 + 非机读报错）。

**执行预算（D8 口径待确认）**：task #2 已登记 5 个 run（1 个是 P1.5-c 的存根、1 build、2 个失败的 test、1 个通过的 test）。**benchmark + profile 再各 1 次就会摸到「总共不超过 6 次」的默认线** —— 见「未决与风险」。

## P1.5-c 实测记录（2026-10-15，门禁与环境 · commit `3cc37b8`）

一次对 v1.1 的复查（事实修正 + 缺失补齐 + 前后矛盾）落地并实测，逐条证据：

- **门禁校验器 v1**：`backend/app/services/gate.py` + `GET /api/runs/{id}/gate`（`?write=true` 落 `gate_report.json`）；fail code 与三个契约文件（manifest / correctness.json / metrics.json）见 **`docs/gate-report-v1.md`**。
- **设备互斥**：`backend/app/services/devices.py` 读 `npu-smi info -t proc-mem` 判占用（**HBM 占用不是判据**：8 张卡都常驻 ~3.4 GB 却无进程），plan 选空卡；`scripts/nm_test.sh` 在远端 `flock -n $OPX_REMOTE_ROOT/.locks/dev<N>.lock` 拿独占锁，拿不到 **exit 3（env_failed）不跑**。实测：`OPX_LOCK_OK devices=0 mode=exclusive`，锁文件与 owner 记录落在 `tk36:/data/lfx/op-agent/runs/.locks/dev0.{lock,owner}`，跑完容器无残留。
- **版本对齐**：源码快照改成**内容哈希**并带 `truncated` 标志（原 size+mtime 会被 rsync/touch 误判「版本不一致」）；与 `nm_test.sh` 里那份实现**逐字节一致**（`test_snapshot.py` 直接抽脚本里的代码对比）。判据档落到 `projects/tasks.criteria_ref`（`档名@档版本`）；镜像 digest 回填 **94/98**（`backend/scripts/sync_image_meta.py`：`docker images --digests` 30 条 + 本地构建镜像 image ID 兜底 64 条）；`runs.suite_version_id`（指向不存在的表）→ `suite_ref`。
- **事实修正（F1–F9）**：默认镜像从「起得来但跑不了算子」的 `ascendhub/cann:9.2.0-beta.2-910b` 切到 `lfx-op-dev-910b:torch210-cann910`（并摘掉工具链镜像的 `is_op_dev`、回填 torch/digest）；P0 手工登记的假 usage 行标 `is_demo=1`；删双真相源的 `tasks.spec_json`（旧值搬进 `spec_versions`）；501 的理由改成「四动作适配器未实装」（不再说容器没验过）；V5 改 ✅（方案 §9.1 已回填）。
- **CANNBot skills 落地**：`scripts/install_skills.sh` 装进**任务工作区**（原来在 Hermes scratch，空闲 24h 会清；且 symlink 指向同样会被清的 `~/.cache/cannbot`）→ 40 skills / 3 agents / 1274 文件、零残留 symlink、`ops/dsh/skills.lock` 记 commit + 逐文件 sha256。
- **控制台自测与一键起服务**：`backend/tests/` 38 项（迁移往返 + 数据修正、plan 缺项分支、门禁 fail code、快照一致性、501）；`scripts/dev.sh` 一键起前后端。
- **完整链路实测**（run `r20261015-1040-test`，task #2）：`nm_test.sh --action test` → 取锁 → 容器里跑 `test.sh`（当时还是 NOT IMPLEMENTED 存根，exit 2）→ 回读产物 → manifest 带 digest/快照/判据/设备锁 → 自动登记 run #1 → 门禁判 **fail**：`E_EXIT_NONZERO`、`E_REQUIRED_ARTIFACT_MISSING(correctness)`、`E_SUITE_REF_MISSING`、`E_CORRECTNESS_REPORT_MISSING`。**「存根跑过一次」不会被判过** —— 这正是 M1 要填的洞。

## P1.5-a 实测记录（2026-10-14，容器闭环）

- **结果**：走正式管线 `scripts/nm_test.sh` → `VERDICT npu: OK` + `VERDICT torch_npu: OK`，`torch 2.10.0+cpu / torch_npu 2.10.0.post4 / npu devices 8`、device 0 = `Ascend910B1`；exit 0、产物回读、`manifest.json` 带 sha256（run `r20261014-205455`）。
- **可用配方**（已固化进 `ops/nm.conf` + 镜像 #52 的 `env_recipe`）：`--privileged` + `-e ASCEND_RT_VISIBLE_DEVICES=0,...,7` + 11 个设备节点 + `driver/dcmi/npu-smi/ascend_install.info` 挂载 + `--entrypoint bash`（绕开自带 ENTRYPOINT）+ 容器内 source 完整 CANN `set_env` 链 + 容器执行必须带 `timeout`。
- **镜像结论**：`lfx-op-dev-910b:torch210-cann910`（openEuler 24.03，torch 2.10 + torch_npu 2.10.0.post4）＝能用那个；`cann:9.2.0-beta.2-910b` 只适合做编译/工具链底座（npu-smi 正常，但**不带 torch**）。两者都已在控制台标记为算子开发镜像并写清 note。
- **踩到的 6 条（全进脚本/配置的注释）**：
  1. 只挂 `davinci_manager/devmm_svm/hisi_hdc` → `dcmi module initialize failed ret=-8005`；
  2. 缺 `davinci0..7` 与 `/etc/ascend_install.info`（toolkit 靠它找驱动）→ 看不到卡；
  3. 自带 CANN 的镜像光靠挂载不够，要 `--privileged`；
  4. 不给 `ASCEND_RT_VISIBLE_DEVICES` 时 torch_npu 只认 **1** 张卡，而同一容器里 `npu-smi` 能看到 8 张 —— 给了才是 8；
  5. 镜像自带 ENTRYPOINT 会 `source` 一堆 set_env 并**卡死**；且用 `--entrypoint bash` 之后命令里不能再传 `bash`（`bash bash -lc` → exit 126）；
  6. **`timeout` 只杀 `docker run` 客户端，容器会继续跑并占住卡** —— 必须显式 `docker rm -f` 兜底清理。我泄漏过两次，其中一次反过来导致后续 `dcmi ... device is used ret=-8020`（自己堵自己的路）。
- **脚本加固**：`nm_test.sh` 加了 `--name opx-<run_id>` + 超时兜底 `docker rm -f`、`timeout` 包裹容器执行、`OPX_ENTRYPOINT` 开关、`OPX_DOCKER_ARGS`；**加载顺序改成「先 source 配置再补默认值」**（否则配置文件里的 `${VAR:-默认}` 会被脚本已赋的值静默吃掉——我为此把工作区指错过一次）；修掉 `$RC，` 这类「变量名后紧跟全角字符」被 bash 吞进变量名的坑。
- **判定口径修正**：`smoke.sh` 的 `VERDICT` 改成**按输出判定** —— 实测 `npu-smi` 会「打印报错却返回 0」，只看退出码会把 FAIL 判成 OK。

## P1.5-b 实测记录（2026-10-14，契约最小集）

- **迁移真跑**：老 `runs` 5 行 → `agent_jobs`（语义正确：它们本来就是智能体调用）；新 `runs`（真实执行）建空；`artifacts` 按既定口径清掉；`tasks` 加 `current_spec_version_id`/`next_action`；`host_images` 加 `digest`。降级往返实测无损。
- **seed 落首版规格**：`elementwise_add` v1（sha256 `b29b1852…`，含 assumptions/unknowns）+ design Loop #1 + `next_action`；**不造假 run**。
- **四动作契约**：`POST /api/tasks/{id}/actions` 算六步 plan，实测 `ready=true`；`dry_run=false` → **501**（故意不实现）。**前置检查真抓到问题**：先报 `项目未绑定执行机（hosts）`，修 seed 绑定后才齐备。
- **前端**：任务详情按「过程（agent_jobs）/ 结果（runs）」分列 + 四动作 plan 面板（浏览器实测点通）；SSE 端点改名 `/api/agent-jobs/{id}/events`。
- **回归**：拆分后 agent 调用 job#6 `ok`（in 1351 / out 195 / cache 4096），`runs` 保持 0。
- **契约模板入仓**：`ops/workspace-template/{build,test,benchmark,profile}.sh` + `smoke.sh`。

## P1 实测记录（2026-10-14）

- **CLI / SSE 直连**：事件流按类型逐条到：`session` → `text` → `tool_call: bash ls -A1` → `usage{in 5451, out 70}` → 正文 → `usage{in 164, cacheRead 5376}` → `final` → done。
- **台账回填**：`status=ok`、`exit_code=0`、`dsh_session_id`、`tokens_in=5615 / out=146 / cache_read=5376`。
- **浏览器端到端**：界面填 prompt → 点「发起」→ 面板实时刷出思考 / 工具调用 / step 级 usage / `✔ 最终回答`。
- **期间修的 3 个真 bug**：① 端点写成同步 `def` → 线程池里没事件循环，`asyncio.create_task` 直接 500；② 进程重启后孤儿 `running` 永不完结；③ `dsh_jsonl_path` 少一个尾横线。

## dsh 接模型端点（opencode go）—— 已跑通并固化

- 固化位置：`ops/dsh/llm-opencode-go.patch.yml` + `ops/dsh/README.md`（6 条实测要点）。
- 两个反直觉点：① 必须带 **`x-opencode-session`** 头（缺了 400 `MissingSessionID`；任意稳定 id 即可）；② 不能用 `dsh-llm-deepseek` 适配器，要用 **`pi-ai` + `api: openai-completions`**。
- `--patch` 语义：**同 id 的裸条目＝覆写整行 `config`**（不深合并）；默认模型是独立条目 **`agent-default-model`**。
- 凭据只从环境变量取（`OPX_LLM_KEY`），不进库、不进仓库。

## 待办（建议顺序）

> **2026-10-19 口径调整（用户定）**：**M2 案例库延后，放最后做**（原排在「下一步」）。理由：它工作量最大且可分批（框架 / 内容可分开），先把闭环与页面能力铺开更划算。下方顺序已按此重排；`PLAN-web-console-v1.1.md` §13.3 的排期同样以此为准。

1. **自动优化增强**（M3 余项）✅ **机制已闭环（2026-10-19，M3-b）**：`POST /api/loops/{id}/optimize`（agent 在工作区改代码 + 逐文件 diff 证据 + 一键回退）+ 前端入口 + 从建议「开新 Loop 并让 agent 执行」。**三轮真机 A/B 已跑完**：Loop #2（TILE 4096→8192，no_significant_change）、Loop #3（满块 DataCopy，regressed）、**Loop #4（自适应 tile，64k −30.4%、1M 零退化 → 采纳）**。**余下**：把「按形状补测 + 自适应阈值」做成常规动作（新案例按各自 shape 扫一遍）；其它 shape 的阈值未扫。
2. **P0-2 对话页** ✅ **已完成（2026-10-20，自研 B）**：会话/消息 + 复用 `agent_runner` + 逐 token 打字机；任务会话默认挂 skills、轻对话用空目录。**余下**：附件/终端/审批 UI、无任务通用对话、会话搜索。
3. **P0-3「任务开始」页** ✅ **已闭环**：草稿生命周期 + 对话式澄清（**走 dsh 会话**，`--session-id` 续会话）+ **逐 token 流式**（两段式契约 + `prose_relay` + 插件回流）+ 定版门禁（`/finalize`）。增强项（未做）：澄清会话挂 skills、`/context` 装载清单每轮拼进 prompt、`agent_jobs.context_ref`、**过程区视觉重设计（要先出样机）**、正式 agent run 也开流式。
4. **P2 结果视图** ✅ **已完成（2026-10-16）**：三类报告渲染 + 极限对比/Roofline + 结构化优化建议 + 导出。
5. **补齐四页占位** ✅ **全部完成（2026-10-19）**：**消耗情况**（`GET /api/usage`，排除 `is_demo`）+ **任务详情「轮次与关联 / 交付」** + **测试管理（读侧）**（`GET /api/tasks/{id}/cases`）+ **阶段看板**（`GET /api/stages` + `POST /api/tasks/{id}/stage/sync`）+ **API 管理**（`endpoints` 表 + CRUD + 探针）。
6. **四动作「点一下就真执行」** ✅ **已完成（2026-10-20）**：`dry_run=false` 起异步 `action_jobs` 作业，跑 `nm_test.sh` + 找回 run + 门禁 verdict + 前端实时日志/取消。**余下**：取消传到远端（现在只杀本地进程组）。
7. **小尾巴**：证据回读压缩（`expected=reject` 的用例即使通过也会把输入 bin 拉回本机，`add_bad_002` 一次 8 MB —— 改成「拒绝类用例只留 stdout/stderr 片段」）；`interrupted` 状态胶囊映射；`web/` 工具函数复用面整理。
8. **反馈 CANNBot 上游**：`ascendc-direct-invoke-template` 的 add 样例在「每核 >1 个 TILE」时不推进 GM 偏移（M1-a 实测踩到，见上）；顺带把「样例 host 的 tiling 恰好掩盖这个洞」写进反馈。
9. **部署硬化（D4 定后）**：~~600 环境文件注入 `OPX_LLM_KEY`~~ ✅ 已接（`~/.config/op-agent/env` + `scripts/dev.sh` 自动 source，2026-10-15）；**launchd 自启仍未做**。
10. **M2 案例库（最后做 · 2026-10-19 定）**：**10 个案例全部真机全链路（硬性，2026-10-20 定）**；**交付形态 = 独立 zip bundle + `scripts/import_cases.py`**（仓库不预置案例；页面已做说明占位）。**算子清单不写死**（同场景内可替换，但保持 10 个 + 四场景覆盖 + 全部真机）。框架（导入后索引 + 列表 + 一键发起）1–2 天 + 10 个案例内容与真机调试 **15–25 天**。**建议顺序（风险前置）**：relu（验框架）→ attention 核心 → flash-attention → matmul → batched matmul → softmax → layer_norm → rms_norm → gelu → silu。依赖：各案例基线支持范围确认（§4.6）。
11. **任务页审查遗留（2026-10-21）**：① `agent_jobs` 无 `error` 列 → 失败摘要只在 SSE/`/live`；② `/agent-jobs/{id}/events` 只重放最近 32 个结束 job；③ 线上若项目默认 host/image 错配，相关任务 PATCH 会 400（先修项目配置）；④ SSE 断流重连未在真实代理/浏览器下压测；⑤ D8 预算（`fix_rounds_max` / `budget_per_action`）仍只落库；⑥ 多项目假设（App 固定 `projects[0]`）与 `stage` 手工 PATCH 语义待产品确认；⑦ 审查待定项未复现：`perf_diff` 异常安全、`delete-preview` 大目录耗时、`_delete_files` 非 OSError。~~create_spec 改名不迁目录~~ / ~~侧栏切走丢编辑~~ 已修（定版后 op_name 冻结、侧栏导航前 flush）。
12. **任务外页面审查遗留（2026-10-21，收口后）**：**已修**：① key 明文 → 本机密钥库；② 探针 per-host 并发锁 + 部分失败告警落 `probe_error`；③ 会话僵尸 running 删除做 stale 兜底（对齐 task_deletion）；④ `is_active=false` 明确拒绝、端点名大小写不敏感；⑤ SSH 私钥上传分块读 + 严格 PEM/OpenSSH 头校验；⑥ 模型路由「页面字段不改变实际调用」在 API 页明确提示。**仍留（接受/后续）**：① ~~模型路由仍由 dsh patch 写死~~ ✅ 已修（2026-10-21 晚，按当前端点动态生成 patch）；② 对话发送锁是单进程 `threading.Lock`（多进程部署需 DB 唯一约束）；③ 旧 `changes_json`（无 `after_sha256`）回退跳过冲突校验；④ 探针重定向/SSRF 限制、删除服务器对 ActionJob 的引用检查/解绑路径、私钥路径回显、消耗页端点/时间维度、TaskRecords 金额精度、前端自动化测试、真实代理下 SSE 压测；⑤ M2 案例内容 0/10、一键发起落规格、多项目假设。

## 未决与风险

- **排期（2026-10-19 用户定）**：**M2 案例库延后，放最后做**（原排在「下一步」）—— 见上「待办」；内容口径（`PLAN §13.2` 的 10 案例清单 + 验收 #11「内置 ≥10 案例」）**不变**，只改先后。
- **案例口径（2026-10-20 定）**：**10 个案例全部真机全链路（硬性）**，不设 ≥3 底线、无离线降级出口；**算子可在同场景内替换**（保持 10 个 + 四场景覆盖）。排期 15–25 天；M2 完成 = 10 个全部通过门禁。**风险**：attention / flash-attention 最高，已按「风险前置」排序。
- **D4 ✅ 已定**：本机（这台 Mac）+ 单人使用 → SQLite 保持；不做登录/鉴权；V6 关闭。
- **D6 ✅ 已定**：首条流程算子 = `elementwise_add`；`complex64 mm` 留作后续代表验证。
- **D8 ✅ 已定（2026-10-15，M1-a 实测后修正；**同日需求调整后再次重定**）**：同一 Loop 最多 **3 轮**受控修复；~~一个任务总共不超过 6 次~~ → **改按「每案例 / 每 Loop」计**（每案例 `build`/`test`/`benchmark`/`profile` 各 1–2 次）；**环境/契约类失败不计入**（设备被占 `exit 3`、容器干净导致的契约缺口、存根式跑法 —— 属 §4.4 的基础设施/环境问题），但仍要如实登记 run 行。方案 §11.2 已同步。**当前用量**：task #2 计入 3 次（build 1 / 暴露算子缺陷的 test 1 / 通过的 test 1），修复 1 轮（≤3）；M1-b 用 2 次（benchmark / profile）；**M3-a 的 perf Loop #2 用 4 次（优化前 test/benchmark + 优化后 test/benchmark，各 1 次，均在「每案例四动作各 1–2 次」额度内）**；Loop #3 用 4 次（另含 3 条环境/冒烟 run，不计入）；**Loop #4 用 4 次**。跨 Loop 各自独立计。
- **机器是共享的**：tk36 load 长期 30+，还有别人的容器（vllm 等）在跑。性能采集必须**按设备互斥**、避开被占用的卡；跑完必须确认没留下容器（脚本已加兜底清理）。**2026-10-20 起「互斥」才真正成立**：原来容器固定暴露 0–7、算子写死 `aclrtSetDevice(0)`，锁 dev1 却跑 dev0（实测 `ret=507033`、单例卡 151s）—— 已修成「容器内可见设备 = 锁定的卡」；`perf_diff` 的设备一致性检查仍只比台账字符串，待升级为比实际可见设备。
- **`--privileged` 是必需的**（实测少了它 torch_npu 看不到设备）：这是个安全面取舍，记在这里备查。
- **对话页形态 ✅ 已定（2026-10-20）**：**自研（B）** —— 复用 `agent_runner` + SSE；任务会话默认挂 skills、轻对话用空目录。不反代 `dsh web`（其 WS/ cookie/版本耦合成本高）；若以后需要 dsh 原生终端/附件，可另开「新窗口高级入口」而不进主链路。
- **P0-3 创建向导的对话面板** ✅ 已定（2026-10-15）：**自研轻面板**（复用 headless 的 SSE 事件流）—— 创建向导要「结构化产出 + 字段联动」，反代 iframe 里的 `dsh web` 做不到；长对话页（P0-2）的形态仍是独立待定项。
- **D9 ✅ 已定（2026-10-16，M1-b 实测）**：`msopprof` 四类指标（耗时/数据搬运/存储使用/计算单元负载）**均可采**（详见「M1-b 实测记录」）；门禁对 `profile` 从「只要 `log`」收紧为**必需 `log`+`metrics`**，`W_METRICS_MISSING` → `E_METRICS_REPORT_MISSING`（fail）。命令与字段固化在 `workspaces/cann-ops/tools/run_profile.py`。
- **`runs`/`artifacts` 已不再空**（M1-a 起）：task #2 有 5 条真实 run（`artifacts` 逐条带 sha256），页面「结果」分区能看真数据了。**`stage_events` 自阶段看板起可写入**（`POST /api/tasks/{id}/stage/sync` 按证据推进时留痕）——但**自动推进/打回仍未做**，任务 `stage` 默认仍是种子值，靠看板的「按证据同步」落正。
- **凭据轮换**：opencode go 的 key 曾出现在聊天记录里，需轮换。
- **容器是干净的（每次全新）**：build 产物不跨 run 存在，任何依赖上一动作产物的动作都要自己处理（`test.sh` 已按此补编；`benchmark`/`profile` 同样要处理）。~~若以后要省编译时间，再考虑「按源码快照哈希做远端编译缓存」~~ → **已实装（2026-10-20）**：按 `op + 源码快照 sha256 + 镜像 digest` 做远端编译缓存，`OPX_BUILD_CACHE=0` 关闭、`OPX_FORCE_BUILD=1` 强制重编（见「服务器 / 镜像 / API key 增强」）。

---

## 2026-10-23 任务生命周期 v2 实施记录（P0–P6 已落地；真机验收待跑）

**需求与设计**：用户 2026-10-23 口径（18 条）→ `REQUIREMENT.md` §二 + `docs/plan-task-lifecycle-v2.md`（v2 设计）+ `docs/impl-contract-v2.md`（并行实现契约）+ `docs/templates/算子设计文档模板.{docx,md}`（五段主线 + 可选用例附录，docx 已渲染检查）。

**已实现（代码 + 418 项后端测试 + 前端 tsc/build 全绿）**：
1. **文档输入**：任务开始页模板下载（docx/md）+ 设计文档上传/替换/删除；后端纯 stdlib 解析 docx（段落/表格→Markdown，零新依赖），原文与解析稿落 `<ws>/task-docs/task-<id>/`，DB `task_documents`；创建期对话自动注入文档解析稿（12k 截断）。
2. **用例组确认**：任务对话内整组确认精度/性能用例（对话 prompt 注入四态与 curl 指引）；`POST /api/tasks/{id}/cases/confirm` 校验两文件 sha256 + 写 `tasks.cases_confirmed_*` + StageEvent，并**按证据自动推进阶段**（确认后通常从「需求与设计」进入「开发」）；**四动作 plan 硬门禁**「用例组未确认」（未确认跑不了任何真机动作）；阶段一推导要求已确认。
3. **版本模型**：大版本=规格版本 / 小版本=采纳优化·交付前人工变更；`tasks.version_minor` + `task_versions` 表；证据按 `(spec_version_id, code_snapshot_sha256, suite_ref)` 与 run 匹配 → **优化采纳直接复用候选 run，不重跑**；落新规格 = 大版本+1、小版本归 0、用例需重确认；手工小版本 / 对话确认豁免（`waivers_json`）API。
4. **交付**：新增本地 `delivery` 动作（`POST /api/tasks/{id}/deliveries`）——门禁（版本证据 + 用例确认 + 5 件产物）→ zip（代码/Python 接口/pytest/说明文档/规划文档/manifest/证据）→ run+artifacts+版本 `delivered_at` 冻结；`GET /deliveries`、`/deliveries/{run_id}/download`、`/deliveries/preview`；阶段四由「永远不达成」改为 delivery 门禁。
5. **UI**：任务详情标题 `v大.小`；对话全屏放大（Esc、不断 SSE）；规格修改走对话（移除 JSON 编辑器）；测试管理/测试与结果并列精度与性能两类用例；版本分区任务版本列表 + savepoints；交付面板（门禁预览/生成/历史）。
6. **Agent 协议**：`docs/reference/deliverable-standards.md`（Python ctypes C ABI + CMake 双 target + pytest 口径 + 五件产物路径）；`workspaces/cann-ops/AGENTS.md` 增交付物规范；规格修改/用例确认/开发交付物/版本重跑提示写进对话 prompt（API base 只从 `settings.console_api` 读）。
7. **迁移清理（用户定）**：`scripts/reset_tasks_v2.py`（备份→删 5 个存量任务→清旧算子产物，保留共享资产）；已执行：DB 备份 `backend/opx.db.bak-20261023-032547-pre-v2`、工作区备份 `archives/workspaces-cann-ops-20261023-032547-pre-v2.tar.gz`；DB 任务/run/spec/version 全为 0，`cases/` 仅留 README，旧 ops/tools 基线/上下文空间已清。
8. 控制台后端已用新代码重启（8077），新路由（documents/versions/deliveries/templates/cases confirm）在线；live smoke：模板下载、文档上传/列表、版本接口、web 首页、任务删除均通过。

**真机待验收（代码已就位，需一轮真机）**：
- Python ctypes 调用接口先做 **spike**：同一进程 torch_npu 已初始化 ACL 上下文时，ctypes 直调 host API 是否可用；不通过则按设计退回子进程 wrapper。
- 新任务完整链路：对话生成 spec → 对话确认用例组 → agent 生成 5 件产物 → build/test/benchmark → 采纳优化（小版本+1、不重跑）→ 交付包（版本证据 + 下载）。
- 真机验收前需确认 8077 进程环境里有 `OPX_LLM_KEY`（已从 `~/.config/op-agent/env` 注入当前后台进程）。

### 2026-10-23 真机验收（部分）：build 全链路通过，test/benchmark 被共享机占用卡住

用备份样例（elementwise_add，2026-10-23 清库备份里恢复）走 **live 控制台 API** 验收：

| 步骤 | 结果 |
|---|---|
| 建任务 → 写 spec_draft → `finalize` | OK；落 spec v1、任务版本 **v1.0** |
| 对话确认用例（cases+benchmark 双文件 sha256） | OK；`in_sync=true` |
| `build` plan → 真执行（`POST /api/tasks/1/actions`） | **ready=true；action_job #1 ok，run #1 exit=0，门禁 pass**；容器内真编 + 构建缓存命中；产物 `runs/r1-20261023-112947/manifest.json` |
| `test` plan | ready=false：`没有空闲设备（8 张卡上都有别人的进程）` —— 8 张卡均被 VLLM::Worker_TP 占约 13h，未强占 |
| 交付门禁 preview / POST | 409 且 codes 正确：`E_VERSION_EVIDENCE_STALE`（test/benchmark 未跑）+ `E_DELIVERY_PYTHON_IFACE_MISSING` + `E_DELIVERY_TESTS_MISSING` + `E_DELIVERY_DOCS_MISSING` |

验收脚本：`scripts/acceptance_v2_live.py`（live API；跑完删除验收任务、恢复工作区为空）。
**剩余真机验收**：等有空闲卡后跑 test/benchmark → 交付包 → 优化采纳（小版本+1 不重跑）；以及 Python ctypes 调用接口 spike。


### 2026-10-23 Python 调用接口（交付物）真机编译 spike

用备份样例 elementwise_add 在 cann 910b 算子开发镜像里验证《交付物规范》§2 的可编译/可加载路径：

- 新增 `op_host/elementwise_add_pyapi.asc`（`extern "C"` 导出 `elementwise_add_run` + ABI 版本；不调
  `aclInit/aclFinalize`，生命周期交给 Python 进程），CMake 增加 `SHARED libelementwise_add_host.so` +
  `add_dependencies(elementwise_add elementwise_add_host)`，**未修改共享 `build.sh`**；
- 控制台 `build` 动作真跑：**exit=0、门禁 pass**，容器日志出现
  `Built target elementwise_add_host` → `PYAPI_OK symbol=elementwise_add_run abi=opx-python-abi/1`
  （run `r3-20261023-114033`，`.so` sha256 `8a31e87c…`）—— 证明「可执行文件 + 共享库」双 target 与
  ctypes 符号加载可在镜像内成立；
- **未验证**：torch_npu 初始化 ACL 上下文后实际调 kernel（当时 8 张卡均被 VLLM::Worker_TP 占用，不能强占）；
  有卡后按规范 §2.1 做最小 spike。
- 另发现：**镜像未预装 pytest**（`import pytest` 失败）；pytest 仍是必需交付件，四动作门禁不依赖它。
- 参考实现已归档：`docs/reference/examples/elementwise_add_pyapi.asc`、`python_elementwise_add.py`。


### 2026-10-23 开发内冒烟（smoke）：先编译 + 最小用例跑通，再进正式测试

用户要求：开发对话里算子写完先编译，并用一条最小用例确认能跑通，避免后续 test/benchmark 才发现
构建失败/跑不通。已落地为第 5 个受控动作（不是四动作的替代，不参与交付必需证据）：

- **工作区**：新增 `smoke_check.sh`（动作入口；根目录旧 `smoke.sh` 仍是环境自检，已注释说明别混）+
  `tools/smoke_case.py`（挑最小 required match 用例，复用 `run_cases.py` 的输入生成/候选调用/golden/判据）。
  产物 `smoke.json`（版本对齐字段 + case + verdict）+ `smoke.md` + log。
- **执行器**：`scripts/nm_test.sh --action smoke`（映射到 `smoke_check.sh`，产物类型 `smoke`；仅 `make <op>`
  时靠 `add_dependencies` 顺带编出 `.so` 的既有口径不受影响）；`scripts/nm.sh --action smoke` 可直接用。
- **门禁**：`gate.REQUIRED_ARTIFACTS["smoke"]=("log","smoke")`，`smoke.json.verdict=PASS` 才通过
  （fail code：`E_SMOKE_FAILED` / `E_SMOKE_REPORT_MISSING` / `E_SMOKE_CASE_MISSING`）。
- **阶段**：阶段「开发」门禁由「build pass」升级为「build + smoke 都 pass」。
- **Agent 协议**：对话 prompt / 交付物规范 / 工作区 AGENTS.md 均写明「build → smoke PASS 后才允许
  test/benchmark，也才允许说开发完成；失败先修，设备被占如实报阻塞」；前端执行步骤加「冒烟」。
- **离线验证**：用假算子（CPU 脚本）跑 `smoke_check.sh`，正确挑到最小用例（n=4）并 PASS、产物字段齐全；
  真机 smoke 需空闲卡（当前 8 卡仍被占，未强占）。


### 2026-10-23 端到端验收（合成证据，live 控制台）通过

真机 test/benchmark 依赖空闲卡（共享机 8 卡连续 3 轮被 VLLM::Worker_TP 占用，未强占），先用**明确标注为
合成的 run 证据**在真实控制台/数据库上验证 v2 全链路接线（不代表真机正确性/性能结论）：

- 脚本：`scripts/acceptance_v2_synth_e2e.py`（临时算子 `e2e_synth`，结束后任务/文件全清）。
- 链路：建任务 → finalize（spec v1 / version v1.0）→ 用例组确认（suite `cases@b7ecf5e7aadb596e`）→
  登记 build/smoke/test/benchmark 合成 run（版本三键一致、门禁全 pass）→ `GET /versions` evidence
  `deliverable=true / missing=[]`、阶段推导 = **交付** → 创建并采纳 perf Loop → **小版本 v1.1
  （source=optimization，候选证据不重跑）** → `POST /deliveries` 生成交付包（run #5，3.8KB，13 个 entry，
  含 manifest / Python 接口 / pytest / 说明文档 / 规划文档）→ 下载校验 → `GET /deliveries` 门禁 pass。
- 收尾：任务删除后 DB tasks/runs/versions/spec 全 0，工作区 `cases/` 仅剩 README、`ops/` 仅 chips，
  本地 run 目录已清；`E2E_SYNTH PASS`。
- **仍待真机**（外部条件，非代码问题）：用空闲 NPU 卡跑真实 build+smoke+test+benchmark 与交付包，
  以及 Python ctypes 运行时 spike。


### 2026-10-23 真机端到端验收（真实跑通，elementwise_add）

用户释放卡后，用备份样例 elementwise_add 在 live 控制台跑完整生命周期（任务 #1，保留作验收证据）：

| 步骤 | 结果 |
|---|---|
| 定版 | spec v1 / 任务版本 **v1.0** |
| 用例组确认 | 精度 + 性能两文件确认，`in_sync=true` |
| **build** | run #1，exit=0，门禁 **pass**（容器内真编，`libelementwise_add_host.so` 也由 add_dependencies 顺带编出） |
| **smoke（新动作）** | run #2，exit=0，门禁 **pass**；自动挑最小用例 `add_edge_001`（N=1）**PASS** |
| **test** | run #3，exit=0，门禁 **pass**；**19/19 PASS**（correctness.json 全绿） |
| **benchmark** | run #4，exit=0，门禁 **pass**；`bench_1m` 候选 8.14us / 基线 8.38us（**1.029×**），`bench_64k` 候选 5.48us / 基线 3.80us（0.693×，如实记录，无加速比硬线） |
| 版本模型 | 采纳优化（验收用临时 Loop）→ **v1.1（source=optimization，小版本+1 且不重跑）**；临时 Loop 验收后删除，`task_versions.loop_id` 置空 |
| **delivery** | run #5，门禁 **pass**；`elementwise_add_v1.1_20261023.zip` 33,619 B，18 个 entry（代码/Python 接口/pytest/说明文档/规划文档/cases/manifest 全在），下载校验通过；阶段 = **交付** |
| **Python ctypes 运行时 spike** | 真机 PASS：torch_npu 初始化 ACL 后 ctypes 直调 host 共享库，n=1/4096/65536/1048576 全部 PASS，max_abs_err=0；`OPX_LOCK_OK dev0 mode=exclusive` |

真机 smoke 用的就是新增的 `smoke_check.sh` + `tools/smoke_case.py`；Python 接口 spike 证明《交付物规范》§2 的 ctypes 路线在运行时可行。
`pytest` 脚本仍为交付件；算子开发镜像未装 pytest（已记录在规范 §3）。任务 #1 留在控制台供查看，工作区对应 `ops/elementwise_add/`、`cases/elementwise_add/`、`tests/test_elementwise_add.py`、`docs/elementwise_add/design.md`。


### 2026-10-23 新建算子开发镜像 v2 并设为项目默认

旧基线镜像 `lfx-op-dev-910b:torch210-cann910`（host_images #52）的 pytest 9.1.1 **import 失败**
（缺 legacy `py` 模块，`python3 -m pytest` 报 `No module named 'py'`），v2 交付物里的 pytest 脚本在镜像内跑不了。

- **新镜像**：`lfx-op-dev-910b:torch210-cann910-v2`（image id `sha256:313b20aa34b4…`，14.24GB，基于基线镜像）
  - `pip install py` 修复 pytest（实测 `python3 -m pytest --version` → pytest 9.1.1 可用）；
  - 默认 WORKDIR 从 `/opt` 改为 `/tmp`（`/opt` 下的同名目录会遮蔽标准库/三方库导入；执行侧仍 `-w /ws`）；
  - 其余环境与基线一致：CANN 9.1、Python 3.12.13、numpy 1.26.4、torch 2.10.0+cpu / torch_npu 2.10.0.post4。
  - Dockerfile 在远端构建（`docker build -t lfx-op-dev-910b:torch210-cann910-v2`）。
- **控制台登记**：探针扫描后 `host_images #125`，已标记 `is_op_dev=1`、`is_test=1`，写 digest
  `local-id:sha256:313b20aa34b4…` / cann / python / torch 版本 / env_recipe，**项目 #1 默认镜像改为 #125**
  （新增 `PATCH /api/projects/{id}` 接口 + 测试；旧镜像 #52 保留用于回滚）。`ops/nm.conf` 本机默认镜像同步改为 v2。
- **验证**：
  - `POST /api/tasks/1/actions {build,dry_run}` → `image_ref=…-v2`、`image_id=125`、`ready=true`（项目默认生效）；
  - 用 v2 真机跑 `build + smoke`：build OK、**SMOKE_VERDICT PASS**（最小用例 add_edge_001，设备 0 独占锁，
    run `r20261023-124200`，构建缓存命中新镜像 digest）；
  - v2 容器内 `python3 -m pytest --version` → 9.1.1，numpy/torch/torch_npu 导入正常。
