# 变更记录

格式：`版本 — 日期 — 一句话`，下面列**影响运维的改动**（改文件/改配置/改行为），
纯文档修改不列。回退点看 git tag。

---

## v0.1.1 — 2026-11-04 — 清掉明暗切换残留 + 模型侧工作区块对齐算子侧 + 仓库改名

**改了什么**

1. **去掉两侧的明暗主题切换（残留功能）**
   - 模型平台：`model-platform/static/index.html` 删掉 `#theme-btn` 按钮、`applyTheme/initTheme`、
     `THEME_KEY`（`mb-theme`）持久化与点击监听、`initTheme()` 调用。
     **注意**：原来没有 localStorage 时会跟随 `prefers-color-scheme`，所以只删按钮不够——
     不删这段逻辑，系统深色用户仍然会被顶成深色。
   - 算子平台：`operator/web/src/App.tsx` 删掉 `theme` state、`data-theme` 副作用、右上角「暗/亮」按钮；
     `operator/web/dist` 用 vite 重新构建（tsc --noEmit 通过，新 `index-Cy1qBUO3.js`，
     CSS 哈希未变 `index-DHQX1TAc.css`）。前端 `index.html` 本来就静态声明 `data-theme="light"`。
   - 结果：无论系统是深色还是浏览器里残留旧偏好，两个页面都固定浅色（Playwright 用
     `color_scheme='dark'` 上下文实测：`dark_class=false`、`theme_attr=light`、无任何切换按钮）。

2. **模型平台左侧「默认工作空间」区块改成算子侧「📁 CANN 算子开发」同款**（网关注入 CSS，未改业务源码）
   - 原来：36px 渐变方块头像「M」+ 两行文字（默认工作空间 / 小模型迁移计划），透明底。
   - 现在：单行 `📁 默认工作空间 ▾`，`height:36px`、`border:1px solid #e5e3ef`、`border-radius:13px`、
     `background:rgba(251,250,255,.91)`、`box-shadow:0 10px 30px rgba(20,26,46,.09)`、
     文字 14px/400、副标题隐藏、头像隐藏、箭头 14px 右对齐。
   - 箭头也统一了：算子侧是文本字符 `▾`（实心小三角），模型侧原来是细线 chevron 图标，
     并排看形状不一样；现在隐藏 SVG 图标，用 `::after` 输出同一个 `▾` 字。
   - 实测与算子侧卡片逐项一致（36px 高 / 13px 圆角 / 同边框同底色同阴影 / 同字号字重）。
   - 过程中踩到一个坑：CSS 里写 `content:"\1F4C1"` 被 Python 当成八进制转义，
     发到页面变成 `\u0001F4C1`（控制字符 + 字面量），图标不显示。改成**直接写 📁 字面量**。

3. **仓库改名** `tk-unified-platform` → **`tk-operator-platform`**
   （GitHub 旧地址自动 301 重定向；README/LEDGER 里的 clone 地址已同步）

4. **补上 `operator/web/dist/`（v0.1.0 的实际缺陷）**
   v0.1.0 的 README/`.gitignore` 写着「dist 故意入库」，但首次同步时用了
   `--exclude='dist'`，**实际没进库**。后果很严重：`deploy/setup-operator.sh` 用
   `rsync --delete` 同步 `operator/web/`，仓库里没有 dist 就会把线上正常的 dist **删掉**，
   算子前端直接 404。本次把构建产物入库（`index-Cy1qBUO3.js` / `index-DHQX1TAc.css` / `index.html`），
   README 指纹表同步更新到 v0.1.1。

5. **文档修订**
   - `docs/01-architecture.md` 新增 §4.3：没有 agent 的模型平台**未知模型完整链路**（9 步）+
     `npu_shim.py` 7 条规则补丁清单 + 与算子平台的优劣对比。
   - `docs/04-known-issues.md` 第 2 条按实物证据**改写结论**：OBS 上传其实成功了
     （`obsutil ls -s` 实测 758.46MB 对象存在、`weights_image.json` `ok:true`），
     平台记的"未完成"是超时记账错误；补上两层根因与正确修法。

**影响面**

- 需要重启：`tk-gateway-preview`（CSS 注入变了）。算子平台**前端资源**换了一个 js 文件，
  无需重启后端（静态文件按请求读盘）。模型平台只换了 `static/index.html`，同样无需重启。
- 数据/数据库：无变化，不涉及 migration。
- 回退：
  - 网关 `bash deploy/rollback.sh gateway`
  - 算子前端 `cp -a app/web/dist.bak-before-notheme-<ts> app/web/dist`
  - 模型页面 `cp -a static/index.html.bak-before-notheme-<ts> static/index.html`

**验证**

- 网关 `43 passed, 3 subtests passed`（新增 2 个契约测试：工作区块对齐、两侧不得再有明暗切换）
- 浏览器（1600×1000，`color_scheme='dark'`）：两侧 `toggles.found=[]`、`dark_class=false`、
  `page_errors=[]`、无 `http>=400`
- 工作区块实测：`box 177×36`、`radius 13px`、`border 1px rgb(229,227,239)`、
  `bg rgba(251,250,255,.91)`、`shadow rgba(20,26,46,.09) 0 10px 30px`、`fs 14px/400`、
  `avatar_display=none`、`small_display=none`、`icon_content="📁"`

---

## v0.1.0 — 2026-11-04 — 首个可复现基线

融合系统首次入库：1 个外壳 + 2 个平台 + 算力机侧探针，附部署/运维/验证文档。

**入库时线上状态**（用于核对是否漂移）：

| 文件 | SHA-256 前 16 位 |
|---|---|
| `shell/gateway.py` | `f427b024b4a4a9c9` |
| `shell/portal.html` | `78fb21207384477d` |
| `shell/tanku-logo.png` | `cd75e9b829711b8c` |
| `model-platform/app.py` | `071229de59bf99fc` |
| `model-platform/static/index.html` | `92d190b4ff58feba` |

**组件版本**

- 算子平台后端 `op-agent-console 0.1.0`（FastAPI，143 个 .py）
- 算子平台前端 `op-agent-web 0.1.0`（React + Vite，dist 已入库）
- agent 运行时 `@deepseek-ai/dsh 0.2.0-rc.2`（Node ≥ 24.2）
- 算子平台 LLM `deepseek-v4.1-flash-1017` @ `https://tokens.tiankuan.com/v1`
- 模型平台 `_platform/app.py` 1432 行（纯 stdlib，**无 LLM**）
- 外壳 `gateway.py` 24.5KB（纯 stdlib）
- 依赖锁 `deploy/requirements.lock.txt`（34 项精确版本）

**已验证**（详见 `docs/03-verification.md`）

- 算子平台六动作全通（build/smoke/test 22-22/benchmark/profile/delivery），卡 4 真机
- 交付包 `bid_elementwise_sub_20261101_v2.0_20261103_r30.zip` sha256 `5b9d6ce5b7d4…`
- 模型平台真迁移到卡 4（加载/验证/性能/镜像构建全通），仅 OBS 上传受外部隧道故障阻塞
- 外壳 42 项单测 + 浏览器实测无报错
- 他人卡 0–3 全程未受影响

**已知未闭合**（详见 `docs/04-known-issues.md`）

1. 两平台之间无卡互斥（离线原型 10 tests OK，未接入）
2. 模型平台 OBS 上传走死代理隧道（外部故障）
3. 模型平台命令拼接未净化
4. 入口匿名可写（知情决定，上线前必改）
5. 整机冷启动、反代真实链路、数据切换 均未测

---

## 模板（下次改动用这个）

```
## v0.1.x — YYYY-MM-DD — 一句话

**改了什么**
- 文件 + 行为变化

**为什么改**
- 触发问题（附现象与根因）

**影响面**
- 哪些组件要重启 / 哪些数据要迁移 / 回退是否涉及 schema

**验证**
- 跑了什么，结果如何（数字）

**回退**
- bash deploy/rollback.sh <组件>
```

### v0.1.0 修正记录（同一 tag 内）

首次推送后发现两处遗漏，已修并**移动 tag**（仓库刚建、尚无人 clone）：

1. **漏了算子工作区模板** `operator/workspaces/cann-ops/`。少了它
   `tests/test_deployment_run_root.py` 找不到 `workspaces/cann-ops/scripts/nm.sh` 而失败。
   现已入库（含 scripts/cases/criteria/tools/docs 与 5 个算子实现，排除 `build/` 与会话目录）。
2. **shell 测试路径失效**。三个测试用 `Path(__file__).with_name()` 找 `gateway.py`/`portal.html`，
   那是开发期「测试与源码同目录」的写法；入库后变成 `shell/` + `shell/tests/` 分层就找不到了。
   新增 `shell/tests/_paths.py`，两个布局都兼容。

顺带把 `.gitignore` 里的运行数据目录改成前导 `/` 锚定，避免误伤 `operator/workspaces`。

**验证**（在 172.16.111.102 上，从 GitHub 拉一份干净副本跑）：
`shell` 42 passed + 3 subtests；`operator/backend` 571 collected、rc=0；`model-platform/app.py` 1432 行 AST 通过。

### v0.1.0 第二轮修正：部署脚本改端口的逻辑其实是坏的

自测发现 `setup-shell.sh` 里改上游端口那段**从来没生效过**：

- 写的是 `sed s/("127.0.0.1", 8090)/…/`（双引号）
- 源码里是 `('127.0.0.1', 8090)`（**单引号**）
- 一条都没替换成功，而脚本只 `warn` 了一句就继续往下跑

后果：换端口部署后网关仍然打 8090/18078，两个页签全 502，而且日志里没有任何报错。

**修法**：改用 python 正则 `subn` + 断言「必须正好命中 1 处」+ 回读 grep 双重确认，
替换不到直接 `die`。同时新增两条离线自测把这段逻辑钉住：

- `deploy/selftest-render.sh` —— 用假 site.env 渲染三个 unit 并逐项断言 16 项，实测全过
- `deploy/selftest-run.sh` —— 起两个假上游（28090/28078）真跑一遍，验路由与 404，实测 5 项全过

两条都在 `/tmp` + 非默认端口上跑，不安装 unit、不重启服务、不碰生产。
