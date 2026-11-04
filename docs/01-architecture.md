# 架构与边界

## 1. 四台机器/四个角色，别搞混

| 角色 | 现场实值 | 跑什么 | 边界 |
|---|---|---|---|
| 浏览器 | 内网任意 PC | 只连外壳 | 必须能到 `172.16.111.0/24` |
| **通算机（落地机）** | `172.16.111.102` / `ubuntu-zg` | 外壳 + 两个平台后端 | 全部服务只监听本机回环，只有外壳对外 |
| **算力服务器** | `61.241.77.3:31236` / `bms-fae` | 昇腾 910B ×8、模型权重、算子容器 | 没有 Web 服务，只被 SSH 派发 |
| 跳板机（运维用） | 公网入口 + 端口转发 | ssh / scp | 不是运行组件，不装任何服务 |

> 现场真值：**通算机是 `172.16.111.102`，算力机是 `61.241.77.3:31236`**。
> 换环境时这两个地址都要改，且**算力机上的两个数据根路径必须与通算机配置里写的完全一致**
> （`/data/model_corpus`、`/data/lfx/op-agent`），因为容器挂载用的是同一路径字符串。

## 2. 进程清单（一共 3 个服务，1 个端口对外）

```
0.0.0.0:18080   tk-gateway-preview.service      /usr/bin/python3 shell/gateway.py
127.0.0.1:18078 tk-operator-preview.service     control-venv312/bin/python -m uvicorn app.main:app
127.0.0.1:8090  model-migration-platform.service /usr/bin/python3 model-platform/app.py
```

三者都是 `Restart=on-failure`/`always`，`UMask=0077`。开机自启由 systemd 保证。
**没有任何一个服务需要 root 跑**（都是 `User=hzzg`），只有装 unit 时需要 sudo。

## 3. 外壳到底做了什么（`shell/gateway.py`，单文件 24KB）

### 3.1 路由

`route()` 只认两个前缀，多一个都不认：

| 请求 | 行为 |
|---|---|
| `GET /` | 读同目录 `portal.html` **无条件**返回（缺文件才 503） |
| `GET /brand/tanku.png` | 读同目录 `tanku-logo.png`，网关自己发，不打上游 |
| `/operator/*` | 剥掉 `/operator` 前缀，转发到 `127.0.0.1:18078` |
| `/model/*` | 剥掉 `/model` 前缀，转发到 `127.0.0.1:8090` |
| 其他 | 404（防止用户绕过页签直接访问别的路径） |

上游地址在 `main()` 里写死为 `{'model': ('127.0.0.1', 8090), 'operator': ('127.0.0.1', 18078)}`，
且 `make_server()` 会断言「必须正好两个回环上游」——**这两个端口若改动，`setup-shell.sh` 会用 sed 重写**。

### 3.2 为什么要重写响应体（三个真实踩过的坑）

| 坑 | 现象 | 修法 |
|---|---|---|
| 子路径资源找不到 | 算子前端是 Vite 构建，HTML 里写的是 `/assets/x.js`，套在 `/operator/` 下就 404 | `rewrite_text()` 把**引号或 `url(` 开头的** `/api`、`/assets`、`/static` 前插 `/operator`（故意窄，不做全文本替换） |
| 改了样式页面还是旧的 | 算子页只有 `Last-Modified`/`ETag`，没有 `Cache-Control` → 浏览器启发式缓存旧 HTML | 对**被重写过**的 HTML 强制 `Cache-Control: no-store, must-revalidate`，并剥掉 `ETag/Last-Modified/Expires`，同时不转发 `If-None-Match/If-Modified-Since` |
| 注入的 `!important` 压不过业务样式 | 算子用了 Tailwind，`!text-[10px]` 是层内 `!important` | 注入的 CSS 放进 `@layer tk-unified{...}`，并放在 `<head>` 起始；`body,body *` 权重 (0,0,1) 低于 `body table *` (0,0,2)，正好让表格 13px 覆盖全局 14px |

### 3.3 统一视觉是「注入」，不是改源码

`apply_visual_theme(data, name)` 给两侧各注入一段 `<style id="tk-unified-theme-*">`。
**业务前端源码一行没改**——这是本方案能随时回退的前提。要调视觉只改 `gateway.py`，
改坏了 `rollback.sh gateway` 一条命令回去。

定稿的视觉参数（用户三轮反馈后确认，改动前先问）：

| 项 | 值 |
|---|---|
| 主色 | `#6557d2`（深 `#5144bd`，浅底 `#f0edff`） |
| 正文 / H1 / 控件 / 表格 | 14px / 24px / 13px / 13px |
| 导航字重 | `400`（两侧一致） |
| Logo | 118×29（源图 180×44，**不上采样**） |
| 顶栏底色 | `rgb(247,246,251)`（与内容同色，无渐变带） |
| 侧栏宽 | 200px |
| 表格 | 短列 nowrap、首列 `max-width:300px`、操作列 `min-width:270px` |

### 3.4 写闸与安全边界

两个开关，默认可写（`0`）：

```
WORKBENCH_OPERATOR_READONLY=0   置 1 → 算子侧所有非 GET 返回 503
WORKBENCH_MODEL_READONLY=0      置 1 → 模型侧所有非 GET 返回 503
```

`OPERATOR_READONLY=1` 时额外拦 `GET /operator/api/proto`（这个 GET 会写文件，是已知副作用）。

真正的安全边界只有两层，**都不强**：

1. `WORKBENCH_ALLOWED_CIDRS` —— 来源网段（现场 `127.0.0.0/8,172.16.111.0/24`），不匹配返回 **403**
2. `WORKBENCH_ALLOWED_HOSTS` —— Host 头白名单（现场 `*`），不匹配返回 **400**

> 实测教训：反代报 400 时先怀疑 Host 白名单，别去查 CIDR——CIDR 拒绝是 403，两者现象不同。
> 另外写操作会校验 `Origin` 与 `Host` 一致，不一致 403（CSRF 保护，曾把我自己的 curl 拦下来）。

## 4. 两个平台的本质差别（排查时最重要的一张表）

| | 算子开发平台 | 模型迁移工作台 |
|---|---|---|
| 后端框架 | FastAPI + uvicorn | 纯 `http.server.ThreadingHTTPServer` |
| 代码规模 | 143 个 `.py`（约 6MB）+ React 前端 | **单文件 1432 行** |
| 数据 | SQLite `data/opx.db` | SQLite `_platform/platform.db`（jobs/events） |
| **有 agent** | **有** | **没有** |
| agent 实现 | `@deepseek-ai/dsh` 0.2.0-rc.2（DeepSeek 官方 CLI） | 无 |
| **调用 LLM** | 是，`deepseek-v4.1-flash-1017` | **否**（连 HTTP 客户端都没 import） |
| "智能"来源 | LLM + 工具调用 + skills | 配方库 + 坑库关键词打分 + 远端真推理 |
| 派发方式 | SSH 到算力机跑 `scripts/nm_test.sh` → docker 容器 | SSH 到算力机跑 `_probe/probe.sh` / `mk_image.sh` |
| 独占机制 | `flock` 在算力机 `/data/lfx/op-agent/runs/.locks/` | 进程内 `_CARD_BUSY` 字典 + `npu-smi` 探空闲卡 |

### 4.1 算子平台的 agent 链路

```
页面操作 → FastAPI 建 agent_job
        → agent_runner.build_command()
             dsh --profile headless
                 --patch app/backend/.runtime/dsh-llm.patch.yml   ← 按「API 管理」页当前端点动态生成
                 --patch ops/dsh/opx-typewriter.patch.yml         ← 逐 token 流式插件
                 --json "<任务>"
        → DSH_HOME=app/workspaces/.dsh   （会话 JSONL 落这里，是真相源）
        → 子进程 JSONL 事件流 → SSE 推给页面
```

关键设计：**模型路由和 Key 必须来自同一个端点**。历史踩坑记录（在 `llm_config.py` 注释里）：
静态 patch 写死 opencode 网关、页面配 DeepSeek 官方端点时，页面 key 被灌给 dsh，
dsh 拿 DeepSeek key 去请求 opencode 网关 → `401 Invalid credential` → 页面报「本轮没有得到回答」。

dsh 版本有个坑：`dsh` 是指向 `studio-dsh-launch.sh` 的软链，脚本里 pin 了 Node ≥ 24.2。
否则 `lib/bin.js` 在 `import.meta.main` 不可用时**静默 exit 0**，Studio 侧只看到
「DSH exited with code 0」。装完必须验一遍 `dsh --version` 有输出。

### 4.2 模型平台为什么"看起来像 agent"

它的 7 个 phase（resolve → match_recipe → prepare_env → adapt → validate_infer → quality → deliver）
全部来自 `emit()` 写进 `events` 表然后 SSE 推给前端。**没有一句是模型生成的**。

真实的三层能力：

1. **配方库** `_manifest/recipes.json` —— 命中就走 fast route，直接复用上次成功的加载路线与依赖（页面会显示「历史耗时 18.6s」）
2. **坑库** `_manifest/pitfalls.json` + `diagnose()` —— 纯**关键词子串打分**，按 bucket A/B/C/E 加权取 top3
3. **远端真推理** —— `_probe/probe_model.py` 在算力机上用 `torch_npu` 真跑一遍

判断依据（可复核）：`app.py` 的全部 import 只有
`argparse json os re sqlite3 subprocess threading time uuid http.server urllib.parse`。
没有 `requests`/`httpx`/`urllib.request`，**物理上无法发任何 HTTP 请求**。

### 4.3 没有 agent，未知模型是怎么迁移的（完整链路）

没有任何"模型自己思考"的环节，全部是**规则 + 配方复用 + 经验库检索 + 远端真跑**：

```
① find_corpus_dir(model)        语料库有没有这个模型？目录名把 / 换成 __ 匹配
   ├─ 有 → 直接用
   └─ 没有 ↓
② guess_category(model)         按名字关键词归类（nlp / cv / audio / multi-modal / scientific）
   + download_model()           在算力机上 modelscope.snapshot_download 到 /tmp/ms_dl
                                · 同模型加互斥，避免 N 份并发下载抢带宽
                                · 并行查体积，超过 MAX_MODEL_GB(512GB) 立刻 kill
                                · 归位到 /data/model_corpus/<cat>/<org__name>/
③ load_recipes()                查有没有"已验证配方"
                                来源：_probe/<模型>/recipe.json（优先）或 probe.json，
                                只要 loaded=true 就算配方命中
   ├─ 命中 → fast route：直接复用上次的加载路线/依赖/耗时，跳过画像
   └─ 未命中 ↓
④ profile_model(mdir)           本地画像：文件数、有哪些标准 config、arch、权重文件数
   （只是数文件和读 config.json，不是模型推理）
⑤ wait_for_card()               排队抢卡（进程内 _CARD_BUSY + npu-smi 探空闲）
⑥ 远端 probe.sh → probe_model.py 在算力机容器里**真加载 + 真推理**
   · probe_model.py 会 import npu_shim 打 7 条兼容补丁（见下）
   · 输出里出现 VERDICT=LOAD-OK 才算过
⑦ 远端日志用 STAGE_RULES 正则映射成阶段事件 → 这就是页面上"看起来像 agent"的进度
⑧ 失败时：classify_error() 分桶 + diagnose(err) 在 41 条坑库里按关键词打分取 top2
⑨ 成功时：perf.sh 测 P50/P95/P99 → mk_image.sh 造权重镜像传 OBS → make_delivery 出交付包
          → 配方归档，**下次同模型就走 fast route**
```

**"自动适配"到底自动在哪**：`compute-side/_probe/npu_shim.py` 里有 **7 条规则补丁**，
每条都对应一个实测踩过的坑，带现象/根因/影响面的注释：

| # | 补丁 | 治什么 |
|---|---|---|
| P1 | `modelscope.verify_device` 接受 npu | 白名单写死 cpu/cuda/gpu，传 npu 直接 assert |
| P2 | `torch.cuda.*` 重定向到 npu | 模型代码硬编码 `.cuda()` |
| P3 | 屏蔽 `npu-smi` 缺失告警 | 容器里没 npu-smi 会刷屏 |
| P4 | transformers 远程代码免交互 | 卡在 `Do you wish to run the custom code? [y/N]` 被答 N |
| P5 | 补齐旧模型缺的 `all_tied_weights_keys` | 新版 transformers 属性缺失 |
| P6 | 补齐 config 缺的生成参数默认值 | `forced_bos_token_id` 等 AttributeError |
| P7 | 补齐 torchaudio 被移除的旧 API | 旧音频模型 import 失败 |

超出这 7 条的坑，平台**不会自己修**：它把 `datasets`/桶位/坑库建议推给你，
由人改配置或改代码，修好之后把结论回写 `adaptations.json` / `pitfalls.json`——
下次遇到同类就能自动命中。所以它的"经验增长"是**人写进去的**，不是模型学出来的。

**这套东西的好处与代价**（和算子平台对比时最有意义）：

| | 好处 | 代价 |
|---|---|---|
| 规则 + 配方 | 同样输入必得同样结果，可审计、可复现、零 token 成本 | 遇到 7 条补丁覆盖不到的新坑就停住，等人介入 |
| 坑库关键词打分 | 命中即给现成根因与处置方向，比人翻日志快 | 只能匹配"见过的现象"，同义不同词的报错会漏 |
| 远端真跑 | 判据是真实 LOAD-OK / 真推理，不是猜测 | 每个模型都要占一张卡，慢；没法像 LLM 那样"先推理再试" |

## 5. 算力机侧布局

```
/data/lfx/op-agent/                    ← 算子平台用
  runs/<run-id>/{run.sh,lockrun.sh,out/,ws/}
  runs/.locks/dev<N>.lock              ← ★ 跨进程共享锁，绝不能删/替换
  op-agent-console-20261101 -> ...     ← 软链，容器内与宿主同路径

/data/model_corpus/                    ← 模型平台用
  nlp/ multi-modal/ audio/ cv/ scientific-computing/   ← 权重（143GB）
  _probe/{probe.sh,mk_image.sh,npu_shim.py,perf_test.py,probe_model.py}
  _probe/<模型名>/probe.json, recipe.json, weights_image.json
```

### 5.1 设备隔离怎么做（这里最容易出事）

- 卡的选择**由锁驱动**：拿到 `dev4.lock` → 传 `ASCEND_RT_VISIBLE_DEVICES=4`。
- **不要**在 `OPX_DOCKER_ARGS` 里写 `--device=/dev/davinciN`。历史 bug：锁了卡 4 却挂载 `/dev/davinci2`，
  容器里看到的设备号和锁的不一致。现场已从 `ops/nm.conf` 删除该参数，只剩 3 个管理设备
  （`davinci_manager` / `devmm_svm` / `hisi_hdc`）。
- `--privileged` 是**实测必需**的：非特权容器下 `npu-smi` 报 `dcmi module initialize failed ret=-8005`，
  容器 exit 187。这是宿主驱动 26.0.rc1 的行为，不是配错了。
- 真机验证不能只看 `npu-smi` 抓到进程（时间太短抓不到）。可靠判据是**负例/正例对照**：
  传 `ASCEND_RT_VISIBLE_DEVICES=99` 必须 `npu_device_count=0` + `aclrtSetDevice failed ret=107001`；
  传 `=4`（持锁）必须 `npu_device_count=1` 且输出正确。脚本见 `tools/npu_device_discriminator.sh`。

### 5.2 锁的现状与风险（必须知道）

| 平台 | 独占机制 | 在哪个命名空间 |
|---|---|---|
| 算子平台 | `flock` on `/data/lfx/op-agent/runs/.locks/dev<N>.lock` | **跨进程、跨平台**（算力机文件系统） |
| 模型平台 | `_CARD_BUSY` 集合 + `npu-smi` 探空闲 | **仅进程内** |

→ **两个平台之间没有互斥**。极端情况下算子平台锁住卡 4，模型平台在同一毫秒扫到 4 空闲，
两边可能同时用卡 4。离线原型 `shared-lock` 已验证可行（10 tests OK），但**尚未部署**，
见 `docs/04-known-issues.md` 第 1 条。
