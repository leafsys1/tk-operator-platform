# 一键模型迁移平台

## 部署拓扑（重要）

```
虚拟机 172.16.111.102  ← 平台在这里跑       算力服务器 61.241.77.3  ← 只做算力
├── ~/workspace/model-corpus/                └── /data/model_corpus/
│   ├── _platform/   平台代码                      ├── nlp/ multi-modal/ audio/
│   ├── _manifest/   清单与索引                    │   cv/ scientific-computing/
│   ├── _probe/      配方 + 回收的探针结果          │   （模型权重，143GB）
│   ├── _delivery/   53 个交付包                   └── _probe/probe.sh + npu_shim.py
│   └── _tools/      运维脚本                          （探针执行体）
└── 你直接访问 http://172.16.111.102:8090/
```

- **代码与数据在虚拟机**：平台、清单、配方、交付包、坑库索引都在本机，读写快、可备份。
- **权重与 NPU 在算力服务器**：探针必须在有 NPU 和推理镜像的那侧跑。
- **平台只做派发与回收**：提交任务 → SSH 到算力服务器执行探针 → rsync 回收 `probe.json` / `recipe.json` → 本机做诊断。

### 相关环境变量

| 变量 | 缺省 | 说明 |
|---|---|---|
| `CORPUS_ROOT` | `~/workspace/model-corpus` | 本机代码与数据根目录 |
| `COMPUTE_SSH` | `root@61.241.77.3` | 算力服务器 |
| `COMPUTE_PORT` | `31236` | 算力服务器 SSH 端口 |
| `COMPUTE_KEY` | `~/.ssh/KeyPair-fae2027.pem` | SSH 私钥 |
| `COMPUTE_ROOT` | `/data/model_corpus` | 算力服务器上的语料根目录（权重所在） |
| `COMPUTE_CARD` | `7` | 用哪张 NPU 卡 |

## 启动与自启

```bash
cd ~/workspace/model-corpus
./start-platform.sh            # 前台
./start-platform.sh daemon     # 后台
./start-platform.sh status     # 状态
./start-platform.sh stop       # 停止
./start-platform.sh restart    # 重启
```

**开机自启**（需要 sudo，沙箱内无法代劳，请手动执行一次）：

```bash
sudo cp ~/workspace/model-corpus/model-migration-platform.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable --now model-migration-platform
systemctl status model-migration-platform
```

装好 systemd 后可以停掉临时看门狗：

```bash
pkill -f platform-watchdog.sh     # 注意：确认匹配到的只有看门狗
```

---

## 依赖

**仅 Python 3 标准库**（`http.server` + `sqlite3`），无需 `pip install` 任何东西。

---

## 依赖

**仅 Python 3 标准库**（`http.server` + `sqlite3`），无需 `pip install` 任何东西。

## 启动

```bash
./start.sh              # 前台启动
./start.sh daemon       # 后台启动（写 server.pid）
./start.sh status       # 查看状态
./start.sh stop         # 停止
```

浏览器打开 `http://<主机>:8090/`。

若平台跑在远端、想在本机浏览器访问：

```bash
./start.sh tunnel 8090  # 起 SSH 隧道（需已配置免密登录）
```

## 前端界面（2026-10 改版）

首页按《ModelBridge 优化版 UI Prototype》落地，静态文件只有两个：

| 文件 | 说明 |
|---|---|
| `_platform/static/index.html` | 工作台页面：指标卡 / 发起迁移 / 七步流程 / 执行日志 / 迁移任务 / 模型语料库 / 交付产物 / 任务详情弹窗 |
| `_platform/static/app.css` | 设计令牌与组件样式（柔和天蓝、圆角卡片、深色主题、响应式断点） |

- 所有数字都来自 `/api/*`（`/api/corpus`、`/api/jobs`、`/api/jobs/{id}`、`/api/capacity`、`/api/deliveries`），没有前端假数据。
- 提交任务后每 2.5s 拉取当前任务的阶段事件并追加日志；任务列表每 5s、算力容量每 8s 刷新。
- 浅色/深色主题跟随系统并可手动切换（`localStorage` 记忆）；≤720px 为抽屉式侧栏。
- 旧版备份：`static/index.pre-redesign.html`；`static/preview.css` 已不被首页引用。
- 改完前端无需重启服务，浏览器强刷（`Ctrl/⌘ + Shift + R`）即可；`ui-preview/` 是本次改版的页面截图。

## 配置

| 环境变量 | 缺省 | 说明 |
|---|---|---|
| `CORPUS_ROOT` | `/data/model_corpus` | 语料库根目录（平台读它下面的 `_manifest/`、`_probe/`、`_delivery/` 等） |
| `PORT` | `8090` | 监听端口 |
| `HOST` | `0.0.0.0` | 监听地址 |
| `PY` | `python3` | 解释器 |

等价命令行参数：`python3 app.py --port 8090 --root /path/to/corpus`

## 目录依赖

平台不自己存业务数据，全部读 `CORPUS_ROOT`：

| 路径 | 用途 | 缺失时 |
|---|---|---|
| `_manifest/corpus_status.json` | 语料库总览 | 对应面板显示空 |
| `_manifest/pitfalls.json` | 坑库索引（诊断用） | 诊断助手提示未命中 |
| `_manifest/split.json` | train/test 划分 | 划分列显示 — |
| `_manifest/testset_validation.json` | 能力验收报告 | 验收面板隐藏 |
| `_manifest/adaptations.json` | 适配记录 | 记录面板显示空 |
| `_probe/<模型>/recipe.json` | 配方（快速路径依据） | 该模型走逐步迁移路径 |
| `_delivery/` | 交付包清单 | 交付包面板隐藏 |
| `_platform/platform.db` | 任务与阶段事件（SQLite，自动建） | 自动创建 |

## API

| 端点 | 方法 | 说明 |
|---|---|---|
| `/api/jobs` | POST | 提交迁移任务，body `{"model": "<模型名>"}` |
| `/api/jobs/{id}` | GET | 任务详情 + 全部阶段事件 |
| `/api/jobs/{id}/stream?after=N` | GET | SSE 实时推送（`after` 为已收到的最大 seq） |
| `/api/recipes` | GET | 配方库 |
| `/api/pitfalls` | GET | 坑库索引 |
| `/api/diagnose` | POST | 报错 → 坑库匹配，body `{"error": "..."}` |
| `/api/deliveries` | GET | 交付包清单 |
| `/api/corpus` | GET | 语料库状态总览（支持 `?status=` `?category=`） |
| `/api/validation` | GET | 平台能力验收报告 |
| `/api/rescues` | GET | 适配记录 |
| `/api/phases` | GET | 阶段定义 |

## 阶段

`resolve → match_recipe → prepare_env → adapt → validate_infer → quality → deliver`

每个阶段发 `running / ok / warn / failed` 四态事件，前端据此渲染进度条与实时日志。
`match_recipe` 的结论决定 `route` 字段是 `fast` 还是 `slow`。

## 扩展

| 想做什么 | 怎么做 |
|---|---|
| 新增一个已迁移模型 | 在该模型 `_probe/<模型>/` 放 `recipe.json`（含 `loaded: true`），平台自动走快速路径 |
| 新增一条经验 | 往 skill 仓库 `pitfall-index/references/known-pitfalls.md` 加条目，再跑 `_tools/build_pitfall_index.py` |
| 刷新语料库状态 | 跑 `_tools/make_corpus_status.py` |
| 重新生成交付包 | 跑 `_tools/make_delivery.py` |
