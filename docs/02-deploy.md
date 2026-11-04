# 部署（等价服务器一次复现）

## 0. 只改一个文件

```bash
cp deploy/site.env.example deploy/site.env
vi deploy/site.env        # 改 5 处：TARGET_HOST / RUN_USER / 三个 *_ROOT / 算力机四行
```

`site.env` 已在 `.gitignore` 里，不会误提交。所有脚本从它读，**脚本本身不用改**。

现场真值（`172.16.111.102` 那台）标在注释里，**换机器不能照抄**：
`OPERATOR_ROOT` / `MODEL_ROOT` / `SHELL_ROOT` / `COMPUTE_SSH` / `COMPUTE_KEY`。

## 1. 三条命令

```bash
bash deploy/setup-operator.sh          # 算子后端（先起，外壳依赖它）
bash deploy/setup-model-platform.sh    # 模型平台
bash deploy/setup-shell.sh             # 外壳
bash deploy/verify.sh                  # 验收（不通过就不要交付）
```

每个脚本都是**幂等**的：重跑只更新代码与 unit，不动数据库、密钥、`runs/`、`archives/`。

### 1.1 脚本各自的关键动作与失败证据

| 脚本 | 关键动作 | 失败时看哪里 |
|---|---|---|
| `setup-operator.sh` | 建 venv（缺 ensurepip 走 `--without-pip` + get-pip 兜底）→ 装 34 项锁版依赖 → `pip check` → **后端 571 tests** → alembic → 写 Key → 装 unit | `pip check` 有冲突会 `die`；测试失败会打印失败用例并拒绝启动 |
| `setup-model-platform.sh` | 同步 `app.py`+`static/` → **把 `compute-side/_probe` 推到算力机** → AST 自检 → 装 unit | 探针推送失败会警告「迁移任务会在远端失败」 |
| `setup-shell.sh` | 装网关/页面/Logo → **sed 改上游端口** → **跑 42 项单测** → 装 unit → 端到端冒烟（壳页必须含 `company-logo` 与两个 `data-product`） | 单测失败直接 `die`，不部署 |
| `verify.sh` | 进程/端口/回环/壳页/缓存头/两侧 API/端点探测，`--write` 加写链路，`--gpu` 查卡 | 汇总表 `PASS/FAIL/SKIP`，明细落 `/tmp/tk-verify-*.log` |

### 1.2 首次部署前的 5 项前置（缺一项就会卡住）

1. **Python 3.12**（生产实测版本）。3.10/3.11 要同步下调 `deploy/requirements.lock.txt` 里的版本。
2. **能 SSH 到算力机**，且该私钥已在算力机的 `authorized_keys` 里（`setup-operator.sh` 会把它拷成 `$OPERATOR_ROOT/keys/executor.pem`，权限 0600）。
3. **算力机已装 Ascend 驱动 + CANN**，且 `npu-smi info` 能列出 8 张卡。
4. **算力机上这两个路径与 site.env 完全一致**：`COMPUTE_MODEL_ROOT`、`COMPUTE_OPAGENT_ROOT`。容器挂载用的是同一字符串，不一致会挂空。
5. **三个端口空闲**，且两平台上游客能互相 SSH？——不需要，两平台之间零耦合，只有外壳连着它们。

## 2. 大模型端点（只有算子平台用得到）

模型迁移平台**不需要**任何 LLM，别给它配 Key，配了也不会用。

算子平台的端点写在数据库 `endpoints` 表里，`llm_config.py` 每次跑 agent 前把它渲染成
`app/backend/.runtime/dsh-llm.patch.yml`。三种配置方式，任选：

**方式 A（推荐）部署时交互输入**：`setup-operator.sh` 第 6 步会问你要 Key，写进
`$OPERATOR_ROOT/secrets/values/server-active-endpoint.key`（0600）。

**方式 B 网页配置**：打开外壳 → 算子平台 → 「API 管理」→ 填 base_url / 模型名 / Key → 点探测。
探测成功 `probe_state=ok` 才生效。

**方式 C 直接改库**（应急）：

```bash
sqlite3 "$OPERATOR_ROOT/data/opx.db" \
  "update endpoints set base_url='https://tokens.tiankuan.com/v1',
   model='deepseek-v4.1-flash-1017', is_active=1 where id=1;"
```

### 2.1 怎么确认「实际用的」是哪个模型（不要只信配置）

三重取证，缺一不算：

```bash
# ① 端点探测：返回的 models 列表里有没有目标模型
curl -s -X POST "http://127.0.0.1:18078/api/endpoints/1/probe" | python3 -m json.tool

# ② 近期 agent_jobs 落库的 model 字段
sqlite3 "$OPERATOR_ROOT/data/opx.db" \
  "select id,kind,model,endpoint_id,tokens_in,tokens_out from agent_jobs order by id desc limit 5;"

# ③ dsh 自己的会话 JSONL（真相源）
cd "$OPERATOR_ROOT/app/workspaces/.dsh/sessions"/*/
zstd -dc session.*.jsonl.zstd | grep -o '"model":"[^"]*"' | sort | uniq -c
```

第 ③ 条最关键——前两条只证明「配置对了」，第 ③ 条证明「模型真的被调了」。

> 实测坑：`deepseek-v4.1-flash-1017` 开了 thinking。`max_tokens=32` 时 32 个 token 全被
> `reasoning_content` 吃掉，`content` 为空。**别据此判断模型挂了**，把 `max_tokens` 提到 200 再看。

## 3. 防火墙与对外暴露

部署完只有 `0.0.0.0:18080` 对外，两个平台都是回环。`verify.sh` 会断言这一点。

```bash
sudo ufw allow from 172.16.111.0/24 to any port 18080 proto tcp
sudo ufw deny 18080            # 如果不用 ufw 就写 iptables，别把 18078/8090 放出去
```

对外（域名+证书+鉴权）建议走反代，反代只需注意两点：

- **保留原始 Host 头**，或把域名写进 `WORKBENCH_ALLOWED_HOSTS`。否则 400。
- 反代到 `http://127.0.0.1:18080`，**不要**改成 `https` 终止后再发明文到公网。

反向代理配置样例（Lucky / Nginx 同理）：

```nginx
location / {
    proxy_pass http://127.0.0.1:18080;
    proxy_set_header Host $host;
    proxy_set_header X-Real-IP $remote_addr;
    proxy_http_version 1.1;
    proxy_set_header Upgrade $http_upgrade;      # 算子平台的 SSE 需要
    proxy_set_header Connection "";
    proxy_buffering off;                          # SSE 必须关，否则进度条不动
    proxy_read_timeout 3600s;                     # 真机任务要跑 10 分钟以上
}
```

## 4. 装完之后必须做的三件事

1. `bash deploy/verify.sh` —— 全 PASS 才算装好，FAIL 一条都不许放过。
2. 在浏览器上手工点一遍：两个页签都打开、算子平台建一个任务、模型平台提交一个已迁移过的模型。
3. **关掉匿名可写**（如果暂时做不到，至少把 `ALLOWED_CIDRS` 收紧到具体网段）：

```bash
sudo sed -i 's/WORKBENCH_AUTH_REQUIRED=0/WORKBENCH_AUTH_REQUIRED=1/' /etc/systemd/system/tk-gateway-preview.service
sudo systemctl daemon-reload && sudo systemctl restart tk-gateway-preview
```

## 5. 从旧机迁移数据（不是首次安装时看这段）

三个组件的数据要分开搬，别整目录打包：

| 数据 | 位置 | 怎么搬 | 注意 |
|---|---|---|---|
| 算子平台数据库 | `$OPERATOR_ROOT/data/opx.db` | **`sqlite3 <db> ".backup <目标>"`**，不要 `cp` | 直接 `cp` 会拿到正在写的 WAL，可能半条事务 |
| 算子工作区 | `$OPERATOR_ROOT/app/workspaces/` | `rsync -a` | 含 `.dsh/` 会话历史，别漏 |
| 密钥 | `$OPERATOR_ROOT/keys/`、`secrets/` | `rsync -a` 后确认 0600/0700 | 目标目录必须先 `mkdir -m 700` |
| 模型语料索引 | `$MODEL_ROOT/_manifest/`、`_delivery/` | `rsync -a` | 权重不在本机，不用搬 |
| 权重 | 算力机 `$COMPUTE_MODEL_ROOT` | 不搬 | 已在算力机上 |

**切换窗口**：旧机停写 → 做一次 `.backup` → 起新机 → 双向核对 `runs` 条数与最近 run 的 sha256 →
DNS/入口切过去 → 旧机只读保留一周。**不要双写**，两个 SQLite 会分叉。

回退：入口切回旧机，新机保留现场供排查。详见 `docs/05-operations.md`。
