# 运维、升级与回退

## 1. 日常命令

```bash
# 状态
systemctl status tk-operator-preview model-migration-platform tk-gateway-preview --no-pager
ss -ltn | grep -E '18078|8090|18080'

# 日志（按组件分开看，别混）
journalctl -u tk-gateway-preview -f                      # 网关：路由/403/400/503 都在这
journalctl -u tk-operator-preview -f                     # 算子后端：API 调用
journalctl -u model-migration-platform -f                # 模型平台
tail -f $MODEL_ROOT/_platform/server.log                 # 模型平台另有一份

# 重启（顺序：先业务，再外壳）
sudo systemctl restart tk-operator-preview model-migration-platform
sudo systemctl restart tk-gateway-preview

# 整机重启后自查（未实测过冷启动，这是第一次跑时要重点看的）
bash deploy/verify.sh
```

## 2. 出问题时的定位顺序

**不要一上来就重启。** 按这个顺序走，每步都有明确的判别信号：

| 现象 | 先查 | 判别信号 | 常见原因 |
|---|---|---|---|
| 打开是空白/文字提示页 | `curl -s localhost:18080/ \| head -3` | 有 `portal.html` 内容？ | `portal.html` 丢了 → 从仓库 `shell/portal.html` 补 |
| 页面能开但一个平台 502 | `curl -o /dev/null -w '%{http_code}' localhost:18078/api/health` | 200 还是 000 | 业务平台没起 / 端口不符 |
| 改样式没生效 | `curl -sI localhost:18080/operator/ \| grep -i cache` | 有 `no-store`？ | 网关没重启，或浏览器强缓存（硬刷） |
| 反代后 400 | 看 `journalctl -u tk-gateway-preview` | 日志里有 Host 拒绝 | `ALLOWED_HOSTS` 没加域名（**不是 CIDR**，CIDR 拒绝是 403） |
| 反代后 403 | 同上 | 日志里有 CIDR 拒绝 | 来源网段不在 `ALLOWED_CIDRS` |
| 提交写操作 403 | 看请求头 | `Origin` 与 `Host` 不一致 | 反代没传对 Host，或跨域调用 |
| 写操作 503 | `systemctl show tk-gateway-preview -p Environment` | `*_READONLY=1`？ | 只读闸开着 |
| 任务点不动/进度条卡住 | 浏览器 Network 里看 SSE | 长连接是否被反代断 | 反代缺 `proxy_buffering off` / `proxy_read_timeout` 太短 |
| 算子 agent 答复为空 | 看 `agent_jobs.tokens_out` | 是否只消耗了 reasoning | 模型开了 thinking，`max_tokens` 太小 |

## 3. 升级流程（一次改动的完整闭环）

```bash
# 1) 在仓库里改，跑单测
vi shell/gateway.py
cd shell && python3 -m pytest tests -q          # 必须全绿

# 2) 提交，打草稿 tag（不推远端也可以先本地试）
git commit -am "gateway: 表格列宽再收 4px"
git tag -a v0.1.1-rc1 -m "候选：表格列宽"

# 3) 灰度：先只改文件，不重启；备份现有
cp -a $SHELL_ROOT/gateway-v2.py $SHELL_ROOT/gateway-v2.py.bak-prerelease-$(date +%Y%m%d-%H%M%S)
install -m 0700 shell/gateway.py $SHELL_ROOT/gateway-v2.py
sudo systemctl restart tk-gateway-preview

# 4) 验收（不通过立刻回滚）
bash deploy/verify.sh

# 5) 通过 → 正式 tag；不通过 → 回滚
git tag -a v0.1.1 -m "表格列宽"
# 或
bash deploy/rollback.sh gateway
```

**规矩**：`verify.sh` 不 PASS 不许打正式 tag。tag 就是回退点，别把没验过的状态打上 tag。

### 3.1 只改一个组件时，不要动其他两个

三个组件之间只有 HTTP 依赖，没有共享代码。改网关不需要重启业务平台，反之亦然。
唯一要注意顺序：**业务先起，外壳后起**（外壳启动时会检查上游可达性）。

## 4. 回退

```bash
bash deploy/rollback.sh list        # 看有哪些备份/tag 可用
bash deploy/rollback.sh gateway     # 网关退回上一份（自动 curl 验证壳页）
bash deploy/rollback.sh gateway /path/to/gateway-v2.before-xxx.py   # 退到指定文件
bash deploy/rollback.sh model       # 模型平台退回上一份 app.py
bash deploy/rollback.sh operator v0.1.0   # 算子后端退回某个 tag
```

### 4.1 回退的边界（必须知道）

| 回退什么 | 能回退 | 不能回退 |
|---|---|---|
| 网关代码 | ✅ 完全（纯文件替换） | — |
| 模型平台代码 | ✅ 完全（单文件） | 已生成的交付包/镜像不删 |
| 算子后端代码 | ✅ 按 tag | **数据库 schema 不回退**（tag 间有 migration 时要手工 `alembic downgrade`） |
| 算子前端 | ✅ 按 tag（`app/web/dist` 一起换） | — |
| 数据库数据 | ❌ 本脚本不管 | 需要单独 `.backup` 恢复 |
| 算力机上的 run 目录 | ❌ | 保留原始证据，不要删 |

## 5. 备份

**至少要备三样**，缺一个都还原不回来：

```bash
B=/backup/$(date +%Y%m%d)
mkdir -p $B
# ① 数据库（用 .backup，不要 cp）
sqlite3 $OPERATOR_ROOT/data/opx.db ".backup $B/opx.db"
sqlite3 $MODEL_ROOT/_platform/platform.db ".backup $B/platform.db"
# ② 密钥与配置
tar -czf $B/secrets.tgz -C $OPERATOR_ROOT secrets keys config
# ③ 工作区（含 dsh 会话历史，这是排查的依据）
rsync -a $OPERATOR_ROOT/app/workspaces/ $B/workspaces/
```

**不备**：`runs/`（80MB 且会持续增长，是过程证据不是状态）、`node_modules`、`control-venv312`。

## 6. 数据切换（从源机搬到新机时）

一次性窗口，别双写：

```bash
# 1) 源机停写（不要停服，只把入口切走即可）
#    实际操作：先在反代/入口层把流量切到新机，源机留只读给排查

# 2) 源机做一致性快照
ssh src 'sqlite3 .../data/opx.db ".backup /tmp/opx.snapshot"'
ssh src 'sqlite3 /tmp/opx.snapshot "pragma integrity_check"'      # 必须 ok
ssh src 'sqlite3 /tmp/opx.snapshot "pragma foreign_key_check"'    # 必须 0 行

# 3) 搬
rsync -a src:/tmp/opx.snapshot $OPERATOR_ROOT/data/opx.db.new
rsync -a src:.../app/workspaces/  $OPERATOR_ROOT/app/workspaces/

# 4) 切换（留旧库）
mv $OPERATOR_ROOT/data/opx.db $OPERATOR_ROOT/data/opx.db.bak-before-cutover
mv $OPERATOR_ROOT/data/opx.db.new $OPERATOR_ROOT/data/opx.db
sudo systemctl restart tk-operator-preview && bash deploy/verify.sh

# 5) 双向核对（少了任何一项都算没切完）
#    - 任务条数
sqlite3 $OPERATOR_ROOT/data/opx.db "select count(*) from tasks;"
#    - 最近 5 个 run 的 id / action / status / exit_code
#    - 抽取 3 个 run 的 manifest.json sha256 与源机比对
```

**回滚**：`mv` 回来 + 重启 + 入口切回。源机在切换后一周内**保持可启动状态**。

## 7. 提交与检视（给团队的口径）

```bash
# 看两个基线之间动了什么（这是接手的人最常问的）
git log --oneline v0.1.0..HEAD
git diff --stat v0.1.0..HEAD

# 只看某个组件的改动
git diff v0.1.0..HEAD -- shell/
git diff v0.1.0..HEAD -- operator/backend/app/services/

# 核对线上文件是否与仓库一致（防"手改没进库"）
sha256sum $SHELL_ROOT/gateway-v2.py $MODEL_ROOT/_platform/app.py
git show HEAD:shell/gateway.py | sha256sum
```

最后那条最有用：**线上文件和仓库不一致 = 有人绕过流程手工改了**。
发现这种情况应该把改动补进仓库，而不是继续手改。

## 8. 严禁的操作

| 操作 | 为什么 |
|---|---|
| `pkill -f gateway` / `pkill -f app.py` | 命令本身会匹配到自己，可能自杀或误杀别人的进程；用 `systemctl` |
| 删/替换 `/data/lfx/op-agent/runs/.locks/*.lock` | 别的事件的锁；删了等于放开互斥，会让两个任务抢同一张卡 |
| 任务超时前手工 kill 容器 | 会丢掉交付包；让平台自己走完降级流程 |
| 直接 `cp` 正在写的 SQLite | 会拿到未提交的 WAL；用 `.backup` |
| 在别人的容器/进程上重启或清理 | 只报告，不动。卡 0–3 常驻他人的 VLLM |
| 为了让测试过而改 `cases.json` 的期望值 | 那是改被测对象；口径冲突要改规格并留证据 |
