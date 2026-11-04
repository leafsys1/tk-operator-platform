# 已知问题与风险（按优先级）

每一条都写清：现象 → 根因（已测到哪一层）→ 影响 → 现在怎么绕 → 正确修法。
**没有标"已证实"的都是候选，不要当结论用。**

---

## 1. 【高】两个平台之间没有卡互斥，可能同时抢同一张卡

**现象**：算子平台锁住卡 4，模型平台在同一时刻扫到卡 4 空闲（`npu-smi` 还没看到容器进程），
两边同时往卡 4 上跑。真发生会看到显存争抢、`aclrtSetDevice` 失败或推理精度异常。

**根因（已证实到代码层）**：

| 平台 | 机制 | 生效范围 |
|---|---|---|
| 算子 | `flock` on `$COMPUTE_OPAGENT_ROOT/runs/.locks/dev<N>.lock` | 跨进程、跨平台（文件系统级） |
| 模型 | `_CARD_BUSY` 字典 + 8 秒 TTL 的 `npu-smi` 缓存 | **仅本进程** |

两者不在同一个命名空间，谈不上互斥。这不是"配置不对"，是设计缺口。

**影响**：中。目前单人使用、一次只跑一个任务时不会触发。多人同时用就会。

**现在怎么绕**：约定同一时间只让一个人在网页上跑真机任务；或在跑之前手工
`flock $COMPUTE_OPAGENT_ROOT/runs/.locks/dev4.lock -c 'npu-smi info'` 确认锁空闲。

**正确修法**（离线原型已验证）：
把模型平台的 `_CARD_BUSY` 换成与算子平台**同一个 inode** 的 `flock`，
并在整个任务生命周期（含 probe + perf 两段）持有同一把锁，不能中途放了再拿。

离线原型 10 个测试已通过（并发互斥、崩溃后残留隔离、inode 漂移 fail-closed、
多设备按序加锁逆序释放、部分失败释放已持有锁…），但**尚未接入两个平台**。
接入前还要解决：锁文件在两台机器？不——两个平台都在同一台通算机上，所以本地 `flock` 就够。

---

## 2. 【高·外部】模型平台上传 OBS 走死掉的代理隧道

**现象**：`deliver` 阶段卡在「构建模型权重镜像并上传 OBS…」，事件长时间不推进，
`obssdk.log` 反复重试。

**实测证据**：

```
$ tail /root/.obsutil_log/obssdk.log
Failed to send request with reason: Put "https://61.241.77.58:443/model-corpus/images%2Fweights%
2Fqwen__qwen2.5-0.5b-instruct.tar.gz?partNumber=5&uploadId=..." :
  use of closed network connection and will try again
  ... EOF and will try again
  ... context deadline exceeded and will try again

$ curl -x http://127.0.0.1:17891 http://example.com
curl: (28) Operation timed out after 5000 milliseconds with 0 bytes received
```

**根因（已证实）**：算力机 root 的环境里有 `HTTPS_PROXY=http://127.0.0.1:17891`，
这个本地端口是**别人**（`codex` 进程）起的 sshd 动态转发隧道，隧道已死。
`obsutil` 走代理 → 代理不通 → 上传永远重试。

**对照证据（关键）**：同日 07:20 同一模型、同一路径成功过
（`权重镜像：OK 759 MB`、`交付包已生成（10 个文件）`），说明配置没错，是隧道在那之后挂的。

**影响**：模型平台上"权重镜像上传 + OBS 归档"这一步不可用；镜像本地构建、性能测试、
`probe.json`、配方归档**都正常**。不阻塞 `docs/03-verification.md` 里已列的其他能力。

**现在怎么绕**：
1. 平台自身会在 `npu(timeout=3600)` 到点后降级为 `warn` 并继续生成交付包——
   **不要在超时前手工 kill，会丢掉交付包**。
2. 真要上传，让隧道主人在算力机上重连隧道，或者让 `obsutil` 绕过代理：

```bash
# 只在确认不走代理也能到 OBS 时这么做
env -u HTTPS_PROXY -u https_proxy -u ALL_PROXY -u all_proxy \
  obsutil cp /tmp/_imgone/<x>.tar.gz obs://model-corpus/images/weights/<x>.tar.gz
```

**正确修法**：`mk_image.sh` 里不要继承 `HTTPS_PROXY`，显式 `env -u`；或把隧道纳入
systemd 管理（`ClientAliveInterval` + 自动重连），别让别人的临时进程承载生产依赖。

**注意**：这条隧道被别人的进程共用。**不要自己去 kill 或重启别人的进程**，报告给对应的人。

---

## 3. 【高·安全】模型平台有命令拼接，输入未净化

**现象**：`app.py` 里 `download_model()` / `run_migration()` 把用户提交的模型名
拼进 shell 命令（SSH 派发到算力机执行）。构造特殊的模型名可以注入命令。

**已测**：离线候选 `model-security-dev` 26 个测试里 **5 失败 + 1 报错**，不可部署。
另有两个独立审查结论（都是只读审查，未部署）：

- **下载互斥候选不宜部署**：① `app.py:722–736` 用 glob 取模型目录**首个子项**而不是下载返回的
  精确源路径，可能把不完整模型判为可用（离线 mock 已复现）；② 锁只覆盖下载、不覆盖
  "已有目录被发现使用"，别的任务能读到半成品；③ 异常时 `finally` 只释放占位，
  不保证远端下载进程真停了。
- **旧配方身份**：96 条已加载配方中 **51 条没有显式 `model` 字段**。不能用目录名或
  registry 唯一匹配去"猜"身份并放行——那等于把猜测当授权。必须把
  "旧配方展示" 与 "执行授权" 拆开：可展示，不可据此执行。

**影响**：只在内网可信输入下可用。当前入口是**匿名可写**，等于把注入面暴露给整个
`172.16.111.0/24`。

**现在怎么绕**：收紧 `ALLOWED_CIDRS` 到具体机器；或临时把 `WORKBENCH_MODEL_READONLY=1`。

**正确修法**：模型名先按白名单正则校验（`^[\w.-]+/[\w.-]+$`），再 `shlex.quote`；
远端路径用 `realpath` 校验必须落在 `COMPUTE_MODEL_ROOT` 内。
（这条和网关层的 403/400 校验是两件事，网关挡不住已授权的内部人。）

---

## 4. 【中·安全】入口匿名可写可执行

**现象**：`WORKBENCH_AUTH_REQUIRED=0`、`PREVIEW_GATE=0`、两侧 `*_READONLY=0`。
内网任何人可以：提交真机算子任务（起 `--privileged` 容器）、提交模型迁移任务。

**为什么现在这样**：用户明确要求先跑通流程、鉴权自己用反代做。这是**知情决定**，不是疏忽。

**上线前必做**（按顺序）：

1. 反代 + 域名 + 证书，反代侧做认证。
2. 打开网关 Basic auth 作为兜底：
   `WORKBENCH_AUTH_REQUIRED=1`，凭据文件 `$SHELL_ROOT/access.txt`（网关自读）。
3. 收紧 `ALLOWED_CIDRS`，把 `ALLOWED_HOSTS` 从 `*` 改成实际域名。
4. 改完跑 `bash deploy/verify.sh`，确认匿名 GET 变 401 而不是页面打不开。

---

## 5. 【中】`--privileged` 容器是必须的，不是配错了

**现象**：去掉 `--privileged` 后容器内 `npu-smi` 报
`dcmi module initialize failed ret=-8005`，容器 exit 187。

**根因**：宿主驱动 26.0.rc1 在非特权设备直通容器里拒绝 ACL init。
（另有一处历史注释写 `-8020 "device is used"`，是同一类行为的另一个错误码。）

**影响**：任务容器等于宿主机 root 权限。配合第 4 条（匿名可写）风险叠加。

**不要做**：不要为了"降低风险"去掉 `--privileged`，那会让所有任务失败。要降风险就收紧入口。

---

## 6. 【中】设备映射曾经错位（已修，但要知道怎么查）

**历史现象**：锁了卡 4，容器里却挂载 `/dev/davinci2`。任务能跑但用的是错的卡。

**根因**：`ops/nm.conf` 的 `OPX_DOCKER_ARGS` 里写死了 `--device=/dev/davinci2`，
而卡号实际由 `ASCEND_RT_VISIBLE_DEVICES=<锁到的卡>` 决定，两者打架。

**已修**：从 `nm.conf` 删掉 `--device=/dev/davinciN`，只保留 3 个管理设备
（`davinci_manager` / `devmm_svm` / `hisi_hdc`）。

**怎么自查**：跑完任务看 run 目录的 `docker_cmd.txt`：

```bash
grep -o 'ASCEND_RT_VISIBLE_DEVICES=[0-9]*' $OPERATOR_ROOT/app/runs/<run-id>/docker_cmd.txt
grep -c 'device=/dev/davinci[0-9]' $OPERATOR_ROOT/app/runs/<run-id>/docker_cmd.txt   # 必须是 0
```

---

## 7. 【低】算子页有 GET 请求会写文件

`GET /operator/api/proto` 会写文件（生成 proto 定义），被 `OPERATOR_READONLY=1` 时单独拦掉。
`GET /api/capacity` 会 SSH 到算力机跑 `npu-smi`（有副作用但是只读查询），
`GET /operator/api/jobs` 会碰数据库 WAL。

这三个是**已知副作用**，做只读发布时不要以为 GET 就一定安全。

---

## 8. 【低】任务 #22 是一次失败的历史记录，别当 bug 修

`r4-20261103-124331` 的 `smoke` 是 `failed`。原因是当初给的规格（支持广播、空张量返回空）
与算子二进制实现（`n_min=1`、不支持广播、空张量返回 `REJECT_*`）冲突。

**处理方式是改规格而不是改算子**——以现有实现为准，产出 v2.0 规格并重跑。这条 run 保留作为证据。
若以后有人跑同类算子又失败，先核对"规格口径 vs 二进制口径"，不要先怀疑环境。

---

## 9. 【低】`gate` 接口的 `checks=0` 不代表没检查

`/api/runs/<id>/gate` 返回 `{"checks":0,"fail":0,"warn":0}` 是**门禁项统计为空**，
不等于"全部检查通过"。判据要看 `verdict` 字段和 run 目录里的机读报告
（`correctness.json` / `performance.json` / `metrics.json`）。

---

## 10. 【低】交付包门禁会因"版本证据过期"拒收

**现象**：`POST /operator/api/tasks/<id>/deliveries` 返回
`409 E_VERSION_EVIDENCE_STALE`，提示「编译（没有与当前规格/算子代码/用例集一致且门禁通过的证据）」。

**根因（正常行为）**：编译证据是旧规格版本跑的 run。改了规格/用例后必须重跑 build，
否则交付的代码与验证过的代码不是同一份。

**做法**：按提示重跑 build 出新的 run，再提交付。**不要绕过门禁**。

---

## 11. 【低】`access.txt` 与凭据文件

`WORKBENCH_CREDENTIAL_FILE` 指向 `$SHELL_ROOT/access.txt`（42 字节，`0600`）。
`AUTH_REQUIRED=0` 时它不被读，但文件留着。里面是明文 `user:pass`。

**本仓库不含该文件**。换机器时由部署者自己创建：

```bash
umask 077; printf 'admin:%s' "$(openssl rand -base64 18)" > "$SHELL_ROOT/access.txt"
```

---

## 12. 【提示】模型平台的"agent"是假的，别按 agent 方式排查

它没有 LLM、没有工具调用、没有会话。看到"分析中/适配中"是 `emit()` 打的事件。
要看真实进展只能看 `events` 表和算力机上的进程：

```bash
sqlite3 $MODEL_ROOT/_platform/platform.db "select seq,phase,status,message from events where job_id='<id>' order by seq;"
ssh ... 'docker ps | grep opx; ps -ef | grep -E "probe|mk_image"'
```

要它具备真正的 agent 能力（自动读错误日志给根因、自动改写配方），
**当前代码里没有任何基础，得新写**，不是改配置能出来的。
