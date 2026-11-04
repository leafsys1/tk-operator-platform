"""全局配置：一切可通过环境变量覆盖（OPX_ 前缀）。"""

from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="OPX_", env_file=".env", extra="ignore")

    # 数据：默认落在 backend/opx.db（SQLite）。多人部署时换 PostgreSQL 连接串即可。
    db_url: str = "sqlite:///./opx.db"

    # 访问裸金属：SSH 别名 + 私钥（凭据只存引用，页面不明文回显）
    ssh_key: str = str(Path.home() / "work/pem/KeyPair-fae2027.pem")
    # 页面「选择私钥文件」上传后的落点（目录 0700 / 文件 0600）；host.cred_ref 指向这里
    ssh_key_dir: str = str(Path.home() / ".local/share/op-agent/keys")
    ssh_opts: str = "-o BatchMode=yes -o ConnectTimeout=8 -o StrictHostKeyChecking=accept-new"
    ssh_timeout: int = 25

    # 每 run 一个远端目录的根（上机测试用，见 scripts/nm_test.sh）
    remote_run_root: str = "/data/lfx/op-agent/runs"

    app_name: str = "op-agent console"
    # 仓库根（backend/app/config.py → 上溯三层）
    repo_root: str = str(Path(__file__).resolve().parents[2])

    # dsh 运行时（agent run 用）
    dsh_profile: str = "headless"
    dsh_profile_patch: str = "ops/dsh/llm-opencode-go.patch.yml"  # 兜底：没有端点/端点信息不全时用
    dsh_generated_patch: str = "backend/.runtime/dsh-llm.patch.yml"  # 按当前端点动态生成（llm_config）
    dsh_stream_patch: str = "ops/dsh/opx-typewriter.patch.yml"  # 逐 token 流式插件（P0-3 对话）
    llm_key_env: str = "OPX_LLM_KEY"  # 凭据只从环境变量取，不进库

    # 逐 token 流式：插件要把增量 POST 回控制台自己（本机回环）
    chat_stream: bool = True  # 总开关（关掉就退回「过程流式 + 兜底补发」）
    delta_base_url: str = "http://127.0.0.1:8077"

    # 四动作真执行：nm_test.sh 跑完把 run 回填给哪个控制台（本机回环）
    console_api: str = "http://127.0.0.1:8077"

    # 案例库（外置 bundle；仓库不预置案例）：scripts/import_cases.py 的默认库目录
    case_library: str = str(Path.home() / ".local/share/op-agent/case-library")

    # 版本存档（savepoints v1）：内容寻址源码归档（tar.gz）的根目录。
    # 空 = <repo_root>/archives（见 services/savepoints.py::archive_root）。
    savepoint_root: str = ""

    # 优化报告归档（Step 3）：append-only 报告仓库根目录。
    # 空 = <repo_root>/reports（见 services/optimization_reports.py::root）。
    report_root: str = ""

    # P0-3 第二步「对话式需求澄清」：走 dsh 会话（--session-id 续会话），单轮超时
    chat_timeout: int = 180

    # 智能体调用（agent_jobs）**总超时**：dsh 卡住/LLM 无响应时到点杀进程并落 failed，
    # 避免任务永久停在「对话中」、/live 永久 running、任务永久删不掉。见 agent_runner._consume。
    agent_timeout: int = 1800

    # 模型 API key 的本机密钥存储（页面配置的 key 不再落 SQLite）。
    # auto = macOS Keychain → Linux Secret Service（secret-tool）→ 0600 文件；
    # 也可显式指定 keychain / secret-service / file（headless/SSH 建议 file）。
    secret_backend: str = "auto"
    secret_dir: str = str(Path.home() / ".local/share/op-agent/secrets")
    keychain_service: str = "op-agent"


settings = Settings()
