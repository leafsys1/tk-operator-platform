"""端点 API key 的**本机密钥存储**：key 不落 SQLite，DB 只存引用名。

设计（2026-10-21，替代 `endpoints.api_key` 明文落库）：

- **auto** 顺序：macOS Keychain → Linux Secret Service（`secret-tool`）→ 0600 文件回退。
  可用 `OPX_SECRET_BACKEND=keychain|secret-service|file` 显式指定（headless/SSH 建议 file）。
- `set_secret()` 写成功才返回后端名；某个后端失败会继续试下一个（auto 模式），最后才报错。
- `index.json`（0600）记录 `ref → backend`，保证「写时回退到文件、读时也知道去文件找」。
- key 不经过命令行参数（Keychain 走 stdin、secret-tool 走 stdin），避免同机 `ps` 看到。
- 文件后端：目录 0700，单个 key 文件 0600，写入用「临时文件 + os.replace」原子替换。
- 注意：这只是「不落 DB + 系统密钥库优先」；进程内存/环境变量里仍然有明文 key（dsh 需要）。
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
from pathlib import Path

from ..config import settings

KeyBackend = str  # keychain | secret-service | file

_INDEX_NAME = "index.json"
_VALUES_DIR = "values"
_REFS_ALLOWED = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789._-")


class SecretStoreError(RuntimeError):
    """密钥库不可用/读写失败。"""


# ---------------------------------------------------------------- 基础路径


def _root(*, create: bool = True) -> Path:
    root = Path(settings.secret_dir).expanduser()
    if not create:
        return root
    try:
        root.mkdir(parents=True, exist_ok=True, mode=0o700)
        root.chmod(0o700)
    except OSError as e:
        raise SecretStoreError(f"创建密钥目录失败：{e}") from e
    return root


def _index_path(*, create: bool = False) -> Path:
    return _root(create=create) / _INDEX_NAME


def _values_dir(*, create: bool = False) -> Path:
    d = _root(create=create) / _VALUES_DIR
    if create:
        d.mkdir(parents=True, exist_ok=True, mode=0o700)
    return d


def _safe_ref(ref: str) -> str:
    value = (ref or "").strip()
    if not value or len(value) > 120 or any(ch not in _REFS_ALLOWED for ch in value):
        raise SecretStoreError(f"非法密钥引用名：{ref!r}")
    return value


def _load_index() -> dict[str, str]:
    path = _index_path()
    if not path.is_file():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return {str(k): str(v) for k, v in data.items()} if isinstance(data, dict) else {}


def _save_index(index: dict[str, str]) -> None:
    path = _index_path(create=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(index, ensure_ascii=False, indent=1), encoding="utf-8")
    os.chmod(tmp, 0o600)
    os.replace(tmp, path)


# ---------------------------------------------------------------- 后端探测


def _has_keychain() -> bool:
    return shutil.which("security") is not None


def _has_secret_service() -> bool:
    return shutil.which("secret-tool") is not None and bool(os.environ.get("DBUS_SESSION_BUS_ADDRESS"))


def _candidates() -> list[KeyBackend]:
    choice = (settings.secret_backend or "auto").strip().lower()
    if choice not in ("auto", "keychain", "secret-service", "file"):
        raise SecretStoreError(f"OPX_SECRET_BACKEND 只能是 auto/keychain/secret-service/file（收到 {choice!r}）")
    if choice != "auto":
        if choice == "keychain" and not _has_keychain():
            raise SecretStoreError("指定了 keychain，但系统里没有 security 命令")
        if choice == "secret-service" and not _has_secret_service():
            raise SecretStoreError("指定了 secret-service，但没有 secret-tool 或 D-Bus 会话")
        return [choice]
    order: list[KeyBackend] = []
    if _has_keychain():
        order.append("keychain")
    if _has_secret_service():
        order.append("secret-service")
    order.append("file")
    return order


def backend_name() -> str:
    """当前会优先使用的后端名（auto 时返回探测结果；不保证写一定成功）。"""
    return _candidates()[0]


def backend_for(ref: str | None) -> str | None:
    if not ref:
        return None
    return _load_index().get(ref)


# ---------------------------------------------------------------- Keychain


def _keychain_write(ref: str, value: str) -> None:
    sec = shutil.which("security")
    assert sec is not None
    service = settings.keychain_service
    # 先删旧条目（不存在也忽略）；再 add。-w 不带参数时 security 从 stdin 读两遍密码。
    subprocess.run(
        [sec, "delete-generic-password", "-s", service, "-a", ref],
        capture_output=True, text=True, timeout=8,
    )
    proc = subprocess.run(
        [sec, "add-generic-password", "-s", service, "-a", ref, "-w"],
        input=f"{value}\n{value}\n", capture_output=True, text=True, timeout=8,
    )
    # security 在部分失败场景仍返回 0，必须读回校验，避免「以为写成功」。
    if _keychain_read(ref) != value:
        raise SecretStoreError(f"Keychain 写入失败：{(proc.stderr or proc.stdout or '').strip()[:200]}")


def _keychain_read(ref: str) -> str | None:
    sec = shutil.which("security")
    if sec is None:
        raise SecretStoreError("没有 security 命令")
    proc = subprocess.run(
        [sec, "find-generic-password", "-s", settings.keychain_service, "-a", ref, "-w"],
        capture_output=True, text=True, timeout=8,
    )
    if proc.returncode != 0:
        return None
    return proc.stdout.rstrip("\n") or None


def _keychain_delete(ref: str) -> None:
    sec = shutil.which("security")
    if sec is None:
        return
    subprocess.run(
        [sec, "delete-generic-password", "-s", settings.keychain_service, "-a", ref],
        capture_output=True, text=True, timeout=8,
    )


# ---------------------------------------------------------------- Secret Service


def _secret_service_write(ref: str, value: str) -> None:
    tool = shutil.which("secret-tool")
    if tool is None:
        raise SecretStoreError("没有 secret-tool")
    proc = subprocess.run(
        [tool, "store", "--label", f"{settings.keychain_service}:{ref}",
         "service", settings.keychain_service, "ref", ref],
        input=value, capture_output=True, text=True, timeout=8,
    )
    if proc.returncode != 0:
        raise SecretStoreError(f"secret-tool store 失败：{(proc.stderr or '').strip()[:200]}")


def _secret_service_read(ref: str) -> str | None:
    tool = shutil.which("secret-tool")
    if tool is None:
        raise SecretStoreError("没有 secret-tool")
    proc = subprocess.run(
        [tool, "lookup", "service", settings.keychain_service, "ref", ref],
        capture_output=True, text=True, timeout=8,
    )
    if proc.returncode != 0:
        return None
    return proc.stdout.rstrip("\n") or None


def _secret_service_delete(ref: str) -> None:
    tool = shutil.which("secret-tool")
    if tool is None:
        return
    subprocess.run(
        [tool, "clear", "service", settings.keychain_service, "ref", ref],
        capture_output=True, text=True, timeout=8,
    )


# ---------------------------------------------------------------- 文件回退


def _file_path(ref: str, *, create: bool = False) -> Path:
    return _values_dir(create=create) / f"{_safe_ref(ref)}.key"


def _file_write(ref: str, value: str) -> None:
    path = _file_path(ref, create=True)
    tmp = path.with_suffix(".tmp")
    fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "w", encoding="utf-8") as fh:
        fh.write(value)
    os.chmod(tmp, 0o600)
    os.replace(tmp, path)


def _file_read(ref: str) -> str | None:
    path = _file_path(ref)
    if not path.is_file():
        return None
    try:
        return path.read_text(encoding="utf-8") or None
    except OSError as e:
        raise SecretStoreError(f"读取密钥文件失败：{e}") from e


def _file_delete(ref: str) -> None:
    try:
        _file_path(ref).unlink(missing_ok=True)
    except OSError:
        pass


# ---------------------------------------------------------------- 对外 API


def set_secret(ref: str, value: str) -> str:
    """写入一个 key，返回实际使用的后端名（keychain/secret-service/file）。"""
    ref = _safe_ref(ref)
    if not value:
        raise SecretStoreError("key 不能为空")
    errors: list[str] = []
    for backend in _candidates():
        try:
            if backend == "keychain":
                _keychain_write(ref, value)
            elif backend == "secret-service":
                _secret_service_write(ref, value)
            else:
                _file_write(ref, value)
            index = _load_index()
            index[ref] = backend
            _save_index(index)
            return backend
        except (SecretStoreError, OSError, subprocess.SubprocessError) as e:
            errors.append(f"{backend}: {e}")
    raise SecretStoreError("没有可用的密钥后端：" + "；".join(errors))


def get_secret(ref: str | None) -> str | None:
    """按索引读 key；未知引用按 auto 顺序探测一次（兼容外部/旧数据）。

    读取不创建目录、不修改权限或索引；受限子进程只需已有密钥的读取权限。
    密钥库不可读或系统后端异常时返回 None，不让列表/详情接口 500。
    """
    if not ref:
        return None
    try:
        return _get_secret(ref)
    except (SecretStoreError, OSError, subprocess.SubprocessError):
        return None


def _get_secret(ref: str) -> str | None:
    ref = _safe_ref(ref)
    index = _load_index()
    backend = index.get(ref)
    if backend:
        try:
            return _read_with(backend, ref)  # type: ignore[arg-type]
        except (SecretStoreError, OSError, subprocess.SubprocessError):
            return None
    for candidate in _candidates():
        try:
            value = _read_with(candidate, ref)
        except (SecretStoreError, OSError, subprocess.SubprocessError):
            continue
        if value:
            return value
    return None


def _read_with(backend: KeyBackend, ref: str) -> str | None:
    if backend == "keychain":
        return _keychain_read(ref)
    if backend == "secret-service":
        return _secret_service_read(ref)
    return _file_read(ref)


def delete_secret(ref: str | None) -> None:
    if not ref:
        return
    ref = _safe_ref(ref)
    index = _load_index()
    backend = index.get(ref)
    try:
        if backend == "keychain":
            _keychain_delete(ref)
        elif backend == "secret-service":
            _secret_service_delete(ref)
        elif backend == "file":
            _file_delete(ref)
        else:
            # 未知引用：删文件 + 尽力删两个系统后端（不报错）
            _file_delete(ref)
            _keychain_delete(ref)
            _secret_service_delete(ref)
    finally:
        if ref in index:
            index.pop(ref, None)
            _save_index(index)


def has_secret(ref: str | None) -> bool:
    try:
        return bool(get_secret(ref))
    except (SecretStoreError, OSError, subprocess.SubprocessError):
        return False


def reset_for_tests() -> None:
    """测试辅助：清空索引与文件后端（只在 OPX_SECRET_BACKEND=file 的临时目录里用）。"""
    try:
        root = _root()
        for path in root.glob("values/*.key"):
            path.unlink(missing_ok=True)
        (root / _INDEX_NAME).unlink(missing_ok=True)
    except (OSError, SecretStoreError):
        pass
