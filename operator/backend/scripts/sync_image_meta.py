#!/usr/bin/env python
"""把裸金属上的镜像元数据回填进库：digest（三硬规则要求）、CANN 版本。

为什么要有这一步（2026-10-15 复查 F6）：
- 方案 §3 的硬规则第 3 条要求「镜像 digest 必须留」——判「两次跑的是不是同一个镜像」靠 digest，
  靠 tag 不行（tag 会被重新 push 覆盖）。P1.5-b 加了 `host_images.digest` 列，但一直是空的。
- `cann_version` 是镜像所属工具链版本的结构化记录（实际只采集 CANN；torch/python 版本
  目前不在本脚本能力内，后续由探针/PATCH 元数据补齐）。

连接目标 / 端口 / 私钥统一走 `services/ssh.py`（与页面、探针、真执行同源）：
`root@1.2.3.4:50023` 这类写法会拆成目标 + `-p`，私钥按 `host.cred_ref` 解析（空则回退全局）。

只读：`docker images --digests` + `docker image inspect`，不拉镜像、不删镜像。

用法（在 backend/ 下）：
    .venv/bin/python scripts/sync_image_meta.py             # 全部镜像
    .venv/bin/python scripts/sync_image_meta.py --op-dev     # 只同步 is_op_dev=1 的
    .venv/bin/python scripts/sync_image_meta.py --dry-run
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sqlalchemy import select  # noqa: E402

from app.config import settings  # noqa: E402
from app.db import SessionLocal  # noqa: E402
from app.models import Host, HostImage  # noqa: E402
from app.services.probe import ssh_exec  # noqa: E402
from app.services.ssh import ssh_key, ssh_port, ssh_target  # noqa: E402

# 一次 ssh 拉回「仓库 标签 digest」；digest 为空的行（本地构建、无 repo digest）后面单查 image id
DIGESTS_CMD = "docker images --digests --format '{{.Repository}}\t{{.Tag}}\t{{.Digest}}' 2>&1"
# 顺带取 CANN / torch 版本（拿不到就留空，不猜）
VERSIONS_CMD = (
    "for p in /usr/local/Ascend/ascend-toolkit/latest /usr/local/Ascend/latest; do "
    "v=$(basename \"$(readlink -f $p 2>/dev/null)\" 2>/dev/null); "
    "[ -n \"$v\" ] && [ \"$v\" != \".\" ] && { echo \"CANN=$v\"; break; }; done"
)


def parse_digests(text: str) -> dict[tuple[str, str], str]:
    out: dict[tuple[str, str], str] = {}
    for line in text.splitlines():
        parts = line.split("\t")
        if len(parts) != 3:
            continue
        repo, tag, digest = (p.strip() for p in parts)
        if repo.startswith("REPOSITORY") or not digest or digest == "<none>":
            continue
        out[(repo, tag)] = digest
    return out


def local_id_batch_cmd(refs: list[str]) -> str:
    """本地构建的镜像没有 repo digest → 用镜像 ID 作内容标识（`local-id:sha256:...`）。

    一次 ssh 里循环查完，避免 68 个镜像 68 次连接。
    """
    quoted = " ".join(f"'{r}'" for r in refs)
    return (
        f"for r in {quoted}; do "
        "id=$(docker image inspect \"$r\" --format '{{.Id}}' 2>/dev/null); "
        "[ -n \"$id\" ] && echo \"$r\t$id\"; done"
    )


def parse_local_ids(text: str) -> dict[str, str]:
    out: dict[str, str] = {}
    for line in text.splitlines():
        parts = line.split("\t")
        if len(parts) == 2 and parts[1].startswith("sha256:"):
            out[parts[0]] = "local-id:" + parts[1]
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--op-dev", action="store_true", help="只同步 is_op_dev=1 的镜像")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    with SessionLocal() as db:
        hosts = {h.id: h for h in db.scalars(select(Host))}
        q = select(HostImage)
        if args.op_dev:
            q = q.where(HostImage.is_op_dev.is_(True))
        images = list(db.scalars(q))

        by_host: dict[int, list[HostImage]] = {}
        for im in images:
            by_host.setdefault(im.host_id, []).append(im)

        total = 0
        for host_id, imgs in by_host.items():
            host = hosts.get(host_id)
            if host is None:
                continue
            # 与页面/探针/真执行同源：拆端口、按 host.cred_ref 取私钥（空则回退全局）
            target = ssh_target(host)
            if not target:
                print(f"跳过 host #{host_id}：没有 ssh_alias/address")
                continue
            key = ssh_key(host)
            port = ssh_port(host)
            rc, out = ssh_exec(target, DIGESTS_CMD, key, settings.ssh_opts, settings.ssh_timeout, port=port)
            if rc != 0:
                print(f"host {target}: docker images 失败 rc={rc}: {out[:200]}")
                continue
            digests = parse_digests(out)
            rc, ver_out = ssh_exec(target, VERSIONS_CMD, key, settings.ssh_opts, settings.ssh_timeout, port=port)
            cann_ver = None
            if rc == 0:
                for line in ver_out.splitlines():
                    if line.startswith("CANN="):
                        cann_ver = line.split("=", 1)[1][:120]
            updated = 0
            missing = [f"{im.image_ref}:{im.tag}" for im in imgs
                       if not digests.get((im.image_ref, im.tag or ""))]
            local_ids: dict[str, str] = {}
            if missing:
                rc2, id_out = ssh_exec(target, local_id_batch_cmd(missing), key,
                                       settings.ssh_opts, max(settings.ssh_timeout, 60), port=port)
                if rc2 == 0:
                    local_ids = parse_local_ids(id_out)
            for im in imgs:
                ref = f"{im.image_ref}:{im.tag}"
                d = digests.get((im.image_ref, im.tag or "")) or local_ids.get(ref)
                if d and im.digest != d:
                    im.digest = d
                    updated += 1
                if cann_ver and not im.cann_version:
                    im.cann_version = cann_ver
            print(f"host {target}: repo digest {len(digests)} 条 + 本地镜像 ID {len(local_ids)} 条，"
                  f"更新 {updated} 个镜像" + (f"，CANN={cann_ver}" if cann_ver else ""))
            total += updated
        if args.dry_run:
            db.rollback()
            print(f"[dry-run] 会更新 {total} 行，未提交")
        else:
            db.commit()
            print(f"已提交：更新 {total} 行（digest / cann_version）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
