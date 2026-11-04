"""任务级文档（任务生命周期 v2）：上传 / 解析 / 存储 / 下载辅助。

契约：`docs/impl-contract-v2.md` §3.1/§3.2；设计：`docs/plan-task-lifecycle-v2.md` §3.1。

要点：
- 只收 `.docx` / `.md`，单文件 ≤ 20MB（路由层流式校验，这里再兜一次）。
- docx **只用标准库**解析（`zipfile` + `xml.etree.ElementTree` 读 `word/document.xml`），
  按文档顺序抽段落 + 表格（表格转 Markdown）；不引入 python-docx（backend 依赖里没有）。
- 存储：`<project.workspace_path>/task-docs/task-<id>/<filename>` 原文 +
  `<stem>.extracted.md` 解析稿（md 原文直接复制为解析稿）。
- 同名覆盖：先删旧行 + 旧文件再写；解析失败 → `DocumentError` 且不落库、不落文件。
- 文件名/路径全部做穿越校验：只接受单段文件名，落盘前再确认解析后的路径仍在该任务目录内。
"""

from __future__ import annotations

import hashlib
import io
import re
import uuid
import xml.etree.ElementTree as ET
import zipfile
from pathlib import Path

from sqlalchemy import select
from sqlalchemy.orm import Session

from ..models import Project, Task, TaskDocument, utcnow

#: 单文件大小上限（与对话附件一致，契约 §3.2）
MAX_BYTES = 20 * 1024 * 1024
#: 允许的扩展名（小写比较）
ALLOWED_EXTS = (".docx", ".md")
#: 工作区里的文档根目录名
DOCS_ROOT = "task-docs"
#: 默认文档类型（契约 task_documents.kind）
DEFAULT_KIND = "design"

_DOCX_MIME = "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
_MD_MIME = "text/markdown; charset=utf-8"
_MIME_BY_EXT = {".docx": _DOCX_MIME, ".md": _MD_MIME}

#: WordprocessingML 命名空间（读 `word/document.xml` 只需要这一个）
_W = "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}"
_HEADING_RE = re.compile(r"^(?:heading|Heading|标题)(\d+)$")
#: 单段文件名里一律拒绝的字符（路径分隔符 + 控制字符）
_BAD_FILENAME_CHARS = re.compile(r"[\x00-\x1f\x7f]")


class DocumentError(ValueError):
    """文档不合法 / 解析失败 / 工作区不可用。`status_code` 给路由映射 HTTP 状态码。"""

    def __init__(self, message: str, status_code: int = 400) -> None:
        super().__init__(message)
        self.status_code = status_code


# ---------------------------------------------------------------------------
# 工作区 / 路径
# ---------------------------------------------------------------------------

def workspace_of(db: Session, task: Task) -> Path:
    """任务所属项目的工作区（必须是已存在的目录）。"""
    project = db.get(Project, task.project_id)
    if project is None or not project.workspace_path:
        raise DocumentError("项目的 workspace_path 未配置")
    workspace = Path(project.workspace_path).expanduser()
    if not workspace.is_dir():
        raise DocumentError(f"工作区不存在：{workspace}")
    return workspace


def docs_dir(workspace: Path, task_id: int) -> Path:
    """任务的文档目录：`<workspace>/task-docs/task-<id>/`。"""
    return workspace / DOCS_ROOT / f"task-{int(task_id)}"


def _safe_name(filename: str) -> str:
    """只接受**单段**文件名：拒绝空名 / 路径分隔符 / `.`、`..` / 控制字符 / 过长。

    刻意不做「自动取 basename」：客户端传了路径说明调用方有问题，直接 400 比静默改写更安全。
    """
    name = str(filename or "").strip()
    if not name:
        raise DocumentError("缺少文件名")
    if name in (".", "..") or "/" in name or "\\" in name or ".." in name:
        raise DocumentError(f"非法文件名：{name!r}")
    if _BAD_FILENAME_CHARS.search(name):
        raise DocumentError(f"非法文件名：{name!r}")
    if len(name) > 300:  # 与 task_documents.filename 列宽一致
        raise DocumentError("文件名过长（上限 300 字符）")
    if not Path(name).stem or Path(name).stem in (".", ""):
        raise DocumentError(f"非法文件名：{name!r}")
    return name


def _check_ext(filename: str) -> str:
    ext = Path(filename).suffix.lower()
    if ext not in ALLOWED_EXTS:
        raise DocumentError(f"不支持的文档格式：{ext or '（无扩展名）'}（仅支持 .docx / .md）")
    return ext


def _within(base: Path, name: str) -> Path:
    """`base/name` 的解析路径必须仍在 `base` 下（防 `..` / 符号链接穿越）。"""
    base_resolved = base.resolve()
    target = (base / name).resolve()
    if target != base_resolved and base_resolved not in target.parents:
        raise DocumentError("文档路径越界")
    return base / name


def _resolve_rel(workspace: Path, rel_path: str | None) -> Path | None:
    """把 DB 里存的相对工作区路径还原成绝对路径，并确认没跑出工作区。"""
    if not rel_path:
        return None
    base = workspace.resolve()
    target = (workspace / rel_path).resolve()
    if target != base and base not in target.parents:
        raise DocumentError("文档路径越界")
    return workspace / rel_path


def absolute_path(workspace: Path, row: TaskDocument) -> Path:
    """某个文档原文的绝对路径（路径来自 DB，仍按工作区做穿越校验）。"""
    path = _resolve_rel(workspace, row.path)
    if path is None:  # pragma: no cover - 模型里 path 非空；防御历史脏数据
        raise DocumentError("文档路径为空")
    return path


# ---------------------------------------------------------------------------
# docx → Markdown（纯标准库）
# ---------------------------------------------------------------------------

def _text_of(element: ET.Element) -> str:
    """按文档顺序取一个元素里的可见文字（w:t 文本 + 制表符/换行）。"""
    parts: list[str] = []
    for node in element.iter():
        if node.tag == f"{_W}t":
            parts.append(node.text or "")
        elif node.tag == f"{_W}tab":
            parts.append("\t")
        elif node.tag in (f"{_W}br", f"{_W}cr"):
            parts.append("\n")
    return "".join(parts)


def _paragraph_markdown(paragraph: ET.Element) -> str:
    """一个 `w:p` → Markdown 行（标题 / 列表 / 普通段落）。

    标题按 `w:pStyle`（Heading1..9 / Title）或 `w:outlineLvl` 识别；
    列表按 ListBullet / ListNumber 样式的常见 id 识别。识别不出就按普通段落输出。
    """
    text = _text_of(paragraph).strip()
    if not text:
        return ""
    style = ""
    level: int | None = None
    ppr = paragraph.find(f"{_W}pPr")
    if ppr is not None:
        pstyle = ppr.find(f"{_W}pStyle")
        if pstyle is not None:
            style = pstyle.get(f"{_W}val") or ""
        match = _HEADING_RE.match(style)
        if match:
            level = int(match.group(1))
        elif style == "Title":
            level = 1
        if level is None:
            outline = ppr.find(f"{_W}outlineLvl")
            if outline is not None:
                raw = outline.get(f"{_W}val")
                if raw is not None and raw.isdigit() and int(raw) < 9:
                    level = int(raw) + 1
    if level is not None:
        return "#" * min(max(level, 1), 6) + " " + text
    lowered = style.lower()
    if lowered.startswith("listbullet"):
        return "- " + text
    if lowered.startswith("listnumber"):
        return "1. " + text
    return text


def _cell_text(cell: ET.Element) -> str:
    """表格单元格文字：多段落用换行拼（Markdown 里再转 `<br>`）。"""
    lines = []
    for paragraph in cell.iter(f"{_W}p"):
        text = _text_of(paragraph).strip()
        if text:
            lines.append(text)
    return "\n".join(lines)


def _table_markdown(table: ET.Element) -> str:
    """一个 `w:tbl` → Markdown 表格（首行当表头，空行跳过，列数取最大）。"""
    rows: list[list[str]] = []
    for tr in table.findall(f"{_W}tr"):
        cells = [_cell_text(tc) for tc in tr.findall(f"{_W}tc")]
        if any(cell.strip() for cell in cells):
            rows.append(cells)
    if not rows:
        return ""
    width = max(len(row) for row in rows)
    rows = [row + [""] * (width - len(row)) for row in rows]

    def _esc(value: str) -> str:
        return value.replace("|", "\\|").replace("\n", "<br>").strip()

    lines = ["| " + " | ".join(_esc(cell) for cell in rows[0]) + " |",
             "| " + " | ".join(["---"] * width) + " |"]
    for row in rows[1:]:
        lines.append("| " + " | ".join(_esc(cell) for cell in row) + " |")
    return "\n".join(lines)


def _iter_body_blocks(container: ET.Element):
    """按文档顺序展开 body 级块：段落 `w:p` / 表格 `w:tbl`。

    真实 Word 文档常把正文包一层结构标签（内容控件 `w:sdt/w:sdtContent`、`w:customXml`、
    修订插入 `w:ins`），这里递归展开它们；修订删除 `w:del` 里的内容不输出。
    """
    for child in container:
        if child.tag == f"{_W}p":
            yield "p", child
        elif child.tag == f"{_W}tbl":
            yield "tbl", child
        elif child.tag in (f"{_W}sdt", f"{_W}customXml", f"{_W}ins"):
            if child.tag == f"{_W}sdt":
                child = child.find(f"{_W}sdtContent")
                if child is None:
                    continue
            yield from _iter_body_blocks(child)


def parse_docx(data: bytes) -> str:
    """docx 字节流 → Markdown 文本：按 body 文档顺序输出段落与表格。

    任何结构问题（不是 zip / 缺 `word/document.xml` / XML 非法 / 缺 body）都抛
    `DocumentError`（路由映射 400，调用方保证不落库）。
    """
    try:
        with zipfile.ZipFile(io.BytesIO(data)) as zf:
            try:
                xml = zf.read("word/document.xml")
            except KeyError as e:
                raise DocumentError("docx 解析失败：缺少 word/document.xml") from e
    except zipfile.BadZipFile as e:
        raise DocumentError("docx 解析失败：不是有效的 .docx 文件（zip 头损坏）") from e
    try:
        root = ET.fromstring(xml)
    except ET.ParseError as e:
        raise DocumentError(f"docx 解析失败：word/document.xml 不是合法 XML（{e}）") from e

    body = root.find(f"{_W}body")
    if body is None:
        raise DocumentError("docx 解析失败：缺少 w:body")

    blocks: list[str] = []
    for kind, element in _iter_body_blocks(body):
        text = _paragraph_markdown(element) if kind == "p" else _table_markdown(element)
        if text:
            blocks.append(text)
    text = "\n\n".join(blocks)
    text = re.sub(r"\n{3,}", "\n\n", text).strip()
    return text + "\n" if text else ""


def extract_markdown(filename: str, data: bytes) -> bytes:
    """原文 → 解析稿字节：docx 走纯标准库解析，md 原文直接复制。"""
    ext = _check_ext(filename)
    if len(data) == 0:
        raise DocumentError("文件为空")
    if len(data) > MAX_BYTES:
        raise DocumentError("文档超过 20MB 上限", status_code=413)
    if ext == ".docx":
        text = parse_docx(data)
        return text.encode("utf-8")
    return bytes(data)


def media_type(filename: str) -> str:
    return _MIME_BY_EXT.get(Path(filename).suffix.lower(), "application/octet-stream")


# ---------------------------------------------------------------------------
# 存储
# ---------------------------------------------------------------------------

def _remove_file(workspace: Path | None, rel_path: str | None) -> None:
    """尽力删除工作区内的一个相对路径文件（不存在 / 越界 / 权限问题都忽略）。"""
    if workspace is None or not rel_path:
        return
    try:
        path = _resolve_rel(workspace, rel_path)
    except DocumentError:
        return
    if path is None:
        return
    try:
        if path.is_file():
            path.unlink()
    except OSError:  # pragma: no cover - 删不掉不影响 DB 行删除
        pass


def _row_paths(row: TaskDocument) -> list[str]:
    return [path for path in (row.path, row.extracted_path) if path]


def save(db: Session, task: Task, filename: str, data: bytes) -> TaskDocument:
    """校验 + 解析 + 落盘 + 落库；同名覆盖（先删旧行/旧文件再写）。

    解析在**任何写入之前**完成：docx 解析失败时不留文件、不留 DB 行。
    """
    name = _safe_name(filename)
    _check_ext(name)
    if len(data) > MAX_BYTES:
        raise DocumentError("文档超过 20MB 上限", status_code=413)
    if not data:
        raise DocumentError("文件为空")
    extracted = extract_markdown(name, data)  # 解析失败 → DocumentError（不落库）

    workspace = workspace_of(db, task)
    # 一次性校验完整目录。若 task-docs/task-N 已被换成指向工作区外的符号链接，
    # 只校验 task-docs 会把上传内容写到任意外部目录。
    directory = _within(workspace, f"{DOCS_ROOT}/task-{int(task.id)}")
    target = _within(directory, name)
    stem = Path(name).stem
    extracted_name = f"{stem}.extracted.md"
    target_extracted = _within(directory, extracted_name)
    rel_path = f"{DOCS_ROOT}/task-{int(task.id)}/{name}"
    rel_extracted = f"{DOCS_ROOT}/task-{int(task.id)}/{extracted_name}"

    # 同名文件覆盖：连解析稿同名的旧行一起替换（例如 a.docx 与 a.md 的解析稿都叫
    # a.extracted.md）。这些旧行对应的原文/解析稿文件在写入成功后统一清掉。
    old_rows = list(db.scalars(select(TaskDocument).where(
        TaskDocument.task_id == task.id,
        (TaskDocument.filename == name) | (TaskDocument.path == rel_path)
        | (TaskDocument.extracted_path == rel_extracted),
    )))
    old_paths = [path for row in old_rows for path in _row_paths(row)]
    new_paths = {rel_path, rel_extracted}

    directory.mkdir(parents=True, exist_ok=True)
    tag = uuid.uuid4().hex
    tmp_original = directory / f".{tag}.upload.tmp"
    tmp_extracted = directory / f".{tag}.extracted.tmp"
    written: list[Path] = []
    try:
        tmp_original.write_bytes(data)
        written.append(tmp_original)
        tmp_extracted.write_bytes(extracted)
        written.append(tmp_extracted)
        tmp_original.replace(target)
        tmp_extracted.replace(target_extracted)
        for row in old_rows:
            db.delete(row)
        row = TaskDocument(
            task_id=task.id,
            filename=name,
            kind=DEFAULT_KIND,
            path=rel_path,
            extracted_path=rel_extracted,
            sha256=hashlib.sha256(data).hexdigest(),
            size_bytes=len(data),
            created_at=utcnow(),
        )
        db.add(row)
        db.commit()
        db.refresh(row)
    except OSError as e:
        db.rollback()
        raise DocumentError(f"文档保存失败：{e}", status_code=500) from e
    except Exception:
        db.rollback()
        raise
    finally:
        for path in written:
            try:
                if path.exists():
                    path.unlink()
            except OSError:  # pragma: no cover
                pass

    # 写入 + 提交都成功后才清旧文件（失败的旧行已随 rollback 恢复，文件不能先删）。
    for old_rel in old_paths:
        if old_rel not in new_paths:
            _remove_file(workspace, old_rel)
    return row


def list_for_task(db: Session, task: Task) -> list[TaskDocument]:
    """任务文档列表（新→旧）。"""
    return list(db.scalars(select(TaskDocument)
                           .where(TaskDocument.task_id == task.id)
                           .order_by(TaskDocument.created_at.desc(), TaskDocument.id.desc())))


def get(db: Session, task: Task, doc_id: int) -> TaskDocument | None:
    row = db.get(TaskDocument, doc_id)
    if row is None or row.task_id != task.id:
        return None
    return row


def delete(db: Session, task: Task, doc_id: int) -> bool:
    """删除文档（定版后也允许，契约 §3.2）：先删行，再尽力删磁盘文件。"""
    row = get(db, task, doc_id)
    if row is None:
        return False
    try:
        workspace = workspace_of(db, task)
    except DocumentError:
        workspace = None  # 工作区没了也要能删掉 DB 行
    paths = _row_paths(row)
    db.delete(row)
    db.commit()
    for rel_path in paths:
        _remove_file(workspace, rel_path)
    return True


def as_dict(row: TaskDocument) -> dict:
    """契约 §3.2 的响应结构（路径均为相对工作区）。"""
    return {
        "id": row.id,
        "task_id": row.task_id,
        "filename": row.filename,
        "kind": row.kind,
        "path": row.path,
        "extracted_path": row.extracted_path,
        "sha256": row.sha256,
        "size_bytes": row.size_bytes,
        "created_at": row.created_at,
    }
