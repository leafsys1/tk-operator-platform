"""任务文档上传/解析/下载/删除 + 设计文档模板下载（契约 §3.1/§3.2）。

覆盖点（与 docs/impl-contract-v2.md 对齐）：
- `.md` 上传：原文=解析稿（直接复制）、列表/下载/删除；
- `.docx` 上传：**纯标准库**解析出段落 + 表格（Markdown 表格）；
- 模板下载：docx / md 与 `<repo_root>/docs/templates/` 源文件逐字节一致；
- 同名覆盖：旧行/旧文件删掉，列表只留一条，磁盘只有一对文件；
- 非法格式 / 空文件 / 超限 / 解析失败 → 4xx 且不落库、不落文件；
- 路径穿越、任务不存在、跨任务文档 404。
"""

from __future__ import annotations

import hashlib
import io
import zipfile
from pathlib import Path

import pytest

from app.config import settings
from app.models import Project, Task, TaskDocument

W_NS = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"
DOCX_MIME = "application/vnd.openxmlformats-officedocument.wordprocessingml.document"


# ---------------------------------------------------------------------------
# 夹具 / 构造器
# ---------------------------------------------------------------------------

def _seed_task(Session, workspace: Path, *, is_draft: bool = True, op: str = "add",
               title: str = "elementwise_add") -> int:
    """建项目（指向测试工作区）+ 任务；与 conftest 的 fixture 风格一致（Service 层直接建）。"""
    with Session() as db:
        project = Project(name=f"p-{title}-{is_draft}", workspace_path=str(workspace))
        db.add(project)
        db.flush()
        task = Task(project_id=project.id, title=title, op_name=op, is_draft=is_draft)
        db.add(task)
        db.commit()
        return task.id


def _paragraph(text: str, style: str | None = None) -> str:
    ppr = f'<w:pPr><w:pStyle w:val="{style}"/></w:pPr>' if style else ""
    return f'<w:p>{ppr}<w:r><w:t xml:space="preserve">{text}</w:t></w:r></w:p>'


def _table(rows: list[list[str]]) -> str:
    body = []
    for row in rows:
        cells = "".join(
            f"<w:tc><w:p><w:r><w:t>{cell}</w:t></w:r></w:p></w:tc>" for cell in row
        )
        body.append(f"<w:tr>{cells}</w:tr>")
    return f"<w:tbl>{''.join(body)}</w:tbl>"


def _docx(*body_xml: str) -> bytes:
    """手工构造最小 docx（只有 word/document.xml 参与解析；不依赖 python-docx）。"""
    document = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        f'<w:document xmlns:w="{W_NS}"><w:body>{"".join(body_xml)}'
        '<w:sectPr/></w:body></w:document>'
    )
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("[Content_Types].xml",
                    '<?xml version="1.0"?><Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types"/>')
        zf.writestr("word/document.xml", document)
    return buf.getvalue()


def _upload(c, task_id: int, filename: str, data: bytes, mime: str = "application/octet-stream"):
    return c.post(f"/api/tasks/{task_id}/documents",
                  files={"file": (filename, data, mime)})


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


# ---------------------------------------------------------------------------
# md
# ---------------------------------------------------------------------------

def test_upload_md_lists_downloads_and_deletes(client, workspace: Path):
    c, Session = client
    task_id = _seed_task(Session, workspace)
    body = "# 需求背景\n\n算子：elementwise_add\n".encode("utf-8")

    resp = _upload(c, task_id, "算子设计文档.md", body, "text/markdown")
    assert resp.status_code == 201, resp.text
    doc = resp.json()
    assert doc["task_id"] == task_id
    assert doc["filename"] == "算子设计文档.md"
    assert doc["kind"] == "design"
    assert doc["path"] == f"task-docs/task-{task_id}/算子设计文档.md"
    assert doc["extracted_path"] == f"task-docs/task-{task_id}/算子设计文档.extracted.md"
    assert doc["sha256"] == _sha(body)
    assert doc["size_bytes"] == len(body)
    assert doc["created_at"]

    # 原文与解析稿都落盘；md 的解析稿 = 原文复制
    original = workspace / doc["path"]
    extracted = workspace / doc["extracted_path"]
    assert original.read_bytes() == body
    assert extracted.read_bytes() == body

    listed = c.get(f"/api/tasks/{task_id}/documents")
    assert listed.status_code == 200
    assert [item["id"] for item in listed.json()] == [doc["id"]]

    download = c.get(f"/api/tasks/{task_id}/documents/{doc['id']}/download")
    assert download.status_code == 200
    assert download.content == body
    assert download.headers["content-type"].startswith("text/markdown")

    assert c.delete(f"/api/tasks/{task_id}/documents/{doc['id']}").status_code == 204
    assert c.get(f"/api/tasks/{task_id}/documents").json() == []
    assert not original.exists() and not extracted.exists()
    # 再删 → 404
    assert c.delete(f"/api/tasks/{task_id}/documents/{doc['id']}").status_code == 404


def test_draft_task_allows_upload(client, workspace: Path):
    """草稿任务（is_draft=1）允许上传（契约 §3.2；任务开始页的核心入口）。"""
    c, Session = client
    task_id = _seed_task(Session, workspace, is_draft=True)
    resp = _upload(c, task_id, "notes.md", b"draft doc", "text/markdown")
    assert resp.status_code == 201, resp.text
    assert resp.json()["filename"] == "notes.md"


def test_same_filename_overwrites(client, workspace: Path):
    """同名文件覆盖：删除旧行 + 旧文件再写，列表不留两条。"""
    c, Session = client
    task_id = _seed_task(Session, workspace)
    first = _upload(c, task_id, "note.md", b"version 1", "text/markdown")
    assert first.status_code == 201, first.text
    doc_dir = workspace / "task-docs" / f"task-{task_id}"
    assert sorted(p.name for p in doc_dir.iterdir()) == ["note.extracted.md", "note.md"]

    second = _upload(c, task_id, "note.md", b"version 2!", "text/markdown")
    assert second.status_code == 201, second.text
    assert second.json()["id"] != first.json()["id"]
    assert second.json()["sha256"] == _sha(b"version 2!")

    listed = c.get(f"/api/tasks/{task_id}/documents").json()
    assert len(listed) == 1, "同名覆盖后列表应只剩一条"
    assert listed[0]["id"] == second.json()["id"]
    assert (workspace / listed[0]["path"]).read_bytes() == b"version 2!"
    assert (workspace / listed[0]["extracted_path"]).read_bytes() == b"version 2!"
    assert sorted(p.name for p in doc_dir.iterdir()) == ["note.extracted.md", "note.md"]

    with Session() as db:
        rows = db.query(TaskDocument).filter(TaskDocument.task_id == task_id).all()
        assert len(rows) == 1


# ---------------------------------------------------------------------------
# docx
# ---------------------------------------------------------------------------

def test_upload_docx_extracts_paragraphs_and_tables(client, workspace: Path):
    c, Session = client
    task_id = _seed_task(Session, workspace)
    data = _docx(
        _paragraph("《add》算子设计文档", "Title"),
        _paragraph("1. 需求背景", "Heading1"),
        _paragraph("1.1 数学语义", "Heading2"),
        _paragraph("普通段落：out = a + b"),
        _table([["输入", "输出"], ["a | b", "out"]]),
    )

    resp = _upload(c, task_id, "算子设计文档.docx", data, DOCX_MIME)
    assert resp.status_code == 201, resp.text
    doc = resp.json()
    assert doc["path"] == f"task-docs/task-{task_id}/算子设计文档.docx"
    assert doc["extracted_path"] == f"task-docs/task-{task_id}/算子设计文档.extracted.md"
    assert (workspace / doc["path"]).read_bytes() == data  # 原文原样保存

    text = (workspace / doc["extracted_path"]).read_text(encoding="utf-8")
    assert "# 《add》算子设计文档" in text  # Title → 一级标题
    assert "# 1. 需求背景" in text  # Heading1
    assert "## 1.1 数学语义" in text  # Heading2
    assert "普通段落：out = a + b" in text
    # 表格转 Markdown（首行表头；单元格内的 | 转义）
    assert "| 输入 | 输出 |" in text
    assert "| --- | --- |" in text
    assert r"| a \| b | out |" in text

    download = c.get(f"/api/tasks/{task_id}/documents/{doc['id']}/download")
    assert download.status_code == 200
    assert download.content == data
    assert download.headers["content-type"].startswith(DOCX_MIME)


def test_docx_content_controls_and_missing_body(client, workspace: Path):
    """真实 Word 文档可能把正文包在内容控件（w:sdt）里；缺 body / 缺 document.xml → 解析失败。"""
    from app.services import documents

    wrapped = _docx(
        "<w:sdt><w:sdtContent>" + _paragraph("内容控件里的段落", "Heading1") + "</w:sdtContent></w:sdt>",
        _table([["a", "b"], ["1", "2"]]),
    )
    parsed = documents.parse_docx(wrapped)
    assert "# 内容控件里的段落" in parsed
    assert "| 1 | 2 |" in parsed

    no_body = io.BytesIO()
    with zipfile.ZipFile(no_body, "w") as zf:
        zf.writestr("word/document.xml", f'<w:document xmlns:w="{W_NS}"/>')
    with pytest.raises(documents.DocumentError):
        documents.parse_docx(no_body.getvalue())

    no_document = io.BytesIO()
    with zipfile.ZipFile(no_document, "w") as zf:
        zf.writestr("word/styles.xml", "<styles/>")
    with pytest.raises(documents.DocumentError):
        documents.parse_docx(no_document.getvalue())


def test_docx_parse_failure_is_400_and_persists_nothing(client, workspace: Path):
    """解析失败 → 400，不落库、不落文件（解析在任何写入之前）。"""
    c, Session = client
    task_id = _seed_task(Session, workspace)

    bad_zip = _upload(c, task_id, "坏文件.docx", b"not a zip at all", DOCX_MIME)
    assert bad_zip.status_code == 400
    assert "docx" in bad_zip.json()["detail"].lower() or "解析" in bad_zip.json()["detail"]

    # 合法 zip 但 document.xml 不是合法 XML
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr("word/document.xml", "<w:document")
    malformed = _upload(c, task_id, "坏XML.docx", buf.getvalue(), DOCX_MIME)
    assert malformed.status_code == 400

    assert c.get(f"/api/tasks/{task_id}/documents").json() == []
    assert not (workspace / "task-docs" / f"task-{task_id}").exists()
    with Session() as db:
        assert db.query(TaskDocument).filter(TaskDocument.task_id == task_id).count() == 0


# ---------------------------------------------------------------------------
# 校验：格式 / 空文件 / 超限 / 路径穿越 / 404
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("filename", ["设计.txt", "设计.pdf", "noext"])
def test_rejects_unsupported_format(client, workspace: Path, filename: str):
    c, Session = client
    task_id = _seed_task(Session, workspace)
    resp = _upload(c, task_id, filename, b"content")
    assert resp.status_code == 400
    assert "仅支持" in resp.json()["detail"]
    assert c.get(f"/api/tasks/{task_id}/documents").json() == []


def test_rejects_empty_file(client, workspace: Path):
    c, Session = client
    task_id = _seed_task(Session, workspace)
    resp = _upload(c, task_id, "empty.md", b"", "text/markdown")
    assert resp.status_code == 400
    assert c.get(f"/api/tasks/{task_id}/documents").json() == []


def test_rejects_too_large(client, workspace: Path, monkeypatch):
    """≤20MB：路由层流式校验（这里把上限调小，避免测试真造 20MB 请求体）。"""
    from app.services import documents

    monkeypatch.setattr(documents, "MAX_BYTES", 32)
    c, Session = client
    task_id = _seed_task(Session, workspace)
    resp = _upload(c, task_id, "big.md", b"x" * 64, "text/markdown")
    assert resp.status_code == 413
    assert "20MB" in resp.json()["detail"]
    assert c.get(f"/api/tasks/{task_id}/documents").json() == []


@pytest.mark.parametrize("filename", ["../evil.md", "..\\evil.md", "sub/evil.md", "/etc/evil.md"])
def test_rejects_path_traversal(client, workspace: Path, filename: str):
    c, Session = client
    task_id = _seed_task(Session, workspace)
    resp = _upload(c, task_id, filename, b"pwned", "text/markdown")
    assert resp.status_code == 400, resp.text
    assert not (workspace.parent / "evil.md").exists()
    assert not (workspace / "evil.md").exists()
    assert c.get(f"/api/tasks/{task_id}/documents").json() == []


def test_rejects_symlinked_task_document_directory(client, workspace: Path, tmp_path: Path):
    """既有 task-N 目录若指向工作区外，上传不能把文件写到链接目标。"""
    c, Session = client
    task_id = _seed_task(Session, workspace)
    outside = tmp_path / "outside-docs"
    outside.mkdir()
    docs_root = workspace / "task-docs"
    docs_root.mkdir()
    (docs_root / f"task-{task_id}").symlink_to(outside, target_is_directory=True)

    response = _upload(c, task_id, "escape.md", b"must stay inside", "text/markdown")
    assert response.status_code == 400, response.text
    assert not (outside / "escape.md").exists()
    assert c.get(f"/api/tasks/{task_id}/documents").json() == []


def test_task_not_found(client):
    c, _ = client
    assert c.get("/api/tasks/999999/documents").status_code == 404
    assert _upload(c, 999999, "a.md", b"x", "text/markdown").status_code == 404


def test_document_belongs_to_task(client, workspace: Path):
    """跨任务的 doc_id 不能读到（否则等于绕过任务隔离）。"""
    c, Session = client
    task_a = _seed_task(Session, workspace, title="a")
    task_b = _seed_task(Session, workspace, title="b")
    doc = _upload(c, task_a, "a.md", b"aaa", "text/markdown").json()
    assert c.get(f"/api/tasks/{task_b}/documents/{doc['id']}/download").status_code == 404
    assert c.delete(f"/api/tasks/{task_b}/documents/{doc['id']}").status_code == 404


# ---------------------------------------------------------------------------
# 模板下载
# ---------------------------------------------------------------------------

def test_download_operator_design_doc_templates(client):
    c, _ = client
    cases = {
        "docx": ("算子设计文档模板.docx",
                 "application/vnd.openxmlformats-officedocument.wordprocessingml.document"),
        "md": ("算子设计文档模板.md", "text/markdown"),
    }
    for fmt, (filename, media_type) in cases.items():
        resp = c.get("/api/templates/operator-design-doc", params={"format": fmt})
        assert resp.status_code == 200, resp.text
        source = Path(settings.repo_root) / "docs" / "templates" / filename
        assert source.is_file(), f"模板源文件不存在：{source}"
        assert resp.content == source.read_bytes()
        assert resp.headers["content-type"].startswith(media_type)
        assert "attachment" in resp.headers["content-disposition"]


def test_download_operator_design_doc_rejects_bad_format(client):
    c, _ = client
    resp = c.get("/api/templates/operator-design-doc", params={"format": "pdf"})
    assert resp.status_code == 400
    assert "format" in resp.json()["detail"]
