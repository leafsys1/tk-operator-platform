#!/usr/bin/env python3
"""生成《算子设计文档模板》docx（docs/templates/算子设计文档模板.docx）。

与同名 .md 模板同构：背景—分析—设计—验证 五段 + 可选用例附录。
重新生成：用 DSH 自带 python（python-docx）执行本脚本，无需其它依赖。
"""

from __future__ import annotations

from pathlib import Path

from docx import Document
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Cm, Pt, RGBColor

OUT = Path(__file__).resolve().parents[1] / "docs" / "templates" / "算子设计文档模板.docx"

CN_FONT = "PingFang SC"
MONO_FONT = "Consolas"
GRAY = RGBColor(0x66, 0x66, 0x66)
HEADER_FILL = "D9E2F3"
BOX_FILL = "F2F2F2"


def set_run_font(run, *, size: float | None = None, bold: bool | None = None,
                 italic: bool | None = None, color: RGBColor | None = None,
                 mono: bool = False) -> None:
    """同时设置西文与东亚字体（只设 font.name 中文会回退）。"""
    name = MONO_FONT if mono else CN_FONT
    run.font.name = name
    rpr = run._element.get_or_add_rPr()
    rfonts = rpr.find(qn("w:rFonts"))
    if rfonts is None:
        rfonts = OxmlElement("w:rFonts")
        rpr.append(rfonts)
    rfonts.set(qn("w:eastAsia"), name)
    rfonts.set(qn("w:ascii"), name)
    rfonts.set(qn("w:hAnsi"), name)
    if size is not None:
        run.font.size = Pt(size)
    if bold is not None:
        run.font.bold = bold
    if italic is not None:
        run.font.italic = italic
    if color is not None:
        run.font.color.rgb = color


def shade_cell(cell, fill: str) -> None:
    tc_pr = cell._tc.get_or_add_tcPr()
    shd = OxmlElement("w:shd")
    shd.set(qn("w:val"), "clear")
    shd.set(qn("w:color"), "auto")
    shd.set(qn("w:fill"), fill)
    tc_pr.append(shd)


def cell_text(cell, text: str, *, bold: bool = False, size: float = 9,
              color: RGBColor | None = None) -> None:
    cell.text = ""
    p = cell.paragraphs[0]
    p.paragraph_format.space_before = Pt(1)
    p.paragraph_format.space_after = Pt(1)
    run = p.add_run(text)
    set_run_font(run, size=size, bold=bold, color=color)


def add_table(doc, headers: list[str], rows: list[list[str]], *, widths: list[float] | None = None,
              font_size: float = 9) -> None:
    table = doc.add_table(rows=1, cols=len(headers))
    table.style = "Table Grid"
    table.alignment = WD_TABLE_ALIGNMENT.CENTER
    table.autofit = True
    for i, header in enumerate(headers):
        cell = table.rows[0].cells[i]
        cell_text(cell, header, bold=True, size=font_size)
        shade_cell(cell, HEADER_FILL)
    for row in rows:
        cells = table.add_row().cells
        for i, value in enumerate(row):
            cell_text(cells[i], value, size=font_size)
    if widths:
        for row in table.rows:
            for i, width in enumerate(widths):
                row.cells[i].width = Cm(width)
    doc.add_paragraph()  # 表后留白


def heading(doc, text: str, level: int) -> None:
    p = doc.add_heading(level=level)
    run = p.add_run(text)
    size = {0: 20, 1: 14, 2: 12, 3: 11}[level]
    set_run_font(run, size=size, bold=True, color=RGBColor(0x1F, 0x38, 0x64))


def hint(doc, text: str) -> None:
    p = doc.add_paragraph()
    p.paragraph_format.space_after = Pt(4)
    run = p.add_run("填写提示：" + text)
    set_run_font(run, size=9, italic=True, color=GRAY)


def body(doc, text: str) -> None:
    p = doc.add_paragraph()
    run = p.add_run(text)
    set_run_font(run, size=10.5)


def instruction_box(doc, lines: list[str]) -> None:
    table = doc.add_table(rows=1, cols=1)
    table.style = "Table Grid"
    table.alignment = WD_TABLE_ALIGNMENT.CENTER
    cell = table.rows[0].cells[0]
    shade_cell(cell, BOX_FILL)
    cell.text = ""
    for i, line in enumerate(lines):
        p = cell.paragraphs[0] if i == 0 else cell.add_paragraph()
        p.paragraph_format.space_before = Pt(0)
        p.paragraph_format.space_after = Pt(2)
        run = p.add_run(line)
        bold = i == 0
        set_run_font(run, size=9, bold=bold, color=RGBColor(0x33, 0x33, 0x33))
    doc.add_paragraph()


def set_margins(doc) -> None:
    for section in doc.sections:
        section.top_margin = Cm(2.2)
        section.bottom_margin = Cm(2.0)
        section.left_margin = Cm(2.2)
        section.right_margin = Cm(2.0)


def set_normal_style(doc) -> None:
    style = doc.styles["Normal"]
    style.font.name = CN_FONT
    style.font.size = Pt(10.5)
    style.element.rPr.rFonts.set(qn("w:eastAsia"), CN_FONT)


def add_footer(doc) -> None:
    section = doc.sections[0]
    p = section.footer.paragraphs[0]
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run = p.add_run("op-agent · 算子设计文档模板（AscendC）")
    set_run_font(run, size=8, color=GRAY)
    # 页码：PAGE 域
    p.add_run("  ")
    run = p.add_run()
    fld_begin = OxmlElement("w:fldChar")
    fld_begin.set(qn("w:fldCharType"), "begin")
    instr = OxmlElement("w:instrText")
    instr.set(qn("xml:space"), "preserve")
    instr.text = "PAGE"
    fld_end = OxmlElement("w:fldChar")
    fld_end.set(qn("w:fldCharType"), "end")
    run._r.append(fld_begin)
    run._r.append(instr)
    run._r.append(fld_end)
    set_run_font(run, size=8, color=GRAY)


def build() -> None:
    doc = Document()
    doc.core_properties.title = "算子设计文档模板（AscendC）"
    doc.core_properties.subject = "op-agent 任务开始页 · 设计文档模板"
    doc.core_properties.author = "op-agent"
    set_margins(doc)
    set_normal_style(doc)
    add_footer(doc)

    heading(doc, "《算子名》算子设计文档（AscendC 实现）", 0)

    instruction_box(doc, [
        "填写说明（下载模板后请删除本框）",
        "1. 本文档面向通用算子的 AscendC 实现，按「背景—分析—设计—验证」组织；带（选填）的章节可以不填，由 op-agent 在补全稿里补。",
        "2. 不确定的内容写 UNKNOWN 或「待确认」，不要留空导致语义歧义。",
        "3. 附录 A 的测试用例表可选填：填了会作为用例生成依据；不填则由 agent 根据正文生成后交你确认。",
        "4. 表格可增行；本模板同时提供 docx 与 md 两个版本，任选其一填写回传。",
    ])

    add_table(doc, ["项目", "内容"], [
        ["算子名称", "<op_name>"],
        ["目标芯片 / 架构", "<如 910B1 / dav-2201>"],
        ["文档版本", "v0.1"],
        ["作者 / 日期", "<name> / <YYYY-MM-DD>"],
    ], widths=[5.0, 11.5])
    heading(doc, "修订记录", 3)
    add_table(doc, ["版本", "日期", "修改人", "修改说明"], [
        ["v0.1", "", "", "初稿"],
    ], widths=[2.5, 3.0, 3.0, 8.0])

    # ---------------- 1 需求背景 ----------------
    heading(doc, "1. 需求背景", 1)

    heading(doc, "1.1 需求来源", 2)
    hint(doc, "说明需求从哪来（产品 / 模型 / 客户 / 内部优化）、为什么现在做、验收方是谁。")
    body(doc, "")

    heading(doc, "1.2 业务背景与使用场景", 2)
    hint(doc, "该算子在什么模型 / 网络中承载什么计算；典型调用方式（算子直调 / 图内节点 / 融合子图）；对上下层的影响。")
    body(doc, "")

    heading(doc, "1.3 算子数学语义", 2)
    hint(doc, "用公式 + 文字给出严格定义，例如 y = max(a + b, 0) + alpha × min(a + b, 0)；明确 NaN / Inf / 空张量 / 边界值语义；与 PyTorch 或其他框架同名算子语义是否一致（对齐到哪个版本）。")
    body(doc, "")

    heading(doc, "1.4 历史 TBE 实现与现状分析", 2)
    hint(doc, "若已有 TBE / 其他实现，给出：实现位置、功能范围、输入输出定义（dtype、format、shape）、计算流程概述、已知功能缺口与性能瓶颈；没有历史实现则写「无，首版实现」并说明参考依据。")
    body(doc, "")

    # ---------------- 2 需求分析 ----------------
    heading(doc, "2. 需求分析", 1)

    heading(doc, "2.1 外部组件依赖", 2)
    hint(doc, "列出依赖的 CANN 组件 / 版本、算子开发套件、torch_npu、编译器、驱动等；给出最低版本要求与已知限制。")
    body(doc, "")

    heading(doc, "2.2 内部适配模块", 2)
    hint(doc, "与现有工程 / 框架的适配点：调用接口、图引擎接入方式、workspace 管理、错误码体系、日志 / 维测接口等。")
    body(doc, "")

    heading(doc, "2.3 需求模块设计", 2)
    hint(doc, "本需求的模块拆分（host / kernel / tiling / 校验 / 参考实现 / 测试），每个模块的职责边界与交付物。")
    body(doc, "")

    heading(doc, "2.4 算子原型", 2)
    heading(doc, "2.4.1 输入", 3)
    add_table(doc, ["参数名", "dtype", "format", "shape", "说明"], [
        ["", "", "", "", ""],
    ], widths=[2.8, 2.2, 2.0, 4.5, 5.0])
    heading(doc, "2.4.2 输出", 3)
    add_table(doc, ["参数名", "dtype", "format", "shape", "说明"], [
        ["", "", "", "", ""],
    ], widths=[2.8, 2.2, 2.0, 4.5, 5.0])
    heading(doc, "2.4.3 属性 / 参数", 3)
    add_table(doc, ["名称", "类型", "默认值", "取值范围", "说明"], [
        ["", "", "", "", ""],
    ], widths=[3.0, 2.5, 2.5, 3.5, 5.0])
    heading(doc, "2.4.4 推导规则", 3)
    hint(doc, "输出 shape / dtype 推导、属性推导、约束校验优先级（如 Range → shape → dtype → 参数）、拒绝码与拒绝时机（host 侧还是 kernel 侧）。")
    body(doc, "")

    # ---------------- 3 需求详细设计 ----------------
    heading(doc, "3. 需求详细设计", 1)

    heading(doc, "3.1 使能方式", 2)
    hint(doc, "选择并说明：Ascend C 直调 / Ascend C 注册调用（ACLNN、GEIR）/ 其他；与现有工程的接入点。")
    body(doc, "")

    heading(doc, "3.2 总体设计方案", 2)
    heading(doc, "3.2.1 Host 侧", 3)
    for item in [
        "tiling 与分核策略：tile 大小、分核方式、尾块处理、不同 shape 的自适应策略。",
        "shape / dtype 校验：合法性校验、拒绝码、边界 shape 处理。",
        "workspace 规划：是否需要 workspace、大小估算与生命周期。",
    ]:
        p = doc.add_paragraph(style="List Bullet")
        run = p.add_run(item)
        set_run_font(run, size=10.5)
    heading(doc, "3.2.2 Kernel 侧", 3)
    for item in [
        "数据搬运：GM↔UB 搬运方式（DataCopy / DataCopyPad）、对齐要求、非对齐处理。",
        "向量计算流水：CopyIn → Compute → CopyOut 的流水组织与同步。",
        "精度处理：计算精度、累加 / 舍入、特殊值（NaN / Inf）、与参考实现的误差预期。",
        "结果写回：写回顺序、对齐、尾块处理。",
    ]:
        p = doc.add_paragraph(style="List Bullet")
        run = p.add_run(item)
        set_run_font(run, size=10.5)

    heading(doc, "3.3 支持硬件", 2)
    hint(doc, "目标芯片型号、架构（如 dav-2201）、AIC / AIV 核数、HBM 带宽等；是否支持多芯片形态。")
    body(doc, "")

    heading(doc, "3.4 算子约束与限制", 2)
    hint(doc, "明确不支持的 shape / dtype / 属性组合及报错行为；性能前提假设；已知边界。")
    body(doc, "")

    # ---------------- 4 特性交叉分析 ----------------
    heading(doc, "4. 特性交叉分析", 1)
    hint(doc, "逐项说明本算子与其他特性的相互影响；不涉及的写「不涉及」并说明原因。")
    heading(doc, "4.1 混合精度", 2)
    hint(doc, "是否支持混合 dtype 输入 / 输出；精度转换点与对齐要求。")
    body(doc, "")
    heading(doc, "4.2 图模式", 2)
    hint(doc, "图模式下的形状推导、动态 shape 支持、图融合影响、属性传递。")
    body(doc, "")
    heading(doc, "4.3 动态 shape", 2)
    hint(doc, "动态维度支持范围、tiling 在运行时的适应性、workspace 变化。")
    body(doc, "")
    heading(doc, "4.4 其他特性（选填）", 2)
    hint(doc, "如多流、内存复用、切分 / 广播、量化等。")
    body(doc, "")

    # ---------------- 5 可维可测分析 ----------------
    heading(doc, "5. 可维可测分析", 1)

    heading(doc, "5.1 精度标准", 2)
    hint(doc, "参考实现（oracle）是什么（如 torch CPU 实现）、判据档（如 ops-precision-standard@1）、容差（rtol / atol）、特殊值（NaN / Inf）比较口径、异常输入的期望拒绝码。")
    body(doc, "")

    heading(doc, "5.2 性能标准", 2)
    hint(doc, "基线（官方对应实现 / 既有可信实现）、测量口径（设备侧 Task Duration、预热与重复次数、聚合方式）、目标（如无硬性目标请明确写「仅记录」）、代表用例（shape / dtype）。")
    body(doc, "")

    heading(doc, "5.3 兼容性分析", 2)
    for item in [
        "接口兼容性：调用形式、参数顺序、返回值、错误码是否与既有接口 / 框架一致。",
        "数据类型兼容性：支持的 dtype 范围、精度损失、类型提升规则。",
        "硬件形态兼容性：不同芯片 / 架构 / 核数下的行为与性能差异预期。",
    ]:
        p = doc.add_paragraph(style="List Bullet")
        run = p.add_run(item)
        set_run_font(run, size=10.5)
    doc.add_paragraph()

    # ---------------- 附录 A ----------------
    heading(doc, "附录 A（选填）：测试用例要点", 1)
    hint(doc, "填了会作为 agent 生成用例的依据；不填则由 agent 按规格生成后交你确认。精度用例至少覆盖：常规 / 边界 / 数值边界 / 异常；性能用例给出 n、dtype、是否主用例。")
    heading(doc, "A.1 精度用例", 2)
    add_table(doc, ["Case ID", "分组", "shape", "dtype", "输入生成方式 / 种子", "期望（match / reject + 拒绝码）", "说明"], [
        ["", "常规", "", "", "", "", ""],
        ["", "边界", "", "", "", "", ""],
        ["", "数值边界", "", "", "", "", ""],
        ["", "异常", "", "", "", "", ""],
    ], widths=[2.2, 1.8, 2.4, 1.6, 3.2, 3.2, 3.0], font_size=8)
    heading(doc, "A.2 性能用例", 2)
    add_table(doc, ["Case ID", "n", "dtype", "主用例", "必需", "说明"], [
        ["", "", "", "", "", ""],
    ], widths=[2.6, 2.6, 2.0, 1.8, 1.8, 5.0])

    # ---------------- 附录 B ----------------
    heading(doc, "附录 B（选填）：参考资料", 1)
    hint(doc, "相关设计文档、官方算子文档、历史实现链接、论文 / 公式出处等。")
    body(doc, "")

    doc.save(OUT)
    print(f"generated: {OUT}")


if __name__ == "__main__":
    build()
