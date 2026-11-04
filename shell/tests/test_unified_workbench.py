import re
import unittest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from _paths import ROOT, text as _asset_text  # noqa: E402

import gateway

HTML = _asset_text('portal.html')
GATEWAY = _asset_text('gateway.py')


class UnifiedWorkbenchContractTest(unittest.TestCase):
    def test_company_brand_and_two_real_tabs(self):
        self.assertIn('src="/brand/tanku.png"', HTML)
        self.assertIn('role="tablist"', HTML)
        self.assertIn('data-product="model"', HTML)
        self.assertIn('data-product="operator"', HTML)
        self.assertIn('src="/model/"', HTML)
        self.assertIn('data-src="/operator/"', HTML)
        self.assertIn('history.replaceState', HTML)

    def test_no_preview_copy_in_shell(self):
        for phrase in ('内网预览', '只读预览', '此页面不代表全部功能已验收', '内网预览环境'):
            self.assertNotIn(phrase, HTML)

    def test_compact_rail_and_single_header_brand(self):
        self.assertIn('grid-template-columns:64px minmax(0,1fr)', HTML)
        self.assertIn('white-space:nowrap;font-size:10px', HTML)
        self.assertIn('class="brandbar"', HTML)
        self.assertIn('class="company-logo"', HTML)

    def test_product_theme_keeps_both_app_sidebars_compact_and_brand_in_shell(self):
        self.assertIn('--sidebar:200px!important', GATEWAY)
        self.assertIn('.sidebar>.brand{display:none!important}', GATEWAY)
        self.assertIn('#root>.grid.min-h-screen>aside>div:first-child{display:none!important}', GATEWAY)
        self.assertIn('--brand:#6557d2!important', GATEWAY)
        self.assertIn('--accent:#6557d2!important', GATEWAY)

    def test_brandbar_is_light_blue_and_readable(self):
        # Header now merges into the content canvas instead of a separate gradient band.
        self.assertIn('.brandbar{grid-column:2;display:flex;align-items:center;gap:18px;padding:0 22px;background:var(--canvas)', HTML)
        self.assertIn('color:#202f4b', HTML)
        self.assertNotIn('background:#302758', HTML)
        self.assertNotIn('linear-gradient(110deg,#dff3ff', HTML)

    def test_logo_is_small_and_not_upscaled(self):
        self.assertIn('.company-logo{display:block;width:118px;height:29px', HTML)
        self.assertNotIn('filter:drop-shadow(0 1px 1px #fff9)', HTML)

    def test_both_embedded_apps_force_same_readable_type_scale(self):
        css = gateway.THEME_CSS
        self.assertIn('body,body *{font-size:14px!important;line-height:1.45!important}', css)
        for tag, size in (('h1', '24px'), ('h2', '18px'), ('h3', '16px'), ('h4', '15px')):
            self.assertIn(f'body [class] {tag},body [class] {tag}[class]{{font-size:{size}!important', css)
        for tag in ('button', 'input', 'select', 'textarea'):
            self.assertIn(f'body [class] {tag},body [class] {tag}[class]', css)
        self.assertIn('body table,body table *{font-size:13px!important', css)

    def test_tables_do_not_squeeze_action_and_metric_cells(self):
        css = gateway.THEME_CSS
        self.assertIn('body table th,body table td{white-space:nowrap!important}', css)
        self.assertIn('body table td:first-child,body table td:nth-child(2)', css)
        self.assertIn('body table button,body table .chip,body table .badge', css)
        self.assertIn('padding:6px 9px!important', css)
        self.assertIn('body table td>div[class*="whitespace-pre"]{max-height:2.6em!important', css)
        self.assertIn('body table td:first-child,body table th:first-child{max-width:300px!important}', css)
        self.assertIn('body table td:last-child,body table th:last-child{min-width:270px!important}', css)
        self.assertIn('body table td:last-child>div,body table td:last-child>div[class]{flex-wrap:nowrap!important}', css)

    def test_navigation_weight_is_identical_on_both_products(self):
        css = gateway.THEME_CSS
        self.assertIn('.side-link,.side-link span,.sidebar nav button,.sidebar nav button span{font-weight:400!important}', css)
        self.assertIn('.side-link.active,.side-link.active span{font-weight:400!important}', css)

    def test_model_workspace_block_matches_operator_cann_card(self):
        """模型侧左上角工作区块要和算子侧「📁 CANN 算子开发」卡片同款：
        36px 高、13px 圆角、1px 描边、半透明底、同款阴影、单行 14px/400 文字、
        前置文件夹图标、右侧色弱 chevron。原来的「M 渐变方块 + 两行文字」已废弃。"""
        self.assertIn('.workspace-select.workspace-select{display:flex!important', GATEWAY)
        self.assertIn('height:36px!important', GATEWAY)
        self.assertIn('border:1px solid #e5e3ef!important;border-radius:13px!important', GATEWAY)
        self.assertIn('background:rgba(251,250,255,.91)!important', GATEWAY)
        self.assertIn('box-shadow:0 10px 30px rgba(20,26,46,.09)!important', GATEWAY)
        # 图标换成文件夹（CSS 转义 U+1F4C1），M 渐变方块隐藏
        self.assertIn('.ws-avatar{display:none!important}', GATEWAY)
        self.assertIn('content:"\U0001F4C1"', GATEWAY)
        # 单行 14px/400，副标题隐藏
        self.assertIn('.workspace-select .txt b{display:block!important;font-size:14px!important;font-weight:400!important', GATEWAY)
        # 卡片要和算子侧同宽同高：sidebar 内边距 11px、workspace-wrap 不再额外缩进
        self.assertIn('.workspace-wrap{margin:0 0 14px!important}', GATEWAY)
        self.assertIn('.workspace-select .txt small{display:none!important}', GATEWAY)
        # 不再出现旧的渐变头像与两行布局
        self.assertNotIn('linear-gradient(135deg,#6557d2,#38bdf8)!important', GATEWAY)
        self.assertNotIn('.workspace-select .txt b{font-size:15px!important', GATEWAY)

    def test_no_light_dark_toggle_left_in_either_product(self):
        """明暗切换是残留功能。两侧源码里都不许再有切换入口，
        也不许再读写主题持久化（否则删了按钮仍可能被系统深色顶起来）。"""
        repo = ROOT.parent
        if not (repo / 'operator/web/src/App.tsx').is_file():
            raise unittest.SkipTest('只跑在完整仓库布局下（shell/ + model-platform/ + operator/ 同级）')

        model_html = (repo / 'model-platform/static/index.html').read_text(encoding='utf-8')
        self.assertNotIn('id="theme-btn"', model_html)
        self.assertNotIn('THEME_KEY', model_html)              # 主题持久化常量必须一起删
        self.assertNotIn('applyTheme(', model_html)
        self.assertNotIn('initTheme(', model_html)
        self.assertNotIn('prefers-color-scheme', model_html)

        operator_app = (repo / 'operator/web/src/App.tsx').read_text(encoding='utf-8')
        self.assertNotIn('setTheme', operator_app)
        self.assertNotIn('data.theme = theme', operator_app)
        # 算子前端 index.html 静态声明 light，删掉状态后仍然固定浅色
        operator_index = (repo / 'operator/web/index.html').read_text(encoding='utf-8')
        self.assertIn('data-theme="light"', operator_index)


if __name__ == '__main__':
    unittest.main()
