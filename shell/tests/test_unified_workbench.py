import re
import unittest
from pathlib import Path

import gateway

HTML = Path(__file__).with_name('portal.html').read_text()
GATEWAY = Path(__file__).with_name('gateway.py').read_text()


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

    def test_model_workspace_block_uses_operator_brand_style(self):
        self.assertIn('.workspace-select{display:flex!important', GATEWAY)
        self.assertIn('border:0!important;border-radius:9px!important;background:transparent!important', GATEWAY)
        self.assertIn('.ws-avatar{width:26px!important', GATEWAY)
        self.assertIn('background:linear-gradient(135deg,#6557d2,#38bdf8)!important', GATEWAY)
        self.assertIn('.workspace-select .txt b{font-size:15px!important;font-weight:650!important', GATEWAY)


if __name__ == '__main__':
    unittest.main()
