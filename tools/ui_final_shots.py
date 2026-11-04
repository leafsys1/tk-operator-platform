import re

from playwright.sync_api import sync_playwright

CHROME = '/home/fn-ubuntu/.cache/ms-playwright/chromium-1243/chrome-linux64/chrome'
BASE = 'http://172.16.111.102:18080'
OUT = '/home/fn-ubuntu/.hermes/workspace/tk-unified-workbench'

with sync_playwright() as p:
    b = p.chromium.launch(headless=True, executable_path=CHROME,
        args=['--no-sandbox', '--disable-dev-shm-usage', '--no-proxy-server',
              '--host-resolver-rules=MAP 172.16.111.102 127.0.0.1'])
    page = b.new_page(viewport={'width': 1600, 'height': 1000})
    errs, bad = [], []
    page.on('pageerror', lambda e: errs.append(str(e)))
    page.on('response', lambda r: bad.append((r.status, r.url.split('?')[0])) if r.status >= 400 else None)
    page.goto(BASE + '/#operator', wait_until='networkidle', timeout=40000)
    op = page.frame_locator('#panel-operator')
    op.get_by_role('heading', name='任务', exact=True).wait_for(timeout=25000)
    op.locator('tbody tr').filter(has_text='统一入口真机流程验证').first.get_by_role('button', name='打开').click()
    op.get_by_role('heading', name=re.compile('统一入口真机流程验证')).first.wait_for(timeout=25000)
    page.wait_for_timeout(3000)
    page.screenshot(path=OUT + '/75-overview-after-runs.png', full_page=False)
    print('OVERVIEW_STEPS', [t.strip().replace('\n', ' ') for t in op.locator('button').all_text_contents() if re.search(r'(编译|冒烟|正确性)', t)][:6])
    op.locator('button', has_text='测试与结果').first.click()
    page.wait_for_timeout(3000)
    page.screenshot(path=OUT + '/76-test-results.png', full_page=False)
    txt = op.locator('body').inner_text()
    idx = txt.find('正确性（test）')
    print('RESULT_BLOCK', txt[idx:idx + 400].replace('\n', ' | ') if idx >= 0 else 'n/a')
    print('page_errors', errs[:5], 'http>=400', bad[:6])
    b.close()
