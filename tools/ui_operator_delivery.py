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
    page.on('response', lambda r: bad.append((r.status, r.url.split('?')[0]))
            if r.status >= 400 else None)
    page.goto(BASE + '/#operator', wait_until='networkidle', timeout=45000)
    op = page.frame_locator('#panel-operator')
    op.get_by_role('heading', name='任务', exact=True).wait_for(timeout=25000)
    op.locator('tbody tr').filter(has_text='统一入口真机流程验证').first.get_by_role('button', name='打开').click()
    short = op.get_by_role('heading', name=re.compile('统一入口真机流程验证')).first
    short.wait_for(timeout=25000)
    page.wait_for_timeout(2500)
    op.locator('button, [role=tab], a').filter(has_text=re.compile('记录与交付')).first.click()
    page.wait_for_timeout(3000)
    page.screenshot(path=OUT + '/83-delivery-tab.png', full_page=False)
    body = op.locator('body').inner_text()
    i = body.find('交付')
    print('DELIVERY_PANEL', body[i:i + 700].replace('\n', ' | '))
    btn = op.locator('button').filter(has_text=re.compile('生成交付包')).first
    print('BTN_COUNT', btn.count(), 'disabled?', btn.is_disabled() if btn.count() else 'n/a')
    if btn.count() and not btn.is_disabled():
        btn.click()
        page.wait_for_timeout(8000)
        body2 = op.locator('body').inner_text()
        j = body2.find('交付')
        print('AFTER_CLICK', body2[j:j + 800].replace('\n', ' | '))
        page.screenshot(path=OUT + '/84-delivery-created.png', full_page=False)
    print('page_errors', errs[:5], 'http>=400', bad[:6])
    b.close()
