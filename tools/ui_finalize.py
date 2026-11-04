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
    op.locator('tbody tr').filter(has_text='统一入口真机流程验证').first.get_by_role('button', name='继续编辑').click()
    op.get_by_role('heading', name='任务开始', exact=True).wait_for(timeout=20000)
    page.wait_for_timeout(2000)

    op.locator('button', has_text='确认并定版').first.click()
    print('finalize_clicked')
    deadline = 60
    for _ in range(deadline // 2):
        if '任务列表' in op.locator('body').inner_text() and '任务开始' not in op.locator('body').inner_text():
            break
        page.wait_for_timeout(2000)
    page.wait_for_timeout(3000)
    print('HEADINGS', op.locator('h1,h2').all_text_contents()[:8])
    print('BUTTONS', op.locator('button').evaluate_all(
        "els=>els.filter(e=>e.offsetWidth).map(e=>(e.innerText||'').trim().slice(0,30)).filter(Boolean)"))
    page.screenshot(path=OUT + '/59-finalized.png', full_page=False)
    print('page_errors', errs[:5], 'http>=400', bad[:6])
    b.close()
