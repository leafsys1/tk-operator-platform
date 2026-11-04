from playwright.sync_api import sync_playwright

BASE = 'http://172.16.111.102:18080'
CHROME = '/home/fn-ubuntu/.cache/ms-playwright/chromium-1243/chrome-linux64/chrome'
OUT = '/home/fn-ubuntu/.hermes/workspace/tk-unified-workbench'

with sync_playwright() as p:
    browser = p.chromium.launch(headless=True, executable_path=CHROME,
        args=['--no-sandbox', '--disable-dev-shm-usage', '--no-proxy-server',
              '--host-resolver-rules=MAP 172.16.111.102 127.0.0.1'])
    page = browser.new_page(viewport={'width': 1440, 'height': 900})
    errors, bad = [], []
    page.on('pageerror', lambda e: errors.append('shell:' + str(e)))
    page.on('console', lambda m: errors.append(('console', m.text, m.location)) if m.type == 'error' else None)
    page.on('response', lambda r: bad.append((r.status, r.url.split('?')[0])) if r.status >= 400 else None)

    page.goto(BASE + '/', wait_until='networkidle', timeout=35000)
    model = page.frame_locator('#panel-model')
    model.get_by_role('heading', name='模型迁移工作台', exact=True).wait_for(state='visible', timeout=15000)
    page.screenshot(path=OUT + '/41-model-overview.png', full_page=False)

    model.get_by_role('link', name='迁移任务').click()
    page.wait_for_timeout(1200)
    print('MODEL_TASKS_TABLE', model.locator('body').evaluate('''() => {const t=document.querySelector('table'); if(!t) return null; const r=t.tBodies[0].rows[0];
      return {cols:r.cells.length, rowH:Math.round(r.getBoundingClientRect().height), tableW:Math.round(t.getBoundingClientRect().width),
              wrapW:Math.round(t.parentElement.getBoundingClientRect().width), fs:getComputedStyle(t).fontSize};}'''))
    page.screenshot(path=OUT + '/42-model-tasks.png', full_page=False)

    page.locator('#tab-operator').click()
    operator = page.frame_locator('#panel-operator')
    operator.get_by_role('heading', name='任务', exact=True).wait_for(state='visible', timeout=20000)
    page.screenshot(path=OUT + '/43-operator-tasks.png', full_page=False)
    operator.get_by_role('button', name='服务器与镜像').click()
    operator.get_by_role('heading', name='服务器与镜像', exact=True).wait_for(state='visible', timeout=20000)
    page.wait_for_timeout(1500)
    page.screenshot(path=OUT + '/44-operator-images.png', full_page=False)
    print('page_errors', errors, 'http_errors', bad)
    browser.close()
    assert not errors, errors
    assert not bad, bad
