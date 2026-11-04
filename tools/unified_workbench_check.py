from playwright.sync_api import sync_playwright

BASE = 'http://172.16.111.102:18080'
CHROME = '/home/fn-ubuntu/.cache/ms-playwright/chromium-1243/chrome-linux64/chrome'
with sync_playwright() as p:
    browser = p.chromium.launch(headless=True, executable_path=CHROME,
        args=['--no-sandbox', '--disable-dev-shm-usage', '--no-proxy-server',
              '--host-resolver-rules=MAP 172.16.111.102 127.0.0.1'])
    page = browser.new_page(viewport={'width': 1440, 'height': 960})
    errors, bad = [], []
    page.on('pageerror', lambda e: errors.append('shell:'+str(e)))
    page.on('console', lambda m: errors.append(('console', m.text, m.location)) if m.type == 'error' else None)
    page.on('response', lambda r: bad.append((r.status, r.url.split('?')[0])) if r.status >= 400 else None)
    page.on('requestfailed', lambda r: bad.append(('failed', r.url.split('?')[0], r.failure)) )
    response = page.goto(BASE+'/', wait_until='networkidle', timeout=35000)
    assert response.status == 200
    assert page.locator('.company-logo').evaluate('(e)=>e.complete && e.naturalWidth>0')
    assert page.locator('.workspace').evaluate('(e)=>getComputedStyle(e).gridTemplateColumns.startsWith("64px")')
    assert '内网预览' not in page.locator('body').inner_text()
    model = page.frame_locator('#panel-model')
    model.get_by_role('heading', name='模型迁移工作台', exact=True).wait_for(state='visible', timeout=15000)
    print('model', model.locator('body').evaluate('(e)=>{const s=x=>x?getComputedStyle(x).fontSize:null;return {font:getComputedStyle(e).fontFamily,size:getComputedStyle(e).fontSize,h1:s(e.querySelector("h1")),h2:s(e.querySelector("h2")),p:s(e.querySelector("p")),button:s(e.querySelector("button")),brand:getComputedStyle(document.documentElement).getPropertyValue("--brand").trim(),sidebar:Math.round(document.querySelector(".sidebar").getBoundingClientRect().width)}}'))
    print('company_logo',page.locator('.company-logo').evaluate('(e)=>({width:e.naturalWidth,height:e.naturalHeight})'))
    page.screenshot(path='/home/fn-ubuntu/.hermes/workspace/tk-unified-workbench/25-unified-model.png', full_page=True)
    page.locator('#tab-operator').click()
    operator = page.frame_locator('#panel-operator')
    operator.get_by_role('heading', name='任务', exact=True).wait_for(state='visible', timeout=20000)
    print('operator', operator.locator('body').evaluate('(e)=>{const s=x=>x?getComputedStyle(x).fontSize:null;return {font:getComputedStyle(e).fontFamily,size:getComputedStyle(e).fontSize,h1:s(e.querySelector("h1")),h2:s(e.querySelector("h2")),p:s(e.querySelector("p")),button:s(e.querySelector("button")),accent:getComputedStyle(document.documentElement).getPropertyValue("--accent").trim(),sidebar:Math.round(document.querySelector("#root>.grid.min-h-screen>aside").getBoundingClientRect().width)}}'))
    operator.get_by_role('button', name='阶段看板').click()
    operator.get_by_role('heading', name='阶段看板', exact=True).wait_for(state='visible', timeout=10000)
    print('operator_function_nav=PASS')
    page.screenshot(path='/home/fn-ubuntu/.hermes/workspace/tk-unified-workbench/26-unified-operator.png', full_page=True)
    page.locator('#tab-model').click()
    assert page.locator('#tab-model').get_attribute('aria-selected') == 'true'
    model.get_by_role('link', name='迁移任务').click()
    page.wait_for_timeout(600)
    assert model.get_by_role('heading', name='迁移任务', exact=True).is_visible()
    print('model_function_nav=PASS')
    page.locator('#tab-operator').click()
    operator.get_by_role('heading', name='阶段看板', exact=True).wait_for(state='visible', timeout=5000)
    print('preserved_app_state=PASS')
    print('shell_http', response.status, 'page_errors', errors, 'http_errors', bad)
    browser.close()
    assert not errors, errors
    assert not bad, bad
