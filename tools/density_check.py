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
    header = page.locator('.brandbar').evaluate('''(e)=>{const r=e.getBoundingClientRect();const cs=getComputedStyle(e);
      return {h:Math.round(r.height),bg:cs.backgroundColor,border:cs.borderBottomColor};}''')
    logo = page.locator('.company-logo').evaluate('''(e)=>{const r=e.getBoundingClientRect();
      return {cssW:Math.round(r.width),cssH:Math.round(r.height),natW:e.naturalWidth,natH:e.naturalHeight};}''')
    viewport_bg = page.evaluate('getComputedStyle(document.querySelector(".viewport")).backgroundColor')
    print('HEADER', header, 'LOGO', logo, 'CONTENT_BG', viewport_bg)
    assert logo['cssW'] <= logo['natW'], 'logo must not be upscaled'

    # model workbench: nav weight + type scale
    model = page.frame_locator('#panel-model')
    model.get_by_role('heading', name='模型迁移工作台', exact=True).wait_for(state='visible', timeout=15000)
    print('MODEL_NAV', model.locator('body').evaluate('''() => [...document.querySelectorAll('.side-link')].slice(0,4).map(e=>({t:(e.innerText||'').trim().slice(0,8),w:getComputedStyle(e).fontWeight,fs:getComputedStyle(e).fontSize}))'''))
    print('MODEL_TYPE', model.locator('body').evaluate('''() => ({body:getComputedStyle(document.body).fontSize,h1:getComputedStyle(document.querySelector('h1')).fontSize,table:document.querySelector('table')?getComputedStyle(document.querySelector('table')).fontSize:null})'''))
    page.screenshot(path=OUT + '/31-model-density.png', full_page=False)

    page.locator('#tab-operator').click()
    operator = page.frame_locator('#panel-operator')
    operator.get_by_role('heading', name='任务', exact=True).wait_for(state='visible', timeout=20000)
    print('OPERATOR_NAV', operator.locator('body').evaluate('''() => [...document.querySelectorAll('#root>.grid.min-h-screen>aside nav button')].slice(0,4).map(e=>({t:(e.innerText||'').trim().slice(0,8),w:getComputedStyle(e).fontWeight,fs:getComputedStyle(e).fontSize}))'''))
    print('OPERATOR_TYPE', operator.locator('body').evaluate('''() => ({body:getComputedStyle(document.body).fontSize,h1:getComputedStyle(document.querySelector('h1')).fontSize,table:document.querySelector('table')?getComputedStyle(document.querySelector('table')).fontSize:null})'''))

    # dense table page that was squeezed
    operator.get_by_role('button', name='服务器与镜像').click()
    operator.get_by_role('heading', name='服务器与镜像', exact=True).wait_for(state='visible', timeout=20000)
    page.wait_for_timeout(1500)
    print('IMAGE_TABLE', operator.locator('body').evaluate('''() => {const t=document.querySelector('table'); if(!t) return null;
      const row=t.tBodies[0].rows[0];
      return {tableW:Math.round(t.getBoundingClientRect().width),wrapW:Math.round(t.parentElement.getBoundingClientRect().width),
              scrollW:t.parentElement.scrollWidth,rowH:Math.round(row.getBoundingClientRect().height),
              cells:[...row.cells].map(c=>({w:Math.round(c.getBoundingClientRect().width),ws:getComputedStyle(c).whiteSpace,lines:Math.round(c.getBoundingClientRect().height/20)}))};}'''))
    page.screenshot(path=OUT + '/32-operator-table.png', full_page=False)
    print('page_errors', errors, 'http_errors', bad)
    browser.close()
    assert not errors, errors
    assert not bad, bad
