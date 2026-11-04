from playwright.sync_api import sync_playwright

BASE = 'http://172.16.111.102:18080'
CHROME = '/home/fn-ubuntu/.cache/ms-playwright/chromium-1243/chrome-linux64/chrome'
SAMPLE = '''() => {
  const sizes = {}; const labels = {};
  const els = [...document.querySelectorAll('body *')].filter(e => {
    const t = (e.childElementCount === 0 ? e.textContent : '').trim();
    const r = e.getBoundingClientRect();
    return t && r.width > 0 && r.height > 0 && getComputedStyle(e).visibility !== 'hidden';
  });
  for (const e of els) {
    const s = getComputedStyle(e).fontSize;
    sizes[s] = (sizes[s] || 0) + 1;
    if (!labels[s]) labels[s] = (e.textContent || '').trim().slice(0, 18);
  }
  return {total: els.length, sizes, labels, family: getComputedStyle(document.body).fontFamily};
}'''

with sync_playwright() as p:
    browser = p.chromium.launch(headless=True, executable_path=CHROME,
        args=['--no-sandbox', '--disable-dev-shm-usage', '--no-proxy-server',
              '--host-resolver-rules=MAP 172.16.111.102 127.0.0.1'])
    context = browser.new_context(viewport={'width': 1440, 'height': 960}, bypass_csp=True)
    page = context.new_page()
    errors, bad = [], []
    page.on('pageerror', lambda e: errors.append('shell:' + str(e)))
    page.on('console', lambda m: errors.append(('console', m.text, m.location)) if m.type == 'error' else None)
    page.on('response', lambda r: bad.append((r.status, r.url.split('?')[0])) if r.status >= 400 else None)

    response = page.goto(BASE + '/', wait_until='networkidle', timeout=35000)
    assert response.status == 200
    assert page.locator('.company-logo').evaluate('(e)=>e.complete && e.naturalWidth>0')

    model = page.frame_locator('#panel-model')
    model.get_by_role('heading', name='模型迁移工作台', exact=True).wait_for(state='visible', timeout=15000)
    model_report = model.locator('body').evaluate(SAMPLE)
    print('MODEL total', model_report['total'], 'sizes', model_report['sizes'])
    print('MODEL family', model_report['family'])
    print('MODEL workspace block', model.locator('body').evaluate('''() => {
      const el = document.querySelector('.workspace-select');
      const avatar = document.querySelector('.ws-avatar');
      const name = document.querySelector('.workspace-select .txt b');
      const cs = getComputedStyle(el), ca = getComputedStyle(avatar);
      return {border: cs.borderTopWidth, radius: cs.borderRadius, bg: cs.backgroundColor,
              avatar: [ca.width, ca.height, ca.borderRadius, ca.backgroundImage.slice(0, 60)],
              nameSize: getComputedStyle(name).fontSize, nameWeight: getComputedStyle(name).fontWeight};
    }'''))
    page.screenshot(path='/home/fn-ubuntu/.hermes/workspace/tk-unified-workbench/27-model-final.png', full_page=False)

    page.locator('#tab-operator').click()
    operator = page.frame_locator('#panel-operator')
    operator.get_by_role('heading', name='任务', exact=True).wait_for(state='visible', timeout=20000)
    operator_report = operator.locator('body').evaluate(SAMPLE)
    print('OPERATOR total', operator_report['total'], 'sizes', operator_report['sizes'])
    print('OPERATOR family', operator_report['family'])
    page.screenshot(path='/home/fn-ubuntu/.hermes/workspace/tk-unified-workbench/28-operator-final.png', full_page=False)

    assert model_report['family'] == operator_report['family']
    model_main = model_report['sizes'].get('15px', 0) / model_report['total']
    operator_main = operator_report['sizes'].get('15px', 0) / operator_report['total']
    print('15px share model %.2f operator %.2f' % (model_main, operator_main))
    print('page_errors', errors, 'http_errors', bad)
    browser.close()
    assert not errors, errors
    assert not bad, bad
