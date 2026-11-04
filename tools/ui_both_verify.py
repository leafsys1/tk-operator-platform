import re

from playwright.sync_api import sync_playwright

CHROME = '/home/fn-ubuntu/.cache/ms-playwright/chromium-1243/chrome-linux64/chrome'
BASE = 'http://172.16.111.102:18080'
OUT = '/home/fn-ubuntu/.hermes/workspace/tk-unified-workbench'

with sync_playwright() as p:
    b = p.chromium.launch(headless=True, executable_path=CHROME,
        args=['--no-sandbox', '--disable-dev-shm-usage', '--no-proxy-server',
              '--host-resolver-rules=MAP 172.16.111.102 127.0.0.1'])
    page = b.new_page(viewport={'width': 1680, 'height': 1050})
    errs, bad = [], []
    page.on('pageerror', lambda e: errs.append(str(e)))
    page.on('response', lambda r: bad.append((r.status, r.url.split('?')[0]))
            if r.status >= 400 else None)

    # ---- 算子侧：记录与交付 ----
    page.goto(BASE + '/#operator', wait_until='networkidle', timeout=45000)
    op = page.frame_locator('#panel-operator')
    op.get_by_role('heading', name='任务', exact=True).wait_for(timeout=25000)
    op.locator('tbody tr').filter(has_text='统一入口真机流程验证').first.get_by_role('button', name='打开').click()
    op.get_by_role('heading', name=re.compile('统一入口真机流程验证')).first.wait_for(timeout=25000)
    page.wait_for_timeout(2500)
    print('OP_GATES', ' | '.join(
        t.strip() for t in op.locator('body').inner_text().split('\n')
        if re.search(r'(编译|冒烟|正确性|基准|采集)\s*·', t))[:300])
    op.locator('button, [role=tab], a').filter(has_text=re.compile('记录与交付')).first.click()
    page.wait_for_timeout(3500)
    txt = op.locator('body').inner_text()
    i = txt.find('交付包')
    j = txt.find('v2.0')
    print('DELIVERY_TEXT', txt[max(0, min(x for x in (i, j) if x > 0) - 120):][:500].replace('\n', ' | '))
    page.screenshot(path=OUT + '/85-operator-delivery.png', full_page=False)

    # ---- 模型侧：迁移任务 ----
    page.locator('[data-product="model"]').first.click()
    page.wait_for_timeout(5000)
    ml = page.frame_locator('#panel-model')
    ml.locator('body').wait_for(state='visible', timeout=25000)
    page.screenshot(path=OUT + '/86-model-tasks-unified.png', full_page=False)
    mtxt = ml.locator('body').inner_text()
    for kw in ('Qwen2.5-0.5B', '运行中', '已完成'):
        k = mtxt.find(kw)
        print('MODEL_CTX', kw, '->', mtxt[max(0, k - 60):k + 120].replace('\n', ' | ') if k >= 0 else 'n/a')
    print('page_errors', errs[:5], 'http>=400', bad[:6])
    b.close()
