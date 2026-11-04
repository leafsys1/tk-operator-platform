import json
import re
import sys
import time

from playwright.sync_api import sync_playwright

CHROME = '/home/fn-ubuntu/.cache/ms-playwright/chromium-1243/chrome-linux64/chrome'
BASE = 'http://172.16.111.102:18080'
OUT = '/home/fn-ubuntu/.hermes/workspace/tk-unified-workbench'
STEP = sys.argv[1] if len(sys.argv) > 1 else '编译'
WAIT_S = int(sys.argv[2]) if len(sys.argv) > 2 else 420

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
    page.wait_for_timeout(2500)

    op.locator('button', has_text=STEP).first.click()
    print('step_clicked', STEP)

    # 等 plan 计算完、出现「真执行」
    clicked = False
    for _ in range(40):
        page.wait_for_timeout(3000)
        labels = op.locator('button').evaluate_all(
            "els=>els.filter(e=>e.offsetWidth).map(e=>(e.innerText||'').trim())")
        if any(x == '真执行' for x in labels):
            op.locator('button', has_text='真执行').first.click()
            clicked = True
            print('real_execute_clicked')
            break
    if not clicked:
        body = op.locator('body').inner_text()
        print('NO_REAL_EXECUTE; plan tail:', json.dumps(body[-500:], ensure_ascii=False))
        page.screenshot(path=OUT + '/70-noexecute-%s.png' % STEP, full_page=False)
        b.close()
        raise SystemExit(2)

    # 等页面执行结束
    deadline = time.time() + WAIT_S
    last = ''
    while time.time() < deadline:
        page.wait_for_timeout(4000)
        btns = op.locator('button').evaluate_all(
            "els=>els.filter(e=>e.offsetWidth).map(e=>((e.innerText||'').trim().replace(/\\n/g,' ')))")
        cur = [x for x in btns if STEP in x or '执行中' in x or '进行中' in x][:3]
        cur = ' ;; '.join(cur)
        if cur != last:
            print('BTN', cur[:160])
            last = cur
        if not any('执行中' in x or '进行中' in x for x in btns):
            break
    page.wait_for_timeout(3000)
    print('FINAL_BTNS', [x for x in op.locator('button').all_text_contents() if STEP in x][:2])
    page.screenshot(path=OUT + '/71-done-%s.png' % STEP, full_page=False)
    print('page_errors', errs[:5], 'http>=400', bad[:6])
    b.close()
