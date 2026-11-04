import json
import re
import time

from playwright.sync_api import sync_playwright

CHROME = '/home/fn-ubuntu/.cache/ms-playwright/chromium-1243/chrome-linux64/chrome'
BASE = 'http://172.16.111.102:18080'
OUT = '/home/fn-ubuntu/.hermes/workspace/tk-unified-workbench'
MSG = ('请按当前任务已定版的规格，整组确认精度用例与性能用例（用例来源 cases/bid_elementwise_sub_20261101/'
       'cases.json 与 benchmark.json，判据 ops-precision-standard@1；正确性口径 torch CPU 一一对应）。'
       '确认成功后回复“用例组已确认”。不要改代码、不要编译、不要触发真机动作。')

with sync_playwright() as p:
    b = p.chromium.launch(headless=True, executable_path=CHROME,
        args=['--no-sandbox', '--disable-dev-shm-usage', '--no-proxy-server',
              '--host-resolver-rules=MAP 172.16.111.102 127.0.0.1'])
    page = b.new_page(viewport={'width': 1600, 'height': 1000})
    errs = []
    page.on('pageerror', lambda e: errs.append(str(e)))
    page.goto(BASE + '/#operator', wait_until='networkidle', timeout=40000)
    op = page.frame_locator('#panel-operator')
    op.get_by_role('heading', name='任务', exact=True).wait_for(timeout=25000)
    op.locator('tbody tr').filter(has_text='统一入口真机流程验证').first.get_by_role('button', name='打开').click()
    op.get_by_role('heading', name=re.compile('统一入口真机流程验证')).first.wait_for(timeout=25000)
    page.wait_for_timeout(2000)
    op.locator('button', has_text='对话').first.click()
    page.wait_for_timeout(3000)
    cands = [t.strip() for t in op.locator('button').all_text_contents() if ('会话' in t or '新建' in t)]
    print('SESSION_BUTTONS', cands)
    if cands:
        op.locator('button', has_text=re.compile('新建会话|新建对话|开始会话')).first.click()
        page.wait_for_timeout(4000)
    ta = op.locator('textarea').first
    print('TA_STATE', op.locator('textarea').evaluate_all(
        "els=>els.map(e=>({ph:(e.placeholder||'').slice(0,26),dis:e.disabled}))"))
    ta.wait_for(state='visible', timeout=15000)
    if ta.is_disabled():
        print('STILL_DISABLED')
    ta.fill(MSG)
    op.get_by_role('button', name='发送', exact=True).click()
    print('msg_sent')
    deadline = time.time() + 220
    while time.time() < deadline:
        body = op.locator('body').inner_text()
        if '用例组已确认' in body and op.get_by_role('button', name='发送', exact=True).is_enabled():
            break
        time.sleep(4)
    page.wait_for_timeout(2000)
    print('TAIL', json.dumps(op.locator('body').inner_text()[-600:], ensure_ascii=False))
    page.screenshot(path=OUT + '/64-case-confirm.png', full_page=False)
    print('page_errors', errs[:5])
    b.close()
