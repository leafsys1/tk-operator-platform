import re
import time

from playwright.sync_api import sync_playwright

CHROME = '/home/fn-ubuntu/.cache/ms-playwright/chromium-1243/chrome-linux64/chrome'
BASE = 'http://172.16.111.102:18080'
OUT = '/home/fn-ubuntu/.hermes/workspace/tk-unified-workbench'
MSG = ('冒烟实测失败：最小用例 sub_bnd_002 n=0 返回 rc=1。现有算子二进制是旧口径实现'
       '（n_min=1、不支持广播、空张量按拒绝处理、拒绝码 REJECT_*），与我上一轮 v1.0 规格冲突。'
       '请以现有实现为准修正：规格改 v2（n_min=1、不支持广播、空张量/非法参数返回拒绝码），'
       '把 cases.json 里空张量等用例改回 expected=reject，然后重新整组确认精度/性能用例。'
       '只改规格与用例文件，不要改算子源码、不要编译、不要触发真机动作。')

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

    # 选中「挂 skills · 2 条」那个会话（做过用例确认的）
    target = op.locator('aside.card button').filter(has_text='挂 skills · 2 条').first
    print('target_count', target.count())
    target.click()
    page.wait_for_timeout(3500)
    ta = op.locator('textarea').first
    print('TA', op.locator('textarea').evaluate_all(
        "els=>els.map(e=>({ph:(e.placeholder||'').slice(0,26),dis:e.disabled}))"))
    ta.fill(MSG)
    op.get_by_role('button', name='发送', exact=True).click()
    print('msg_sent')
    page.wait_for_timeout(5000)
    print('TAIL', op.locator('body').inner_text()[-300:].replace('\n', ' | '))
    page.screenshot(path=OUT + '/74-taskmode-msg.png', full_page=False)
    print('page_errors', errs[:5])
    b.close()
