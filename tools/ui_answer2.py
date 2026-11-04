import json
import re
import time

from playwright.sync_api import sync_playwright

CHROME = '/home/fn-ubuntu/.cache/ms-playwright/chromium-1243/chrome-linux64/chrome'
BASE = 'http://172.16.111.102:18080'
OUT = '/home/fn-ubuntu/.hermes/workspace/tk-unified-workbench'
ANSWER = ('1) 广播最大秩限 8 维；广播后元素总数不设额外上限，按显存/内存自然约束；超限统一返回拒绝码。\n'
          '2) 拒绝码就用这三类固定值：非法布局 EPARAM_LAYOUT、秩超限 EPARAM_RANK、dtype 不支持 EPARAM_DTYPE；'
          '错误信息统一格式 "ERR <code>: <reason>"。正确性口径采用项目默认：参考实现 torch CPU 的 a-b 一一对应，'
          '输出 shape/dtype 完全一致、NaN/Inf 逐元素精确比对。\n'
          '3) 本轮性能基线与性能目标不作要求，先只要求 build/smoke/正确性通过；交付物包含算子源码、测试与文档；'
          '目标环境 CANN 9.2.0（与执行镜像一致）。本轮只做规格确认，仍然不要修改文件、不要编译、不要执行任何命令。')

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
    op.locator('tbody tr').filter(has_text='统一入口真机流程验证').first.get_by_role('button', name='继续编辑').click()
    op.get_by_role('heading', name='任务开始', exact=True).wait_for(timeout=20000)
    op.locator('textarea').first.wait_for(state='visible', timeout=15000)

    op.locator('textarea').first.fill(ANSWER)
    op.get_by_role('button', name='发送', exact=True).click()
    print('answer_round2_sent')

    deadline = time.time() + 220
    state = None
    while time.time() < deadline:
        state = op.locator('button', has_text='确认创建').evaluate_all(
            "els=>els.map(e=>({t:e.innerText.trim(),dis:e.disabled}))")
        if state and not state[0]['dis']:
            break
        time.sleep(4)
    print('CONFIRM_STATE', state)
    body = op.locator('body').inner_text()
    pend = re.search(r'待确认\s*(\d+)\s*项', body)
    print('PENDING', pend.group(1) if pend else '0')
    page.screenshot(path=OUT + '/56-clarified2.png', full_page=False)

    if state and not state[0]['dis']:
        op.locator('button', has_text='确认创建').first.click()
        page.wait_for_timeout(7000)
        print('AFTER_CREATE', op.locator('h1,h2').all_text_contents()[:6])
        page.screenshot(path=OUT + '/57-created.png', full_page=False)
    print('page_errors', errs[:5])
    b.close()
