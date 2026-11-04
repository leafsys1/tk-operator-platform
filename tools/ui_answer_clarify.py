import json
import time

from playwright.sync_api import sync_playwright

CHROME = '/home/fn-ubuntu/.cache/ms-playwright/chromium-1243/chrome-linux64/chrome'
BASE = 'http://172.16.111.102:18080'
OUT = '/home/fn-ubuntu/.hermes/workspace/tk-unified-workbench'
ANSWER = ('1) op_name 就用 bid_elementwise_sub_20261101（单段名，沿用工作区已有算子目录与源码，不加库前缀）。\n'
          '2) 输出 dtype 恒为 float32；广播按完整 numpy 广播规则（含 0 维标量）；布局只要求连续内存输入，暂不要求非连续/原地写回。\n'
          '3) 空张量（0 元素）返回空结果而不是报错；NaN/Inf 按逐元素精确透传；芯片是 910B1（用 dav-2201 硬件画像）；不支持的组合返回明确拒绝码。'
          '本轮仍然只做规格确认，不要改文件、不要编译、不要执行命令。')

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

    # 进入刚创建的草稿任务
    row = op.get_by_role('row').filter(has_text='统一入口真机流程验证').first
    if row.count():
        row.get_by_role('button', name='继续编辑').click()
    else:
        op.locator('tbody tr', has_text='统一入口真机流程验证').first.get_by_role('button').first.click()
    page.wait_for_timeout(3000)
    print('OPENED', op.locator('h1').all_text_contents()[:2])

    ta = op.locator('textarea[placeholder*="继续说"]')
    if ta.count():
        ta.fill(ANSWER)
        op.get_by_role('button', name='发送', exact=True).click()
        print('clarification_sent')
        deadline = time.time() + 180
        while time.time() < deadline:
            if op.get_by_role('button', name='发送', exact=True).is_enabled():
                txt = op.locator('body').inner_text()
                if '待确认 0' in txt or '待澄清 0' in txt:
                    break
            time.sleep(3)
        page.wait_for_timeout(2000)
        print('after_answer_pending', '待确认 0' in op.locator('body').inner_text())
        page.screenshot(path=OUT + '/52-task-answered.png', full_page=False)

    body = op.locator('body').inner_text()
    print('BODY_TAIL', json.dumps(body[-900:], ensure_ascii=False))
    print('page_errors', errs[:5])
    b.close()
