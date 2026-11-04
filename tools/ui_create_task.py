import json
import time

from playwright.sync_api import sync_playwright

CHROME = '/home/fn-ubuntu/.cache/ms-playwright/chromium-1243/chrome-linux64/chrome'
BASE = 'http://172.16.111.102:18080'
OUT = '/home/fn-ubuntu/.hermes/workspace/tk-unified-workbench'
TITLE = '统一入口真机流程验证 20261103'
OP = 'bid_elementwise_sub_20261101'
PROMPT = ('我要验证向量减法（elementwise sub）算子在 910B 上的完整流程。'
          '输入两个 float32 张量 a、b，输出 a-b，允许形状广播为同 shape；精度判据按 ops-precision-standard@1。'
          '本轮只做需求澄清与规格确认，请不要修改任何文件、不要编译、不要执行任何命令或真机动作。')

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
    op.get_by_role('button', name='+ 新建任务').click()
    op.get_by_role('heading', name='任务开始', exact=True).wait_for(timeout=15000)

    # 任务字段：标题 / 算子名
    op.locator('input[placeholder="未命名任务"]').fill(TITLE)
    op.locator('input[placeholder="rms_norm"]').fill(OP)
    print('fields_filled', TITLE, OP)

    # 需求澄清对话：发一条真实需求给 agent
    ta = op.locator('textarea[placeholder*="继续说"]')
    ta.fill(PROMPT)
    op.get_by_role('button', name='发送', exact=True).click()
    print('message_sent')

    # 等 agent 回复（最多 150s），期间盯着对话区文本变化
    deadline = time.time() + 150
    last = ''
    while time.time() < deadline:
        txt = op.locator('body').inner_text()
        if '发送' in txt:
            # 对话区出现助手回复的常见标志
            pass
        if txt != last:
            last = txt
        if ('待澄清' in txt and '确认创建' in txt):
            pass
        if '正在' not in txt[-400:] and '思考' not in txt:
            pass
        time.sleep(2)
        # 以“发送”按钮重新可用 + 出现较长回复为完成信号
        if op.get_by_role('button', name='发送', exact=True).is_enabled():
            body = op.locator('body').inner_text()
            if len(body) > 1200 and 'elementwise' in body.lower() or '向量减法' in body:
                break
    page.wait_for_timeout(3000)

    body = op.locator('body').inner_text()
    print('BODY_LEN', len(body))
    print('SNIPPET', json.dumps(body[:1500], ensure_ascii=False))
    page.screenshot(path=OUT + '/51-task-clarify.png', full_page=False)
    print('page_errors', errs[:5])
    b.close()
