import sys
import time

from playwright.sync_api import sync_playwright

CHROME = '/home/fn-ubuntu/.cache/ms-playwright/chromium-1243/chrome-linux64/chrome'
BASE = 'http://172.16.111.102:18080'
OUT = '/home/fn-ubuntu/.hermes/workspace/tk-unified-workbench'
MODEL = sys.argv[1] if len(sys.argv) > 1 else 'Qwen/Qwen2.5-0.5B-Instruct'
WAIT_S = int(sys.argv[2]) if len(sys.argv) > 2 else 60

with sync_playwright() as p:
    b = p.chromium.launch(headless=True, executable_path=CHROME,
        args=['--no-sandbox', '--disable-dev-shm-usage', '--no-proxy-server',
              '--host-resolver-rules=MAP 172.16.111.102 127.0.0.1'])
    page = b.new_page(viewport={'width': 1600, 'height': 1000})
    errs, bad, posts = [], [], []
    page.on('pageerror', lambda e: errs.append(str(e)))
    page.on('response', lambda r: (bad.append((r.status, r.url.split('?')[0]))
                                   if r.status >= 400 else None))
    page.on('request', lambda r: posts.append((r.method, r.url.split('?')[0]))
            if r.method == 'POST' else None)
    page.goto(BASE + '/#model', wait_until='networkidle', timeout=45000)
    page.wait_for_timeout(3000)
    ml = page.frame_locator('#panel-model')
    ml.locator('nav').get_by_text('新建迁移').first.click()
    page.wait_for_timeout(1500)
    ta = ml.locator('#model-input')
    ta.wait_for(timeout=15000)
    ta.fill(MODEL)
    page.wait_for_timeout(500)
    ml.locator('button', has_text='开始迁移').first.click()
    print('submitted', MODEL)
    page.wait_for_timeout(6000)
    print('POSTS', posts[:6])
    print('ERRORS', errs[:3], 'BAD', bad[:5])
    body = ml.locator('body').inner_text()
    idx = body.find('迁移')
    print('BODY_HEAD', body[:600].replace('\n', ' | '))
    page.screenshot(path=OUT + '/81-model-submitted.png', full_page=False)

    # 抓 job id：页面出现 job-... 或 从 /api/jobs 顶部拿
    jobs = page.evaluate("""async () => {
        const r = await fetch('/model/api/jobs', {headers: {'Accept':'application/json'}});
        return (await r.json()).slice(0, 3);
    }""")
    print('TOP_JOBS', [{k: j.get(k) for k in ('job_id', 'model', 'status', 'phase')} for j in jobs])
    deadline = time.time() + WAIT_S
    last = None
    while time.time() < deadline:
        page.wait_for_timeout(5000)
        j = page.evaluate("""async (id) => {
            const r = await fetch('/model/api/jobs/' + id);
            const d = await r.json();
            return {status: d.job.status, phase: d.job.phase,
                    n_events: d.events.length,
                    last: (d.events.slice(-1)[0] || {}).message};
        }""", jobs[0]['job_id'])
        if j != last:
            print('JOB', j)
            last = j
        if j['status'] in ('succeeded', 'failed', 'cancelled'):
            break
    page.screenshot(path=OUT + '/82-model-running.png', full_page=False)
    print('page_errors', errs[:5], 'http>=400', bad[:6])
    b.close()
