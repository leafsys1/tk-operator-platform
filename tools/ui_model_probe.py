import sys

from playwright.sync_api import sync_playwright

CHROME = '/home/fn-ubuntu/.cache/ms-playwright/chromium-1243/chrome-linux64/chrome'
BASE = 'http://172.16.111.102:18080'
OUT = '/home/fn-ubuntu/.hermes/workspace/tk-unified-workbench'

with sync_playwright() as p:
    b = p.chromium.launch(headless=True, executable_path=CHROME,
        args=['--no-sandbox', '--disable-dev-shm-usage', '--no-proxy-server',
              '--host-resolver-rules=MAP 172.16.111.102 127.0.0.1'])
    page = b.new_page(viewport={'width': 1600, 'height': 1000})
    page.goto(BASE + '/#model', wait_until='networkidle', timeout=45000)
    page.wait_for_timeout(4000)
    ml = page.frame_locator('#panel-model')
    ml.locator('body').wait_for(timeout=20000)
    print('URL', page.url)
    print('--- inputs ---')
    print(ml.locator('input,textarea,select').evaluate_all(
        "els=>els.filter(e=>e.offsetWidth).map(e=>({tag:e.tagName,type:e.type,id:e.id,ph:e.placeholder,cls:(e.className||'').slice(0,40)}))"))
    print('--- buttons ---')
    print(ml.locator('button').evaluate_all(
        "els=>els.filter(e=>e.offsetWidth).map(e=>(e.innerText||'').trim().replace(/\\n/g,' ')).slice(0,25)"))
    print('--- tabs/nav ---')
    print(ml.locator('nav a,[role=tab],.tab').evaluate_all(
        "els=>els.filter(e=>e.offsetWidth).map(e=>(e.innerText||'').trim()).slice(0,20)"))
    page.screenshot(path=OUT + '/80-model-panel.png', full_page=False)
    b.close()
