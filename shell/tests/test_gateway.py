import base64
import http.client
import threading
import time
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import gateway  # noqa: E402


class Upstream(BaseHTTPRequestHandler):
    protocol_version = 'HTTP/1.1'
    def log_message(self, *args): pass
    def do_GET(self):
        if self.path == '/events':
            self.send_response(200); self.send_header('Content-Type', 'text/event-stream')
            self.send_header('Connection', 'close'); self.end_headers()
            self.wfile.write(b'data: first\n\n'); self.wfile.flush()
            time.sleep(0.6)
            self.wfile.write(b'data: second\n\n'); self.wfile.flush()
            self.close_connection = True
            return
        if self.path == '/headers':
            body = str((self.headers.get('X-Forwarded-Server'), self.headers.get('X-Real-IP'))).encode()
            self.send_response(200); self.send_header('Content-Length', str(len(body)))
            self.end_headers(); self.wfile.write(body); return
        if self.path == '/prefixed-location':
            self.send_response(302); self.send_header('Location', '/model/api/next'); self.send_header('Content-Length', '0')
            self.end_headers(); return
        if self.path == '/cookie':
            body = self.headers.get('Cookie', '').encode()
            self.send_response(200); self.send_header('Set-Cookie', 'sid=abc; Path=/; HttpOnly')
            self.send_header('Location', '/api/next'); self.send_header('Content-Length', str(len(body)))
            self.end_headers(); self.wfile.write(body); return
        if self.path == '/page':
            body = b'<html><head><title>fixture</title></head><body><script src="/assets/app.js"></script><script>fetch("/api/jobs")</script><img src="/static/x"></body></html>'
            typ = 'text/html'
        elif self.path == '/cached-page':
            if self.headers.get('If-None-Match') == '"stale-v1"':
                self.send_response(304); self.send_header('Content-Length', '0'); self.end_headers(); return
            body = b'<html><head><title>fixture</title></head><body>cached shell</body></html>'
            self.send_response(200); self.send_header('Content-Type', 'text/html; charset=utf-8')
            self.send_header('ETag', '"stale-v1"')
            self.send_header('Last-Modified', 'Sun, 01 Nov 2026 06:06:38 GMT')
            self.send_header('Cache-Control', 'public, max-age=600')
            self.send_header('Content-Length', str(len(body))); self.end_headers(); self.wfile.write(body)
            return
        elif self.path == '/large':
            body = b'x' * (9 * 1024 * 1024); typ = 'application/octet-stream'
        elif self.path == '/download':
            body = bytes(range(256)); typ = 'application/octet-stream'
        else:
            body = (self.server.label + ':' + self.path + ':' + str(self.headers.get('Authorization'))).encode()
            typ = 'application/json'
        self.send_response(200); self.send_header('Content-Type', typ)
        self.send_header('Content-Length', str(len(body))); self.end_headers(); self.wfile.write(body)
    def do_POST(self):
        body = self.rfile.read(int(self.headers.get('Content-Length', '0')))
        result = self.server.label.encode() + b':' + self.path.encode() + b':' + body
        self.send_response(201); self.send_header('Content-Length', str(len(result)))
        self.end_headers(); self.wfile.write(result)


class GatewayTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.servers = []
        for label in ('model', 'operator'):
            s = ThreadingHTTPServer(('127.0.0.1', 0), Upstream); s.label = label
            threading.Thread(target=s.serve_forever, daemon=True).start(); cls.servers.append(s)
        cls.g = gateway.make_server(('127.0.0.1', 0),
            {n: ('127.0.0.1', s.server_port) for n, s in zip(('model', 'operator'), cls.servers)},
            username='tester', password='test-only-password', max_body=32, timeout=3)
        threading.Thread(target=cls.g.serve_forever, daemon=True).start()
        cls.host = '127.0.0.1:' + str(cls.g.server_port)
        cls.auth = 'Basic ' + base64.b64encode(b'tester:test-only-password').decode()
    @classmethod
    def tearDownClass(cls):
        for s in [cls.g] + cls.servers: s.shutdown(); s.server_close()
    def request(self, method, path, body=None, headers=None):
        conn = http.client.HTTPConnection('127.0.0.1', self.g.server_port, timeout=2)
        h = {'Authorization': self.auth}; h.update(headers or {})
        conn.request(method, path, body=body, headers=h)
        res = conn.getresponse(); result = res.status, dict(res.getheaders()), res.read(); conn.close()
        return result
    def test_actual_source_address_controls_access(self):
        import ipaddress
        old = self.g.allowed_networks
        self.g.allowed_networks = [ipaddress.ip_network('127.0.0.1/32')]
        try:
            c = http.client.HTTPConnection('127.0.0.1', self.g.server_port, timeout=2, source_address=('127.0.0.2', 0))
            c.request('GET', '/', headers={'Authorization': self.auth, 'X-Forwarded-For': '127.0.0.1'})
            r = c.getresponse(); self.assertEqual(r.status, 403); r.read(); c.close()
            self.assertEqual(self.request('GET', '/')[0], 200)
        finally:
            self.g.allowed_networks = old

    def test_socket_peer_cidr_denies_forwarded_spoof(self):
        old = getattr(self.g, 'allowed_networks', None)
        try:
            import ipaddress
            self.g.allowed_networks = [ipaddress.ip_network('192.0.2.0/24')]
            self.assertEqual(self.request('GET', '/', headers={'X-Forwarded-For': '192.0.2.1'})[0], 403)
        finally:
            self.g.allowed_networks = old

    def test_ambiguous_duplicate_slashes_are_rejected(self):
        for path in ('/operator//api/proto/dsh/stream', '/model//api/jobs', '/operator/api//proto/dsh/stream'):
            with self.subTest(path=path):
                self.assertEqual(self.request('GET', path)[0], 400)

    def test_preview_gate_blocks_real_actions_and_rejects_ambiguous_json(self):
        self.g.preview_gate = True
        old_limit = self.g.max_body
        self.g.max_body = 1024
        try:
            h = {'Origin': 'http://' + self.host, 'Content-Type': 'application/json'}
            path = '/operator/api/tasks/123/actions'
            status, _, body = self.request('POST', path, b'{"dry_run":false}', h)
            self.assertEqual(status, 503)
            self.assertIn('预览', body.decode())
            self.assertEqual(self.request('POST', path, b'{"dry_run":true}', h)[0], 201)
            for b in (b'{}', b'{"dry_run":"true"}', b'{', b'{"dry_run":true,"dry_run":false}'):
                self.assertEqual(self.request('POST', path, b, h)[0], 503)
        finally:
            self.g.preview_gate = False
            self.g.max_body = old_limit

    def test_explicit_wildcard_preview_host_origin_and_peer_policy(self):
        upstreams = {n: ('127.0.0.1', s.server_port) for n, s in zip(('model', 'operator'), self.servers)}
        g = gateway.make_server(('0.0.0.0', 0), upstreams, username='tester', password='test-only-password',
                                allowed_cidrs=['127.0.0.0/8'], allowed_hosts=['preview.internal:8080'], preview_gate=True)
        threading.Thread(target=g.serve_forever, daemon=True).start()
        try:
            def req(host, origin=None, method='GET'):
                c = http.client.HTTPConnection('127.0.0.1', g.server_port, timeout=2)
                h = {'Host': host, 'Authorization': self.auth}
                if origin is not None: h['Origin'] = origin
                c.request(method, '/operator/api/run', body=b'x' if method == 'POST' else None, headers=h)
                r = c.getresponse(); status = r.status; r.read(); c.close(); return status
            self.assertEqual(req('preview.internal:8080'), 200)
            self.assertEqual(req('evil:8080'), 400)
            self.assertEqual(req('preview.internal:8080', 'http://evil:8080', 'POST'), 403)
            self.assertEqual(req('preview.internal:8080', 'http://preview.internal:8080', 'POST'), 201)
        finally:
            g.shutdown(); g.server_close()

    def test_reverse_proxy_host_wildcard_accepts_standard_host_forms(self):
        upstreams = {n: ('127.0.0.1', s.server_port) for n, s in zip(('model', 'operator'), self.servers)}
        g = gateway.make_server(('0.0.0.0', 0), upstreams, username='tester', password='test-only-password',
                                allowed_cidrs=['127.0.0.0/8'], allowed_hosts=['*'], preview_gate=True)
        threading.Thread(target=g.serve_forever, daemon=True).start()
        try:
            c = http.client.HTTPConnection('127.0.0.1', g.server_port, timeout=2)
            c.request('GET', '/', headers={'Host': 'portal.company.example:8080', 'Authorization': self.auth})
            r = c.getresponse(); status = r.status; r.read(); c.close()
            self.assertEqual(status, 200)
            for host in ('portal.company.example', 'portal.company.example:8080'):
                c = http.client.HTTPConnection('127.0.0.1', g.server_port, timeout=2)
                c.request('GET', '/', headers={'Host': host, 'Authorization': self.auth})
                r = c.getresponse(); status = r.status; r.read(); c.close()
                self.assertEqual(status, 200)
            for host in ('portal.company.example:0', 'portal.company.example:99999', 'bad_host:8080'):
                c = http.client.HTTPConnection('127.0.0.1', g.server_port, timeout=2)
                c.request('GET', '/', headers={'Host': host, 'Authorization': self.auth})
                r = c.getresponse(); status = r.status; r.read(); c.close()
                self.assertEqual(status, 400)
        finally:
            g.shutdown(); g.server_close()

    def test_auth_can_be_disabled_without_disabling_readonly_and_origin_gates(self):
        upstreams = {n: ('127.0.0.1', s.server_port) for n, s in zip(('model', 'operator'), self.servers)}
        g = gateway.make_server(('0.0.0.0', 0), upstreams, username='', password='',
                                allowed_cidrs=['127.0.0.0/8'], allowed_hosts=['*'], preview_gate=True,
                                operator_readonly=True, model_readonly=True, auth_required=False)
        threading.Thread(target=g.serve_forever, daemon=True).start()
        try:
            c = http.client.HTTPConnection('127.0.0.1', g.server_port, timeout=2)
            c.request('GET', '/', headers={'Host': 'portal.company.example'})
            r = c.getresponse(); status = r.status; r.read(); c.close()
            self.assertEqual(status, 200)
            for origin, expected in ((None, 403), ('http://portal.company.example', 503), ('http://evil.invalid', 403)):
                c = http.client.HTTPConnection('127.0.0.1', g.server_port, timeout=2)
                headers = {'Host': 'portal.company.example', 'Content-Type': 'application/json'}
                if origin is not None: headers['Origin'] = origin
                c.request('POST', '/operator/api/chat', body=b'{}', headers=headers)
                r = c.getresponse(); status = r.status; r.read(); c.close()
                self.assertEqual(status, expected)
        finally:
            g.shutdown(); g.server_close()

    def test_explicit_operator_readonly_blocks_writes_and_write_query(self):
        self.g.operator_readonly = True
        try:
            h = {'Origin': 'http://' + self.host}
            for p in ('/operator/api/tasks/123/shell', '/operator/api/loops/1/start', '/operator/api/chat'):
                status, _, body = self.request('POST', p, b'{}', h)
                self.assertEqual(status, 503); self.assertIn('只读', body.decode())
            self.assertEqual(self.request('GET', '/operator/api/workspace?write=true')[0], 503)
            self.assertEqual(self.request('GET', '/operator/api/workspace?%77rite=1')[0], 503)
            self.assertEqual(self.request('GET', '/operator/api/proto/dsh/stream?prompt=run')[0], 503)
            self.assertEqual(self.request('GET', '/operator/api/workspace')[0], 200)
            # Shell copy no longer advertises read-only; enforcement is server-side only.
            self.assertIn(b'data-product="operator"', self.request('GET', '/')[2])
        finally:
            self.g.operator_readonly = False

    def test_explicit_model_readonly_prevents_task_execution(self):
        self.g.model_readonly = True
        try:
            h = {'Origin': 'http://' + self.host}
            status, _, body = self.request('POST', '/model/api/jobs', b'{}', h)
            self.assertEqual(status, 503)
            self.assertIn('只读', body.decode())
            self.assertEqual(self.request('GET', '/model/api/jobs?write=true')[0], 503)
            self.assertEqual(self.request('GET', '/model/api/jobs')[0], 200)
            self.assertIn(b'data-product="model"', self.request('GET', '/')[2])
        finally:
            self.g.model_readonly = False

    def test_unified_portal_has_real_product_tabs_without_preview_banners(self):
        self.g.operator_readonly = True
        self.g.model_readonly = True
        try:
            status, _, body = self.request('GET', '/')
            self.assertEqual(status, 200)
            self.assertIn('天宽科技'.encode(), body)
            self.assertNotIn('内网预览'.encode(), body)
            self.assertNotIn('只读预览'.encode(), body)
            self.assertIn(b'data-product="model"', body)
            self.assertIn(b'data-product="operator"', body)
            self.assertIn(b'src="/model/"', body)
            self.assertIn(b'data-src="/operator/"', body)
            self.assertIn(b'<!doctype html>', body.lower())
            self.assertIn(b'<style', body.lower())
        finally:
            self.g.operator_readonly = False
            self.g.model_readonly = False

    def test_wildcard_bind_allowed_without_preview_gate_but_requires_source_restriction(self):
        upstreams = {n: ('127.0.0.1', s.server_port) for n, s in zip(('model', 'operator'), self.servers)}
        # Full-function mode: wildcard bind is allowed once source CIDRs and Hosts are explicit.
        g = gateway.make_server(('0.0.0.0', 0), upstreams, username='tester', password='test-only-password',
                                allowed_cidrs=['127.0.0.0/8'], allowed_hosts=['*'], preview_gate=False)
        self.assertFalse(g.preview_gate)
        g.server_close()
        with self.assertRaises(ValueError):
            gateway.make_server(('0.0.0.0', 0), upstreams, username='tester', password='test-only-password',
                                preview_gate=False)

    def test_portal_shell_survives_readonly_flags_being_off(self):
        # Regression: turning operator writes on must not degrade / to a notice page.
        self.g.operator_readonly = False
        self.g.model_readonly = False
        status, _, body = self.request('GET', '/')
        self.assertEqual(status, 200)
        self.assertIn(b'data-product="model"', body)
        self.assertIn(b'data-product="operator"', body)
        self.assertIn(b'class="company-logo"', body)

    def test_rewrite_can_be_disabled_for_prefix_aware_build(self):
        self.g.rewrite = False
        try:
            _, _, body = self.request('GET', '/model/page')
            self.assertIn(b'src="/assets/app.js"', body)
            self.assertNotIn(b'/model/assets', body)
        finally:
            self.g.rewrite = True

    def test_route_and_no_root_api_fallback(self):
        for name in ('model', 'operator'):
            status, _, body = self.request('GET', f'/{name}/api/jobs?q=1')
            self.assertEqual(status, 200)
            self.assertEqual(body.decode(), f'{name}:/api/jobs?q=1:None')
        self.assertEqual(self.request('GET', '/api/jobs')[0], 404)
        self.assertEqual(self.request('GET', '/assets/app.js')[0], 404)
    def test_page_rewrites_recognized_absolute_paths(self):
        for name in ('model', 'operator'):
            status, _, body = self.request('GET', f'/{name}/page')
            self.assertEqual(status, 200)
            for path in ('assets/app.js', 'api/jobs', 'static/x'):
                self.assertIn(f'/{name}/{path}'.encode(), body)
            self.assertIn(b'id="tk-unified-theme"', body)
        self.assertIn(b'--brand:#6557d2', self.request('GET', '/model/page')[2])
        self.assertIn(b'--accent:#6557d2!important', self.request('GET', '/operator/page')[2])

    def test_visual_theme_is_scoped_to_known_product_and_valid_head(self):
        original = b'<html><head><title>x</title></head><body>x</body></html>'
        self.assertIn(b'--brand:#6557d2', gateway.apply_visual_theme(original, 'model'))
        self.assertIn(b'--accent:#6557d2!important', gateway.apply_visual_theme(original, 'operator'))
        self.assertEqual(gateway.apply_visual_theme(original, 'other'), original)
        self.assertEqual(gateway.apply_visual_theme(b'<html>x</html>', 'model'), b'<html>x</html>')

    def test_theme_layer_precedes_app_stylesheet_and_beats_app_important(self):
        original = b'<html><head><link rel="stylesheet" href="/assets/app.css"></head><body><span class="chip !text-[10px]">P0</span></body></html>'
        themed = gateway.apply_visual_theme(original, 'operator')
        self.assertLess(themed.index(b'@layer tk-unified'), themed.index(b'rel="stylesheet"'))
        self.assertIn(b'body,body *{font-size:14px!important', themed)

    def test_company_logo_is_served_from_fixed_local_asset(self):
        status, headers, body = self.request('GET', '/brand/tanku.png')
        self.assertEqual(status, 200)
        self.assertEqual(headers.get('Content-Type'), 'image/png')
        self.assertTrue(body.startswith(b'\x89PNG\r\n\x1a\n'))

    def test_theme_html_is_never_served_from_browser_cache(self):
        status, headers, body = self.request('GET', '/model/cached-page',
                                             headers={'If-None-Match': '"stale-v1"'})
        self.assertEqual(status, 200)
        self.assertIn(b'id="tk-unified-theme"', body)
        self.assertEqual(headers.get('Cache-Control'), 'no-store, must-revalidate')
        self.assertNotIn('ETag', headers)
        self.assertNotIn('Last-Modified', headers)
    def test_post_origin_body_and_limit(self):
        self.assertEqual(self.request('POST', '/model/api/run', b'run')[0], 403)
        h = {'Origin': 'http://' + self.host}
        self.assertEqual(self.request('POST', '/operator/api/run', b'run', h)[2], b'operator:/api/run:run')
        self.assertEqual(self.request('POST', '/model/api/run', b'x'*33, h)[0], 413)
        self.assertEqual(self.request('POST', '/model/api/run', b'x', {'Origin': 'http://evil.invalid'})[0], 403)
    def test_auth_host_and_path_validation(self):
        self.assertEqual(self.request('GET', '/model/api/x', headers={'Authorization': ''})[0], 401)
        self.assertEqual(self.request('GET', '/model/api/x', headers={'Host': 'attacker.invalid'})[0], 400)
        for p in ('/model/../operator/api/x', '/model/%2e%2e/operator/api/x', '/model/%2foperator/api/x'):
            self.assertEqual(self.request('GET', p)[0], 400)
    def test_spoofed_forwarding_headers_are_stripped(self):
        _, _, body = self.request('GET', '/model/headers', headers={'X-Forwarded-Server':'fake', 'X-Real-IP':'1.2.3.4'})
        self.assertEqual(body, b'(None, None)')
    def test_already_prefixed_redirect_is_not_prefixed_again(self):
        self.assertEqual(self.request('GET', '/model/prefixed-location')[1]['Location'], '/model/api/next')

    def test_cookie_namespace_and_redirect(self):
        _, h, body = self.request('GET', '/model/cookie', headers={'Cookie': 'model__sid=m; operator__sid=o; sid=raw'})
        self.assertEqual(body, b'sid=m')
        self.assertEqual(h['Location'], '/model/api/next')
        self.assertEqual(h['Set-Cookie'], 'model__sid=abc; Path=/model/; HttpOnly')
    def test_binary_download(self):
        status, headers, body = self.request('GET', '/operator/download')
        self.assertEqual(status, 200); self.assertEqual(body, bytes(range(256)))
    def test_large_response_and_multipart_upload(self):
        status, _, body = self.request('GET', '/model/large')
        self.assertEqual(status, 200); self.assertEqual(len(body), 9 * 1024 * 1024); self.assertEqual(body, b'x' * len(body))
        h = {'Origin': 'http://' + self.host, 'Content-Type': 'multipart/form-data; boundary=x'}
        payload = b'--x\r\nblob\x00\xff\r\n--x--\r\n'
        self.assertEqual(self.request('POST', '/operator/upload', payload, h)[2], b'operator:/upload:' + payload)

    def test_sse_first_event_arrives_before_upstream_finishes(self):
        for name in ('model', 'operator'):
            c = http.client.HTTPConnection('127.0.0.1', self.g.server_port, timeout=2)
            start = time.monotonic(); c.request('GET', f'/{name}/events', headers={'Authorization': self.auth})
            r = c.getresponse(); first = r.readline(); elapsed = time.monotonic()-start
            self.assertEqual(first, b'data: first\n'); self.assertLess(elapsed, .5)
            self.assertIn(b'data: second', r.read()); c.close()


class GatewayRoutingTest(unittest.TestCase):
    def test_credential_file_permissions_and_symlink(self):
        import os, tempfile
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, 'auth')
            with open(p, 'w') as f: f.write('tester:test-password\n')
            os.chmod(p, 0o600)
            self.assertEqual(gateway.load_credentials(p), ('tester', 'test-password'))
            os.chmod(p, 0o644)
            with self.assertRaises(ValueError): gateway.load_credentials(p)
            os.chmod(p, 0o600)
            os.symlink(p, p + '.link')
            with self.assertRaises((ValueError, OSError)): gateway.load_credentials(p + '.link')

    def test_environment_preview_configuration_requires_explicit_settings(self):
        cfg = gateway.config_from_env({'WORKBENCH_BIND': '0.0.0.0', 'WORKBENCH_ALLOWED_CIDRS': '127.0.0.0/8,172.16.111.0/24',
                                       'WORKBENCH_ALLOWED_HOSTS': 'preview.internal:18080', 'WORKBENCH_PREVIEW_GATE': '1',
                                       'WORKBENCH_OPERATOR_READONLY': '1', 'WORKBENCH_REWRITE': '0'})
        self.assertEqual(cfg['address'], ('0.0.0.0', 18080))
        self.assertEqual(cfg['allowed_cidrs'], ['127.0.0.0/8', '172.16.111.0/24'])
        self.assertTrue(cfg['operator_readonly']); self.assertFalse(cfg['rewrite'])
        self.assertTrue(cfg['auth_required'])
        self.assertFalse(gateway.config_from_env({'WORKBENCH_AUTH_REQUIRED': '0'})['auth_required'])
        with self.assertRaises(ValueError): gateway.config_from_env({'WORKBENCH_AUTH_REQUIRED': 'maybe'})
        with self.assertRaises(ValueError): gateway.config_from_env({'WORKBENCH_PREVIEW_GATE': 'yes'})

    def test_prefixed_api_routes_are_unambiguous(self):
        self.assertEqual(gateway.route('/model/api/tasks?x=1'), ('model', '/api/tasks?x=1'))
        self.assertEqual(gateway.route('/operator/api/tasks?x=1'), ('operator', '/api/tasks?x=1'))
        self.assertIsNone(gateway.route('/api/tasks?x=1'))


if __name__ == '__main__': unittest.main()
