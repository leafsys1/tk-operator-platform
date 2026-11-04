"""Loopback-only experimental workbench gateway; see README for hard limits."""
import base64
import hmac
import http.client
import ipaddress
import json
import os
import re
import socket
import stat
from pathlib import Path
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qsl, unquote, urlsplit

HOP = {'connection', 'keep-alive', 'proxy-authenticate', 'proxy-authorization',
       'te', 'trailer', 'transfer-encoding', 'upgrade'}
TEXT_LIMIT = 8 * 1024 * 1024
def _rule(tags, body):
    """Every app nests content in classed wrappers, so match both bare and nested tags."""
    selectors = []
    for tag in tags:
        selectors += [f'body {tag}', f'body {tag}[class]', f'body [class] {tag}', f'body [class] {tag}[class]']
    return ','.join(selectors) + '{' + body + '}'


THEME_CSS = '''<style id="tk-unified-theme">
:root{--tk-purple:#6557d2;--tk-purple-strong:#5144bd;--tk-purple-soft:#f0edff;--tk-font-ui:Inter,"Noto Sans CJK SC","PingFang SC","Microsoft YaHei",system-ui,-apple-system,"Segoe UI",sans-serif}
html{font-size:14px!important}html,body,body *{font-family:var(--tk-font-ui)!important;-webkit-font-smoothing:antialiased}
body,body *{font-size:14px!important;line-height:1.45!important}
''' + _rule(['h1'], 'font-size:24px!important;line-height:1.25!important;font-weight:700!important') + '\n' \
    + _rule(['h2'], 'font-size:18px!important;line-height:1.3!important;font-weight:650!important') + '\n' \
    + _rule(['h3'], 'font-size:16px!important;line-height:1.35!important;font-weight:650!important') + '\n' \
    + _rule(['h4'], 'font-size:15px!important;line-height:1.35!important;font-weight:650!important') + '\n' \
    + _rule(['button', 'input', 'select', 'textarea'], 'font-size:13px!important;line-height:1.35!important') + '''
body table,body table *{font-size:13px!important;line-height:1.4!important}
body table th,body table td{padding:6px 9px!important;vertical-align:middle!important}
body table th,body table td{white-space:nowrap!important}
body table td:first-child,body table td:nth-child(2),body table th:first-child,body table th:nth-child(2){white-space:normal!important}
body table button,body table .chip,body table .badge,body table [class*="chip"]{white-space:nowrap!important}
body table [class*="whitespace-pre"],body table pre,body table code{font-size:11px!important;line-height:1.3!important}
body table td>div[class*="whitespace-pre"]{max-height:2.6em!important;overflow:hidden!important}
body table td:first-child,body table th:first-child{max-width:300px!important}
body table td:last-child,body table th:last-child{min-width:270px!important}
body table td:last-child>div,body table td:last-child>div[class]{flex-wrap:nowrap!important}
button:focus-visible,a:focus-visible,input:focus-visible,select:focus-visible,textarea:focus-visible{outline:2px solid var(--tk-purple)!important;outline-offset:2px}
.topbar{background:#fbfaff!important;border-color:#e5e3ef!important;color:#24243a!important}
.sidebar{background:#fbfaff!important;border-color:#e5e3ef!important;color:#514c71!important}
.side-link,.sidebar nav button{border-radius:9px!important;color:#5f5b78!important}
.side-link,.side-link span,.sidebar nav button,.sidebar nav button span{font-weight:400!important}
.side-link.active,.side-link.active span{font-weight:400!important}
#root>.grid.min-h-screen>aside nav button,#root>.grid.min-h-screen>aside nav button span{font-weight:400!important}
.side-link:hover,.sidebar nav button:hover{background:#f4f1ff!important;color:#5144bd!important}
.side-link.active,.sidebar nav button[aria-current="page"],.sidebar nav button[style*="--accent-soft"]{background:#f0edff!important;color:#5144bd!important;box-shadow:none!important}
.side-label,.nav-label{color:#77738f!important;font-size:13px!important;letter-spacing:.04em!important}
</style>'''


def route(target):
    for name in ('model', 'operator'):
        prefix = '/' + name + '/'
        if target.startswith(prefix):
            return name, '/' + target[len(prefix):]
    return None


def rewrite_text(data, name):
    # Deliberately narrow: quoted or CSS url() literal root paths only.
    return re.sub(rb'(["\'`(])/(api|assets|static)(?=[/"\'`)?#])',
                  lambda m: m.group(1) + b'/' + name.encode() + b'/' + m.group(2), data)


def apply_visual_theme(data, name):
    """Keep the two apps' behavior intact while aligning typography and accents."""
    if name == 'model':
        css = '''<style id="tk-unified-theme-vars">:root{--sidebar:200px!important;--brand:#6557d2!important;--brand-dark:#5144bd!important;--brand-faint:#f0edff!important;--nav:#f0edff!important;--blue:#7468df!important;--page:#f7f6fb!important;--surface2:#fbfaff!important;--surface3:#f3f0ff!important;--line:#e5e3ef!important;--line2:#ddd8ef!important;--ink:#24243a!important;--ink-soft:#514c71!important;--muted:#77738f!important} .sidebar{width:200px!important;padding:18px 11px!important;background:#fbfaff!important;border-color:#e5e3ef!important;color:#514c71!important}.sidebar>.brand{display:none!important}.main-wrap{margin-left:200px!important;width:calc(100% - 200px)!important}.side-link{min-height:37px!important;padding:0 11px!important}.workspace-wrap{margin:0 0 14px!important}.side-divider{margin:12px!important}.sidebar-bottom{padding-top:8px!important}.stat>strong{font-size:24px!important;line-height:1.25!important}.workspace-select.workspace-select{display:flex!important;align-items:center!important;gap:7px!important;width:100%!important;height:36px!important;min-height:36px!important;padding:7px 10.5px!important;border:1px solid #e5e3ef!important;border-radius:13px!important;background:rgba(251,250,255,.91)!important;box-shadow:0 10px 30px rgba(20,26,46,.09)!important;transform:none!important;color:#24243a!important;font-size:14px!important;font-weight:400!important;text-align:left!important}.workspace-select.workspace-select:hover,.workspace-select.workspace-select:active,.workspace-select[aria-expanded="true"]{border:1px solid #e5e3ef!important;background:rgba(251,250,255,.91)!important;box-shadow:0 10px 30px rgba(20,26,46,.09)!important;transform:none!important}.workspace-select:focus-visible{outline:2px solid var(--tk-purple)!important;outline-offset:2px}#workspace-btn{font-size:14px!important}.ws-avatar{display:none!important}.workspace-select::before{content:"📁";font-size:14px!important;line-height:1!important;flex:none!important}.workspace-select .txt{display:block!important;flex:1 1 auto!important;min-width:0!important;text-align:left!important}.workspace-select .txt b{display:block!important;font-size:14px!important;font-weight:400!important;color:#24243a!important;line-height:1.2!important;white-space:nowrap!important;overflow:hidden!important;text-overflow:ellipsis!important}.workspace-select .txt small{display:none!important}.ws-chev{margin-left:auto!important;width:auto!important;height:auto!important;flex:none!important;background:transparent!important;color:#77738f!important}.ws-chev .icon{display:none!important}.ws-chev::after{content:"▾";font-size:14px!important;line-height:1!important;color:#77738f!important}</style>'''
    elif name == 'operator':
        css = '''<style id="tk-unified-theme-vars">:root{--accent:#6557d2!important;--accent-soft:#6557d21f!important;--run:#6557d2!important;--panel-3:#6557d212!important;--bg:#f7f6fb!important;--panel:#fbfaffe8!important;--panel-2:#fff!important;--border:#e5e3ef!important;--border-strong:#d9d4ea!important;--fg:#24243a!important;--fg-dim:#514c71!important;--fg-mute:#77738f!important;--aurora:radial-gradient(900px 520px at 10% -10%,#9b8cff22,transparent 62%),radial-gradient(760px 480px at 94% 2%,#b7aaff1a,transparent 60%),radial-gradient(720px 460px at 62% 112%,#e3ddff22,transparent 62%)!important} body{background-color:#f7f6fb!important} #root>.grid.min-h-screen{grid-template-columns:200px minmax(0,1fr)!important} #root>.grid.min-h-screen>aside{background:#fbfaff!important;border-color:#e5e3ef!important} #root>.grid.min-h-screen>aside>div:first-child{display:none!important} #root>.grid.min-h-screen>aside nav button{min-height:37px!important;border-radius:9px!important;font-size:14px!important} #root>.grid.min-h-screen>aside nav button[style*="background"]{border-color:transparent!important} </style>'''
    else:
        return data
    payload = data.decode('utf-8', errors='strict')
    head = re.search(r'<head\b[^>]*>', payload, re.I)
    if head:
        marker = head.end()
    else:
        marker = payload.lower().find('</head>')
        if marker < 0:
            return data
    payload = payload[:marker] + _in_layer(THEME_CSS) + _in_layer(css) + payload[marker:]
    return payload.encode('utf-8')


def _in_layer(style_html):
    """Declare our rules in a cascade layer placed first, so app `!important` utilities lose."""
    return re.sub(r'(<style[^>]*>)(.*?)(</style>)',
                  lambda m: m.group(1) + '@layer tk-unified{' + m.group(2) + '}' + m.group(3),
                  style_html, flags=re.S)


def load_credentials(path):
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    try:
        info = os.fstat(fd)
        if not stat.S_ISREG(info.st_mode) or stat.S_IMODE(info.st_mode) != 0o600 or info.st_uid != os.geteuid() or info.st_size > 4096:
            raise ValueError('credential file must be owned by this user and mode 600')
        with os.fdopen(fd, 'rb', closefd=False) as f:
            raw = f.read(4097).decode('utf-8')
        if raw.endswith('\n'): raw = raw[:-1]
        if '\n' in raw or '\r' in raw: raise ValueError('invalid credential file')
        user, sep, password = raw.partition(':')
        if not sep or not user or not password: raise ValueError('invalid credential file')
        return user, password
    finally:
        os.close(fd)


def safe_dry_run(body):
    def unique(pairs):
        obj = {}
        for key, value in pairs:
            if key in obj:
                raise ValueError('duplicate key')
            obj[key] = value
        return obj
    try:
        parsed = json.loads(body.decode('utf-8'), object_pairs_hook=unique,
                            parse_constant=lambda value: (_ for _ in ()).throw(ValueError(value)))
        return isinstance(parsed, dict) and parsed.get('dry_run') is True
    except (ValueError, UnicodeError, RecursionError):
        return False


def make_server(address, upstreams, *, username, password, max_body=8*1024*1024, timeout=30,
                allowed_cidrs=None, allowed_hosts=None, preview_gate=False, rewrite=True, operator_readonly=False, model_readonly=False, auth_required=True):
    if address[0] not in ('127.0.0.1', '0.0.0.0') or (auth_required and (not username or not password or ':' in username)):
        raise ValueError('invalid bind address or credentials')
    if address[0] == '0.0.0.0' and (not allowed_cidrs or not allowed_hosts):
        raise ValueError('wildcard bind requires explicit CIDRs and hosts')
    networks = [ipaddress.ip_network(cidr, strict=True) for cidr in (allowed_cidrs or ['127.0.0.0/8'])]
    if any(n.version != 4 for n in networks): raise ValueError('only IPv4 CIDRs supported')
    if allowed_hosts is not None:
        if not allowed_hosts or ('*' in allowed_hosts and len(allowed_hosts) != 1) or any(h != '*' and (not re.fullmatch(r'[A-Za-z0-9.-]+(?::[0-9]{1,5})?', h) or (':' in h and not 1 <= int(h.rsplit(':', 1)[1]) <= 65535)) for h in allowed_hosts):
            raise ValueError('hosts must be explicit host values or a sole wildcard')
    if set(upstreams) != {'model', 'operator'} or any(h != '127.0.0.1' for h, p in upstreams.values()):
        raise ValueError('exactly two loopback upstreams required')
    if max_body < 0 or timeout <= 0:
        raise ValueError('invalid limits')
    server = ThreadingHTTPServer(address, Handler)
    server.daemon_threads = True
    server.upstreams, server.max_body, server.timeout = upstreams, max_body, timeout
    server.auth = ('Basic ' + base64.b64encode((username + ':' + password).encode()).decode()) if auth_required else None
    server.auth_required = auth_required
    server.allowed_networks = networks
    server.allowed_hosts = set(allowed_hosts) if allowed_hosts is not None else {'127.0.0.1:' + str(server.server_port)}
    server.preview_gate = preview_gate
    server.rewrite = rewrite
    server.operator_readonly = operator_readonly
    server.model_readonly = model_readonly
    return server


class Handler(BaseHTTPRequestHandler):
    protocol_version = 'HTTP/1.1'
    server_version = 'WorkbenchPrototype'
    sys_version = ''
    def log_message(self, *args):
        pass  # Do not log paths, credentials or requests.

    def fail(self, code):
        self.send_response(code)
        if code == 401: self.send_header('WWW-Authenticate', 'Basic realm="workbench", charset="UTF-8"')
        self.send_header('Content-Length', '0')
        self.send_header('Cache-Control', 'no-store')
        self.send_header('Connection', 'close')
        self.end_headers(); self.close_connection = True

    def unavailable(self, message):
        body = message.encode('utf-8')
        self.send_response(503); self.send_header('Content-Type', 'text/plain; charset=utf-8')
        self.send_header('Content-Length', str(len(body))); self.send_header('Cache-Control', 'no-store')
        self.send_header('Connection', 'close'); self.end_headers(); self.close_connection = True
        if self.command != 'HEAD': self.wfile.write(body)

    def handle_all(self):
        self.connection.settimeout(self.server.timeout)
        if not any(ipaddress.ip_address(self.client_address[0]) in net for net in self.server.allowed_networks):
            return self.fail(403)
        hosts = self.headers.get_all('Host', [])
        if (len(hosts) != 1 or
                ('*' not in self.server.allowed_hosts and hosts[0] not in self.server.allowed_hosts) or
                not re.fullmatch(r'[A-Za-z0-9.-]+(?::[0-9]{1,5})?', hosts[0]) or
                (':' in hosts[0] and not 1 <= int(hosts[0].rsplit(':', 1)[1]) <= 65535)): return self.fail(400)
        host = hosts[0]
        auth = self.headers.get_all('Authorization', [])
        if self.server.auth_required and (len(auth) != 1 or not hmac.compare_digest(auth[0].encode('utf-8'), self.server.auth.encode('ascii'))): return self.fail(401)
        origins = self.headers.get_all('Origin', [])
        if len(origins) > 1: return self.fail(403)
        origin = origins[0] if origins else None
        if (origin is not None and origin != 'http://' + host) or self.headers.get('Sec-Fetch-Site', '') not in ('', 'same-origin', 'none'):
            return self.fail(403)
        if self.command not in ('GET', 'HEAD', 'POST', 'PUT', 'PATCH', 'DELETE', 'OPTIONS'):
            return self.fail(405)
        if self.command not in ('GET', 'HEAD', 'OPTIONS') and origin != 'http://' + host:
            return self.fail(403)
        if self.headers.get('Transfer-Encoding') or len(self.headers.get_all('Content-Length', [])) > 1:
            return self.fail(400)
        rawlen = self.headers.get('Content-Length', '0')
        if not re.fullmatch(r'[0-9]+', rawlen): return self.fail(400)
        length = int(rawlen)
        if length > self.server.max_body: return self.fail(413)
        if self.command in ('GET', 'HEAD', 'OPTIONS') and length: return self.fail(400)
        parts = urlsplit(self.path)
        path = parts.path
        decoded = unquote(path)
        if (not self.path.startswith('/') or '//' in decoded or parts.scheme or parts.netloc
                or '\\' in decoded or any(x in ('.', '..') for x in decoded.split('/'))
                or re.search(r'%(?:2f|5c|2e|25)', path, re.I)):
            return self.fail(400)
        if self.path == '/' and self.command in ('GET', 'HEAD'):
            # The unified shell is the product UI and must never be swapped for a
            # degraded notice page just because a read-only flag changed.
            try:
                page = (Path(__file__).resolve().parent / 'portal.html').read_bytes()
            except OSError:
                return self.fail(503)
            self.send_response(200); self.send_header('Content-Type', 'text/html; charset=utf-8')
            self.send_header('Content-Length', str(len(page))); self.send_header('Cache-Control', 'no-store'); self.end_headers()
            if self.command != 'HEAD': self.wfile.write(page)
            return
        if path == '/brand/tanku.png' and self.command in ('GET', 'HEAD'):
            try:
                logo = (Path(__file__).resolve().parent / 'tanku-logo.png').read_bytes()
            except OSError:
                return self.fail(404)
            if not logo.startswith(b'\x89PNG\r\n\x1a\n') or len(logo) > 1024 * 1024:
                return self.fail(404)
            self.send_response(200)
            self.send_header('Content-Type', 'image/png')
            self.send_header('Content-Length', str(len(logo)))
            self.send_header('Cache-Control', 'public, max-age=3600')
            self.end_headers()
            if self.command != 'HEAD': self.wfile.write(logo)
            return
        if path in ('/model', '/operator') and self.command in ('GET', 'HEAD'):
            self.send_response(308); self.send_header('Location', path + '/'); self.send_header('Content-Length', '0'); self.end_headers(); return
        target = route(self.path)
        if not target: return self.fail(404)
        name, upstream_path = target
        if (self.server.operator_readonly and name == 'operator') or (self.server.model_readonly and name == 'model'):
            query_keys = [k.casefold() for k, _ in parse_qsl(parts.query, keep_blank_values=True)]
            if self.command not in ('GET', 'HEAD') or 'write' in query_keys or (name == 'operator' and (decoded == '/operator/api/proto' or decoded.startswith('/operator/api/proto/'))):
                return self.unavailable('预览只读限制：此操作未执行。')
        try:
            body = self.rfile.read(length) if length else None
            if length and len(body) != length: return self.fail(400)
            if (self.server.preview_gate and self.command == 'POST' and
                    re.fullmatch(r'/operator/api/tasks/[^/]+/actions/*', decoded)):
                media = self.headers.get('Content-Type', '').split(';', 1)[0].strip().lower()
                if media != 'application/json' or not safe_dry_run(body or b''):
                    return self.unavailable('预览安全限制：禁止实际执行，仅允许明确的 dry_run=true。')
            self.proxy(name, upstream_path, body)
        except (OSError, http.client.HTTPException, ValueError):
            self.close_connection = True

    def proxy(self, name, path, body):
        host, port = self.server.upstreams[name]
        connection = http.client.HTTPConnection(host, port, timeout=self.server.timeout)
        headers = {}
        dynamic_hop = {x.strip().lower() for x in self.headers.get('Connection', '').split(',')}
        for key, val in self.headers.items():
            low = key.lower()
            if low.startswith('x-forwarded-') or low in HOP | dynamic_hop | {'host', 'authorization', 'cookie', 'origin', 'referer', 'accept-encoding', 'content-length', 'forwarded', 'x-real-ip', 'if-none-match', 'if-modified-since'}: continue
            headers[key] = val
        headers['Accept-Encoding'] = 'identity'
        if body is not None: headers['Content-Length'] = str(len(body))
        if self.headers.get('Origin'): headers['Origin'] = f'http://{host}:{port}'
        prefix = name + '__'
        cookies = []
        for cookie in self.headers.get('Cookie', '').split(';'):
            k, sep, v = cookie.strip().partition('=')
            if sep and k.startswith(prefix): cookies.append(k[len(prefix):] + '=' + v)
        if cookies: headers['Cookie'] = '; '.join(cookies)
        try:
            connection.request(self.command, path, body=body, headers=headers)
            response = connection.getresponse()
        except (OSError, http.client.HTTPException):
            connection.close(); self.fail(502); return
        typ = response.getheader('Content-Type', '').split(';', 1)[0].strip().lower()
        rewrite = (self.server.rewrite and typ in ('text/html', 'text/css', 'text/javascript', 'application/javascript', 'application/x-javascript') and self.command != 'HEAD' and response.status not in (204, 304))
        payload = None
        if rewrite:
            if response.getheader('Content-Encoding', 'identity').lower() != 'identity':
                connection.close(); return self.fail(502)
            try: payload = response.read(TEXT_LIMIT + 1)
            except (OSError, http.client.HTTPException): connection.close(); return self.fail(502)
            if len(payload) > TEXT_LIMIT: connection.close(); return self.fail(502)
            payload = rewrite_text(payload, name)
            if typ == 'text/html':
                try: payload = apply_visual_theme(payload, name)
                except UnicodeError:
                    connection.close(); return self.fail(502)
        self.send_response(response.status)
        dynamic_hop = {x.strip().lower() for x in response.getheader('Connection', '').split(',')}
        for key, val in response.getheaders():
            low = key.lower()
            if low in HOP | dynamic_hop | {'content-length', 'server', 'date'}: continue
            if rewrite and low in ('etag', 'content-md5', 'content-encoding', 'digest', 'last-modified', 'cache-control', 'expires'): continue
            if low == 'set-cookie':
                fields = [x.strip() for x in val.split(';')]; cookie, sep, value = fields[0].partition('=')
                if not sep or not re.fullmatch(r'[!#$%&\'*+.^_`|~0-9A-Za-z-]+', cookie): continue
                fields = [x for x in fields[1:] if not x.lower().startswith(('path=', 'domain='))]
                val = prefix + cookie + '=' + value + '; Path=/' + name + '/'
                if fields: val += '; ' + '; '.join(fields)
            if low == 'location':
                if val.startswith(f'http://{host}:{port}/'): val = val[len(f'http://{host}:{port}'):]
                if val.startswith('/') and not val.startswith('//') and val != '/' + name and not val.startswith('/' + name + '/'):
                    val = '/' + name + val
            self.send_header(key, val)
        if rewrite and typ == 'text/html':
            self.send_header('Cache-Control', 'no-store, must-revalidate')
        if payload is not None: self.send_header('Content-Length', str(len(payload)))
        else: self.send_header('Connection', 'close'); self.close_connection = True
        self.end_headers()
        try:
            if self.command != 'HEAD':
                if payload is not None: self.wfile.write(payload)
                else:
                    while True:
                        chunk = response.read1(65536)
                        if not chunk: break
                        self.wfile.write(chunk); self.wfile.flush()
        except (OSError, http.client.HTTPException):
            self.close_connection = True
        finally: connection.close()

    do_GET = do_HEAD = do_POST = do_PUT = do_PATCH = do_DELETE = do_OPTIONS = handle_all


def config_from_env(env):
    def flag(name, default):
        value = env.get(name, default)
        if value not in ('0', '1'): raise ValueError(name + ' must be 0 or 1')
        return value == '1'
    def listing(name):
        value = env.get(name)
        return [part.strip() for part in value.split(',')] if value is not None else None
    def auth_required():
        value = env.get('WORKBENCH_AUTH_REQUIRED', '1')
        if value not in ('0', '1'): raise ValueError('WORKBENCH_AUTH_REQUIRED must be 0 or 1')
        return value == '1'
    port = int(env.get('WORKBENCH_PORT', '18080'))
    if not 1 <= port <= 65535: raise ValueError('invalid port')
    return dict(address=(env.get('WORKBENCH_BIND', '127.0.0.1'), port),
                allowed_cidrs=listing('WORKBENCH_ALLOWED_CIDRS'), allowed_hosts=listing('WORKBENCH_ALLOWED_HOSTS'),
                preview_gate=flag('WORKBENCH_PREVIEW_GATE', '0'),
                auth_required=auth_required(),
                operator_readonly=flag('WORKBENCH_OPERATOR_READONLY', '0'), model_readonly=flag('WORKBENCH_MODEL_READONLY', '0'), rewrite=flag('WORKBENCH_REWRITE', '1'))


if __name__ == '__main__':
    try:
        user, password = load_credentials(os.environ['WORKBENCH_CREDENTIAL_FILE'])
        server = make_server(upstreams={'model': ('127.0.0.1', 8090), 'operator': ('127.0.0.1', 18078)},
                             username=user, password=password, **config_from_env(os.environ))
    except (ValueError, KeyError, OSError, UnicodeError) as exc:
        raise SystemExit('Configuration error: ' + str(exc))
    server.serve_forever()
