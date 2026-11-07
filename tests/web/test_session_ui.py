"""Real-browser layout/lifecycle regression against isolated Demo and mock state.

Uses an existing Firefox (VLA_FIREFOX or /opt/firefox/firefox); never downloads
browser assets, probes devices, or touches a live workspace/session.
"""
import asyncio
import base64
import json
import os
from pathlib import Path
import shutil
import socket
import subprocess
import tempfile
import threading
import time
import uuid

import pytest


class Browser:
    def __init__(self, connection):
        self.connection = connection
        self.sequence = 0
        self.receive()
        self.call('WebDriver:NewSession', capabilities={})

    def receive(self):
        size = b''
        while not size.endswith(b':'):
            part = self.connection.recv(1)
            assert part, 'Firefox disconnected'
            size += part
        remaining, data = int(size[:-1]), b''
        while remaining:
            part = self.connection.recv(remaining)
            assert part, 'Firefox disconnected'
            data += part
            remaining -= len(part)
        return json.loads(data)

    def call(self, name, **params):
        self.sequence += 1
        data = json.dumps([0, self.sequence, name, params]).encode()
        self.connection.sendall(str(len(data)).encode() + b':' + data)
        result = self.receive()
        assert result[1] == self.sequence and not result[2], result
        return result[3]

    def js(self, script):
        return self.call('WebDriver:ExecuteScript', script=script, args=[],
                         newSandbox=False, sandbox=None)['value']

    def wait(self, expression, timeout=12):
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            result = self.js('return ' + expression)
            if result:
                return result
            time.sleep(.05)
        raise AssertionError((expression, self.js('return document.body.innerText')))

    def click(self, selector):
        self.js(f'document.querySelector({json.dumps(selector)}).click()')

    def fill(self, selector, value, event='change'):
        self.js(f'const el=document.querySelector({json.dumps(selector)});'
                f'el.value={json.dumps(value)};el.dispatchEvent(new Event({json.dumps(event)},{{bubbles:true}}))')

    def key(self, value):
        self.call('WebDriver:PerformActions', actions=[{'type': 'key', 'id': 'keyboard',
            'actions': [{'type': 'keyDown', 'value': value}, {'type': 'keyUp', 'value': value}]}])

    def shot(self, path):
        path.write_bytes(base64.b64decode(self.call('WebDriver:TakeScreenshot',
            id=None, highlights=[], full=False)['value']))


def unused_port():
    with socket.socket() as listener:
        listener.bind(('127.0.0.1', 0))
        return listener.getsockname()[1]


@pytest.fixture
def short_session_root():
    # AF_UNIX paths must fit the kernel limit, including the session UUID.
    with tempfile.TemporaryDirectory(prefix='vui-') as directory:
        yield Path(directory)


@pytest.fixture
def session_browser(tmp_path, short_session_root, monkeypatch, request):
    firefox = os.environ.get('VLA_FIREFOX') or shutil.which('firefox') or '/opt/firefox/firefox'
    if not Path(firefox).is_file():
        pytest.skip('Existing Firefox required; browser verification is unrun')
    import uvicorn
    import vla.web.hardware as hardware
    from vla.web.app import create_app
    monkeypatch.setattr(hardware, 'inventory', lambda: {
        'serials': [], 'warnings': [], 'cameras': [
            {'value': serial, 'label': serial + ' · Mock RealSense', 'selectable': True}
            for serial in ['BASE', 'LEFT', 'RIGHT', 'SIDE']]})
    app = create_app(short_session_root, demo=True)
    scenario = {'record': None, 'state': 'standby', 'final': False, 'real': False,
                'bootstrap_delay': .8, 'check_delay': 0, 'check_fail': False,
                'create_mode': 'ok', 'posts': [], 'checks': [], 'commands': [], 'snapshots': 0}
    scenario['reduced_motion'] = getattr(request, 'param', False)
    manager = app.state.manager
    original_active, original_snapshot = manager.active, manager.snapshot
    original_create, original_rpc = manager.create, manager.rpc

    def active():
        return original_active() if scenario['real'] else scenario['record']

    def snapshot(since=-1):
        scenario['snapshots'] += 1
        if scenario['real']:
            return original_snapshot(since)
        record = scenario['record']
        if not record:
            return {'active': None, 'connection_state': 'idle', 'status': {'state': 'unavailable'}, 'samples': [], 'now': time.monotonic()}
        now = time.monotonic()
        return {'active': record, 'session_id': record['session_id'], 'config': record['config'],
                'now': now, 'samples': [], 'commands': [], 'revision': 0,
                'status': {'state': scenario['state'], 'final': scenario['final'], 'updated_at': now,
                           'episode': 3, 'frames': 123, 'elapsed_s': 4.1, 'events': ['mock final event'],
                           'error': 'mock fault' if scenario['state'] == 'fault' else '',
                           'available_commands': ['enable', 'pause', 'stop']}}

    def create(data):
        scenario['posts'].append(data)
        if scenario['real']:
            return original_create(data)
        if scenario['create_mode'] == 'fail':
            raise ValueError('mock create rejected')
        time.sleep(.4)
        record = {'session_id': uuid.uuid4().hex, 'alive': True, 'config': data,
                  'directory': str(tmp_path / 'mock-session'), 'demo': True}
        directory = Path(record['directory'])
        directory.mkdir(exist_ok=True)
        (directory / 'config.json').write_text(json.dumps(data))
        scenario.update(record=record, state='starting', final=False)
        return record

    def rpc(kind, **values):
        if scenario['real']:
            return original_rpc(kind, **values)
        if kind == 'command':
            scenario['commands'].append(values)
            if values['operation'] == 'stop':
                scenario.update(state='stopped', final=True)
                scenario['record'] = {**scenario['record'], 'alive': False}
            return {'status': 'completed'}
        raise RuntimeError('Mock preview intentionally unavailable')

    monkeypatch.setattr(manager, 'active', active)
    monkeypatch.setattr(manager, 'snapshot', snapshot)
    monkeypatch.setattr(manager, 'create', create)
    monkeypatch.setattr(manager, 'rpc', rpc)

    @app.middleware('http')
    async def simulate(request, call_next):
        from starlette.responses import JSONResponse
        scenario['loop'] = asyncio.get_running_loop()
        if request.url.path == '/api/bootstrap':
            await asyncio.sleep(scenario['bootstrap_delay'])
        if request.url.path == '/api/preflight':
            scenario['checks'].append(await request.json())
            await asyncio.sleep(scenario['check_delay'])
            if scenario['check_fail']:
                return JSONResponse({'ok': False, 'errors': ['mock preflight rejected']})
        if request.url.path == '/api/sessions' and scenario['create_mode'] == 'lost':
            data = await request.json()
            scenario['posts'].append(data)
            # No acknowledgement and no session yet: a later snapshot resolves it.
            return JSONResponse({'detail': 'mock acknowledgement lost'}, status_code=503)
        return await call_next(request)

    port, marionette = unused_port(), unused_port()
    server = uvicorn.Server(uvicorn.Config(app, host='127.0.0.1', port=port, log_level='critical'))
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    def disconnect():
        for protocol in list(server.server_state.connections):
            if hasattr(protocol, 'ws_server'):
                protocol.transport.close()
    scenario['disconnect'] = lambda: scenario['loop'].call_soon_threadsafe(disconnect)
    profile = tmp_path / 'firefox'
    profile.mkdir()
    (profile / 'user.js').write_text('\n'.join(
        f'user_pref({json.dumps(key)}, {json.dumps(value)});' for key, value in {
            'marionette.port': marionette, 'browser.shell.checkDefaultBrowser': False,
            'browser.startup.homepage_override.mstone': 'ignore',
            'datareporting.policy.dataSubmissionEnabled': False, 'app.update.auto': False,
            'browser.aboutwelcome.enabled': False,
            'ui.prefersReducedMotion': int(scenario['reduced_motion']),
            'toolkit.legacyUserProfileCustomizations.stylesheets': True}.items()))
    (profile / 'chrome').mkdir()
    (profile / 'chrome/userChrome.css').write_text(':root { min-width: 0 !important; }')
    connection = None
    with (tmp_path / 'firefox.log').open('w') as log:
        process = subprocess.Popen([firefox, '--headless', '--no-remote', '--profile', str(profile),
                                    '--marionette', 'about:blank'], stdout=log, stderr=log)
        try:
            deadline = time.monotonic() + 25
            while time.monotonic() < deadline:
                try:
                    connection = socket.create_connection(('127.0.0.1', marionette), timeout=1)
                    break
                except OSError:
                    time.sleep(.1)
            assert connection, (tmp_path / 'firefox.log').read_text()
            connection.settimeout(30)
            browser = Browser(connection)
            browser.call('WebDriver:SetWindowRect', width=1440, height=1100)
            browser.call('WebDriver:Navigate', url=f'http://127.0.0.1:{port}/')
            yield browser, scenario, tmp_path
        finally:
            # This manager belongs solely to this test's unique temporary root.
            if scenario['real']:
                record = original_active()
                if record and record['alive']:
                    try:
                        state = original_rpc('state')
                        original_rpc('command', session_id=record['session_id'], revision=state['revision'],
                                     command_id=uuid.uuid4().hex, operation='stop', args={})
                        deadline = time.monotonic() + 10
                        while original_active()['alive'] and time.monotonic() < deadline:
                            time.sleep(.1)
                    finally:
                        if original_active()['alive']:
                            import signal
                            os.kill(record['pid'], signal.SIGTERM)
            if connection:
                connection.close()
            process.terminate()
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
            server.should_exit = True
            thread.join(timeout=10)


def phase(browser, name):
    browser.wait(f'document.querySelector("#livePanel").dataset.phase==={json.dumps(name)}')


def test_session_layout_and_lifecycle(session_browser):
    b, scenario, root = session_browser
    assert b.js('return document.querySelector("#configHost").hidden')
    phase(b, 'configure')
    scenario['bootstrap_delay'] = 0
    b.wait('document.querySelector("#camera-device-base")?.options.length>1')
    assert b.js('return document.querySelector("#sessionWorkbench").hidden && !document.querySelector("#advancedSetup").open')
    assert b.js('return getComputedStyle(document.querySelector(".config-grid")).gridTemplateColumns.split(" ").length===2')
    # The action bar stays on screen even before the long hardware form is scrolled.
    assert b.js('const r=document.querySelector("#start").getBoundingClientRect();return r.top>=0 && r.bottom<=innerHeight')
    b.shot(root / 'configure-dark.png')
    for workflow in ['collect', 'deploy']:
        b.click('#' + workflow + 'Tab')
        for arms in ['dual', 'single_left']:
            b.fill('[name=arms]', arms)
            for source in ['gello', 'gamepad']:
                b.fill('[name=source]', source)
                b.wait('!!document.querySelector("#hardware-LEFT_ROBOT_IP")')
                assert b.js(f'return !!document.querySelector("#hardware-GELLO_CONFIG")==={str(source == "gello").lower()}')
                assert b.js(f'return document.querySelector("#workspaceSetup").hidden==={str(workflow != "collect" or source != "gamepad").lower()}')
                assert b.js(f'return document.querySelector("#modelFields").hidden==={str(workflow != "deploy").lower()}')
                b.click('#preflight')
                b.wait('!document.querySelector("#preflightResult").hidden && !document.querySelector("#preflight").disabled')
                payload = scenario['checks'][-1]
                assert (payload['workflow'], payload['arms'], payload['source']) == (workflow, arms, source)
    b.click('#collectTab')
    b.fill('[name=task]', '保留任务草稿', 'input')
    b.fill('#camera-device-base', 'BASE')
    b.fill('[name=source]', 'gello')
    b.fill('[name=source]', 'gamepad')
    assert b.js('return document.querySelector("#camera-base").value==="BASE"')
    scenario.update(check_delay=.6, check_fail=True)
    b.click('#preflight')
    assert b.js('return document.querySelector("#start").disabled && document.querySelector("#preflight").disabled && document.querySelector("[name=task]").disabled')
    b.wait('document.querySelector("#preflightResult").textContent.includes("mock preflight rejected")')
    b.wait('!document.querySelector("#start").disabled')
    phase(b, 'configure')
    scenario.update(check_delay=0, check_fail=False, create_mode='fail')
    b.click('#start')
    b.wait('document.querySelector("#error").textContent.includes("mock create rejected")')
    b.wait('!document.querySelector("#start").disabled')
    assert b.js('return document.querySelector("[name=task]").value==="保留任务草稿"')
    # Lost acknowledgement: even repeated idle snapshots cannot unlock a pending POST.
    scenario['create_mode'] = 'lost'
    b.click('#start')
    b.wait('document.querySelector("#requestStatus").textContent.includes("尚未确认")')
    before = len(scenario['posts'])
    b.click('#start')
    time.sleep(.4)
    assert len(scenario['posts']) == before
    assert b.js('return document.querySelector("#start").disabled')
    saved = scenario['posts'][-1]
    assert saved['source'] == 'gamepad' and saved['arms'] == 'single_left'
    assert saved['camera_serials'] == {'base': 'BASE'}
    (root / 'mock-session').mkdir()
    (root / 'mock-session/config.json').write_text(json.dumps(saved))
    scenario.update(record={'session_id': 'confirmed-after-loss', 'config': saved, 'alive': True,
                            'directory': str(root / 'mock-session'), 'demo': True}, state='starting')
    phase(b, 'running')
    assert b.js('return document.querySelector("#state").textContent==="启动中" && document.querySelector("[name=task]").disabled')
    # Keyboard resizing persists only the live collect workbench preference.
    b.js('document.querySelector("#sessionWorkbench .column-resizer").focus()')
    b.key('\ue014')  # ArrowRight
    ratios = b.js('return localStorage.getItem("vla.columnRatios")')
    assert 'collect' in json.loads(ratios)
    # Same form node and handlers, keyboard trap, inert background, focus restoration.
    b.js('window.originalSetup=document.querySelector("#setup")')
    b.click('#viewConfig')
    b.wait('document.querySelector("#configDrawer").open')
    assert b.js('return window.originalSetup===document.querySelector("#drawerBody #setup") && [...document.querySelectorAll("#setup input,#setup select,#setup textarea,#setup button")].every(el=>el.disabled)')
    b.js('document.querySelector("#processingTab").focus()')
    assert b.js('return document.querySelector("#configDrawer").contains(document.activeElement)')
    for _ in range(15):
        b.key('\ue004')  # Tab
        assert b.js('return document.querySelector("#configDrawer").contains(document.activeElement)')
    b.key('\ue00c')  # Escape
    b.wait('!document.querySelector("#configDrawer").open')
    assert b.js('return document.activeElement.id==="viewConfig" && window.originalSetup===document.querySelector("#configHome #setup")')
    assert b.js('return localStorage.getItem("vla.columnRatios")') == ratios
    # Duplicate updates do not replay phase animations or touch width preferences.
    b.js('window.stageAnimations=0;document.querySelector("#sessionSummary").animate=()=>{window.stageAnimations++}')
    scenario['state'] = 'fault'
    b.wait('document.querySelector("#error").textContent.includes("mock fault")')
    phase(b, 'running')
    assert b.js('return document.querySelector("[name=task]").disabled && window.stageAnimations===0')
    b.click('#processingTab')
    b.click('#collectTab')
    assert not scenario['commands']
    b.call('WebDriver:Refresh')
    phase(b, 'running')
    assert b.js('return document.querySelector("[name=task]").disabled && document.getAnimations().length===0')
    b.call('WebDriver:SetWindowRect', width=390, height=844)
    assert b.js('return innerWidth===390')
    assert b.js('return document.documentElement.scrollWidth<=innerWidth')
    b.click('#themeToggle')
    b.click('#viewConfig')
    assert b.js('return Math.abs(document.querySelector("#configDrawer").getBoundingClientRect().width-innerWidth)<2')
    b.wait('document.getAnimations().length===0')
    b.shot(root / 'drawer-mobile-light.png')
    b.key('\ue00c')
    b.wait('!document.querySelector("#configDrawer").open')
    b.click('[data-op=stop]')
    phase(b, 'finished')
    assert b.js('return document.querySelector("#episode").textContent==="3" && document.querySelector("#frames").textContent.includes("123") && document.querySelector("#events").textContent.includes("mock final event")')
    b.click('#nextSession')
    phase(b, 'configure')
    b.fill('[name=task]', '下一会话草稿', 'input')
    start_snapshots = scenario['snapshots']
    time.sleep(.4)
    assert scenario['snapshots'] > start_snapshots
    assert b.js('return document.querySelector("#livePanel").dataset.phase==="configure" && document.querySelector("[name=task]").value==="下一会话草稿"')
    assert b.js('return localStorage.getItem("vla.columnRatios")') == ratios
    b.shot(root / 'configure-mobile-light.png')
    assert b.js('return document.documentElement.scrollWidth<=innerWidth')
    # Even an older non-final snapshot cannot revive the dismissed session.
    scenario.update(final=False, state='standby')
    time.sleep(.3)
    phase(b, 'configure')
    scenario.update(final=True, state='stopped', create_mode='ok')
    b.click('#start')
    b.wait('document.querySelector("#requestStatus").textContent.includes("提交")')
    phase(b, 'configure')
    assert b.js('return document.querySelector("#start").disabled && document.querySelector("[name=task]").disabled')
    before = len(scenario['posts'])
    b.click('#start')
    phase(b, 'running')
    assert len(scenario['posts']) == before
    b.wait('!document.querySelector("[data-op=stop]").disabled')
    b.click('[data-op=stop]')
    phase(b, 'finished')
    b.click('#nextSession')
    phase(b, 'configure')
    # A genuinely new ID must enter the workbench, even after dismissing old finals.
    scenario['record'] = {**scenario['record'], 'session_id': 'new-external-session', 'alive': True}
    scenario.update(final=False, state='standby')
    phase(b, 'running')


@pytest.mark.parametrize('session_browser', [False, True], indirect=True)
def test_isolated_demo_start_reconnect_stop(session_browser):
    b, scenario, root = session_browser
    phase(b, 'configure')
    scenario.update(real=True, bootstrap_delay=0)
    b.wait('document.querySelector("#camera-device-base")?.options.length>1')
    b.click('#deployTab')
    b.fill('[name=arms]', 'dual')
    b.fill('[name=source]', 'gello')
    b.fill('[name=policy_path]', str(root / 'demo-checkpoint'), 'input')
    b.click('#start')
    phase(b, 'running')
    b.wait('document.querySelector("#state").textContent==="待命"')
    assert len(scenario['posts']) == 1
    assert scenario['posts'][0]['workflow'] == 'deploy'
    assert scenario['posts'][0]['arms'] == 'dual' and scenario['posts'][0]['source'] == 'gello'
    assert b.js('return document.querySelector("#summaryFields").textContent.includes("demo-checkpoint") && document.querySelector("#posePanel").hidden')
    scenario['bootstrap_delay'] = 1
    scenario['disconnect']()
    b.wait('document.querySelector("#connection").textContent.includes("中断")')
    phase(b, 'running')
    assert b.js('return document.querySelector("[name=task]").disabled && document.querySelector("[data-op=enable]").disabled')
    b.wait('document.querySelector("#connection").textContent==="已连接"')
    scenario['bootstrap_delay'] = 0
    b.call('WebDriver:Refresh')
    phase(b, 'running')
    b.wait('!document.querySelector("[data-op=stop]").disabled')
    b.click('#viewConfig')
    if scenario['reduced_motion']:
        assert b.js('return matchMedia("(prefers-reduced-motion: reduce)").matches && document.getAnimations().length===0')
    assert b.js('return [...document.querySelectorAll("#setup input,#setup select,#setup textarea,#setup button")].every(el=>el.disabled)')
    b.key('\ue00c')
    b.click('[data-op=stop]')
    phase(b, 'finished')
    b.click('#nextSession')
    phase(b, 'configure')
    assert b.js('return document.querySelector("[name=policy_path]").value.endsWith("demo-checkpoint") && !document.querySelector("[name=policy_path]").disabled')
