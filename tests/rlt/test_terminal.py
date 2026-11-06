"""Terminal presentation and input boundaries, without ROS or robot devices."""
import io
import threading
import time

import pytest

from vla import terminal
from vla.terminal import TerminalConfirmation, TerminalReporter


def wait_until(predicate, timeout=2):
    deadline = time.monotonic() + timeout
    while not predicate():
        if time.monotonic() >= deadline:
            pytest.fail('background reporter did not finish')
        time.sleep(.01)


@pytest.mark.parametrize('tty', [False, True])
def test_state_throttling_and_events_have_separate_output(tmp_path, tty, monkeypatch):
    monkeypatch.setenv('TERM', 'xterm')
    stream = io.StringIO()
    stream.isatty = lambda: tty
    clock = [10.0]
    reporter = TerminalReporter(tmp_path, stream=stream, clock=lambda: clock[0])
    try:
        reporter.status('RECORDING frames=10')
        wait_until(lambda: 'frames=10' in reporter.path.read_text())
        for i in range(100):
            reporter.status(f'RECORDING frames={i + 11}')
        reporter.event('Saved episode=3')
        reporter.event('encoder diagnostics', level='debug')
        reporter.event('fault; commands disabled', level='error', details='Traceback file details')
        wait_until(lambda: 'encoder diagnostics' in reporter.path.read_text())
        assert 'Saved episode=3' in stream.getvalue()
        assert 'fault; commands disabled' in stream.getvalue()
        assert 'encoder diagnostics' not in stream.getvalue()
        assert stream.getvalue().count('RECORDING') == int(tty)
        clock[0] += 1.1
        # A pending old recording snapshot must not return after save/fault.
        reporter.event('status window advanced', level='debug')
        wait_until(lambda: 'status window advanced' in reporter.path.read_text())
        assert 'frames=110' not in reporter.path.read_text()
        reporter.status('STANDBY frames=0')
        wait_until(lambda: 'STANDBY' in reporter.path.read_text())
        assert stream.getvalue().count('RECORDING') == int(tty)
        assert stream.getvalue().count('STANDBY') == int(tty)
        assert ('\x1b' in stream.getvalue()) == tty
        assert 'Traceback file details' not in stream.getvalue()
    finally:
        reporter.close()
    assert 'Traceback file details' in reporter.path.read_text()


def test_slow_console_has_bounded_queue_and_does_not_block_control(tmp_path):
    entered, release = threading.Event(), threading.Event()
    class SlowStream:
        def isatty(self): return False
        def write(self, text):
            entered.set(); release.wait(3)
        def flush(self): pass
    reporter = TerminalReporter(tmp_path, stream=SlowStream(), capacity=4)
    try:
        assert entered.wait(1)
        started = time.monotonic()
        for i in range(1000):
            reporter.event(f'debug {i}', level='debug')
            reporter.status(str(i))
        reporter.event('important fault', level='error')
        assert time.monotonic() - started < .5
        assert len(reporter._events) <= 4
        assert len(reporter._critical) == 1
    finally:
        release.set(); reporter.close()
    assert 'important fault' in reporter.path.read_text()
    assert '日志输出拥堵' in reporter.path.read_text()


def test_confirmation_never_blocks_or_executes_stale_operations():
    messages, now, identity = [], [1.0], [5]
    confirmation = TerminalConfirmation(messages.append, clock=lambda: now[0])
    assert confirmation.route('discard', destructive=True, valid=lambda: identity[0] == 5) is None
    assert confirmation.key_event(b'y') == 'confirm'
    assert confirmation.key_event(b'n') == 'cancel'
    assert confirmation.key_event(b'\x1b') == 'cancel'
    assert confirmation.route('confirm') == 'discard'
    assert not confirmation.pending
    assert confirmation.route('save') == 'save'
    assert confirmation.route('discard', destructive=True) is None
    assert confirmation.route('pause') == 'pause'  # Immediate safety action cancels prompt.
    assert not confirmation.pending
    assert confirmation.route('quit', destructive=True, valid=lambda: identity[0] == 5) is None
    identity[0] = 6
    assert confirmation.route('confirm') is None
    assert confirmation.route('discard', destructive=True) is None
    now[0] += 11
    confirmation.expire()
    assert not confirmation.pending
    assert confirmation.route('confirm') is None
    assert confirmation.route('discard', destructive=True) is None
    assert confirmation.route('cancel') is None
    assert not confirmation.pending


def test_cli_context_logs_survive_and_managed_imports_do_not_activate(tmp_path, monkeypatch):
    monkeypatch.setenv('VLA_ROOT', str(tmp_path))
    monkeypatch.delenv('VLA_TERMINAL_LOG_DIR', raising=False)
    monkeypatch.setenv('VLA_RUN_MODE', 'managed')
    @terminal.terminal_session('test')
    def call():
        return terminal.current_reporter()
    assert call() is None
    assert not (tmp_path / 'runtime').exists()
    monkeypatch.setenv('VLA_RUN_MODE', 'terminal')
    @terminal.terminal_session('test')
    def recorded():
        terminal.console_print('Saved episode=8')
        path = terminal.diagnostic_path('bridge.log')
        path.write_text('persistent sidecar diagnostic')
        return path
    path = recorded()
    assert path.read_text() == 'persistent sidecar diagnostic'
    assert 'Saved episode=8' in (path.parent / 'terminal.log').read_text()
    assert terminal.current_reporter() is None


@pytest.mark.parametrize('events,expected', [
    (['discard','cancel','discard','confirm'], ['discard']),
    (['quit','cancel'], []),
    (['discard','pause'], ['pause']),
    (['quit','confirm'], ['discard','stop']),
    (['force_quit'], ['discard','stop']),
])
def test_ros_terminal_confirmations_continue_polling(events,expected):
    from types import SimpleNamespace
    from vla.cli.ros_session import terminal_command
    accepted=[]
    backend=SimpleNamespace(sample=lambda:{'recording':True,'episode':2}, command=lambda op,args:accepted.append(op))
    confirmation=TerminalConfirmation(lambda *_:None)
    for event in events:
        terminal_command(backend,confirmation,event)
    assert accepted==expected


def test_raw_keyboard_confirmation_and_eof_do_not_replay_keys(monkeypatch):
    from collections import deque
    from vla.datacollection.franka_robotiq_rollout.common import KeyboardEvents
    import threading
    confirmation = TerminalConfirmation(lambda *_: None)
    confirmation.route('left', destructive=True)
    keys = iter([b'y', b''])
    keyboard = object.__new__(KeyboardEvents)
    keyboard.confirmation = confirmation
    keyboard._fd, keyboard._sft = 0, False
    keyboard._stop, keyboard._lock = threading.Event(), threading.Lock()
    keyboard._events = deque()
    monkeypatch.setattr('vla.datacollection.franka_robotiq_rollout.common.select.select', lambda *_: ([0], [], []))
    monkeypatch.setattr('vla.datacollection.franka_robotiq_rollout.common.os.read', lambda *_: next(keys))
    keyboard._worker()
    assert keyboard.consume() == ['confirm', 'force_quit']


@pytest.mark.parametrize('events,expected', [
    (['enable', 'force_quit'], ['force_quit']),
    (['left', 'confirm', 'pause'], ['pause']),
    (['pause', 'force_quit'], ['force_quit']),
    (['save', 'intervention'], ['save', 'intervention']),
])
def test_drained_keyboard_batch_keeps_stop_priority(events, expected):
    assert TerminalConfirmation.prioritize(events) == expected



def test_key_event_invalidates_status_already_copied_by_worker(tmp_path, monkeypatch):
    stream = io.StringIO(); stream.isatty = lambda: True
    monkeypatch.setenv('TERM', 'xterm')
    reporter = TerminalReporter(tmp_path, stream=stream)
    wait_until(lambda: '运行日志' in stream.getvalue())
    original = reporter._record
    def record(item):
        original(item)
        if item[2] == 'trigger transition':
            reporter.event('Saved episode=9')
    reporter._record = record
    try:
        # Place both items in one batch: _worker copies the status before _record
        # publishes the key event, reproducing the unlocked rendering race.
        with reporter._lock:
            reporter._status = 'RECORDING stale'
            reporter._events.append((time.time(), 'debug', 'trigger transition', None))
        wait_until(lambda: 'Saved episode=9' in stream.getvalue())
        assert 'RECORDING stale' not in stream.getvalue()
    finally:
        reporter.close()
