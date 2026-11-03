"""Small CLI-only output and confirmation adapters; compatible with ROS Python 3.10.

Hardware and control code may publish messages, but never waits for stdout or files.
Web runners do not activate this adapter. VLA_TERMINAL_LOG_DIR may be supplied by
an owning shell launcher to keep child diagnostics in one persistent directory.
"""
from __future__ import annotations

import builtins
from collections import deque
from functools import wraps
import logging
import os
from pathlib import Path
import re
import shutil
import sys
import threading
import time
import traceback
import uuid

_current = None


class TerminalReporter:
    def __init__(self, directory, *, stream=None, capacity=256, clock=time.monotonic):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.path = self.directory / 'terminal.log'
        self.stream = stream if stream is not None else sys.stdout
        self.tty = bool(getattr(self.stream, 'isatty', lambda: False)()) and os.environ.get('TERM') != 'dumb'
        self.clock = clock
        self._events = deque(maxlen=capacity)
        self._critical = deque(maxlen=32)
        self._status = None
        self._event_version = 0
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._dropped = 0
        self._last_status = float('-inf')
        self._last_console = {}
        self._line = False
        # Fail before hardware startup if the diagnostic location is unwritable.
        self._file = self.path.open('a', encoding='utf-8')
        self._thread = threading.Thread(target=self._worker, name='vla-terminal', daemon=True)
        self._thread.start()
        self.event(f'运行日志：{self.path}')

    def event(self, message, *, level='info', details=None):
        if self._stop.is_set():
            return
        item = (time.time(), level, str(message), details)
        with self._lock:
            if level != 'debug':
                self._event_version += 1
                self._status = None  # Never repaint an old RECORDING after save/pause/fault.
            target = self._critical if level == 'error' else self._events
            if len(target) == target.maxlen:
                self._dropped += 1
            target.append(item)

    def status(self, message):
        with self._lock:
            self._status = str(message)

    def _write_console(self, text, *, status=False):
        try:
            # Never replay arbitrary control sequences from exceptions or devices.
            text = re.sub(r'\x1b(?:\[[0-?]*[ -/]*[@-~]|.)', '', text).replace('\r', '')
            if status:
                width = max(20, shutil.get_terminal_size((100, 20)).columns - 1)
                self.stream.write('\r\x1b[2K' + text.replace('\n', ' ')[:width])
                self._line = True
            else:
                if self._line:
                    self.stream.write('\r\x1b[2K')
                    self._line = False
                self.stream.write(text + '\n')
            self.stream.flush()
        except Exception:
            pass

    def _record(self, item):
        stamp, level, message, details = item
        prefix = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(stamp))
        try:
            self._file.write(f'{prefix} [{level.upper()}] {message}\n')
            if details:
                self._file.write(str(details) + '\n')
            self._file.flush()
        except OSError:
            if not getattr(self, '_file_failed', False):
                self._file_failed = True
                self._write_console(f'日志文件写入失败：{self.path}；控制流程继续。')
        if level == 'debug':
            return
        now = self.clock()
        key = (level, message)
        last = self._last_console.get(key, float('-inf'))
        if now - last < 5 and not self._line:
            return
        if len(self._last_console) >= 256:
            self._last_console.clear()
        self._last_console[key] = now
        if 'Traceback (most recent call last)' in message:
            message = message.split('Traceback (most recent call last)')[0].strip() + f'（完整堆栈：{self.path}）'
        self._write_console(f'[{"错误" if level == "error" else "提示"}] {message}')

    def _worker(self):
        try:
            while True:
                with self._lock:
                    items = list(self._critical) + list(self._events)
                    self._critical.clear(); self._events.clear()
                    dropped, self._dropped = self._dropped, 0
                    status = None
                    status_version = self._event_version
                    if self._status is not None and self.clock() - self._last_status >= 1:
                        status, self._status = self._status, None
                        self._last_status = self.clock()
                if dropped:
                    self._record((time.time(), 'warning', f'日志输出拥堵，丢弃 {dropped} 条待输出消息；控制流程继续。', None))
                for item in items:
                    self._record(item)
                # Producers may publish a save/fault while this batch is being
                # written. Never show the status copied before that transition.
                with self._lock:
                    status_current = status_version == self._event_version
                if status is not None and status_current:
                    self._record((time.time(), 'debug', status, None))
                    if self.tty:
                        self._write_console(status, status=True)
                if self._stop.is_set():
                    with self._lock:
                        if not self._critical and not self._events:
                            break
                self._stop.wait(.05)
        finally:
            if self._line:
                self._write_console('')
            self._file.close()

    def close(self):
        self._stop.set()
        self._thread.join(timeout=.5)


class _LogHandler(logging.Handler):
    def __init__(self, reporter):
        super().__init__(); self.reporter = reporter

    def emit(self, record):
        try:
            details = ''.join(traceback.format_exception(*record.exc_info)) if record.exc_info else None
            self.reporter.event(record.getMessage(), level=record.levelname.lower(), details=details)
        except Exception:
            pass


def current_reporter():
    return _current


def diagnostic_path(name, fallback=None):
    if _current is not None:
        return _current.directory / name
    return Path(fallback if fallback is not None else name)


def console_print(*values, sep=' ', end='\n', file=None, flush=False):
    if _current is None or (file is not None and file not in (sys.stdout, sys.stderr)):
        return builtins.print(*values, sep=sep, end=end, file=file, flush=flush)
    _current.event(sep.join(str(v) for v in values), level='error' if file is sys.stderr else 'info')


def console_status(message):
    if _current is not None:
        _current.status(message)
    # Web has a structured state channel; no periodic terminal status on import.


def console_debug(*values, **_kwargs):
    if _current is not None:
        _current.event(' '.join(str(v) for v in values), level='debug')


def terminal_session(label):
    """Decorate CLI main only; managed Web imports and business calls stay independent."""
    def decorate(function):
        @wraps(function)
        def run(*args, **kwargs):
            global _current
            if _current is not None or os.environ.get('VLA_RUN_MODE') == 'managed':
                return function(*args, **kwargs)
            directory = os.environ.get('VLA_TERMINAL_LOG_DIR')
            if not directory:
                root = Path(os.environ.get('VLA_ROOT', Path.cwd()))
                directory = root / 'runtime/terminal' / (time.strftime('%Y%m%d-%H%M%S-') + label + '-' + uuid.uuid4().hex[:8])
            reporter = TerminalReporter(directory)
            _current = reporter
            logger = logging.getLogger()
            old_handlers, old_level = logger.handlers[:], logger.level
            logger.handlers = [_LogHandler(reporter)]
            logger.setLevel(logging.INFO)
            try:
                return function(*args, **kwargs)
            except KeyboardInterrupt:
                reporter.event('收到 Ctrl-C，已请求立即停止并清理；未等待数据确认。')
                raise SystemExit(130) from None
            except Exception as error:
                reporter.event(f'{type(error).__name__}: {error}；详细堆栈：{reporter.path}',
                               level='error', details=traceback.format_exc())
                raise SystemExit(1) from None
            finally:
                logger.handlers, logger.level = old_handlers, old_level
                _current = None
                reporter.close()
        return run
    return decorate


class TerminalConfirmation:
    """Nonblocking y/n decision in the CLI event adapter, never in control methods."""
    def __init__(self, report=console_print, *, clock=time.monotonic, timeout=10):
        self.report, self.clock, self.timeout = report, clock, timeout
        self._pending = None

    @staticmethod
    def prioritize(events):
        # Keyboard consumers drain a batch. A one-command-per-tick frontend must
        # not lose EOF/stop behind an enable, save, or confirmation in that batch.
        if 'force_quit' in events:
            return ['force_quit']
        if 'pause' in events:
            return ['pause']
        return events

    @property
    def pending(self):
        return self._pending is not None

    def key_event(self, char):
        if not self.pending:
            return None
        if char.lower() == b'y':
            return 'confirm'
        if char.lower() in (b'n', b'\x1b'):
            return 'cancel'
        return None

    def expire(self):
        pending = self._pending
        if pending is not None and (self.clock() > pending[2] or not pending[3]()):
            self._pending = None
            self.report(f'已取消确认：回合状态已改变或超过 {self.timeout:g} 秒。')

    def route(self, event, *, destructive=False, valid=lambda: True, label='丢弃当前未保存回合'):
        self.expire()
        if self._pending is not None:
            requested, _, _, guard = self._pending
            self._pending = None
            if event == 'confirm':
                if guard():
                    return requested
                self.report('回合状态已改变，未执行丢弃。')
                return None
            self.report('已取消丢弃确认。')
            if event == 'cancel':
                return None
        if event in ('confirm', 'cancel'):
            return None
        if destructive:
            self._pending = (event, label, self.clock() + self.timeout, valid)
            self.report(f'{label}？y=确认，n/Esc=取消（{self.timeout:g} 秒内）；监控继续，Ctrl-C 仍按原退出流程立即处理。')
            return None
        return event
