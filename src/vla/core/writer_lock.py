"""Process-lifetime writer leases; SQLite transactions alone are not sufficient."""

from __future__ import annotations

import contextlib
import fcntl
import functools
import json
import os
from pathlib import Path
import socket


class WriterLease:
    def __init__(self, *roots):
        self.roots = {Path(root).expanduser().resolve() for root in roots}
        self._streams = []
        try:
            for root in sorted(self.roots):
                root.parent.mkdir(parents=True, exist_ok=True)
                path = root.with_name(f".{root.name}.rlt-writer.lock")
                stream = path.open("a+", encoding="utf-8")
                try:
                    fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                except BlockingIOError:
                    stream.seek(0)
                    owner = stream.read()
                    stream.close()
                    raise RuntimeError(f"RLT writer already owns {root}: {owner}") from None
                except BaseException:
                    stream.close()
                    raise
                self._streams.append(stream)
                stream.seek(0)
                stream.truncate()
                json.dump({"pid": os.getpid(), "host": socket.gethostname(), "root": str(root)}, stream)
                stream.flush()
        except BaseException:
            self.close()
            raise

    def require(self, root):
        if Path(root).expanduser().resolve() not in self.roots or not self._streams:
            raise ValueError("writer lease does not cover this root")

    def close(self):
        while self._streams:
            stream = self._streams.pop()
            stream.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()


def cleanup_on_init_error(initialize):
    @functools.wraps(initialize)
    def wrapped(self, *args, **kwargs):
        try:
            initialize(self, *args, **kwargs)
        except BaseException:
            with contextlib.suppress(Exception):
                self.close()
            raise
    return wrapped
