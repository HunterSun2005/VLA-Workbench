"""Bounded subprocess I/O for background media work; no ROS or Torch imports."""
import os
import selectors
import subprocess
import time


class MediaProcessError(RuntimeError):
    pass


def run_media(command, *, timeout_s=30):
    try:
        return subprocess.run(command, check=True, capture_output=True, text=True, timeout=timeout_s)
    except subprocess.TimeoutExpired as error:
        raise MediaProcessError(f'{command[0]} timed out after {timeout_s}s: {command[-1]}') from error
    except subprocess.CalledProcessError as error:
        raise MediaProcessError(f'{command[0]} failed: {error.stderr[-65536:]}') from error


class StreamingEncoder:
    """Drain stderr while writing nonblocking stdin; each operation has a deadline."""
    def __init__(self, command, *, timeout_s=30):
        if timeout_s <= 0:
            raise ValueError('media timeout must be positive')
        self.timeout_s = timeout_s
        self.command = command
        self.process = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL,
                                        stderr=subprocess.PIPE, bufsize=0)
        self.selector = selectors.DefaultSelector()
        self.stderr = bytearray()
        self.closed = False
        self.failure = None
        os.set_blocking(self.process.stdin.fileno(), False)
        os.set_blocking(self.process.stderr.fileno(), False)
        self.selector.register(self.process.stderr, selectors.EVENT_READ)

    def _remaining(self, deadline):
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise MediaProcessError(f'{self.command[0]} timed out after {self.timeout_s}s')
        return remaining

    def _drain(self, key):
        try:
            chunk = os.read(key.fd, 65536)
        except BlockingIOError:
            return
        if chunk:
            self.stderr.extend(chunk)
            del self.stderr[:-65536]
        else:
            self.selector.unregister(key.fileobj)

    def _failed(self, error):
        self.failure = MediaProcessError(
            f'{error}; {self.stderr.decode(errors="replace").strip()}')
        try:
            self.abort()
        except (OSError, subprocess.TimeoutExpired):
            pass  # Preserve the initiating failure; abort has already closed pipes.
        raise self.failure from error

    def write(self, data):
        if self.closed:
            raise self.failure or MediaProcessError('encoder is closed')
        deadline = time.monotonic() + self.timeout_s
        pending = memoryview(data)
        stream = self.process.stdin
        self.selector.register(stream, selectors.EVENT_WRITE)
        try:
            while pending:
                for key, _ in self.selector.select(min(.1, self._remaining(deadline))):
                    if key.fileobj is stream:
                        try:
                            written = os.write(key.fd, pending[:65536])
                        except BlockingIOError:
                            continue
                        pending = pending[written:]
                    else:
                        self._drain(key)
                if self.process.poll() is not None:
                    raise MediaProcessError(f'encoder exited during frame write ({self.process.returncode})')
        except (OSError, MediaProcessError) as error:
            self._failed(error)
        finally:
            if not self.closed:
                self.selector.unregister(stream)

    def close(self):
        if self.closed:
            if self.failure:
                raise self.failure
            return
        deadline = time.monotonic() + self.timeout_s
        try:
            self.process.stdin.close()  # Unbuffered: no implicit blocking flush.
            while self.selector.get_map() or self.process.poll() is None:
                for key, _ in self.selector.select(min(.05, self._remaining(deadline))):
                    self._drain(key)
            returncode = self.process.wait(timeout=self._remaining(deadline))
            if returncode:
                raise MediaProcessError(f'encoder exited with status {returncode}')
        except (OSError, MediaProcessError, subprocess.TimeoutExpired) as error:
            self._failed(error)
        finally:
            self._close_pipes()

    def _close_pipes(self):
        self.closed = True
        self.selector.close()
        for stream in (self.process.stdin, self.process.stderr):
            if stream is not None:
                stream.close()

    def abort(self):
        try:
            if self.process.poll() is None:
                self.process.kill()
            self.process.wait(timeout=5)
        finally:
            self._close_pipes()
