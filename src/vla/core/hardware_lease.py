"""Host-local advisory leases shared by ROS hardware clients."""

from __future__ import annotations

import fcntl
import hashlib
import json
import os
from pathlib import Path
from vla.core.lease_metadata import release_owned_streams


class HardwareLease:
    def __init__(self, robot_ip: str | None, gripper_port: str | None = None, *, directory: Path | None = None):
        directory = directory or Path("/tmp/vla-hardware-locks")
        directory.mkdir(mode=0o755, parents=True, exist_ok=True)
        self._streams = []
        resources = [f"franka:{robot_ip}"] if robot_ip else []
        if gripper_port:
            resources.append(f"serial:{Path(gripper_port).resolve()}")
        if not resources:
            raise ValueError('HardwareLease requires an arm or a peripheral')
        resources.sort()
        try:
            for resource in resources:
                name = hashlib.sha256(resource.encode("utf-8")).hexdigest() + ".lock"
                fd = os.open(directory / name, os.O_RDWR | os.O_CREAT | os.O_NOFOLLOW, 0o600)
                stream = os.fdopen(fd, "r+")
                try:
                    fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
                except BlockingIOError:
                    owner = stream.read()
                    stream.close()
                    raise RuntimeError(f"Hardware already owned: {resource}; {owner}") from None
                except BaseException:
                    stream.close()
                    raise
                self._streams.append(stream)
                stream.seek(0)
                stream.truncate()
                json.dump({"pid": os.getpid(), "resource": resource}, stream)
                stream.flush()
        except BaseException:
            self.close()
            raise

    def close(self):
        # Never unlink flock files: another process may already be waiting on the inode.
        release_owned_streams(self._streams)

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()
