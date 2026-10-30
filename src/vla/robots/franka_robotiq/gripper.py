from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from typing import Any, Callable

import numpy as np


@dataclass(frozen=True)
class RobotiqStatus:
    activation_requested: bool
    activation_status: int
    activated: bool
    moving: bool
    fault: int
    position_request: int
    position: int
    current: int

    @property
    def open_fraction(self) -> float:
        return float(np.clip(1.0 - self.position / 255.0, 0.0, 1.0))


def action_to_request(action: float) -> int:
    if not np.isfinite(action):
        raise ValueError("gripper action must be finite")
    return int(round((1.0 - float(np.clip(action, 0.0, 1.0))) * 255.0))


def command_registers(*, activate: bool, position: int, speed: int, force: int) -> list[int]:
    values = bytes((0x09 if activate else 0x00, 0, 0, position, speed, force))
    return [int.from_bytes(values[index : index + 2], "big") for index in range(0, 6, 2)]


def parse_status_registers(registers: list[int]) -> RobotiqStatus:
    if len(registers) < 3:
        raise ValueError(f"expected three Robotiq status registers, got {len(registers)}")
    raw = b"".join(int(value).to_bytes(2, "big") for value in registers[:3])
    status = raw[0]
    return RobotiqStatus(
        activation_requested=bool(status & 0x01),
        activation_status=(status >> 4) & 0x03,
        activated=bool(status & 0x01) and ((status >> 4) & 0x03) == 0x03,
        moving=((status >> 6) & 0x03) == 0,
        fault=raw[2],
        position_request=raw[3],
        position=raw[4],
        current=raw[5],
    )


class RobotiqModbus:
    """Latest-target-wins Robotiq 2F Modbus RTU worker."""

    COMMAND_ADDRESS = 0x03E8
    STATUS_ADDRESS = 0x07D0
    # These faults are recoverable as part of a fresh reset/activation handshake:
    # 0x05 activation in progress, 0x07 activation not set, 0x09 communication timeout.
    TRANSIENT_ACTIVATION_FAULTS = frozenset((0x05, 0x07, 0x09))

    def __init__(
        self,
        *,
        port: str,
        baudrate: int = 115200,
        bytesize: int = 8,
        parity: str = "N",
        stopbits: int = 1,
        device_id: int = 9,
        timeout_s: float = 0.2,
        state_timeout_s: float = 0.5,
        command_timeout_s: float | None = None,
        deadband: float = 0.01,
        max_delta_per_s: float = 2.0,
        speed: int = 128,
        force: int = 128,
        poll_hz: float = 50.0,
        client_factory: Callable[..., Any] | None = None,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._serial_kwargs = dict(
            port=port, baudrate=baudrate, bytesize=bytesize, parity=parity,
            stopbits=stopbits, timeout=timeout_s,
        )
        self.device_id = int(device_id)
        self.state_timeout_s = float(state_timeout_s)
        if command_timeout_s is not None and (not np.isfinite(command_timeout_s) or command_timeout_s <= 0):
            raise ValueError("command_timeout_s must be finite and positive")
        self.command_timeout_s = command_timeout_s
        self._target_deadline: float | None = None
        self._hold_requested = False
        self.deadband = float(deadband)
        self.max_delta_per_s = float(max_delta_per_s)
        self.speed = int(np.clip(speed, 0, 255))
        self.force = int(np.clip(force, 0, 255))
        self.poll_period = 1.0 / float(poll_hz)
        self._client_factory = client_factory
        self._clock = clock
        self._client = None
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        self._lock = threading.Lock()
        self._target = 1.0
        self._rate_scale = 1.0
        self._command = 1.0
        self._command_time = self._clock()
        self._status: RobotiqStatus | None = None
        self._status_time = 0.0
        self._error: BaseException | None = None

    def _make_client(self):
        if self._client_factory is not None:
            return self._client_factory(**self._serial_kwargs)
        from pymodbus.client import ModbusSerialClient

        return ModbusSerialClient(**self._serial_kwargs)

    def _call(self, method: str, *args, **kwargs):
        function = getattr(self._client, method)
        try:
            return function(*args, device_id=self.device_id, **kwargs)
        except TypeError:
            return function(*args, slave=self.device_id, **kwargs)

    @staticmethod
    def _check_response(response, operation: str) -> None:
        if response is None or (hasattr(response, "isError") and response.isError()):
            raise OSError(f"Robotiq Modbus {operation} failed: {response}")

    def _write(self, action: float, *, activate: bool = True) -> None:
        registers = command_registers(
            activate=activate,
            position=action_to_request(action),
            speed=self.speed,
            force=self.force,
        )
        response = self._call("write_registers", self.COMMAND_ADDRESS, registers)
        self._check_response(response, "write")

    def _read(self) -> RobotiqStatus:
        response = self._call("read_holding_registers", self.STATUS_ADDRESS, count=3)
        self._check_response(response, "read")
        return parse_status_registers(list(response.registers))

    def _record_status(self, status: RobotiqStatus) -> None:
        with self._lock:
            self._status, self._status_time = status, self._clock()

    def _check_activation_fault(self, status: RobotiqStatus) -> None:
        if status.fault and status.fault not in self.TRANSIENT_ACTIVATION_FAULTS:
            raise RuntimeError(f"Robotiq activation fault 0x{status.fault:02x}")

    def _activation_timed_out(self, deadline: float, phase: str) -> None:
        if self._clock() >= deadline:
            raise TimeoutError(f"timed out waiting for Robotiq {phase}")

    def connect(self, activation_timeout_s: float = 8.0) -> None:
        if self._thread is not None:
            return
        self._client = self._make_client()
        try:
            if self._client.connect() is False:
                raise ConnectionError(f"could not open Robotiq serial port {self._serial_kwargs['port']}")
            status = self._read()
            self._record_status(status)
            self._check_activation_fault(status)
            self._command = self._target = status.open_fraction
            self._hold_requested = False
            self._command_time = self._clock()
            deadline = self._clock() + activation_timeout_s

            if not status.activated or status.fault in self.TRANSIENT_ACTIVATION_FAULTS:
                # A reset must be acknowledged before rACT is asserted again.
                # Recoverable startup faults may persist while this handshake settles.
                self._write(self._command, activate=False)
                while True:
                    status = self._read()
                    self._record_status(status)
                    self._check_activation_fault(status)
                    if not status.activation_requested and status.activation_status == 0:
                        break
                    self._activation_timed_out(deadline, "reset acknowledgement")
                    time.sleep(0.05)

                # Hold the latest measured position so connecting for a disarmed
                # preflight does not request avoidable gripper motion.
                self._command = self._target = status.open_fraction
                self._command_time = self._clock()
                self._write(self._command, activate=True)
                while True:
                    status = self._read()
                    self._record_status(status)
                    self._check_activation_fault(status)
                    if status.activated and not status.fault:
                        break
                    self._activation_timed_out(deadline, "activation")
                    time.sleep(0.05)
        except BaseException:
            self._client.close()
            self._client = None
            raise
        self._stop.clear()
        self._thread = threading.Thread(target=self._worker, name="robotiq-modbus", daemon=True)
        self._thread.start()

    def set_target(self, action: float, *, rate_scale: float = 1.0) -> None:
        value = float(np.clip(action, 0.0, 1.0))
        scale = float(rate_scale)
        if not np.isfinite(value) or not np.isfinite(scale):
            raise ValueError("gripper action and rate scale must be finite")
        if not 0.0 <= scale <= 1.0:
            raise ValueError("gripper rate scale must be in [0, 1]")
        with self._lock:
            self._target = value
            self._hold_requested = False
            self._rate_scale = scale
            self._target_deadline = (
                None if self.command_timeout_s is None else self._clock() + self.command_timeout_s
            )

    def hold_position(self) -> None:
        """Queue a measured-position hold, without ramping through old targets.

        The worker samples its latest state and performs serial IO; the caller
        never bypasses the normal worker or requests an arbitrary fast movement.
        """
        self.check_health()
        with self._lock:
            self._hold_requested = True
            self._target_deadline = None

    def _worker(self) -> None:
        try:
            while not self._stop.is_set():
                started = self._clock()
                with self._lock:
                    expired = self._target_deadline is not None and started >= self._target_deadline
                    hold = expired or self._hold_requested
                    if hold:
                        if self._status is None or started - self._status_time > self.state_timeout_s:
                            raise TimeoutError("cannot hold gripper command: state is stale")
                        self._target = self._status.open_fraction
                        self._target_deadline = None
                        self._hold_requested = False
                    target = self._target
                    rate_scale = self._rate_scale
                    previous = self._command
                    previous_time = self._command_time
                max_delta = (
                    self.max_delta_per_s
                    * rate_scale
                    * max(started - previous_time, self.poll_period)
                )
                command = float(np.clip(target, previous - max_delta, previous + max_delta))
                if hold:
                    command = target
                if hold or abs(command - previous) >= self.deadband:
                    self._write(command)
                    with self._lock:
                        self._command, self._command_time = command, started
                status = self._read()
                if status.fault:
                    raise RuntimeError(f"Robotiq fault 0x{status.fault:02x}")
                with self._lock:
                    self._status, self._status_time = status, self._clock()
                self._stop.wait(max(0.0, self.poll_period - (self._clock() - started)))
        except BaseException as error:
            with self._lock:
                self._error = error
            self._stop.set()

    def position(self) -> float:
        with self._lock:
            error, status, age = self._error, self._status, self._clock() - self._status_time
        if error is not None:
            raise RuntimeError(f"Robotiq worker failed: {error}") from error
        if status is None or age > self.state_timeout_s:
            raise TimeoutError(f"Robotiq state is stale ({age:.3f}s)")
        return status.open_fraction

    def check_health(self) -> None:
        self.position()

    def recover(self) -> None:
        """Operator-requested reconnect; reset the failed worker before activation."""
        self.close()
        with self._lock:
            self._error = None
            self._status = None
            self._target_deadline = None
            self._hold_requested = False
        self.connect()

    def close(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            if self._thread.is_alive():
                raise RuntimeError("Robotiq worker did not stop; refusing to release the serial client")
            self._thread = None
        if self._client is not None:
            self._client.close()
            self._client = None
