"""
Actuator interface and the dry-run backend.

No physical actuator protocol exists or is documented for this project, so
only DryRunActuator is implemented: it applies the same rules a real
actuator controller must (saturation, emergency stop latch, watchdog
release) and logs every command, but moves nothing. Physical actuation is an
unresolved hardware dependency; see docs/SAFETY.md for what the physical
backend and its independent hardware safety must provide.
"""

import csv
import math
import threading
from abc import ABC, abstractmethod
from collections import deque
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, TextIO

from .config import ActuatorConfig, SafetyConfig

COMMAND_FIELDS = (
    "sequence",
    "timestamp",
    "requested_stiffness",
    "applied_stiffness",
    "state",
    "armed",
    "estop",
    "watchdog_tripped",
    "faults",
    "note",
)


@dataclass(frozen=True)
class ActuatorCommand:
    """One command from the controller."""

    sequence: int
    timestamp: float
    requested_stiffness: float
    state: str
    armed: bool
    faults: tuple[str, ...] = ()
    diagnostics: Mapping[str, float] = field(default_factory=dict)


@dataclass(frozen=True)
class ActuatorRecord:
    """What the actuator did with a command (or an event such as a watchdog trip)."""

    sequence: int
    timestamp: float
    requested_stiffness: float
    applied_stiffness: float
    state: str
    armed: bool
    estop: bool
    watchdog_tripped: bool
    faults: tuple[str, ...]
    note: str

    def as_row(self) -> list[str]:
        """CSV row matching COMMAND_FIELDS."""
        return [
            str(self.sequence),
            f"{self.timestamp:.6f}",
            f"{self.requested_stiffness:.6f}",
            f"{self.applied_stiffness:.6f}",
            self.state,
            str(int(self.armed)),
            str(int(self.estop)),
            str(int(self.watchdog_tripped)),
            "|".join(self.faults),
            self.note,
        ]


@dataclass(frozen=True)
class ActuatorStatus:
    """Snapshot for the monitor."""

    backend: str
    applied_stiffness: float
    estop: bool
    estop_reason: str
    watchdog_tripped: bool
    commands_received: int
    invalid_commands: int
    last_command_age_s: float | None


class Actuator(ABC):
    """Interface every actuator backend implements."""

    backend = "abstract"

    @abstractmethod
    def send(self, command: ActuatorCommand) -> float:
        """Apply a command; return the stiffness actually applied."""

    @abstractmethod
    def emergency_stop(self, reason: str) -> None:
        """Release immediately and ignore commands until reset()."""

    @abstractmethod
    def reset(self) -> None:
        """Clear the emergency stop and watchdog latches (operator re-arm)."""

    @abstractmethod
    def check_watchdog(self, now: float) -> bool:
        """Release if no command arrived within the timeout. Returns True if tripped."""

    @abstractmethod
    def status(self, now: float) -> ActuatorStatus:
        """Current status."""

    @abstractmethod
    def close(self) -> None:
        """Release and stop accepting commands."""


class DryRunActuator(Actuator):  # pylint: disable=too-many-instance-attributes
    """
    Logs commands without moving hardware. Thread-safe: the control loop calls
    send(), a watchdog thread calls check_watchdog(), the operator calls
    emergency_stop() / reset().

    Rules applied to every command:
      - non-finite or out-of-range requests are saturated to [0, 1]
        (NaN -> 0) and counted as invalid
      - nothing is applied while the e-stop or watchdog is latched, or when
        the command is not armed
      - the watchdog trips when no command arrives within the timeout, and
        stays tripped until reset()
    """

    backend = "dry_run"

    def __init__(
        self, watchdog_timeout_s: float, log_size: int = 100_000, log_path: Path | None = None
    ) -> None:
        self.watchdog_timeout_s = watchdog_timeout_s
        self.records: deque[ActuatorRecord] = deque(maxlen=log_size)
        self.log_path = log_path
        self._lock = threading.Lock()
        self._applied = 0.0
        self._estop = False
        self._estop_reason = ""
        self._watchdog_tripped = False
        self._last_command_time: float | None = None
        self._commands = 0
        self._invalid = 0
        self._closed = False
        self._log_file: TextIO | None = None
        self._log_writer: Any = None
        if log_path is not None:
            log_path.parent.mkdir(parents=True, exist_ok=True)
            # Kept open for the actuator's lifetime and closed in close()
            self._log_file = open(  # pylint: disable=consider-using-with
                log_path, "w", newline="", encoding="utf-8"
            )
            self._log_writer = csv.writer(self._log_file)
            self._log_writer.writerow(COMMAND_FIELDS)

    def _record(self, record: ActuatorRecord) -> None:
        self.records.append(record)
        if self._log_writer is not None:
            self._log_writer.writerow(record.as_row())

    def _event(self, timestamp: float, note: str) -> None:
        self._record(
            ActuatorRecord(
                sequence=-1,
                timestamp=timestamp,
                requested_stiffness=0.0,
                applied_stiffness=self._applied,
                state="",
                armed=False,
                estop=self._estop,
                watchdog_tripped=self._watchdog_tripped,
                faults=(),
                note=note,
            )
        )

    def send(self, command: ActuatorCommand) -> float:
        with self._lock:
            if self._closed:
                return 0.0
            self._commands += 1
            self._last_command_time = command.timestamp
            requested = command.requested_stiffness
            note = ""
            if not math.isfinite(requested) or not 0.0 <= requested <= 1.0:
                self._invalid += 1
                note = f"invalid request {requested!r} saturated"
                requested = 0.0 if not math.isfinite(requested) else min(1.0, max(0.0, requested))
            if self._estop or self._watchdog_tripped or not command.armed:
                applied = 0.0
            else:
                applied = requested
            self._applied = applied
            self._record(
                ActuatorRecord(
                    sequence=command.sequence,
                    timestamp=command.timestamp,
                    requested_stiffness=command.requested_stiffness,
                    applied_stiffness=applied,
                    state=command.state,
                    armed=command.armed,
                    estop=self._estop,
                    watchdog_tripped=self._watchdog_tripped,
                    faults=command.faults,
                    note=note,
                )
            )
            return applied

    def emergency_stop(self, reason: str) -> None:
        with self._lock:
            self._applied = 0.0
            if not self._estop:
                self._estop = True
                self._estop_reason = reason
                self._event(self._last_command_time or 0.0, f"EMERGENCY STOP: {reason}")

    def reset(self) -> None:
        with self._lock:
            if self._estop or self._watchdog_tripped:
                self._event(self._last_command_time or 0.0, "latches reset by operator")
            self._estop = False
            self._estop_reason = ""
            self._watchdog_tripped = False

    def check_watchdog(self, now: float) -> bool:
        with self._lock:
            if self._closed or self._last_command_time is None:
                return self._watchdog_tripped
            age = now - self._last_command_time
            if age > self.watchdog_timeout_s and not self._watchdog_tripped:
                self._watchdog_tripped = True
                self._applied = 0.0
                self._event(now, f"WATCHDOG: no command for {age * 1000:.0f} ms, released")
            return self._watchdog_tripped

    def status(self, now: float) -> ActuatorStatus:
        with self._lock:
            age = None if self._last_command_time is None else now - self._last_command_time
            return ActuatorStatus(
                backend=self.backend,
                applied_stiffness=self._applied,
                estop=self._estop,
                estop_reason=self._estop_reason,
                watchdog_tripped=self._watchdog_tripped,
                commands_received=self._commands,
                invalid_commands=self._invalid,
                last_command_age_s=age,
            )

    def close(self) -> None:
        with self._lock:
            if self._closed:
                return
            self._applied = 0.0
            self._event(self._last_command_time or 0.0, "closed, released")
            self._closed = True
            if self._log_file is not None:
                self._log_file.close()
                self._log_file = None
                self._log_writer = None

    def export_csv(self, path: Path) -> int:
        """Write the in-memory records to a CSV file; returns the number of rows."""
        with self._lock:
            records = list(self.records)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", newline="", encoding="utf-8") as file:
            writer = csv.writer(file)
            writer.writerow(COMMAND_FIELDS)
            for record in records:
                writer.writerow(record.as_row())
        return len(records)


def create_actuator(
    config: ActuatorConfig, safety: SafetyConfig, log_size: int, log_path: Path | None
) -> Actuator:
    """Build the configured backend. Only "dry_run" exists; anything else is refused."""
    if config.backend != "dry_run":
        raise ValueError(
            f"Actuator backend {config.backend!r} is not available: no physical "
            "actuator protocol is documented or verified (see docs/SAFETY.md)"
        )
    return DryRunActuator(safety.actuator_watchdog_timeout_s, log_size, log_path)
