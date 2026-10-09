"""
Software safety supervisor.

Evaluated on every control tick, it turns the health of the pipeline into a
list of active faults. Any fault forces the controller to zero stiffness (and
latches FAULT if it was armed). Arming requires an explicit operator action
and is refused while any fault is active.

This is a software layer only. It cannot protect against a hung operating
system, a crashed Python process or a faulty actuator driver. Physical
actuation additionally needs the independent hardware safety described in
docs/SAFETY.md.
"""

import threading
from dataclasses import dataclass
from enum import Enum

from .config import SafetyConfig


class Fault(str, Enum):
    """Fault conditions; the value is what the monitor and the log show."""

    ESTOP = "E_STOP"
    NO_DATA = "NO_DATA"
    STALE_DATA = "STALE_DATA"
    SOURCE_DISCONNECTED = "SOURCE_DISCONNECTED"
    ACQUISITION_STOPPED = "ACQUISITION_STOPPED"
    INVALID_DATA = "INVALID_DATA"
    NOT_CALIBRATED = "NOT_CALIBRATED"
    ACTUATOR_LATCHED = "ACTUATOR_LATCHED"


@dataclass(frozen=True)
class HealthInputs:  # pylint: disable=too-many-instance-attributes
    """Everything the supervisor needs for one evaluation."""

    now: float
    last_data_time: float | None
    source_connected: bool
    acquisition_running: bool
    invalid_data: bool
    calibrated: bool
    actuator_latched: bool


class SafetySupervisor:
    """Fault evaluation and the emergency-stop latch. Thread-safe."""

    def __init__(self, config: SafetyConfig) -> None:
        self.config = config
        self._lock = threading.Lock()
        self._estop = False
        self._estop_reason = ""

    @property
    def estop_active(self) -> bool:
        """True while the emergency stop is latched."""
        with self._lock:
            return self._estop

    @property
    def estop_reason(self) -> str:
        """Why the emergency stop was triggered."""
        with self._lock:
            return self._estop_reason

    def trigger_estop(self, reason: str) -> None:
        """Latch the emergency stop. Only reset_estop() clears it."""
        with self._lock:
            if not self._estop:
                self._estop = True
                self._estop_reason = reason

    def reset_estop(self) -> None:
        """Clear the emergency stop latch. The system stays disarmed until armed again."""
        with self._lock:
            self._estop = False
            self._estop_reason = ""

    def evaluate(self, inputs: HealthInputs) -> tuple[str, ...]:
        """Active faults, most severe first. Empty means healthy."""
        faults: list[Fault] = []
        if self.estop_active:
            faults.append(Fault.ESTOP)
        if not inputs.acquisition_running:
            faults.append(Fault.ACQUISITION_STOPPED)
        if not inputs.source_connected:
            faults.append(Fault.SOURCE_DISCONNECTED)
        if inputs.last_data_time is None:
            faults.append(Fault.NO_DATA)
        elif inputs.now - inputs.last_data_time > self.config.stale_data_timeout_s:
            faults.append(Fault.STALE_DATA)
        if inputs.invalid_data:
            faults.append(Fault.INVALID_DATA)
        if not inputs.calibrated:
            faults.append(Fault.NOT_CALIBRATED)
        if inputs.actuator_latched:
            faults.append(Fault.ACTUATOR_LATCHED)
        return tuple(fault.value for fault in faults)

    @staticmethod
    def arm_blockers(faults: tuple[str, ...]) -> tuple[str, ...]:
        """
        Faults that prevent arming. A latched actuator (watchdog or e-stop
        latch) is cleared by the arm action itself, so it does not block,
        unless the e-stop itself is still active.
        """
        return tuple(fault for fault in faults if fault != Fault.ACTUATOR_LATCHED.value)
