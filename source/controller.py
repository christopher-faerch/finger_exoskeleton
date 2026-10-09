"""
Exoskeleton controller: the tremor- and grasp-contingent stiffening policy of
docs/Detailed Description.pdf (8.1 option C, 8.4 layer 1 and 3):

  IDLE        --grasp activation above on-threshold for grasp_on_time-->   GRASP_ARMED
  GRASP_ARMED --grasp activation below off-threshold for grasp_off_time--> IDLE
  GRASP_ARMED --tremor level above on-threshold, with tremor ratio at least
                tremor_min_ratio (rhythmic, design doc 7.7), for tremor_on_time--> STIFFENED
  STIFFENED   --tremor level below off-threshold, or tremor ratio below
                tremor_off_ratio (no longer rhythmic), for tremor_off_time--> GRASP_ARMED
  STIFFENED   --grasp activation below off-threshold for grasp_off_time--> IDLE

Separate on/off thresholds plus hold times give hysteresis. In STIFFENED the
stiffness scales with tremor level (semi-proportional, design doc 6.5).
The command rises at most engage_rate_per_s and falls at most
release_rate_per_s, so release takes ~0.5-1 s (design doc 8.4).

Safe states:
  DISARMED  at start-up and after shutdown: stiffness 0, nothing happens
            until arm() is called explicitly.
  FAULT     entered from any armed state when a fault is reported:
            stiffness drops to 0 immediately (no ramp) and stays there
            until the faults clear and arm() is called again.

Stiffness 0 means the semi-active element is released (transparent). The
command is always clamped to [0, max_stiffness], with max_stiffness <= 1.
"""

import math
from collections.abc import Sequence
from dataclasses import dataclass
from enum import Enum

from .activation import ActivationEstimate
from .config import ControllerConfig


class ControllerState(Enum):
    """Controller states; DISARMED and FAULT are the safe states."""

    DISARMED = "DISARMED"
    IDLE = "IDLE"
    GRASP_ARMED = "GRASP_ARMED"
    STIFFENED = "STIFFENED"
    FAULT = "FAULT"


SAFE_STATES = (ControllerState.DISARMED, ControllerState.FAULT)


@dataclass(frozen=True)
class ControllerOutput:
    """Result of one controller update."""

    state: ControllerState
    stiffness: float
    target_stiffness: float
    armed: bool
    faults: tuple[str, ...]
    last_transition: str


class _HoldTimer:
    """True once a condition has held continuously for `duration` seconds."""

    def __init__(self, duration: float) -> None:
        self.duration = duration
        self._since: float | None = None

    def update(self, condition: bool, now: float) -> bool:
        """Feed the condition at time `now`."""
        if not condition:
            self._since = None
            return False
        if self._since is None:
            self._since = now
        return now - self._since >= self.duration

    def reset(self) -> None:
        """Forget how long the condition has held."""
        self._since = None


def clamp(value: float, low: float, high: float) -> float:
    """Clamp to [low, high]; NaN becomes low (the safe value)."""
    if math.isnan(value):
        return low
    return max(low, min(high, value))


class ExoskeletonController:  # pylint: disable=too-many-instance-attributes
    """The state machine described in the module docstring."""

    def __init__(self, config: ControllerConfig) -> None:
        self.config = config
        self.state = ControllerState.DISARMED
        self.stiffness = 0.0
        self.last_transition = "start-up: disarmed"
        self._last_time: float | None = None
        self._grasp_on = _HoldTimer(config.grasp_on_time_s)
        self._grasp_off = _HoldTimer(config.grasp_off_time_s)
        self._tremor_on = _HoldTimer(config.tremor_on_time_s)
        self._tremor_off = _HoldTimer(config.tremor_off_time_s)

    @property
    def armed(self) -> bool:
        """True when the controller may command stiffness."""
        return self.state not in SAFE_STATES

    def _reset_timers(self) -> None:
        for timer in (self._grasp_on, self._grasp_off, self._tremor_on, self._tremor_off):
            timer.reset()

    def _transition(self, state: ControllerState, reason: str) -> None:
        if state != self.state:
            self.last_transition = f"{self.state.value} -> {state.value}: {reason}"
            self.state = state
            self._reset_timers()

    def arm(self) -> None:
        """
        Leave DISARMED or FAULT for IDLE. The caller (runtime safety
        supervisor) must have checked that no fault is active.
        """
        if self.state in SAFE_STATES:
            self.stiffness = 0.0
            self._transition(ControllerState.IDLE, "armed by operator")

    def disarm(self, reason: str) -> None:
        """Go to DISARMED with zero stiffness immediately."""
        self.stiffness = 0.0
        self._transition(ControllerState.DISARMED, reason)

    def _output(self, target: float, faults: Sequence[str]) -> ControllerOutput:
        return ControllerOutput(
            state=self.state,
            stiffness=self.stiffness,
            target_stiffness=target,
            armed=self.armed,
            faults=tuple(faults),
            last_transition=self.last_transition,
        )

    def update(
        self, activation: ActivationEstimate | None, faults: Sequence[str], now: float
    ) -> ControllerOutput:
        """
        Advance the state machine to time `now` (seconds, monotonic).
        Any fault forces stiffness 0 at once; from an armed state it latches FAULT.
        """
        dt = 0.0 if self._last_time is None else clamp(now - self._last_time, 0.0, 0.1)
        self._last_time = now

        if faults:
            self.stiffness = 0.0
            if self.armed:
                self._transition(ControllerState.FAULT, ", ".join(faults))
            return self._output(0.0, faults)
        if not self.armed:
            self.stiffness = 0.0
            return self._output(0.0, faults)
        if activation is None:
            # The supervisor reports missing activation as a fault; this is a second guard
            self.stiffness = 0.0
            self._transition(ControllerState.FAULT, "no activation estimate")
            return self._output(0.0, ("NO_ACTIVATION",))

        self._step_state(activation, now)
        target = self._target_stiffness(activation)
        self._ramp(target, dt)
        return self._output(target, faults)

    def _step_state(self, activation: ActivationEstimate, now: float) -> None:
        config = self.config
        grasp, tremor = activation.grasp_activation, activation.tremor_level
        ratio = activation.tremor_ratio
        if math.isnan(grasp) or math.isnan(tremor) or math.isnan(ratio):
            grasp, tremor, ratio = 0.0, 0.0, 0.0
        grasp_on = self._grasp_on.update(grasp >= config.grasp_on_threshold, now)
        grasp_off = self._grasp_off.update(grasp < config.grasp_off_threshold, now)
        tremor_on = self._tremor_on.update(
            tremor >= config.tremor_on_threshold and ratio >= config.tremor_min_ratio, now
        )
        tremor_off = self._tremor_off.update(
            tremor < config.tremor_off_threshold or ratio < config.tremor_off_ratio, now
        )

        if self.state == ControllerState.IDLE and grasp_on:
            self._transition(ControllerState.GRASP_ARMED, f"grasp activation {grasp:.2f}")
        elif self.state == ControllerState.GRASP_ARMED:
            if grasp_off:
                self._transition(ControllerState.IDLE, f"grasp released ({grasp:.2f})")
            elif tremor_on:
                self._transition(
                    ControllerState.STIFFENED, f"tremor level {tremor:.2f}, ratio {ratio:.2f}"
                )
        elif self.state == ControllerState.STIFFENED:
            if grasp_off:
                self._transition(ControllerState.IDLE, f"grasp released ({grasp:.2f})")
            elif tremor_off:
                self._transition(
                    ControllerState.GRASP_ARMED,
                    f"tremor subsided (level {tremor:.2f}, ratio {ratio:.2f})",
                )

    def _target_stiffness(self, activation: ActivationEstimate) -> float:
        config = self.config
        if self.state != ControllerState.STIFFENED:
            return 0.0
        excess = max(0.0, activation.tremor_level - config.tremor_off_threshold)
        return clamp(
            config.engage_stiffness + config.stiffness_gain * excess, 0.0, config.max_stiffness
        )

    def _ramp(self, target: float, dt: float) -> None:
        config = self.config
        if target > self.stiffness:
            self.stiffness = min(target, self.stiffness + config.engage_rate_per_s * dt)
        else:
            self.stiffness = max(target, self.stiffness - config.release_rate_per_s * dt)
        self.stiffness = clamp(self.stiffness, 0.0, min(1.0, config.max_stiffness))
