"""Controller state machine: transitions, hysteresis, bounds and safe states."""

from dataclasses import replace

import numpy as np
import pytest

from source.activation import ActivationEstimate
from source.config import ControllerConfig
from source.controller import ControllerState, ExoskeletonController, clamp

CONFIG = ControllerConfig()
DT = 1 / 150


def activation(grasp: float, tremor: float = 0.0, ratio: float = 1.0) -> ActivationEstimate:
    """An activation estimate with the given summaries."""
    zeros = np.zeros(2)
    return ActivationEstimate(
        sample_index=0,
        receive_time=0.0,
        slow=zeros,
        fast=zeros,
        tremor=zeros,
        tremor_ratio_per_channel=zeros,
        grasp_activation=grasp,
        tremor_level=tremor,
        tremor_ratio=ratio,
    )


class Clock:  # pylint: disable=too-few-public-methods
    """Drives a controller at the control rate."""

    def __init__(self, controller: ExoskeletonController) -> None:
        self.controller = controller
        self.now = 0.0

    def run(
        self, seconds: float, estimate: ActivationEstimate | None, faults: tuple[str, ...] = ()
    ) -> list[float]:
        """Run for `seconds`; returns the stiffness of every tick."""
        stiffness = []
        for _ in range(max(1, round(seconds / DT))):
            self.now += DT
            stiffness.append(self.controller.update(estimate, faults, self.now).stiffness)
        return stiffness


def state(controller: ExoskeletonController) -> ControllerState:
    """Current state (a call, so the type checker does not narrow it between asserts)."""
    return controller.state


def armed_controller(config: ControllerConfig = CONFIG) -> tuple[ExoskeletonController, Clock]:
    """A controller that has been armed."""
    controller = ExoskeletonController(config)
    controller.arm()
    return controller, Clock(controller)


def test_starts_disarmed_with_zero_stiffness() -> None:
    """Safe start-up: nothing happens until arm()."""
    controller = ExoskeletonController(CONFIG)
    clock = Clock(controller)
    stiffness = clock.run(2.0, activation(grasp=1.0, tremor=1.0))
    assert state(controller) == ControllerState.DISARMED
    assert max(stiffness) == 0.0


def test_full_cycle_idle_armed_stiffened_and_back() -> None:
    """Grasp arms, rhythmic tremor stiffens, tremor subsiding and grasp release go back."""
    controller, clock = armed_controller()
    assert state(controller) == ControllerState.IDLE
    clock.run(0.5, activation(grasp=0.6))
    assert state(controller) == ControllerState.GRASP_ARMED
    stiffness = clock.run(0.5, activation(grasp=0.6, tremor=0.3, ratio=0.95))
    assert state(controller) == ControllerState.STIFFENED
    assert stiffness[-1] > 0.3
    clock.run(1.0, activation(grasp=0.6, tremor=0.02, ratio=0.95))
    assert state(controller) == ControllerState.GRASP_ARMED
    clock.run(0.5, activation(grasp=0.05))
    assert state(controller) == ControllerState.IDLE


def test_grasp_hold_time_debounces() -> None:
    """A grasp shorter than grasp_on_time_s does not arm."""
    controller, clock = armed_controller()
    clock.run(CONFIG.grasp_on_time_s * 0.5, activation(grasp=0.9))
    clock.run(0.2, activation(grasp=0.0))
    assert state(controller) == ControllerState.IDLE


def test_grasp_hysteresis_band_keeps_state() -> None:
    """Between the off and on thresholds the state does not change in either direction."""
    middle = (CONFIG.grasp_on_threshold + CONFIG.grasp_off_threshold) / 2
    controller, clock = armed_controller()
    clock.run(2.0, activation(grasp=middle))
    assert state(controller) == ControllerState.IDLE
    clock.run(0.5, activation(grasp=0.6))
    clock.run(2.0, activation(grasp=middle))
    assert state(controller) == ControllerState.GRASP_ARMED


def test_non_rhythmic_tremor_band_does_not_stiffen() -> None:
    """Large but broadband envelope fluctuation (low ratio) is not treated as tremor."""
    controller, clock = armed_controller()
    clock.run(0.5, activation(grasp=0.6))
    clock.run(2.0, activation(grasp=0.6, tremor=0.5, ratio=0.5))
    assert state(controller) == ControllerState.GRASP_ARMED


def test_tremor_without_grasp_does_not_stiffen() -> None:
    """Stiffening is grasp-contingent."""
    controller, clock = armed_controller()
    stiffness = clock.run(2.0, activation(grasp=0.0, tremor=0.5, ratio=1.0))
    assert state(controller) == ControllerState.IDLE
    assert max(stiffness) == 0.0


def test_release_on_lost_rhythmicity() -> None:
    """Stiffening releases when the fluctuation stops being rhythmic, after tremor_off_time_s."""
    controller, clock = armed_controller()
    clock.run(0.5, activation(grasp=0.6))
    clock.run(0.5, activation(grasp=0.6, tremor=0.3, ratio=0.95))
    clock.run(CONFIG.tremor_off_time_s * 0.5, activation(grasp=0.6, tremor=0.3, ratio=0.3))
    assert state(controller) == ControllerState.STIFFENED
    clock.run(CONFIG.tremor_off_time_s, activation(grasp=0.6, tremor=0.3, ratio=0.3))
    assert state(controller) == ControllerState.GRASP_ARMED


@pytest.mark.parametrize("tremor", [0.5, 5.0, 1e9, float("inf")])
def test_stiffness_is_bounded(tremor: float) -> None:
    """The command never leaves [0, max_stiffness]."""
    config = replace(CONFIG, max_stiffness=0.7)
    _, clock = armed_controller(config)
    clock.run(0.5, activation(grasp=0.6))
    stiffness = clock.run(2.0, activation(grasp=0.6, tremor=tremor, ratio=1.0))
    assert all(0.0 <= value <= 0.7 for value in stiffness)
    assert stiffness[-1] == pytest.approx(0.7)


def test_nan_activation_is_treated_as_rest() -> None:
    """NaN inputs never produce stiffness."""
    controller, clock = armed_controller()
    stiffness = clock.run(
        1.0, activation(grasp=float("nan"), tremor=float("nan"), ratio=float("nan"))
    )
    assert max(stiffness) == 0.0
    assert state(controller) == ControllerState.IDLE


def test_rate_limits() -> None:
    """Engagement is fast but rate limited; release ramps down over ~0.7 s."""
    _, clock = armed_controller()
    clock.run(0.5, activation(grasp=0.6))
    rising = clock.run(0.3, activation(grasp=0.6, tremor=1.0, ratio=1.0))
    steps = np.diff([0.0] + rising)
    assert np.max(steps) <= CONFIG.engage_rate_per_s * DT + 1e-9
    falling = clock.run(3.0, activation(grasp=0.0))
    drops = -np.diff([rising[-1]] + falling)
    assert np.max(drops) <= CONFIG.release_rate_per_s * DT + 1e-9
    assert falling[-1] == 0.0


def test_fault_releases_immediately_and_latches() -> None:
    """A fault zeroes stiffness at once (no ramp), latches FAULT, and needs arm() to recover."""
    controller, clock = armed_controller()
    clock.run(0.5, activation(grasp=0.6))
    clock.run(0.5, activation(grasp=0.6, tremor=0.5, ratio=1.0))
    assert controller.stiffness > 0.5
    stiffness = clock.run(DT, activation(grasp=0.6, tremor=0.5, ratio=1.0), faults=("STALE_DATA",))
    assert stiffness == [0.0]
    assert state(controller) == ControllerState.FAULT
    clock.run(1.0, activation(grasp=0.6, tremor=0.5, ratio=1.0))
    assert state(controller) == ControllerState.FAULT
    controller.arm()
    assert state(controller) == ControllerState.IDLE


def test_missing_activation_while_armed_is_a_fault() -> None:
    """No activation estimate while armed fails safe."""
    controller, clock = armed_controller()
    assert clock.run(DT, None) == [0.0]
    assert state(controller) == ControllerState.FAULT


def test_disarm_from_any_state() -> None:
    """disarm() is an explicit safe transition with zero stiffness."""
    controller, clock = armed_controller()
    clock.run(0.5, activation(grasp=0.6))
    clock.run(0.5, activation(grasp=0.6, tremor=0.5, ratio=1.0))
    controller.disarm("test")
    assert state(controller) == ControllerState.DISARMED and controller.stiffness == 0.0


def test_clamp_handles_nan() -> None:
    """NaN clamps to the safe lower bound."""
    assert clamp(float("nan"), 0.0, 1.0) == 0.0
    assert clamp(2.0, 0.0, 1.0) == 1.0
