"""The grasp signal and the motor connection in source/motor.py, with a fake motor."""

from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

import settings
from source.calibration import Activation
from source.controllers import ThresholdEMGController, VelocityEMGController
from source.motor import GraspSignal, connect_motor
from source.stiffness_controller import State
from tests.fakes import FakeMotors


def fake_pipeline(state: State, grasp: float = 0.6, received: float = 100.0) -> Any:
    """Something with the two attributes GraspSignal reads: state and activation."""
    return SimpleNamespace(state=state, activation=Activation(grasp, 0.0, 0.0, received))


def signal_for(pipeline: Any, now: float = 100.0) -> GraspSignal:
    """A GraspSignal whose clock reads `now`."""
    return GraspSignal(pipeline, clock=lambda: now)


def test_disarmed_and_idle_open_the_finger() -> None:
    """Not calibrated, or no grasp detected: the signal is 0."""
    assert signal_for(fake_pipeline(State.DISARMED))() == 0.0
    assert signal_for(fake_pipeline(State.IDLE))() == 0.0


def test_the_finger_follows_the_grasp() -> None:
    """In GRASP_ARMED the signal is the grasp activation."""
    assert signal_for(fake_pipeline(State.GRASP_ARMED, grasp=0.7))() == 0.7


def test_old_emg_gives_no_signal() -> None:
    """EMG older than STALE_S counts as no signal (None)."""
    pipeline = fake_pipeline(State.GRASP_ARMED, received=100.0)
    assert signal_for(pipeline, now=100.0 + settings.STALE_S + 0.01)() is None


def test_tremor_holds_the_mean_of_the_grasp_before_it() -> None:
    """In STIFFENED the signal is held at the mean grasp of the last second, not the new value."""
    pipeline = fake_pipeline(State.GRASP_ARMED)
    clock = {"now": 100.0}
    signal = GraspSignal(pipeline, clock=lambda: clock["now"])
    for grasp in (0.4, 0.6):
        pipeline.activation = Activation(grasp, 0.0, 0.0, clock["now"])
        signal()
        clock["now"] += 0.1
    pipeline.state = State.STIFFENED
    pipeline.activation = Activation(0.9, 0.0, 0.0, clock["now"])
    assert signal() == pytest.approx(0.5)


def test_idle_forgets_the_previous_grasp() -> None:
    """After IDLE, the hold only averages over the new grasp."""
    pipeline = fake_pipeline(State.GRASP_ARMED, grasp=0.2)
    signal = signal_for(pipeline)
    signal()
    pipeline.state = State.IDLE
    signal()
    pipeline.state, pipeline.activation = State.GRASP_ARMED, Activation(0.8, 0.0, 0.0, 100.0)
    signal()
    pipeline.state = State.STIFFENED
    assert signal() == 0.8


def test_connect_sets_only_the_tendon_motor_to_extended_mode() -> None:
    """Motor MOTOR_ID goes to mode 4 with torque on; motor 1 is not touched."""
    motors, controller = connect_motor("COM0", motors_class=FakeMotors)
    assert motors.init_mode is None
    assert motors.control_mode == [1, FakeMotors.EXT_POS_CONTROL]
    assert motors.torque == [False, True]
    assert (controller.open_pos, controller.closed_pos) == (settings.OPEN_POS, settings.CLOSED_POS)


def test_connect_shifts_the_range_to_the_motors_turn() -> None:
    """A motor counted one turn further gets open and closed one turn further too."""
    def one_turn_on(**kwargs: Any) -> FakeMotors:
        return FakeMotors(position=4600 + 4096, **kwargs)

    _, controller = connect_motor("COM0", motors_class=one_turn_on)
    assert controller.open_pos == settings.OPEN_POS + 4096


def test_connect_refuses_a_motor_far_outside_the_range() -> None:
    """Far outside open-closed the start is refused, with torque left off."""
    made: list[FakeMotors] = []

    def far_away(**kwargs: Any) -> FakeMotors:
        made.append(FakeMotors(position=settings.CLOSED_POS - 1000, **kwargs))
        return made[0]

    with pytest.raises(ValueError, match="outside the open-closed range"):
        connect_motor("COM0", motors_class=far_away)
    assert made[0].torque == [False, False] and made[0].closed


def test_connect_builds_the_velocity_controller() -> None:
    """control="velocity" gives the rate controller with the speeds from settings.py."""
    _, controller = connect_motor("COM0", motors_class=FakeMotors, control="velocity")
    assert isinstance(controller, VelocityEMGController)
    assert controller.close_speed == settings.VELOCITY_CLOSE_SPEED
    with pytest.raises(ValueError, match="Unknown motor control"):
        connect_motor("COM0", motors_class=FakeMotors, control="torque")


def velocity_controller(hold: bool = False) -> tuple[FakeMotors, VelocityEMGController]:
    """A velocity controller on a fake motor that starts at the open position."""
    motors = FakeMotors(position=settings.OPEN_POS)
    motors.control_mode = [1, FakeMotors.EXT_POS_CONTROL]
    controller = VelocityEMGController(
        motors, settings.MOTOR_ID, open_pos=settings.OPEN_POS, closed_pos=settings.CLOSED_POS,
        close_speed=1000.0, open_speed=500.0, deadband=0.1, hold=lambda: hold)
    return motors, controller


def run_steps(controller: VelocityEMGController, activation: float, seconds: float,
              start: float = 0.0) -> float:
    """Step the controller at 50 Hz; returns the time after the last step."""
    now = start
    for _ in range(round(seconds * 50)):
        controller.step(activation, now)
        now += 0.02
    return now


def test_velocity_grasp_closes_at_a_speed_set_by_the_grasp() -> None:
    """Full grasp closes at close_speed, a weaker grasp more slowly, rest opens again."""
    motors, controller = velocity_controller()
    now = run_steps(controller, 1.0, 0.5)
    full = settings.OPEN_POS - motors.position  # closed lies below open
    assert full == pytest.approx(0.5 * 1000.0, abs=25)
    motors, weak = velocity_controller()
    run_steps(weak, 0.55, 0.5)  # (0.55 - 0.1) / 0.9 = half speed
    assert settings.OPEN_POS - motors.position == pytest.approx(full / 2, abs=25)
    before = motors.position
    run_steps(weak, 0.0, 0.2, start=now)
    assert motors.position > before  # rest: the finger opens


def test_velocity_stops_at_closed_and_holds_during_tremor() -> None:
    """The finger never passes closed, and hold() keeps it where it is."""
    motors, controller = velocity_controller()
    run_steps(controller, 1.0, 5.0)
    assert motors.position == settings.CLOSED_POS
    motors, held = velocity_controller(hold=True)
    run_steps(held, 1.0, 1.0)
    assert motors.position == settings.OPEN_POS


def test_connect_builds_the_threshold_controller() -> None:
    """control="threshold" gives the open/closed controller with the thresholds from settings.py."""
    _, controller = connect_motor("COM0", motors_class=FakeMotors, control="threshold")
    assert isinstance(controller, ThresholdEMGController)
    assert (controller.on_threshold, controller.off_threshold) == (
        settings.THRESHOLD_ON, settings.THRESHOLD_OFF)


def test_threshold_does_not_flicker_on_a_noisy_held_grip() -> None:
    """A grip that varies about 20 % around 1.0 keeps the finger closed: the command never reopens."""
    _, controller = connect_motor("COM0", motors_class=FakeMotors, control="threshold")
    rng = np.random.default_rng(0)
    now = 0.0
    for _ in range(100):  # close (takes about 1.3 s)
        controller.step(1.0, now)
        now += 0.02
    commands = []
    for _ in range(500):  # 10 s of a noisy held grip
        _, _, command = controller.step(float(np.clip(rng.normal(1.0, 0.25), 0.0, 1.0)), now)
        commands.append(command)
        now += 0.02
    assert set(commands) == {settings.CLOSED_POS}


def test_threshold_opens_when_the_hand_relaxes() -> None:
    """Below the off threshold for long enough, the finger goes back to open."""
    _, controller = connect_motor("COM0", motors_class=FakeMotors, control="threshold")
    now = 0.0
    for _ in range(100):
        controller.step(1.0, now)
        now += 0.02
    for _ in range(200):
        _, _, command = controller.step(0.05, now)
        now += 0.02
    assert command == settings.OPEN_POS
