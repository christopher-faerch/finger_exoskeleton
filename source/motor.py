"""
Drives the tendon motor from the live EMG.

Emil's ProportionalEMGController (source/controllers.py) moves the finger
between open and closed in proportion to a grasp signal (0 = open, 1 = closed).
It runs its own loop at MOTOR_RATE_HZ in a separate thread (MotorThread), so a
slow serial bus can never hold up the EMG. GraspSignal decides what it sees:

  DISARMED     -> 0, the finger opens (not calibrated yet, or recalibrating)
  no fresh EMG -> None (newest block older than STALE_S); the controller holds the
                  last value for SIGNAL_TIMEOUT_S, then opens
  IDLE         -> 0, the finger opens, so rest activity does not move it
  GRASP_ARMED  -> the grasp activation: the finger follows the grip
  STIFFENED    -> the mean grasp of the last HOLD_WINDOW_S before the tremor was
                  detected: the finger holds its position instead of following the tremor

With velocity control (--control velocity) VelocityEMGController uses the same
signal as a speed: a grasp closes the finger, faster the harder the grip, rest
opens it, and while STIFFENED the finger stands still.

With a log path (run_emg.py --out) MotorThread writes motor.csv in the session folder,
one row per motor step (MOTOR_RATE_HZ): see MOTOR_LOG_COLUMNS. time_s is the stream time
of emg.csv, so the motor can be lined up with the EMG, the cues and the test signal.

The motor runs in extended position mode (4) in both cases, so open and closed may lie on
both sides of the 0/4095 wrap. Only motor MOTOR_ID is touched.
"""

import csv
import math
import threading
import time
from collections import deque
from collections.abc import Callable
from pathlib import Path
from typing import Any

import settings
from source.pipeline import Pipeline
from source.stiffness_controller import State

TICKS_PER_TURN = 4096
MOTOR_LOG_FILE = "motor.csv"
# time_s: stream time; state: the controller state; grasp: the signal the controller got
# (0-1); target/goal: where the controller wants to go and the goal position it sent
# (rate limited); present: where the motor is; closure: present as 0 = open, 1 = closed
MOTOR_LOG_COLUMNS = ["time_s", "state", "grasp", "target_ticks", "goal_ticks",
                     "present_ticks", "closure"]


class GraspSignal:
    """The grasp signal for the motor controller; see the top of this file."""

    def __init__(self, pipeline: Pipeline, clock: Callable[[], float] = time.monotonic) -> None:
        self.pipeline = pipeline
        self.clock = clock
        self.history: deque = deque()  # (time, grasp) during the current grasp
        self.held: float | None = None  # the value held while STIFFENED

    def __call__(self) -> float | None:
        state = self.pipeline.state
        if state == State.DISARMED:
            self.history.clear()
            self.held = None
            return 0.0
        activation = self.pipeline.activation
        now = self.clock()
        if activation is None or now - activation.receive_time > settings.STALE_S:
            return None
        grasp = activation.grasp
        if state == State.STIFFENED:
            if self.held is None:
                values = [value for _, value in self.history]
                self.held = sum(values) / len(values) if values else grasp
            return self.held
        self.held = None
        if state == State.IDLE:
            self.history.clear()  # the hold only averages over the current grasp
            return 0.0
        self.history.append((now, grasp))
        while now - self.history[0][0] > settings.HOLD_WINDOW_S:
            self.history.popleft()
        return grasp


class MotorThread:
    """Runs the motor controller's loop in its own thread."""

    def __init__(self, controller: Any, signal: GraspSignal,
                 message: Callable[[str], None] = print, log_path: Path | None = None) -> None:
        # controller: a Proportional- or VelocityEMGController from source/controllers.py
        self.controller = controller
        self.signal = signal
        self.message = message
        self.log_path = log_path
        self.error: str | None = None
        self.thread = threading.Thread(target=self.run, name="motor", daemon=True)

    def start(self) -> None:
        """Start driving the motor."""
        self.thread.start()
        self.message(f"Motor control started ({type(self.controller).__name__})")

    def run(self) -> None:
        """The thread: the controller's own loop. Any motor error stops only the motor."""
        log = MotorLog(self.log_path, self.controller, self.signal.pipeline) \
            if self.log_path is not None else None
        try:
            self.controller.run(self.signal, verbose=False,
                                on_step=log.write if log is not None else None)
        except Exception as error:  # pylint: disable=broad-exception-caught
            self.error = f"{type(error).__name__}: {error}"
            self.message(f"Motor control stopped: {self.error}")
        finally:
            if log is not None:
                log.close()

    def stop(self) -> None:
        """Stop the controller (the finger returns to open) and wait for the thread."""
        self.controller.stop()
        self.thread.join(timeout=5.0)


class MotorLog:
    """Writes motor.csv: one row per motor step; see MOTOR_LOG_COLUMNS."""

    def __init__(self, path: Path, controller: Any, pipeline: Pipeline) -> None:
        self.controller = controller
        self.pipeline = pipeline
        self.last_flush = time.monotonic()
        # pylint: disable-next=consider-using-with  # closed in close()
        self.file = open(path, "w", newline="", encoding="utf-8")
        self.writer = csv.writer(self.file)
        self.writer.writerow(MOTOR_LOG_COLUMNS)

    def write(self, grasp: float, target: float, goal: int) -> None:
        """One row, after a controller step; the present position is read from the motor."""
        controller = self.controller
        try:
            present = float(controller.motors.get_position(controller.motor_id))
        except Exception:  # pylint: disable=broad-exception-caught  # a missed read is no stop
            present = math.nan
        closure = (present - controller.open_pos) / (controller.closed_pos - controller.open_pos)
        self.writer.writerow([f"{self.pipeline.stream_time:.4f}", self.pipeline.state.name,
                              f"{grasp:.4f}", f"{target:.1f}", goal, f"{present:.0f}",
                              f"{closure:.4f}"])
        if time.monotonic() - self.last_flush >= 1.0:  # keep the data on disk if the PC dies
            self.file.flush()
            self.last_flush = time.monotonic()

    def close(self) -> None:
        """Close motor.csv."""
        self.file.close()


def turn_shift(present: int, open_pos: int, closed_pos: int) -> int:
    """
    Whole turns (in ticks) to add to open_pos and closed_pos so the range lies
    in the turn the motor counts in now: the copy of the range whose middle is
    nearest to the present position.
    """
    middle = (open_pos + closed_pos) / 2
    return round((present - middle) / TICKS_PER_TURN) * TICKS_PER_TURN


def connect_motor(port: str, motors_class: Any = None, control: str = "position",
                  hold: Callable[[], bool] | None = None) -> tuple[Any, Any]:
    """
    Connect to the tendon motor and build the controller. Returns (motors, controller);
    close motors after the controller has stopped.
    control: "position" (the finger follows the grasp), "velocity" (the grasp sets the
    speed) or "threshold" (open or closed, with hysteresis); hold, with velocity control:
    True keeps the finger where it is.
    """
    if control not in ("position", "velocity", "threshold"):
        raise ValueError(f"Unknown motor control {control!r}: use position, velocity or threshold")
    # Imported here so the EMG side works without the Dynamixel SDK installed
    # pylint: disable=import-outside-toplevel
    from source.controllers import (
        ProportionalEMGController,
        ThresholdEMGController,
        VelocityEMGController,
    )

    if motors_class is None:
        from source.DynamixelHardwareInterface import Motors
        motors_class = Motors
    motors = motors_class(port=port, baudrate=settings.MOTOR_BAUDRATE, init_mode=None)
    motor_id = settings.MOTOR_ID
    if motor_id not in motors.motor_ids:
        motors.close()
        raise ValueError(f"Motor ID {motor_id} not found (found {motors.motor_ids})")

    # The operating mode can only be changed with torque off
    motors.disable_torque([motor_id])
    motors.set_cont_mode("ext", [motor_id])
    present = int(motors.get_position(motor_id))  # in the turn count of extended mode
    shift = turn_shift(present, settings.OPEN_POS, settings.CLOSED_POS)
    open_pos, closed_pos = settings.OPEN_POS + shift, settings.CLOSED_POS + shift

    # The controller's first command is the present position limited to the range, sent at full
    # speed. Far outside the range that would be a jump, so refuse while torque is still off.
    low, high = sorted((open_pos, closed_pos))
    margin = max(30, round(0.05 * (high - low)))
    if not low - margin <= present <= high + margin:
        motors.close()
        raise ValueError(f"Motor {motor_id} is at {present} ticks, outside the open-closed range "
                         f"{low}-{high}. Put the finger in the open position, or measure "
                         f"OPEN_POS and CLOSED_POS again. Torque was left off.")
    motors.enable_torque([motor_id])
    common = {"open_pos": open_pos, "closed_pos": closed_pos, "rate_hz": settings.MOTOR_RATE_HZ,
              "signal_timeout_s": settings.SIGNAL_TIMEOUT_S, "deadband": settings.DEADBAND,
              "gamma": settings.GAMMA}
    controller: Any
    if control == "velocity":
        controller = VelocityEMGController(
            motors, motor_id, close_speed=settings.VELOCITY_CLOSE_SPEED,
            open_speed=settings.VELOCITY_OPEN_SPEED, hold=hold, **common)
    elif control == "threshold":
        controller = ThresholdEMGController(
            motors, motor_id, open_pos=open_pos, closed_pos=closed_pos,
            max_speed=settings.MAX_SPEED, rate_hz=settings.MOTOR_RATE_HZ,
            signal_timeout_s=settings.SIGNAL_TIMEOUT_S, on_threshold=settings.THRESHOLD_ON,
            off_threshold=settings.THRESHOLD_OFF, on_dwell_s=settings.THRESHOLD_ON_TIME_S,
            off_dwell_s=settings.THRESHOLD_OFF_TIME_S, mode="hold")
    else:
        controller = ProportionalEMGController(motors, motor_id, max_speed=settings.MAX_SPEED,
                                               **common)
    return motors, controller
