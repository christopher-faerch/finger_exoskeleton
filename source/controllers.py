"""
This module contains the different control systems implemented for the finger exoskeleton.

Two controllers drive the Dynamixel in POSITION mode between an "open" and a
"closed" (grasp) motor position, based on a muscle activation signal that is
supplied from outside (the EMG processing is NOT part of this module):

  * ProportionalEMGController - the finger position follows the muscle
    activation continuously (more effort = more closed).
  * ThresholdEMGController    - activation above a threshold closes the hand,
    below a lower threshold opens it (with hysteresis and dwell times,
    optionally as a toggle).
  * VelocityEMGController     - the muscle activation sets the closing SPEED
    (rate control): grip harder to close faster, relax to open. The motor
    stays in position mode, so the open/closed limits still hold.

INPUT CONTRACT
    activation: a float in 0.0 .. 1.0 where 0 = muscle at rest and 1 = full
    effort. It should already be rectified, smoothed (envelope) and
    normalised per user. Values outside 0..1 are clipped, NaN is treated as
    "no signal".

USAGE
    controller = ProportionalEMGController(motors, motor_id, open_pos, closed_pos)

    # (a) call once per new activation value from your own loop
    controller.step(activation)

    # (b) or let the controller run its own fixed-rate loop, polling a callable
    controller.run(get_activation)      # get_activation() -> float or None
"""

import math
import time

import numpy as np


# --------------------------------------------------------------------------- #
# Controllers
# --------------------------------------------------------------------------- #
class EMGPositionController:
    """Common machinery: EMG -> activation -> target position -> safe motor command.

    The motor must be in POSITION or EXTENDED POSITION control mode. Positions
    are raw Dynamixel ticks (position mode covers one turn, 0..4095; extended
    position mode is multi-turn and may go negative).

    Parameters
    ----------
    motors, motor_id : Motors instance and the ID of the tendon motor
    open_pos         : motor position with the finger open / tendon relaxed (ticks)
    closed_pos       : motor position with the grasp closed (ticks)
    max_speed        : limit on how fast the commanded position may move (ticks/s)
    rate_hz          : loop rate used by run()
    signal_timeout_s : run() only: if the signal source gives no valid value for
                       this long, activation is forced to 0 so the finger opens
    """

    def __init__(self, motors, motor_id, open_pos, closed_pos,
                 max_speed=600.0, rate_hz=50.0, signal_timeout_s=0.5):
        assert motor_id in motors.motor_ids, f"Motor ID {motor_id} not found."
        position_modes = (motors.POS_CONTROL, getattr(motors, "EXT_POS_CONTROL", motors.POS_CONTROL))
        assert motors.control_mode[motors.motor_ids.index(motor_id)] in position_modes, \
            "EMG controllers require the motor to be in (extended) position control mode."
        assert open_pos != closed_pos, "open_pos and closed_pos must differ."

        self.motors = motors
        self.motor_id = motor_id
        self.open_pos = float(open_pos)
        self.closed_pos = float(closed_pos)
        self.lo = min(self.open_pos, self.closed_pos)
        self.hi = max(self.open_pos, self.closed_pos)
        self.max_speed = max_speed
        self.period = 1.0 / rate_hz
        self.signal_timeout_s = signal_timeout_s

        self.target = self.open_pos          # where the controller wants to go
        self.command = None                  # what was last sent (rate limited)
        self._t_last = None
        self._stop = False

    # -- to be implemented by the concrete controllers ------------------------
    def compute_target(self, activation, now):
        """Return the desired motor position in ticks for this activation (0..1)."""
        raise NotImplementedError

    # -------------------------------------------------------------------------
    def _limit(self, target, now):
        """Move the commanded position toward target at no more than max_speed."""
        dt = self.period if self._t_last is None else max(now - self._t_last, 1e-3)
        self._t_last = now
        step = self.max_speed * dt
        self.command += float(np.clip(target - self.command, -step, step))
        self.command = float(np.clip(self.command, self.lo, self.hi))
        return int(round(self.command))

    def step(self, activation, now=None):
        """One control iteration for a new activation value (0..1).
        Returns (activation, target, command)."""
        now = time.perf_counter() if now is None else now
        if self.command is None:             # start from where the motor is: no jump
            self.command = float(np.clip(self.motors.get_position(self.motor_id), self.lo, self.hi))
        activation = 0.0 if activation is None or math.isnan(activation) \
            else float(np.clip(activation, 0.0, 1.0))
        self.target = float(np.clip(self.compute_target(activation, now), self.lo, self.hi))
        command = self._limit(self.target, now)
        self.motors.sendMotorCommand(self.motor_id, command)
        return activation, self.target, command

    def stop(self):
        self._stop = True

    def release(self):
        """Smoothly move back to the open position."""
        if self.command is None:
            return
        while abs(self.command - self.open_pos) > 1:
            time.sleep(self.period)
            now = time.perf_counter()
            self.motors.sendMotorCommand(self.motor_id, self._limit(self.open_pos, now))
        self.motors.sendMotorCommand(self.motor_id, int(round(self.open_pos)))

    def run(self, get_activation, duration=None, verbose=True, on_step=None):
        """Blocking fixed-rate loop. `get_activation()` returns the latest activation
        (0..1) or None/NaN if there is no valid value. Ends after `duration` s, on
        stop(), or Ctrl+C, and always returns the finger to the open position.
        on_step(activation, target, command), if given, is called after every step
        (e.g. to log the motor)."""
        self._stop = False
        self._t_last = None
        t_end = None if duration is None else time.perf_counter() + duration
        next_t = time.perf_counter()
        t_valid = time.perf_counter()
        last = 0.0
        try:
            while not self._stop and (t_end is None or time.perf_counter() < t_end):
                value = get_activation()
                now = time.perf_counter()
                if value is not None and not math.isnan(value):
                    t_valid, last = now, value
                else:                                # no valid value: hold briefly, then open
                    value = last if now - t_valid <= self.signal_timeout_s else 0.0
                activation, target, command = self.step(value)
                if on_step is not None:
                    on_step(activation, target, command)
                if verbose:
                    print(f"emg={activation:5.2f}  target={target:7.1f}  command={command}  "
                          + self.status())
                next_t += self.period
                time.sleep(max(0.0, next_t - time.perf_counter()))
        except KeyboardInterrupt:
            pass
        finally:
            self.release()

    def status(self):
        return ""


class ProportionalEMGController(EMGPositionController):
    """Finger position follows the muscle activation continuously.

        position = open + shape(activation) * (closed - open)

    deadband : activation below this is treated as rest (no motion), the rest
               of the range is rescaled to 0..1
    gamma    : shaping exponent. 1 = linear, >1 = finer control at low effort,
               <1 = more sensitive at low effort
    smooth_s : time constant (s) of a low-pass filter on the shaped activation;
               0 = none. It averages out the ripple of a held grip
    step     : hysteresis on the target, as a share of open-closed (0..1): the
               target only moves when the filtered activation has changed by at
               least this much since the last move, so a held grip stands still.
               When the activation drops to rest the target goes fully open
    The closing speed is limited by max_speed.
    """

    def __init__(self, *args, deadband=0.1, gamma=1.0, smooth_s=0.0, step=0.0, **kwargs):
        super().__init__(*args, **kwargs)
        self.deadband = deadband
        self.gamma = gamma
        self.smooth_s = smooth_s
        self.step_size = step
        self.smoothed = None   # the filtered activation (0..1)
        self.level = None      # the activation the target is set to now (0..1)
        self._t_prev = None

    def compute_target(self, activation, now):
        x = float(np.clip((activation - self.deadband) / (1.0 - self.deadband), 0.0, 1.0))
        x = x ** self.gamma
        dt = self.period if self._t_prev is None else min(max(now - self._t_prev, 0.0), 0.1)
        self._t_prev = now
        if self.smoothed is None or self.smooth_s <= 0:
            self.smoothed = x
        else:
            self.smoothed += (1.0 - math.exp(-dt / self.smooth_s)) * (x - self.smoothed)
        if self.level is None or abs(self.smoothed - self.level) >= self.step_size:
            self.level = self.smoothed
        elif x == 0.0 and self.smoothed < self.step_size:
            self.level = 0.0   # at rest: all the way open, not stuck just short of it
        return self.open_pos + self.level * (self.closed_pos - self.open_pos)


class ThresholdEMGController(EMGPositionController):
    """Open/closed state machine with hysteresis.

    mode "hold"  : closed while the muscle is active, open when relaxed.
                   activation >= on_threshold for on_dwell_s  -> CLOSED
                   activation <= off_threshold for off_dwell_s -> OPEN
    mode "toggle": each contraction flips the state (contract to close, relax,
                   contract again to open), so the grasp is held without
                   effort. A new contraction is only accepted after the muscle
                   has relaxed below off_threshold.

    on_threshold > off_threshold gives hysteresis, the dwell times reject spikes.
    """

    OPEN = "OPEN"
    CLOSED = "CLOSED"

    def __init__(self, *args, on_threshold=0.5, off_threshold=0.3, on_dwell_s=0.1,
                 off_dwell_s=0.2, mode="hold", **kwargs):
        super().__init__(*args, **kwargs)
        assert on_threshold > off_threshold, "on_threshold must be above off_threshold"
        assert mode in ("hold", "toggle")
        self.on_threshold = on_threshold
        self.off_threshold = off_threshold
        self.on_dwell_s = on_dwell_s
        self.off_dwell_s = off_dwell_s
        self.mode = mode
        self.state = self.OPEN
        self._above_since = None
        self._below_since = None
        self._armed = True            # toggle mode: True once the muscle relaxed again

    def _held(self, since, condition, now, dwell):
        """Return (new_since, satisfied) for 'condition true continuously for dwell s'."""
        if not condition:
            return None, False
        since = now if since is None else since
        return since, (now - since) >= dwell

    def compute_target(self, activation, now):
        self._above_since, above = self._held(
            self._above_since, activation >= self.on_threshold, now, self.on_dwell_s)
        self._below_since, below = self._held(
            self._below_since, activation <= self.off_threshold, now, self.off_dwell_s)

        if self.mode == "hold":
            if self.state == self.OPEN and above:
                self.state = self.CLOSED
            elif self.state == self.CLOSED and below:
                self.state = self.OPEN
        else:  # toggle
            if below:
                self._armed = True
            if above and self._armed:
                self.state = self.CLOSED if self.state == self.OPEN else self.OPEN
                self._armed = False
        return self.closed_pos if self.state == self.CLOSED else self.open_pos

    def status(self):
        return f"state={self.state}"


class VelocityEMGController(EMGPositionController):
    """Velocity (rate) control: the activation sets how fast the finger moves.

        activation >= close_above for close_time_s -> close at shape(activation) * close_speed
        activation <= open_below for open_time_s   -> open at open_speed
        in between, or not long enough yet         -> stand still
        hold() is True                             -> stand still (e.g. tremor detected)

    The band between open_below and close_above is the hysteresis: the ripple of
    a held grip never turns closing into opening, and a grip in the band holds
    the finger where it is. shape(a) = ((a - open_below) / (1 - open_below))^gamma.

    The integrated position is sent as a goal position, so the motor stays in
    (extended) position mode and the finger cannot run past open or closed.

    close_speed, open_speed : ticks/s at full activation and when opening
    hold                    : optional callable; True keeps the finger where it is
    """

    def __init__(self, *args, close_speed=1000.0, open_speed=1000.0, close_above=0.30,
                 open_below=0.15, close_time_s=0.10, open_time_s=0.30, gamma=1.0,
                 hold=None, **kwargs):
        assert close_above > open_below, "close_above must be above open_below"
        kwargs.setdefault("max_speed", max(close_speed, open_speed))
        super().__init__(*args, **kwargs)
        self.close_speed = close_speed
        self.open_speed = open_speed
        self.close_above = close_above
        self.open_below = open_below
        self.close_time_s = close_time_s
        self.open_time_s = open_time_s
        self.gamma = gamma
        self.hold = hold
        self.position = None   # the integrated finger position (ticks)
        self.speed = 0.0       # last speed toward closed (ticks/s, negative = opening)
        self._t_prev = None
        self._above_since = None
        self._below_since = None

    def compute_target(self, activation, now):
        if self.position is None:
            self.position = self.command
        dt = self.period if self._t_prev is None else min(max(now - self._t_prev, 0.0), 0.1)
        self._t_prev = now
        self._above_since = (self._above_since if self._above_since is not None else now) \
            if activation >= self.close_above else None
        self._below_since = (self._below_since if self._below_since is not None else now) \
            if activation <= self.open_below else None
        x = np.clip((activation - self.open_below) / (1.0 - self.open_below), 0.0, 1.0)
        if self.hold is not None and self.hold():
            self.speed = 0.0
        elif self._above_since is not None and now - self._above_since >= self.close_time_s:
            self.speed = (x ** self.gamma) * self.close_speed
        elif self._below_since is not None and now - self._below_since >= self.open_time_s:
            self.speed = -self.open_speed
        else:
            self.speed = 0.0
        direction = 1.0 if self.closed_pos > self.open_pos else -1.0
        self.position = float(np.clip(self.position + direction * self.speed * dt,
                                      self.lo, self.hi))
        return self.position

    def status(self):
        return f"speed={self.speed:7.1f}"
