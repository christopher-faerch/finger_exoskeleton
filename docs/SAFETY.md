# Safety requirements before physical actuation

**Status: physical actuation is NOT implemented and NOT allowed.** The only actuator
backend is `DryRunActuator`, which logs commands and moves nothing. The configuration
loader rejects any other `actuator.backend`. No physical actuator protocol is documented
for this project, so no physical backend exists.

This document lists what must exist and be verified before the first energized test.

## What the software already does (dry run)

| Safeguard | Where | Behavior |
|---|---|---|
| Safe start-up | `controller.py` (`DISARMED`) | Starts disarmed with stiffness 0. Nothing is commanded until the operator arms. |
| Explicit arming | `runtime.py` `ControlCore.arm` | Arming is refused while any fault is active (no data, stale data, not calibrated, e-stop, ...). |
| Emergency stop | `Pipeline.trigger_estop`, key `e` | Actuator releases immediately and latches. The controller faults on the next tick. Clearing needs `r` (reset), then `a` (arm). |
| Fault latch | `controller.py` (`FAULT`) | Any fault while armed: stiffness 0 immediately (no ramp), state FAULT. It stays there after the fault clears until re-armed. |
| Data freshness | `safety.py` `STALE_DATA` | No new EMG for `stale_data_timeout_s` (0.15 s): fault. |
| Connection loss | `SOURCE_DISCONNECTED`, `ACQUISITION_STOPPED` | Fault, then reconnection attempts. After reconnecting, filters are reset and re-arming is required. |
| Invalid data | `INVALID_DATA` | Non-finite samples raise a fault and are zeroed, so they cannot corrupt the filter state. |
| Command saturation | `controller.py`, `actuator.py` | The command is clamped to [0, `max_stiffness`] <= 1 and rate-limited. NaN becomes 0. The actuator saturates again and counts invalid requests. |
| Software watchdog | `DryRunActuator.check_watchdog` | No command for `actuator_watchdog_timeout_s` (0.1 s): release and latch. A separate thread also e-stops if the control thread dies. |
| Safe shutdown | `Pipeline.stop` | Final zero command, actuator closed, source stopped. Ctrl+C and window close take the same path. |

## Why this is not enough

Everything above runs in one Python process on a desktop operating system. It does not
protect against:

- the process hanging or being killed (the watchdog thread dies with it)
- Windows scheduling stalls, USB or driver faults
- a bug in the actuator driver or firmware
- a mechanical fault

**A Python software watchdog is not a safety function.**

## Required before enabling a physical actuator

1. **Actuator controller with its own firmware watchdog.** The microcontroller (or drive)
   releases the actuator to its passive state if no valid command arrives within a fixed
   time, for example 50-100 ms. It must not rely on the PC.
2. **Command protocol with integrity checks.** Each command carries a sequence number and
   a checksum. The firmware rejects stale, out-of-order, corrupt or out-of-range commands,
   and enforces its own stiffness, torque, velocity and position limits independently of the PC.
3. **Hardware emergency stop.** A physical button that removes actuator power directly,
   in hardware (a power-cutting relay or contactor), not through software. It must be
   reachable by both the wearer and the operator.
4. **Fail-safe actuator choice.** The design document (sections 6.2, 6.6, 8.2) recommends
   semi-active elements (MR brake or damper, jamming, clutch). These can only dissipate
   energy and are transparent when unpowered. Power loss must leave the finger free.
   If a motor is ever used instead, current limits plus mechanical torque limits are mandatory.
5. **Mechanical end-stops** inside the anatomical range of every joint (design doc 8.5).
6. **Force and torque limits.** Torque capability must be sized to damp tremor, never to
   overpower voluntary motion (design doc 6.6: about 0.1 Nm-class caps in comparable
   finger exoskeletons).
7. **Start-up and reconnection interlock in firmware.** The firmware powers up passive and
   only accepts commands after an explicit arm message, both after power-up and after any
   communication loss.
8. **Verification**, in this order:
   - Bench test without a hand. Test every fault: kill the Python process, unplug USB,
     close the TCU, remove the sensor, press the e-stop, send out-of-range commands.
     Check the actuator releases within the specified time each time.
   - Measure end-to-end latency from EMG to actuator, with a trigger or oscilloscope.
   - Test on healthy volunteers with simulated tremor (design doc 9.3), under ethics
     approval (design doc 9.4).
   - Patients only with clinical partners.

## Implementing the physical backend

Subclass `Actuator` in `source/actuator.py`. `send()` must forward the already-bounded
command and return what was applied. `emergency_stop()` must command release immediately.
`check_watchdog()` must report the firmware watchdog state. Then allow the new backend in
`config.py` `_validate` and `create_actuator`. Keep `dry_run` as the default.
