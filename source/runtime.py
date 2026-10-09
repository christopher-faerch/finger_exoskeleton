"""
Real-time pipeline:

  EMG source -> StreamingProcessor -> ActivationEstimator -> ExoskeletonController -> Actuator
     (acquisition thread)  |  bounded queue  |          (control thread, fixed rate)

- The acquisition thread only reads blocks and puts them in a bounded queue.
  When the queue is full the oldest block is dropped (and counted), so memory
  stays bounded and the controller always works on fresh data.
- The control thread runs at runtime.control_rate_hz: it drains the queue,
  processes the blocks, evaluates safety and sends one command per tick.
- A watchdog thread checks the actuator's command watchdog and whether the
  control thread is still alive.

ControlCore holds all per-tick logic without threads, so tests can drive it
deterministically (run_synchronously).
"""

import queue
import threading
import time
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from .activation import ActivationEstimate, ActivationEstimator, Calibration, Calibrator
from .actuator import Actuator, ActuatorCommand
from .config import AppConfig
from .controller import ControllerOutput, ControllerState, ExoskeletonController
from .safety import Fault, HealthInputs, SafetySupervisor
from .sensor import EMGBlock, EMGSource, SourceConnectionError, SourceError, SourceTimeoutError
from .streaming import ProcessedBlock, StreamingProcessor

FloatArray = NDArray[np.float64]
MONITOR_RATE_HZ = 500.0  # raw/filtered EMG is decimated to about this rate for display


class RollingStat:
    """Last value, mean and max over the most recent `size` values."""

    def __init__(self, size: int = 500) -> None:
        self.values: deque[float] = deque(maxlen=size)

    def add(self, value: float) -> None:
        """Add one value."""
        self.values.append(value)

    def summary(self) -> tuple[float, float, float]:
        """(last, mean, max); zeros when empty."""
        if not self.values:
            return 0.0, 0.0, 0.0
        return self.values[-1], float(np.mean(self.values)), max(self.values)


class Diagnostics:
    """Thread-safe timing statistics and counters."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._stats: dict[str, RollingStat] = {}
        self._counters: dict[str, int] = {}

    def time(self, name: str, seconds: float) -> None:
        """Record a duration."""
        with self._lock:
            self._stats.setdefault(name, RollingStat()).add(seconds)

    def count(self, name: str, increment: int = 1) -> None:
        """Increase a counter."""
        with self._lock:
            self._counters[name] = self._counters.get(name, 0) + increment

    def snapshot(self) -> tuple[dict[str, tuple[float, float, float]], dict[str, int]]:
        """(timing summaries, counters)."""
        with self._lock:
            return (
                {name: stat.summary() for name, stat in self._stats.items()},
                dict(self._counters),
            )


class RingBuffer:
    """Fixed-size history of (n_rows, n) data; oldest values are overwritten."""

    def __init__(self, n_rows: int, capacity: int) -> None:
        self.data = np.zeros((n_rows, capacity))
        self.capacity = capacity
        self._position = 0
        self._filled = 0

    def write(self, values: FloatArray) -> None:
        """Append columns."""
        values = values[:, -self.capacity :]
        n = values.shape[1]
        end = self._position + n
        if end <= self.capacity:
            self.data[:, self._position : end] = values
        else:
            first = self.capacity - self._position
            self.data[:, self._position :] = values[:, :first]
            self.data[:, : n - first] = values[:, first:]
        self._position = end % self.capacity
        self._filled = min(self.capacity, self._filled + n)

    def read(self) -> FloatArray:
        """All stored columns, oldest first."""
        if self._filled < self.capacity:
            return self.data[:, : self._filled].copy()
        return np.concatenate(
            [self.data[:, self._position :], self.data[:, : self._position]], axis=1
        )

    def clear(self) -> None:
        """Remove everything."""
        self._position = 0
        self._filled = 0


@dataclass(frozen=True)
class MonitorSnapshot:  # pylint: disable=too-many-instance-attributes
    """Everything the monitor displays, copied under a lock."""

    channel_names: tuple[str, ...]
    sample_rate: float
    emg_time: FloatArray
    raw: FloatArray
    filtered: FloatArray
    activation: FloatArray
    control_time: FloatArray
    grasp: FloatArray
    tremor: FloatArray
    requested_stiffness: FloatArray
    applied_stiffness: FloatArray
    state: str
    armed: bool
    faults: tuple[str, ...]
    estop: bool
    estop_reason: str
    connected: bool
    calibration_phase: str
    calibration_error: str | None
    last_transition: str
    source_info: dict[str, str]
    timings: dict[str, tuple[float, float, float]]
    counters: dict[str, int]
    messages: tuple[str, ...]
    thresholds: dict[str, float] = field(default_factory=dict)


class ControlCore:  # pylint: disable=too-many-instance-attributes
    """Processing, calibration, safety, control and actuation for one tick at a time."""

    def __init__(
        self,
        config: AppConfig,
        sample_rate: float,
        actuator: Actuator,
        calibration: Calibration | None = None,
        calibration_save_path: Path | None = None,
    ) -> None:
        if sample_rate <= 0:
            raise ValueError("The sample rate must be known before the control core starts")
        self.config = config
        self.sample_rate = sample_rate
        self.actuator = actuator
        self.calibration_save_path = calibration_save_path
        names = config.channel_names
        self.processor = StreamingProcessor(config.processing, len(names), sample_rate)
        self.calibrator = Calibrator(config.calibration, names, sample_rate)
        self.estimator: ActivationEstimator | None = None
        if calibration is not None:
            self.set_calibration(calibration)
        self.supervisor = SafetySupervisor(config.safety)
        self.controller = ExoskeletonController(config.controller)
        self.diagnostics = Diagnostics()
        self.activation: ActivationEstimate | None = None
        self.last_output: ControllerOutput | None = None
        self.last_data_time: float | None = None
        self.messages: deque[str] = deque(maxlen=8)
        self._invalid_data = False
        self._reported_calibration_error: str | None = None
        self._expected_index: int | None = None
        self._sequence = 0
        self._lock = threading.Lock()

        self._decimation = max(1, round(sample_rate / MONITOR_RATE_HZ))
        n_monitor = int(config.runtime.monitor_seconds * sample_rate / self._decimation)
        n_control = int(config.runtime.monitor_seconds * config.runtime.control_rate_hz)
        self._emg_time = RingBuffer(1, n_monitor)
        self._raw = RingBuffer(len(names), n_monitor)
        self._filtered = RingBuffer(len(names), n_monitor)
        self._activation_history = RingBuffer(len(names), n_monitor)
        self._control_history = RingBuffer(5, max(2, n_control))

    # -- calibration ---------------------------------------------------------

    @property
    def calibrated(self) -> bool:
        """True when activation can be estimated."""
        return self.estimator is not None

    def set_calibration(self, calibration: Calibration) -> None:
        """Use a calibration (validated against the configuration)."""
        if calibration.channel_names != self.config.channel_names:
            raise ValueError(
                f"Calibration is for channels {calibration.channel_names}, "
                f"configuration has {self.config.channel_names}"
            )
        calibration.validate(self.config.calibration.min_reference_ratio)
        self.estimator = ActivationEstimator(calibration, self.config.channel_roles)
        self.calibrator.result = calibration

    def recalibrate(self) -> None:
        """Discard the calibration and record a new one; disarms (NOT_CALIBRATED fault)."""
        self.estimator = None
        self.activation = None
        self.calibrator.restart()
        self.message("Recalibration started: rest, then reference contraction")

    def message(self, text: str) -> None:
        """Add a line to the monitor's message log."""
        with self._lock:
            self.messages.append(text)

    # -- data path -----------------------------------------------------------

    def reset_session(self) -> None:
        """New acquisition session (reconnection): reset filters and sample tracking."""
        self.processor.reset()
        self._expected_index = None
        self.message("New acquisition session: filters reset")

    def handle_block(self, block: EMGBlock, now: float) -> None:
        """Process one block from the source."""
        started = time.perf_counter()
        samples = block.samples
        if not np.all(np.isfinite(samples)):
            # Never let NaN/inf into the filter state; it would poison every later output
            self.diagnostics.count("invalid blocks")
            self._invalid_data = True
            samples = np.nan_to_num(samples, nan=0.0, posinf=0.0, neginf=0.0)
            block = EMGBlock(samples, block.first_sample_index, block.receive_time)
        else:
            self._invalid_data = False

        if self._expected_index is not None and block.first_sample_index != self._expected_index:
            missing = block.first_sample_index - self._expected_index
            if missing > 0:
                self.diagnostics.count("missing samples", missing)
                self.diagnostics.count("gaps")
            else:
                self.diagnostics.count("index resets")
        self._expected_index = block.first_sample_index + block.n_samples
        self.last_data_time = (
            block.receive_time
            if self.last_data_time is None
            else max(self.last_data_time, block.receive_time)
        )

        processed = self.processor.process(block)
        if self.estimator is None:
            self.calibrator.update(processed)
            self._finish_calibration()
        activation_trace = None
        if self.estimator is not None:
            self.activation = self.estimator.estimate(processed)
            activation_trace = self.estimator.normalize(processed)[0]

        self._record_emg(processed, activation_trace)
        self.diagnostics.time("processing per block", time.perf_counter() - started)
        self.diagnostics.time("block age at processing", max(0.0, now - block.receive_time))
        self.diagnostics.count("blocks processed")

    def _finish_calibration(self) -> None:
        if self.calibrator.result is not None:
            self.estimator = ActivationEstimator(self.calibrator.result, self.config.channel_roles)
            self.message("Calibration complete")
            if self.calibration_save_path is not None:
                self.calibrator.result.save(self.calibration_save_path)
                self.message(f"Calibration saved to {self.calibration_save_path}")
        elif (
            self.calibrator.error is not None
            and self._reported_calibration_error != self.calibrator.error
        ):
            self._reported_calibration_error = self.calibrator.error
            self.message(f"Calibration failed: {self.calibrator.error} Press 'c' to repeat it.")

    def _record_emg(self, processed: ProcessedBlock, activation: FloatArray | None) -> None:
        indices = processed.first_sample_index + np.arange(processed.n_samples)
        keep = indices % self._decimation == 0
        if not np.any(keep):
            return
        with self._lock:
            self._emg_time.write((indices[keep] / self.sample_rate)[None, :])
            self._raw.write(processed.raw[:, keep])
            self._filtered.write(processed.filtered[:, keep])
            if activation is None:
                activation = np.zeros_like(processed.raw)
            self._activation_history.write(activation[:, keep])

    # -- control path --------------------------------------------------------

    def faults(
        self, now: float, source_connected: bool, acquisition_running: bool
    ) -> tuple[str, ...]:
        """Active faults right now."""
        actuator_status = self.actuator.status(now)
        return self.supervisor.evaluate(
            HealthInputs(
                now=now,
                last_data_time=self.last_data_time,
                source_connected=source_connected,
                acquisition_running=acquisition_running,
                invalid_data=self._invalid_data,
                calibrated=self.calibrated,
                actuator_latched=actuator_status.estop or actuator_status.watchdog_tripped,
            )
        )

    def arm(
        self, now: float, source_connected: bool, acquisition_running: bool
    ) -> tuple[bool, str]:
        """Operator arm request; refused while any fault (except an actuator latch) is active."""
        blockers = self.supervisor.arm_blockers(
            self.faults(now, source_connected, acquisition_running)
        )
        if blockers:
            message = f"Arm refused: {', '.join(blockers)}"
            self.message(message)
            return False, message
        self.actuator.reset()
        self.controller.arm()
        self.message("Armed by operator")
        return True, "armed"

    def tick(
        self, control_time: float, now: float, source_connected: bool, acquisition_running: bool
    ) -> ControllerOutput:
        """
        One control step. control_time drives the controller's timers (monotonic
        time in real time, stream time in run_synchronously); now is monotonic
        time for data freshness.
        """
        started = time.perf_counter()
        faults = self.faults(now, source_connected, acquisition_running)
        output = self.controller.update(self.activation, faults, control_time)
        self._sequence += 1
        applied = self.actuator.send(
            ActuatorCommand(
                sequence=self._sequence,
                timestamp=now,
                requested_stiffness=output.stiffness,
                state=output.state.value,
                armed=output.armed,
                faults=output.faults,
            )
        )
        self.last_output = output
        grasp = self.activation.grasp_activation if self.activation else 0.0
        tremor = self.activation.tremor_level if self.activation else 0.0
        with self._lock:
            self._control_history.write(
                np.array([[control_time], [grasp], [tremor], [output.stiffness], [applied]])
            )
        self.diagnostics.time("controller tick", time.perf_counter() - started)
        return output

    def safe_shutdown(self, now: float, reason: str) -> None:
        """Disarm and send a final zero command."""
        self.controller.disarm(reason)
        self._sequence += 1
        self.actuator.send(
            ActuatorCommand(
                sequence=self._sequence,
                timestamp=now,
                requested_stiffness=0.0,
                state=ControllerState.DISARMED.value,
                armed=False,
                faults=(reason,),
            )
        )

    def snapshot(
        self, source: EMGSource, connected: bool, counters: dict[str, int] | None = None
    ) -> MonitorSnapshot:
        """Copy everything the monitor shows."""
        timings, own_counters = self.diagnostics.snapshot()
        if counters:
            own_counters.update(counters)
        output = self.last_output
        controller = self.config.controller
        with self._lock:
            control = self._control_history.read()
            return MonitorSnapshot(
                channel_names=self.config.channel_names,
                sample_rate=self.sample_rate,
                emg_time=self._emg_time.read()[0],
                raw=self._raw.read(),
                filtered=self._filtered.read(),
                activation=self._activation_history.read(),
                control_time=control[0],
                grasp=control[1],
                tremor=control[2],
                requested_stiffness=control[3],
                applied_stiffness=control[4],
                state=output.state.value if output else self.controller.state.value,
                armed=output.armed if output else False,
                faults=output.faults if output else (),
                estop=self.supervisor.estop_active,
                estop_reason=self.supervisor.estop_reason,
                connected=connected,
                calibration_phase=self.calibrator.phase,
                calibration_error=self.calibrator.error,
                last_transition=self.controller.last_transition,
                source_info=source.info(),
                timings=timings,
                counters=own_counters,
                messages=tuple(self.messages),
                thresholds={
                    "grasp on": controller.grasp_on_threshold,
                    "grasp off": controller.grasp_off_threshold,
                    "tremor on": controller.tremor_on_threshold,
                    "tremor off": controller.tremor_off_threshold,
                },
            )


class Pipeline:  # pylint: disable=too-many-instance-attributes
    """Runs a ControlCore with acquisition, control and watchdog threads."""

    def __init__(
        self,
        config: AppConfig,
        source: EMGSource,
        actuator: Actuator,
        calibration: Calibration | None = None,
        calibration_save_path: Path | None = None,
        auto_arm: bool = False,
    ) -> None:
        self.config = config
        self.source = source
        self.actuator = actuator
        self.auto_arm = auto_arm
        self._calibration = calibration
        self._calibration_save_path = calibration_save_path
        self.core: ControlCore | None = None
        self.error: str | None = None
        self._queue: queue.Queue[tuple[int, EMGBlock]] = queue.Queue(
            config.runtime.queue_size_blocks
        )
        self._requests: queue.Queue[Callable[[], None]] = queue.Queue()
        self._stop = threading.Event()
        self._session = 0
        self._connected = False
        self._acquisition_done = False
        self._auto_armed = False
        self._threads: dict[str, threading.Thread] = {}

    # -- lifecycle -----------------------------------------------------------

    def start(self) -> None:
        """Start the source (raises SourceError if it cannot) and all threads."""
        self.source.start()
        self._connected = True
        self.core = ControlCore(
            self.config,
            self.source.sample_rate,
            self.actuator,
            self._calibration,
            self._calibration_save_path,
        )
        self.core.message(f"Source started: {self.source.kind} at {self.source.sample_rate:g} Hz")
        for name, target in (
            ("acquisition", self._acquisition_loop),
            ("control", self._control_loop),
            ("watchdog", self._watchdog_loop),
        ):
            thread = threading.Thread(target=target, name=name, daemon=True)
            self._threads[name] = thread
            thread.start()

    def stop(self) -> None:
        """Stop everything and leave the actuator released. Safe to call twice."""
        self._stop.set()
        for name in ("control", "acquisition", "watchdog"):
            thread = self._threads.get(name)
            if thread is not None and thread is not threading.current_thread():
                thread.join(timeout=5.0)
        if self.core is not None and self.core.controller.state != ControllerState.DISARMED:
            self.core.safe_shutdown(time.monotonic(), "shutdown")
        self.actuator.close()
        try:
            self.source.stop()
        except SourceError as error:
            self.error = self.error or f"Source stop failed: {error}"
        self._connected = False

    @property
    def running(self) -> bool:
        """True while the control thread is alive."""
        thread = self._threads.get("control")
        return thread is not None and thread.is_alive()

    def run_for(
        self, seconds: float, on_tick: Callable[[], None] | None = None, period_s: float = 0.5
    ) -> None:
        """Block for `seconds` (or until the control loop ends), calling on_tick periodically."""
        deadline = time.monotonic() + seconds
        while time.monotonic() < deadline and self.running and not self._stop.is_set():
            time.sleep(min(period_s, max(0.0, deadline - time.monotonic())))
            if on_tick is not None:
                on_tick()

    # -- operator actions (thread-safe) --------------------------------------

    def _request(self, action: Callable[[], None]) -> None:
        self._requests.put(action)

    def arm(self) -> None:
        """Ask the control thread to arm (refused while faults are active)."""
        self._request(self._do_arm)

    def _do_arm(self) -> None:
        assert self.core is not None
        self.core.arm(time.monotonic(), self._connected, self._acquisition_alive())

    def trigger_estop(self, reason: str = "operator") -> None:
        """Emergency stop: the actuator releases immediately, the controller faults next tick."""
        self.actuator.emergency_stop(reason)
        if self.core is not None:
            self.core.supervisor.trigger_estop(reason)
            self.core.message(f"EMERGENCY STOP: {reason}")

    def reset_estop(self) -> None:
        """Clear the e-stop latch. The system stays disarmed until arm()."""
        if self.core is not None:
            self.core.supervisor.reset_estop()
            self.core.message("E-stop reset; arm again to resume")

    def recalibrate(self) -> None:
        """Ask the control thread to start a new calibration."""
        assert self.core is not None
        self._request(self.core.recalibrate)

    def snapshot(self) -> "MonitorSnapshot":
        """Monitor data."""
        assert self.core is not None
        counters = {"queue depth": self._queue.qsize(), "session": self._session}
        return self.core.snapshot(self.source, self._connected, counters)

    # -- threads -------------------------------------------------------------

    def _acquisition_alive(self) -> bool:
        thread = self._threads.get("acquisition")
        return thread is not None and thread.is_alive() and not self._acquisition_done

    def _put(self, block: EMGBlock) -> None:
        assert self.core is not None
        try:
            self._queue.put_nowait((self._session, block))
        except queue.Full:
            try:
                self._queue.get_nowait()
                self.core.diagnostics.count("dropped blocks")
            except queue.Empty:
                pass
            self._queue.put_nowait((self._session, block))

    def _acquisition_loop(self) -> None:
        assert self.core is not None
        core = self.core
        last_receive: float | None = None
        try:
            while not self._stop.is_set():
                started = time.perf_counter()
                try:
                    block = self.source.read_block()
                except SourceTimeoutError as error:
                    core.diagnostics.count("acquisition timeouts")
                    core.message(f"Acquisition timeout: {error}")
                    continue
                except SourceConnectionError as error:
                    self._connected = False
                    core.message(f"Connection lost: {error}")
                    if not self._reconnect():
                        return
                    continue
                if block is None:
                    core.message("Source finished (end of recording)")
                    return
                core.diagnostics.time("acquisition read", time.perf_counter() - started)
                if last_receive is not None:
                    core.diagnostics.time("block interval", block.receive_time - last_receive)
                last_receive = block.receive_time
                self._put(block)
        except Exception as error:  # pylint: disable=broad-exception-caught
            # Any unexpected error ends acquisition; the control loop sees it as a fault
            core.message(f"Acquisition crashed: {error!r}")
            self.error = self.error or f"Acquisition crashed: {error!r}"
        finally:
            self._acquisition_done = True

    def _reconnect(self) -> bool:
        assert self.core is not None
        safety = self.config.safety
        for attempt in range(1, safety.max_reconnect_attempts + 1):
            if self._stop.wait(safety.reconnect_delay_s):
                return False
            try:
                self.source.stop()
                rate = self.source.sample_rate
                self.source.start()
            except SourceError as error:
                self.core.message(f"Reconnect attempt {attempt} failed: {error}")
                continue
            if self.source.sample_rate != rate:
                self.core.message("Sample rate changed on reconnection; giving up")
                return False
            self._session += 1
            self._connected = True
            self.core.diagnostics.count("reconnections")
            self.core.message(f"Reconnected (attempt {attempt}); re-arm required")
            return True
        self.core.message("Reconnection failed; acquisition stopped")
        return False

    def _control_loop(self) -> None:
        assert self.core is not None
        core = self.core
        period = 1.0 / self.config.runtime.control_rate_hz
        session = self._session
        next_tick = time.monotonic()
        last_tick: float | None = None
        try:
            while not self._stop.is_set():
                now = time.monotonic()
                if last_tick is not None:
                    core.diagnostics.time("control period", now - last_tick)
                last_tick = now
                while True:
                    try:
                        self._requests.get_nowait()()
                    except queue.Empty:
                        break
                while True:
                    try:
                        block_session, block = self._queue.get_nowait()
                    except queue.Empty:
                        break
                    if block_session != session:
                        session = block_session
                        core.reset_session()
                    core.handle_block(block, time.monotonic())
                now = time.monotonic()
                core.tick(now, now, self._connected, self._acquisition_alive())
                self._maybe_auto_arm(now)

                next_tick += period
                delay = next_tick - time.monotonic()
                if delay > 0:
                    time.sleep(delay)
                else:
                    core.diagnostics.count("control overruns")
                    next_tick = time.monotonic()
        except Exception as error:  # pylint: disable=broad-exception-caught
            # Fail safe: release the actuator and latch an emergency stop
            self.error = f"Control loop crashed: {error!r}"
            self.trigger_estop(self.error)
        finally:
            if not self._stop.is_set() and self.error is None:
                self.error = "Control loop ended unexpectedly"

    def _maybe_auto_arm(self, now: float) -> None:
        """Arm once after the first successful calibration (demo / headless use only)."""
        assert self.core is not None
        if self.auto_arm and not self._auto_armed and self.core.calibrated:
            armed, _ = self.core.arm(now, self._connected, self._acquisition_alive())
            self._auto_armed = armed

    def _watchdog_loop(self) -> None:
        while not self._stop.wait(0.005):
            self.actuator.check_watchdog(time.monotonic())
            control = self._threads.get("control")
            if control is not None and not control.is_alive() and not self._stop.is_set():
                self.actuator.emergency_stop("control thread terminated")
                return


@dataclass
class SyncResult:
    """Outputs of run_synchronously, one entry per tick."""

    times: list[float] = field(default_factory=list)
    states: list[ControllerState] = field(default_factory=list)
    stiffness: list[float] = field(default_factory=list)
    grasp: list[float] = field(default_factory=list)
    tremor: list[float] = field(default_factory=list)
    faults: list[tuple[str, ...]] = field(default_factory=list)


def run_synchronously(
    core: ControlCore,
    source: EMGSource,
    duration_s: float,
    on_tick: Callable[[ControlCore, float], None] | None = None,
    arm_when_calibrated: bool = True,
) -> SyncResult:
    """
    Run the full pipeline without threads, as fast as possible, one control
    tick per block, with the controller timed in stream time. Used by tests
    and for offline evaluation of recordings. The source must already be started.
    """
    result = SyncResult()
    connected = True
    while True:
        try:
            block = source.read_block()
        except SourceConnectionError:
            connected = False
            block = None
        if block is not None:
            core.handle_block(block, time.monotonic())
        stream_time = (
            (block.first_sample_index + block.n_samples) / core.sample_rate
            if block is not None
            else (result.times[-1] if result.times else 0.0)
        )
        if (
            arm_when_calibrated
            and core.calibrated
            and core.controller.state == ControllerState.DISARMED
        ):
            core.arm(time.monotonic(), connected, True)
        output = core.tick(stream_time, time.monotonic(), connected, block is not None)
        result.times.append(stream_time)
        result.states.append(output.state)
        result.stiffness.append(output.stiffness)
        result.grasp.append(core.activation.grasp_activation if core.activation else 0.0)
        result.tremor.append(core.activation.tremor_level if core.activation else 0.0)
        result.faults.append(output.faults)
        if on_tick is not None:
            on_tick(core, stream_time)
        if block is None or stream_time >= duration_s:
            return result


__all__ = [
    "ControlCore",
    "Diagnostics",
    "MonitorSnapshot",
    "Pipeline",
    "RingBuffer",
    "SyncResult",
    "run_synchronously",
    "Fault",
]
