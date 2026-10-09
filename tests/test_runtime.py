"""End-to-end pipeline: synchronous and threaded runs, safety behavior, CLI and monitor."""

import time
from collections.abc import Callable
from dataclasses import replace
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pytest
from conftest import offline_synthetic

import run_exoskeleton
from source.activation import Calibration
from source.actuator import ActuatorCommand, DryRunActuator
from source.config import AppConfig
from source.controller import ControllerState
from source.mock_tcu import MockSensor, MockTCU, MockTCUSettings
from source.monitor import ConsoleMonitor, LiveMonitor, status_lines
from source.runtime import ControlCore, Pipeline, RingBuffer, run_synchronously
from source.safety import Fault
from source.sensor import EMGBlock, SyntheticSource
from source.trigno import TrignoSource


# Calibration 7 s, then the cycle: rest 2, grasp 4, grasp + tremor 6, subsides 2, rest 2,
# extension 3, rest 1, rest tremor 3, rest 1 (24 s)
CYCLE_START = 7.0


def labels_by_tick(source: SyntheticSource) -> Callable[[ControlCore, float], None]:
    """on_tick callback recording the scenario label of every tick."""
    labels: list[str] = []

    def record(_core: ControlCore, stream_time: float) -> None:
        labels.append(source.segment_at(max(0.0, stream_time - 1e-6)).label)

    record.labels = labels  # type: ignore[attr-defined]
    return record


def test_end_to_end_synthetic_session(config: AppConfig) -> None:
    """Calibrates, cycles IDLE -> GRASP_ARMED -> STIFFENED; stiffens only with grasp + tremor."""
    source = offline_synthetic(config)
    source.start()
    actuator = DryRunActuator(float("inf"))
    core = ControlCore(config, source.sample_rate, actuator)
    recorder = labels_by_tick(source)
    result = run_synchronously(core, source, CYCLE_START + 48.0, on_tick=recorder)
    labels: list[str] = recorder.labels  # type: ignore[attr-defined]

    assert core.calibrated
    visited = set(result.states)
    assert {ControllerState.IDLE, ControllerState.GRASP_ARMED, ControllerState.STIFFENED} <= visited
    assert ControllerState.FAULT not in visited

    by_label: dict[str, list[ControllerState]] = {}
    for label, state in zip(labels, result.states):
        by_label.setdefault(label, []).append(state)
    stiffened_share = by_label["grasp + tremor"].count(ControllerState.STIFFENED) / len(
        by_label["grasp + tremor"]
    )
    assert stiffened_share > 0.8
    for label in ("grasp", "wrist extension, no grasp", "rest tremor, no grasp"):
        assert ControllerState.STIFFENED not in by_label[label], label
    for label in ("wrist extension, no grasp", "rest tremor, no grasp"):
        assert ControllerState.GRASP_ARMED not in by_label[label], label

    # Every command was logged; stiffness was generated and stayed bounded
    assert len(actuator.records) == len(result.states)
    applied = [record.applied_stiffness for record in actuator.records]
    assert max(applied) > 0.5 and min(applied) >= 0.0 and max(applied) <= 1.0


def test_disconnection_fails_safe_and_needs_rearm(
    config: AppConfig, calibration: Calibration
) -> None:
    """Losing the source during stiffening zeroes the command and latches FAULT."""
    source = offline_synthetic(config)
    source.start()
    core = ControlCore(config, source.sample_rate, DryRunActuator(float("inf")), calibration)

    def disconnect_during_tremor(_core: ControlCore, stream_time: float) -> None:
        if stream_time >= CYCLE_START + 10.0:
            source.inject_disconnect()

    result = run_synchronously(core, source, 60.0, on_tick=disconnect_during_tremor)
    assert ControllerState.STIFFENED in result.states
    assert result.states[-1] == ControllerState.FAULT
    assert result.stiffness[-1] == 0.0
    assert Fault.SOURCE_DISCONNECTED.value in result.faults[-1]


def test_invalid_samples_raise_fault_without_poisoning_filters(
    config: AppConfig, calibration: Calibration
) -> None:
    """A NaN block faults the controller; processing recovers on the next valid block."""
    core = ControlCore(config, 2000.0, DryRunActuator(float("inf")), calibration)
    rng = np.random.default_rng(0)
    core.handle_block(
        EMGBlock(rng.standard_normal((4, 27)) * 0.01, 0, time.monotonic()), time.monotonic()
    )
    core.arm(time.monotonic(), True, True)
    bad = np.full((4, 27), np.nan)
    core.handle_block(EMGBlock(bad, 27, time.monotonic()), time.monotonic())
    output = core.tick(1.0, time.monotonic(), True, True)
    assert Fault.INVALID_DATA.value in output.faults and output.state == ControllerState.FAULT
    core.handle_block(
        EMGBlock(rng.standard_normal((4, 27)) * 0.01, 54, time.monotonic()), time.monotonic()
    )
    assert core.activation is not None and np.isfinite(core.activation.grasp_activation)
    assert Fault.INVALID_DATA.value not in core.tick(1.1, time.monotonic(), True, True).faults


def test_missing_samples_are_counted(config: AppConfig, calibration: Calibration) -> None:
    """Gaps in the sample index are counted as missing samples."""
    source = offline_synthetic(config)
    source.start()
    core = ControlCore(config, source.sample_rate, DryRunActuator(float("inf")), calibration)

    def drop_once(_core: ControlCore, stream_time: float) -> None:
        if 1.0 <= stream_time < 1.02:
            source.inject_dropout(0.1)

    run_synchronously(core, source, 2.0, on_tick=drop_once)
    counters = core.diagnostics.snapshot()[1]
    assert counters["missing samples"] == 200 and counters["gaps"] == 1


def test_arm_refused_while_not_calibrated(config: AppConfig) -> None:
    """Arming before calibration is refused with the reason."""
    core = ControlCore(config, 2000.0, DryRunActuator(float("inf")))
    core.handle_block(EMGBlock(np.zeros((4, 27)), 0, time.monotonic()), time.monotonic())
    armed, message = core.arm(time.monotonic(), True, True)
    assert not armed and Fault.NOT_CALIBRATED.value in message
    assert core.controller.state == ControllerState.DISARMED


def test_ring_buffer_keeps_latest() -> None:
    """The monitor buffers keep the newest values in order and never grow."""
    buffer = RingBuffer(1, 5)
    buffer.write(np.array([[1.0, 2.0, 3.0]]))
    buffer.write(np.array([[4.0, 5.0, 6.0, 7.0]]))
    np.testing.assert_array_equal(buffer.read(), [[3.0, 4.0, 5.0, 6.0, 7.0]])
    buffer.write(np.arange(12.0)[None, :])
    np.testing.assert_array_equal(buffer.read(), [[7.0, 8.0, 9.0, 10.0, 11.0]])


# -- threaded pipeline ----------------------------------------------------------


def realtime_config(config: AppConfig) -> AppConfig:
    """Faster reconnection settings for tests."""
    return replace(
        config, safety=replace(config.safety, reconnect_delay_s=0.05, max_reconnect_attempts=1)
    )


def wait_for(condition: Callable[[], bool], timeout: float = 5.0) -> bool:
    """Poll until condition() is true."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if condition():
            return True
        time.sleep(0.01)
    return False


def arm_when_ready(pipeline: Pipeline) -> None:
    """Wait until data flows without faults (arming is refused before that), then arm."""
    assert wait_for(lambda: not pipeline.snapshot().faults)
    pipeline.arm()
    assert wait_for(lambda: pipeline.snapshot().armed)


def start_threaded(
    config: AppConfig, calibration: Calibration, source: SyntheticSource | None = None
) -> tuple[Pipeline, SyntheticSource, DryRunActuator]:
    """A running threaded pipeline on real-time synthetic EMG with a known calibration."""
    source = source or SyntheticSource(config.synthetic, config.channels)
    actuator = DryRunActuator(config.safety.actuator_watchdog_timeout_s)
    pipeline = Pipeline(realtime_config(config), source, actuator, calibration)
    pipeline.start()
    return pipeline, source, actuator


def test_threaded_safe_start_arm_estop_and_shutdown(
    config: AppConfig, calibration: Calibration
) -> None:
    """Starts disarmed, arms on request, e-stop releases and latches, shutdown releases."""
    pipeline, _, actuator = start_threaded(config, calibration)
    try:
        assert wait_for(lambda: len(actuator.records) > 30)
        assert all(
            not record.armed and record.applied_stiffness == 0.0 for record in actuator.records
        )
        arm_when_ready(pipeline)
        pipeline.trigger_estop("test")
        assert actuator.status(time.monotonic()).estop
        assert wait_for(lambda: pipeline.snapshot().state == ControllerState.FAULT.value)
        pipeline.arm()
        time.sleep(0.1)
        assert pipeline.snapshot().state == ControllerState.FAULT.value  # refused: e-stop active
        pipeline.reset_estop()
        pipeline.arm()
        assert wait_for(lambda: pipeline.snapshot().armed)
        timings = pipeline.snapshot().timings
        assert 1 / timings["control period"][1] == pytest.approx(
            config.runtime.control_rate_hz, rel=0.2
        )
    finally:
        pipeline.stop()
    assert not pipeline.running
    assert actuator.records[-1].note.startswith("closed")
    assert actuator.status(time.monotonic()).applied_stiffness == 0.0
    assert pipeline.error is None


def test_threaded_stale_data_faults(config: AppConfig, calibration: Calibration) -> None:
    """A data dropout longer than the stale timeout faults an armed controller."""
    pipeline, source, _ = start_threaded(config, calibration)
    try:
        arm_when_ready(pipeline)
        source.inject_dropout(0.5)
        assert wait_for(lambda: Fault.STALE_DATA.value in pipeline.snapshot().faults)
        assert wait_for(lambda: pipeline.snapshot().state == ControllerState.FAULT.value)
        assert wait_for(lambda: not pipeline.snapshot().faults)  # data flows again
        assert pipeline.snapshot().state == ControllerState.FAULT.value  # but stays latched
    finally:
        pipeline.stop()


def test_threaded_disconnect_reconnects_but_stays_safe(
    config: AppConfig, calibration: Calibration
) -> None:
    """After a disconnect the pipeline reconnects, resets filters and waits for re-arming."""
    pipeline, source, _ = start_threaded(config, calibration)
    try:
        arm_when_ready(pipeline)
        source.inject_disconnect()
        assert wait_for(lambda: pipeline.snapshot().state == ControllerState.FAULT.value)
        assert wait_for(lambda: pipeline.snapshot().counters.get("reconnections", 0) == 1)
        assert wait_for(lambda: not pipeline.snapshot().faults)
        assert pipeline.snapshot().state == ControllerState.FAULT.value
        pipeline.arm()
        assert wait_for(lambda: pipeline.snapshot().armed)
    finally:
        pipeline.stop()


class FailingActuator(DryRunActuator):
    """Raises inside send() after a number of commands, to simulate a crash in the control loop."""

    def __init__(self, fail_after: int) -> None:
        super().__init__(watchdog_timeout_s=0.05)
        self.fail_after = fail_after

    def send(self, command: ActuatorCommand) -> float:
        if command.sequence > self.fail_after:
            raise RuntimeError("driver failure")
        return super().send(command)


def test_control_loop_crash_triggers_emergency_stop(
    config: AppConfig, calibration: Calibration
) -> None:
    """An exception in the control loop ends it with an e-stop and an error report."""
    actuator = FailingActuator(fail_after=20)
    pipeline = Pipeline(
        config, SyntheticSource(config.synthetic, config.channels), actuator, calibration
    )
    pipeline.start()
    try:
        assert wait_for(lambda: not pipeline.running)
        assert wait_for(lambda: actuator.status(time.monotonic()).estop)
        assert pipeline.error is not None and "driver failure" in pipeline.error
    finally:
        pipeline.stop()


def test_recording_end_stops_safely(
    config: AppConfig, calibration: Calibration, tmp_path: Path
) -> None:
    """When a non-looping recording ends, the controller faults to zero stiffness."""
    from source.config import RecordedConfig  # pylint: disable=import-outside-toplevel
    from source.sensor import RecordedSource  # pylint: disable=import-outside-toplevel

    path = tmp_path / "short.csv"
    path.write_text("\n".join("0.001,0.002,0.003,0.004" for _ in range(400)), encoding="utf-8")
    source = RecordedSource(
        RecordedConfig(file=str(path), loop=False, realtime=True), config.channels
    )
    actuator = DryRunActuator(1.0)
    pipeline = Pipeline(config, source, actuator, calibration)
    pipeline.start()
    try:
        assert wait_for(lambda: Fault.ACQUISITION_STOPPED.value in pipeline.snapshot().faults)
        assert actuator.records[-1].applied_stiffness == 0.0
    finally:
        pipeline.stop()


def test_trigno_backend_end_to_end_with_mock_tcu(
    config: AppConfig, calibration: Calibration
) -> None:
    """The live Trigno backend feeds the same pipeline (against the mock TCU, not hardware)."""
    generator = offline_synthetic(config)
    settings = MockTCUSettings(
        sensors=[
            MockSensor(slot=c.trigno_slot, position=c.trigno_slot - 1) for c in config.channels
        ]
    )
    with MockTCU(settings, lambda first, n: generator.generate(first, n) / 1000.0) as tcu:
        trigno = replace(config.trigno, command_port=tcu.command_port, emg_port=tcu.emg_port)
        actuator = DryRunActuator(config.safety.actuator_watchdog_timeout_s)
        pipeline = Pipeline(config, TrignoSource(trigno, config.channels), actuator, calibration)
        pipeline.start()
        try:
            assert pipeline.core is not None and pipeline.core.sample_rate == 2000.0
            arm_when_ready(pipeline)
            assert wait_for(lambda: pipeline.snapshot().counters.get("blocks processed", 0) > 50)
            tcu.drop_data_connection()
            assert wait_for(lambda: pipeline.snapshot().state == ControllerState.FAULT.value)
        finally:
            pipeline.stop()
        assert tcu.received_commands[-1] == "QUIT"


# -- monitor and CLI ------------------------------------------------------------


def test_monitors_render(
    config: AppConfig, calibration: Calibration, capsys: pytest.CaptureFixture[str]
) -> None:
    """The live monitor draws a frame and the console monitor prints a status line."""
    pipeline, _, _ = start_threaded(config, calibration)
    try:
        assert wait_for(lambda: pipeline.snapshot().emg_time.size > 100)
        monitor = LiveMonitor(pipeline, {"d": ("simulate disconnect", lambda: None)})
        monitor._draw(0)  # pylint: disable=protected-access
        assert "d simulate disconnect" in monitor.help_text()
        plt.close(monitor.figure)
        ConsoleMonitor(pipeline)()
        assert "DISARMED" in capsys.readouterr().out
        assert any(line.startswith("Sample rate") for line in status_lines(pipeline.snapshot()))
    finally:
        pipeline.stop()


def test_cli_offline_synthetic(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """The command-line entry point runs an offline synthetic session and writes the log."""
    log = tmp_path / "commands.csv"
    code = run_exoskeleton.main(
        [
            "--source",
            "synthetic",
            "--offline",
            "--duration",
            "30",
            "--log",
            str(log),
            "--save-calibration",
            str(tmp_path / "cal.json"),
        ]
    )
    assert code == 0
    assert "STIFFENED" in capsys.readouterr().out
    assert log.exists() and len(log.read_text(encoding="utf-8").splitlines()) > 100
    assert (tmp_path / "cal.json").exists()


def test_cli_headless_threaded(tmp_path: Path) -> None:
    """A short headless real-time run exits cleanly."""
    log = tmp_path / "commands.csv"
    assert (
        run_exoskeleton.main(
            ["--source", "synthetic", "--headless", "--duration", "1.5", "--log", str(log)]
        )
        == 0
    )
    assert log.exists()


def test_cli_reports_missing_tcu(tmp_path: Path) -> None:
    """Without a TCU the trigno source fails to start with exit code 3."""
    config_path = tmp_path / "config.toml"
    text = (Path(run_exoskeleton.__file__).parent / "config" / "exoskeleton.toml").read_text(
        encoding="utf-8"
    )
    config_path.write_text(
        text.replace("connect_timeout_s = 3.0", "connect_timeout_s = 0.3").replace(
            "command_port = 50040", "command_port = 1"
        ),
        encoding="utf-8",
    )
    assert (
        run_exoskeleton.main(
            [
                "--source",
                "trigno",
                "--config",
                str(config_path),
                "--headless",
                "--log",
                str(tmp_path / "c.csv"),
            ]
        )
        == run_exoskeleton.EXIT_SOURCE
    )


def test_cli_rejects_bad_config(tmp_path: Path) -> None:
    """Configuration errors give exit code 2."""
    assert (
        run_exoskeleton.main(["--config", str(tmp_path / "missing.toml")])
        == run_exoskeleton.EXIT_CONFIG
    )
