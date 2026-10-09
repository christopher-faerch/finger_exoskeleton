"""Dry-run actuator, safety supervisor and configuration validation."""

import csv
from dataclasses import replace
from pathlib import Path

import pytest

from source.actuator import COMMAND_FIELDS, ActuatorCommand, DryRunActuator, create_actuator
from source.config import (
    ActuatorConfig,
    AppConfig,
    ConfigError,
    DEFAULT_CONFIG_PATH,
    SafetyConfig,
    load_config,
)
from source.safety import Fault, HealthInputs, SafetySupervisor


def command(
    sequence: int, stiffness: float, timestamp: float = 0.0, armed: bool = True
) -> ActuatorCommand:
    """A controller command."""
    return ActuatorCommand(
        sequence=sequence,
        timestamp=timestamp,
        requested_stiffness=stiffness,
        state="STIFFENED" if armed else "DISARMED",
        armed=armed,
    )


# -- dry-run actuator ---------------------------------------------------------


def test_dry_run_logs_exact_commands_and_exports(tmp_path: Path) -> None:
    """Every command is recorded with timestamp, value, state and exported to CSV."""
    log_path = tmp_path / "live.csv"
    actuator = DryRunActuator(watchdog_timeout_s=1.0, log_path=log_path)
    assert actuator.send(command(1, 0.25, 10.0)) == 0.25
    assert actuator.send(command(2, 0.5, 10.01)) == 0.5
    record = actuator.records[-1]
    assert (record.sequence, record.timestamp, record.requested_stiffness, record.state) == (
        2,
        10.01,
        0.5,
        "STIFFENED",
    )
    exported = tmp_path / "export.csv"
    assert actuator.export_csv(exported) == 2
    actuator.close()
    for path in (exported, log_path):
        rows = list(csv.reader(path.open(encoding="utf-8")))
        assert tuple(rows[0]) == COMMAND_FIELDS
        assert rows[1][2] == "0.250000" and rows[2][4] == "STIFFENED"


@pytest.mark.parametrize(
    ("requested", "applied"), [(1.5, 1.0), (-0.2, 0.0), (float("nan"), 0.0), (float("inf"), 0.0)]
)
def test_dry_run_saturates_invalid_requests(requested: float, applied: float) -> None:
    """Out-of-range and non-finite requests are saturated and counted."""
    actuator = DryRunActuator(watchdog_timeout_s=1.0)
    assert actuator.send(command(1, requested)) == applied
    assert actuator.status(0.0).invalid_commands == 1


def test_not_armed_applies_nothing() -> None:
    """A command from a disarmed controller never applies stiffness."""
    actuator = DryRunActuator(watchdog_timeout_s=1.0)
    assert actuator.send(command(1, 0.8, armed=False)) == 0.0


def test_emergency_stop_latches_until_reset() -> None:
    """After an e-stop, commands are logged but nothing is applied until reset()."""
    actuator = DryRunActuator(watchdog_timeout_s=1.0)
    actuator.send(command(1, 0.8))
    actuator.emergency_stop("test")
    assert actuator.status(0.0).applied_stiffness == 0.0
    assert actuator.send(command(2, 0.8)) == 0.0
    assert actuator.records[-1].estop
    actuator.reset()
    assert actuator.send(command(3, 0.8)) == 0.8
    assert any("EMERGENCY STOP" in record.note for record in actuator.records)


def test_watchdog_releases_and_latches() -> None:
    """No command within the timeout releases; it stays released until reset()."""
    actuator = DryRunActuator(watchdog_timeout_s=0.1)
    actuator.send(command(1, 0.8, timestamp=100.0))
    assert not actuator.check_watchdog(100.05)
    assert actuator.check_watchdog(100.2)
    assert actuator.status(100.2).applied_stiffness == 0.0
    assert actuator.send(command(2, 0.8, timestamp=100.21)) == 0.0
    actuator.reset()
    assert actuator.send(command(3, 0.8, timestamp=100.22)) == 0.8


def test_watchdog_idle_before_first_command_and_after_close() -> None:
    """The watchdog does not trip before any command or after close()."""
    actuator = DryRunActuator(watchdog_timeout_s=0.1)
    assert not actuator.check_watchdog(1e6)
    actuator.send(command(1, 0.5, timestamp=0.0))
    actuator.close()
    assert not actuator.check_watchdog(1e6)
    assert actuator.send(command(2, 0.5)) == 0.0


def test_physical_backend_is_refused() -> None:
    """There is no physical backend to create."""
    with pytest.raises(ValueError, match="not available"):
        create_actuator(ActuatorConfig(backend="serial"), SafetyConfig(), 10, None)


# -- supervisor -----------------------------------------------------------------


def health(**changes: object) -> HealthInputs:
    """Healthy inputs plus changes."""
    base = HealthInputs(
        now=10.0,
        last_data_time=9.99,
        source_connected=True,
        acquisition_running=True,
        invalid_data=False,
        calibrated=True,
        actuator_latched=False,
    )
    return replace(base, **changes)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("changes", "fault"),
    [
        ({"last_data_time": None}, Fault.NO_DATA),
        ({"last_data_time": 9.0}, Fault.STALE_DATA),
        ({"source_connected": False}, Fault.SOURCE_DISCONNECTED),
        ({"acquisition_running": False}, Fault.ACQUISITION_STOPPED),
        ({"invalid_data": True}, Fault.INVALID_DATA),
        ({"calibrated": False}, Fault.NOT_CALIBRATED),
        ({"actuator_latched": True}, Fault.ACTUATOR_LATCHED),
    ],
)
def test_each_fault_is_detected(changes: dict[str, object], fault: Fault) -> None:
    """Each unhealthy input produces its fault; healthy inputs produce none."""
    supervisor = SafetySupervisor(SafetyConfig())
    assert not supervisor.evaluate(health())
    assert supervisor.evaluate(health(**changes)) == (fault.value,)


def test_estop_latch_and_arm_blockers() -> None:
    """The e-stop latches until reset and blocks arming; an actuator latch alone does not."""
    supervisor = SafetySupervisor(SafetyConfig())
    supervisor.trigger_estop("button")
    faults = supervisor.evaluate(health())
    assert faults == (Fault.ESTOP.value,) and supervisor.estop_reason == "button"
    assert supervisor.arm_blockers(faults) == (Fault.ESTOP.value,)
    supervisor.reset_estop()
    assert not supervisor.arm_blockers(supervisor.evaluate(health(actuator_latched=True)))


# -- configuration --------------------------------------------------------------


def test_project_config_loads_for_every_source() -> None:
    """The shipped configuration is valid; source calibration overrides apply."""
    for source in ("synthetic", "recorded", "trigno"):
        config = load_config(source=source)
        assert config.actuator.backend == "dry_run"
    assert load_config(source="recorded").calibration.reference_window_s == (4.0, 104.0)


def write_config(tmp_path: Path, old: str, new: str) -> Path:
    """Copy the project configuration with one substitution."""
    text = DEFAULT_CONFIG_PATH.read_text(encoding="utf-8")
    assert old in text
    path = tmp_path / "config.toml"
    path.write_text(text.replace(old, new, 1), encoding="utf-8")
    return path


@pytest.mark.parametrize(
    ("old", "new", "message"),
    [
        ('backend = "dry_run"', 'backend = "physical"', "dry_run"),
        ("grasp_off_threshold = 0.15", "grasp_off_threshold = 0.5", "hysteresis"),
        ("max_stiffness = 1.0", "max_stiffness = 1.5", "max_stiffness"),
        ("control_rate_hz = 150.0", "control_rate_hz = 150.0\nunknown_key = 1", "unknown"),
        ('role = "flexor"', 'role = "biceps"', "role"),
    ],
)
def test_invalid_configs_are_rejected(tmp_path: Path, old: str, new: str, message: str) -> None:
    """Unsafe or misspelled settings stop the program instead of falling back to defaults."""
    with pytest.raises(ConfigError, match=message):
        load_config(write_config(tmp_path, old, new))


def test_missing_config_file(tmp_path: Path) -> None:
    """A missing file is a ConfigError."""
    with pytest.raises(ConfigError, match="not found"):
        load_config(tmp_path / "nope.toml")


def test_config_fixture_has_flexors(config: AppConfig) -> None:
    """The default montage has flexor channels for grasp detection."""
    assert "flexor" in config.channel_roles
