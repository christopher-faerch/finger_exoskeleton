"""
Configuration for the real-time pipeline, loaded from config/exoskeleton.toml.

Every section maps to a frozen dataclass. Unknown keys are rejected so a typo
in the file cannot silently fall back to a default.
"""

import tomllib
from dataclasses import dataclass, fields, replace
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config" / "exoskeleton.toml"

CHANNEL_ROLES = ("flexor", "extensor", "other")
SOURCE_KINDS = ("synthetic", "recorded", "trigno")


class ConfigError(ValueError):
    """Raised for a missing, malformed or unsafe configuration."""


@dataclass(frozen=True)
class ChannelConfig:
    """One EMG channel: its name, role and where each source finds it."""

    name: str
    role: str
    trigno_slot: int
    recorded_column: int


@dataclass(frozen=True)
class TrignoConfig:
    """Delsys Trigno Control Utility TCP/IP SDK settings (MAN-025-3-5)."""

    host: str = "127.0.0.1"
    command_port: int = 50040
    emg_port: int = 50043
    channels_on_port: int = 16
    connect_timeout_s: float = 3.0
    command_timeout_s: float = 2.0
    data_timeout_s: float = 0.5
    start_index_base: int = 0
    required_units: str = "Volts"
    expected_sample_rate_hz: float = 0.0


@dataclass(frozen=True)
class RecordedConfig:
    """Replay of a recorded CSV file."""

    file: str = "10_raw.csv"
    sample_rate_hz: float = 2000.0
    block_size: int = 27
    loop: bool = True
    realtime: bool = True


@dataclass(frozen=True)
class SyntheticConfig:
    """Synthetic EMG generator."""

    sample_rate_hz: float = 2000.0
    block_size: int = 27
    seed: int = 1
    realtime: bool = True
    rest_noise_mv: float = 0.005
    max_contraction_mv: float = 0.4
    tremor_frequency_hz: float = 5.0
    powerline_mv: float = 0.002
    powerline_hz: float = 50.0


@dataclass(frozen=True)
class CalibrationConfig:
    """Rest baseline and reference contraction windows, in seconds of stream time."""

    baseline_window_s: tuple[float, float] = (0.5, 3.0)
    reference_window_s: tuple[float, float] = (3.5, 6.5)
    reference_percentile: float = 95.0
    min_reference_ratio: float = 3.0


@dataclass(frozen=True)
class ProcessingConfig:
    """Streaming (causal) signal processing."""

    bandpass_low_hz: float = 20.0
    bandpass_high_hz: float = 450.0
    bandpass_order: int = 4
    notch_enabled: bool = True
    notch_hz: float = 50.0
    notch_quality: float = 30.0
    slow_window_s: float = 0.2
    fast_cutoff_hz: float = 10.0
    tremor_band_hz: tuple[float, float] = (4.0, 6.0)
    tremor_window_s: float = 0.5
    rhythm_band_hz: tuple[float, float] = (1.0, 9.0)


@dataclass(frozen=True)
class ControllerConfig:
    """State machine thresholds and stiffness shaping (normalized activation units)."""

    grasp_on_threshold: float = 0.25
    grasp_off_threshold: float = 0.15
    grasp_on_time_s: float = 0.10
    grasp_off_time_s: float = 0.30
    tremor_on_threshold: float = 0.15
    tremor_off_threshold: float = 0.08
    tremor_min_ratio: float = 0.8
    tremor_off_ratio: float = 0.6
    tremor_on_time_s: float = 0.05
    tremor_off_time_s: float = 0.75
    engage_stiffness: float = 0.3
    stiffness_gain: float = 2.0
    max_stiffness: float = 1.0
    engage_rate_per_s: float = 20.0
    release_rate_per_s: float = 1.5


@dataclass(frozen=True)
class SafetyConfig:
    """Software safeguards. Not a substitute for the hardware safety in docs/SAFETY.md."""

    stale_data_timeout_s: float = 0.15
    actuator_watchdog_timeout_s: float = 0.1
    max_reconnect_attempts: int = 3
    reconnect_delay_s: float = 1.0


@dataclass(frozen=True)
class RuntimeConfig:
    """Threads, queues and monitoring."""

    control_rate_hz: float = 150.0
    queue_size_blocks: int = 64
    monitor_seconds: float = 5.0
    command_log_size: int = 100_000


@dataclass(frozen=True)
class ActuatorConfig:
    """Actuator backend. Only "dry_run" exists."""

    backend: str = "dry_run"


@dataclass(frozen=True)
class AppConfig:  # pylint: disable=too-many-instance-attributes
    """The complete configuration. calibration is already resolved for one source."""

    channels: tuple[ChannelConfig, ...]
    trigno: TrignoConfig
    recorded: RecordedConfig
    synthetic: SyntheticConfig
    calibration: CalibrationConfig
    processing: ProcessingConfig
    controller: ControllerConfig
    safety: SafetyConfig
    runtime: RuntimeConfig
    actuator: ActuatorConfig

    @property
    def channel_names(self) -> tuple[str, ...]:
        """Channel names in order."""
        return tuple(channel.name for channel in self.channels)

    @property
    def channel_roles(self) -> tuple[str, ...]:
        """Channel roles in order."""
        return tuple(channel.role for channel in self.channels)


def _build[T](cls: type[T], table: dict[str, Any], section: str) -> T:
    """Create dataclass cls from a TOML table, rejecting unknown keys."""
    known = {field.name: field for field in fields(cls)}  # type: ignore[arg-type]
    unknown = set(table) - set(known)
    if unknown:
        raise ConfigError(f"[{section}] has unknown keys: {', '.join(sorted(unknown))}")
    values: dict[str, Any] = {}
    for key, value in table.items():
        if isinstance(value, list):
            value = tuple(value)
        values[key] = value
    try:
        return cls(**values)
    except TypeError as error:
        raise ConfigError(f"[{section}]: {error}") from error


def _check(condition: bool, message: str) -> None:
    if not condition:
        raise ConfigError(message)


def _validate(config: AppConfig) -> None:
    """Reject configurations that are inconsistent or unsafe."""
    names = config.channel_names
    _check(len(names) > 0, "At least one [[channels]] entry is required")
    _check(len(set(names)) == len(names), "Channel names must be unique")
    _check(
        len(names) <= config.trigno.channels_on_port,
        "More channels than the Trigno EMG port carries",
    )
    for channel in config.channels:
        _check(
            channel.role in CHANNEL_ROLES,
            f"Channel {channel.name}: role must be one of {CHANNEL_ROLES}",
        )
        _check(1 <= channel.trigno_slot <= 16, f"Channel {channel.name}: trigno_slot must be 1-16")
        _check(
            channel.recorded_column >= 0, f"Channel {channel.name}: recorded_column must be >= 0"
        )
    slots = [channel.trigno_slot for channel in config.channels]
    _check(len(set(slots)) == len(slots), "Each channel needs its own trigno_slot")
    _check(
        "flexor" in config.channel_roles,
        "At least one flexor channel is needed for grasp detection",
    )

    _check(config.trigno.start_index_base in (0, 1), "trigno.start_index_base must be 0 or 1")

    for window in (config.calibration.baseline_window_s, config.calibration.reference_window_s):
        _check(
            len(window) == 2 and 0 <= window[0] < window[1],
            "Calibration windows must be [start, end] with 0 <= start < end",
        )
    _check(
        config.calibration.baseline_window_s[1] <= config.calibration.reference_window_s[0],
        "The baseline window must end before the reference window starts",
    )
    _check(
        0 < config.calibration.reference_percentile <= 100,
        "reference_percentile must be in (0, 100]",
    )

    processing = config.processing
    _check(
        0 < processing.bandpass_low_hz < processing.bandpass_high_hz,
        "processing.bandpass_low_hz must be below bandpass_high_hz",
    )
    _check(
        len(processing.tremor_band_hz) == 2
        and 0 < processing.tremor_band_hz[0] < processing.tremor_band_hz[1],
        "processing.tremor_band_hz must be [low, high]",
    )
    _check(
        processing.tremor_band_hz[1] < processing.fast_cutoff_hz,
        "The tremor band must lie below the fast envelope cutoff",
    )
    _check(
        len(processing.rhythm_band_hz) == 2
        and 0 < processing.rhythm_band_hz[0] <= processing.tremor_band_hz[0]
        and processing.tremor_band_hz[1]
        <= processing.rhythm_band_hz[1]
        < processing.fast_cutoff_hz,
        "processing.rhythm_band_hz must contain the tremor band and lie below fast_cutoff_hz",
    )

    controller = config.controller
    _check(
        controller.grasp_off_threshold < controller.grasp_on_threshold,
        "grasp_off_threshold must be below grasp_on_threshold (hysteresis)",
    )
    _check(
        controller.tremor_off_threshold < controller.tremor_on_threshold,
        "tremor_off_threshold must be below tremor_on_threshold (hysteresis)",
    )
    _check(0 <= controller.tremor_min_ratio <= 1.0, "tremor_min_ratio must be in [0, 1]")
    _check(
        controller.tremor_off_ratio <= controller.tremor_min_ratio,
        "tremor_off_ratio must not exceed tremor_min_ratio (hysteresis)",
    )
    _check(0 < controller.max_stiffness <= 1.0, "max_stiffness must be in (0, 1]")
    _check(
        0 <= controller.engage_stiffness <= controller.max_stiffness,
        "engage_stiffness must be within [0, max_stiffness]",
    )
    _check(
        controller.engage_rate_per_s > 0 and controller.release_rate_per_s > 0,
        "Stiffness rate limits must be positive",
    )

    _check(config.safety.stale_data_timeout_s > 0, "stale_data_timeout_s must be positive")
    _check(
        config.safety.actuator_watchdog_timeout_s > 0,
        "actuator_watchdog_timeout_s must be positive",
    )
    _check(config.runtime.control_rate_hz > 0, "control_rate_hz must be positive")
    _check(config.runtime.queue_size_blocks > 0, "queue_size_blocks must be positive")
    _check(
        config.actuator.backend == "dry_run",
        'actuator.backend must be "dry_run": no physical actuator protocol is '
        "documented or verified (see docs/SAFETY.md)",
    )


def load_config(path: Path | None = None, source: str = "synthetic") -> AppConfig:
    """
    Load and validate the configuration. The [<source>.calibration] table, if
    present, overrides [calibration] for that source.
    """
    if source not in SOURCE_KINDS:
        raise ConfigError(f"Unknown source {source!r}; expected one of {SOURCE_KINDS}")
    path = path or DEFAULT_CONFIG_PATH
    try:
        with open(path, "rb") as file:
            raw = tomllib.load(file)
    except FileNotFoundError as error:
        raise ConfigError(f"Config file not found: {path}") from error
    except tomllib.TOMLDecodeError as error:
        raise ConfigError(f"Config file {path} is not valid TOML: {error}") from error

    def section(name: str) -> dict[str, Any]:
        value = raw.get(name, {})
        if not isinstance(value, dict):
            raise ConfigError(f"[{name}] must be a table")
        return dict(value)

    source_tables = {kind: section(kind) for kind in SOURCE_KINDS}
    overrides = {kind: table.pop("calibration", {}) for kind, table in source_tables.items()}

    calibration = _build(CalibrationConfig, section("calibration"), "calibration")
    calibration = replace(calibration, **_build_override(overrides[source], source))

    config = AppConfig(
        channels=tuple(
            _build(ChannelConfig, entry, "channels") for entry in raw.get("channels", [])
        ),
        trigno=_build(TrignoConfig, source_tables["trigno"], "trigno"),
        recorded=_build(RecordedConfig, source_tables["recorded"], "recorded"),
        synthetic=_build(SyntheticConfig, source_tables["synthetic"], "synthetic"),
        calibration=calibration,
        processing=_build(ProcessingConfig, section("processing"), "processing"),
        controller=_build(ControllerConfig, section("controller"), "controller"),
        safety=_build(SafetyConfig, section("safety"), "safety"),
        runtime=_build(RuntimeConfig, section("runtime"), "runtime"),
        actuator=_build(ActuatorConfig, section("actuator"), "actuator"),
    )
    _validate(config)
    return config


def _build_override(table: dict[str, Any], source: str) -> dict[str, Any]:
    """Validate a [<source>.calibration] override and return it as keyword arguments."""
    known = {field.name for field in fields(CalibrationConfig)}
    unknown = set(table) - known
    if unknown:
        raise ConfigError(f"[{source}.calibration] has unknown keys: {', '.join(sorted(unknown))}")
    return {key: tuple(value) if isinstance(value, list) else value for key, value in table.items()}
