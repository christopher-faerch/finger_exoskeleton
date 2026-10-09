"""
Run the real-time EMG -> exoskeleton pipeline with the dry-run actuator.

Examples:
  python run_exoskeleton.py --source synthetic
  python run_exoskeleton.py --source recorded --file 10_raw.csv
  python run_exoskeleton.py --source trigno --host 127.0.0.1
  python run_exoskeleton.py --source trigno --mock-tcu        (protocol demo, no hardware)
  python run_exoskeleton.py --source synthetic --headless --duration 40 --auto-arm
  python run_exoskeleton.py --source recorded --offline       (fast offline evaluation)

Physical actuation is not available: every mode uses the dry-run actuator.
"""

import argparse
import sys
import time
from collections import Counter
from collections.abc import Callable
from dataclasses import replace
from pathlib import Path

import numpy as np

from source.activation import Calibration, CalibrationError
from source.actuator import DryRunActuator, create_actuator
from source.config import PROJECT_ROOT, SOURCE_KINDS, AppConfig, ConfigError, load_config
from source.mock_tcu import MockSensor, MockTCU, MockTCUSettings
from source.runtime import ControlCore, Pipeline, run_synchronously
from source.sensor import EMGSource, RecordedSource, SourceError, SyntheticSource
from source.trigno import TrignoSource

EXIT_CONFIG, EXIT_SOURCE, EXIT_RUNTIME = 2, 3, 4


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Command-line options."""
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--source",
        choices=SOURCE_KINDS,
        default="synthetic",
        help="EMG source (default: synthetic)",
    )
    parser.add_argument(
        "--config", type=Path, default=None, help="config file (default: config/exoskeleton.toml)"
    )
    parser.add_argument("--file", help="recorded CSV (name in data/sEMG_online or a path)")
    parser.add_argument("--host", help="host running the Trigno Control Utility")
    parser.add_argument(
        "--mock-tcu",
        action="store_true",
        help="with --source trigno: start a local mock TCU serving synthetic EMG",
    )
    parser.add_argument("--headless", action="store_true", help="no window; print status lines")
    parser.add_argument(
        "--duration", type=float, default=None, help="seconds to run (headless default 40)"
    )
    parser.add_argument(
        "--auto-arm",
        action="store_true",
        help="arm automatically after the first calibration (demo use; re-arming after "
        "a fault is always manual)",
    )
    parser.add_argument(
        "--offline",
        action="store_true",
        help="process a synthetic or recorded source as fast as possible, no threads",
    )
    parser.add_argument(
        "--log",
        type=Path,
        default=None,
        help="CSV log of every actuator command (default: logs/commands-<time>.csv)",
    )
    parser.add_argument(
        "--calibration", type=Path, default=None, help="load a saved calibration (JSON)"
    )
    parser.add_argument(
        "--save-calibration",
        type=Path,
        default=None,
        help="save the calibration recorded in this session (JSON)",
    )
    return parser.parse_args(argv)


def build_source(
    args: argparse.Namespace, config: AppConfig, realtime: bool
) -> tuple[EMGSource, MockTCU | None]:
    """Create the requested source (and a mock TCU if asked for)."""
    if args.source == "synthetic":
        return SyntheticSource(replace(config.synthetic, realtime=realtime), config.channels), None
    if args.source == "recorded":
        recorded = replace(config.recorded, realtime=realtime)
        if args.file:
            recorded = replace(recorded, file=args.file)
        if args.offline:
            recorded = replace(recorded, loop=False)
        return RecordedSource(recorded, config.channels), None

    trigno = config.trigno
    if args.host:
        trigno = replace(trigno, host=args.host)
    mock = None
    if args.mock_tcu:
        generator = SyntheticSource(replace(config.synthetic, realtime=False), config.channels)
        settings = MockTCUSettings(
            sensors=[
                MockSensor(slot=channel.trigno_slot, position=channel.trigno_slot - 1)
                for channel in config.channels
            ],
            start_index_base=trigno.start_index_base,
        )
        mock = MockTCU(settings, signal=lambda first, n: generator.generate(first, n) / 1000.0)
        mock.start()
        trigno = replace(
            trigno, host="127.0.0.1", command_port=mock.command_port, emg_port=mock.emg_port
        )
        print(
            f"Mock TCU listening on 127.0.0.1:{mock.command_port} / {mock.emg_port} "
            "(documented protocol subset, NOT hardware)"
        )
    return TrignoSource(trigno, config.channels), mock


def fault_injectors(
    source: EMGSource, mock: MockTCU | None
) -> dict[str, tuple[str, Callable[[], None]]]:
    """Keys that simulate faults in the monitor."""
    injectors: dict[str, tuple[str, Callable[[], None]]] = {}
    if isinstance(source, SyntheticSource):
        injectors["d"] = ("simulate disconnect", source.inject_disconnect)
        injectors["s"] = ("simulate 0.5 s data dropout", lambda: source.inject_dropout(0.5))
    if mock is not None:
        injectors["d"] = ("drop TCU data connection", mock.drop_data_connection)
        injectors["s"] = ("pause TCU stream 0.5 s", lambda: _pause(mock, 0.5))
    return injectors


def _pause(mock: MockTCU, seconds: float) -> None:
    mock.pause_stream(True)
    time.sleep(seconds)
    mock.pause_stream(False)


def run_offline(
    args: argparse.Namespace, config: AppConfig, calibration: Calibration | None
) -> int:
    """Fast, thread-free evaluation of a synthetic or recorded source."""
    if args.source == "trigno":
        print("--offline works with synthetic or recorded sources only", file=sys.stderr)
        return EXIT_CONFIG
    source, _ = build_source(args, config, realtime=False)
    actuator = DryRunActuator(
        watchdog_timeout_s=float("inf"), log_size=config.runtime.command_log_size, log_path=args.log
    )
    source.start()
    core = ControlCore(config, source.sample_rate, actuator, calibration, args.save_calibration)
    duration = args.duration if args.duration is not None else float("inf")
    result = run_synchronously(core, source, duration)
    actuator.close()
    source.stop()
    states = Counter(state.value for state in result.states)
    print(f"Processed {result.times[-1] if result.times else 0:.1f} s of EMG")
    print("Calibration:", "ok" if core.calibrated else f"FAILED ({core.calibrator.error})")
    print(
        "Time per state:",
        {state: f"{count / len(result.states) * 100:.1f} %" for state, count in states.items()},
    )
    print(
        f"Max stiffness: {max(result.stiffness, default=0.0):.2f}; "
        f"grasp p95 {np.percentile(result.grasp, 95):.2f}; "
        f"tremor p95 {np.percentile(result.tremor, 95):.3f}"
    )
    if args.log:
        print(f"Commands logged to {args.log}")
    return 0


def main(
    argv: list[str] | None = None,
) -> int:  # pylint: disable=too-many-return-statements,too-many-branches
    """Entry point; returns the process exit code."""
    args = parse_args(argv)
    try:
        config = load_config(args.config, args.source)
    except ConfigError as error:
        print(f"Configuration error: {error}", file=sys.stderr)
        return EXIT_CONFIG

    calibration = None
    if args.calibration:
        try:
            calibration = Calibration.load(args.calibration)
            calibration.validate(config.calibration.min_reference_ratio)
        except CalibrationError as error:
            print(f"Calibration error: {error}", file=sys.stderr)
            return EXIT_CONFIG

    if args.log is None:
        args.log = PROJECT_ROOT / "logs" / f"commands-{time.strftime('%Y%m%d-%H%M%S')}.csv"

    if args.offline:
        return run_offline(args, config, calibration)

    source, mock = build_source(args, config, realtime=True)
    actuator = create_actuator(
        config.actuator, config.safety, config.runtime.command_log_size, args.log
    )
    pipeline = Pipeline(
        config, source, actuator, calibration, args.save_calibration, auto_arm=args.auto_arm
    )
    print("DRY RUN: commands are logged, no physical actuator is driven.")
    try:
        pipeline.start()
    except SourceError as error:
        print(f"Cannot start the {args.source} source: {error}", file=sys.stderr)
        actuator.close()
        if mock is not None:
            mock.stop()
        return EXIT_SOURCE

    try:
        if args.headless:
            from source.monitor import ConsoleMonitor  # pylint: disable=import-outside-toplevel

            duration = args.duration if args.duration is not None else 40.0
            pipeline.run_for(duration, on_tick=ConsoleMonitor(pipeline), period_s=1.0)
        else:
            from source.monitor import LiveMonitor  # pylint: disable=import-outside-toplevel

            monitor = LiveMonitor(pipeline, fault_injectors(source, mock))
            print(monitor.help_text())
            monitor.run()
    except KeyboardInterrupt:
        print("Interrupted: shutting down safely")
    finally:
        pipeline.stop()
        if mock is not None:
            mock.stop()

    print(f"Actuator commands logged to {args.log}")
    if pipeline.error:
        print(f"Pipeline error: {pipeline.error}", file=sys.stderr)
        return EXIT_RUNTIME
    return 0


if __name__ == "__main__":
    sys.exit(main())
