"""Command-line entry point for direct Dynamixel motor control."""

import argparse
import time
from collections.abc import Sequence

from source.DynamixelHardwareInterface import Motors


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line parser."""
    parser = argparse.ArgumentParser(
        description="Connect to the Dynamixel motor and optionally send one command."
    )
    parser.add_argument("--port", default="COM4", help="Serial port (default: COM4)")
    parser.add_argument(
        "--baudrate", type=int, default=4_500_000, help="Bus baudrate (default: 4500000)"
    )
    parser.add_argument(
        "--mode",
        choices=("pos", "vel", "cur"),
        default="pos",
        help="Control mode: position, velocity, or current (default: pos)",
    )
    parser.add_argument(
        "--motor-id",
        type=int,
        help="Motor ID to command (default: the first detected motor)",
    )
    parser.add_argument(
        "--command",
        type=int,
        help="Raw Dynamixel command in the selected control mode",
    )
    parser.add_argument(
        "--hold",
        type=float,
        default=0.0,
        help="Keep torque enabled for this many seconds after sending a command",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Connect to the motor, show feedback, and optionally send a command."""
    args = build_parser().parse_args(argv)
    motors = Motors(port=args.port, baudrate=args.baudrate)

    try:
        motor_id = args.motor_id
        if motor_id is None:
            motor_id = motors.motor_ids[0]
        if motor_id not in motors.motor_ids:
            raise ValueError(f"Motor ID {motor_id} was not detected.")

        # The Dynamixel control table requires torque to be disabled while
        # changing the operating mode.
        motors.disable_torque()
        motors.set_cont_mode(args.mode)
        motors.enable_torque()

        print(f"Motor {motor_id} position: {motors.get_position(motor_id)}")
        if args.command is not None:
            motors.sendMotorCommand(motor_id, args.command)
            print(f"Sent {args.mode} command {args.command} to motor {motor_id}.")
            if args.hold > 0:
                time.sleep(args.hold)
    finally:
        motors.close()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
