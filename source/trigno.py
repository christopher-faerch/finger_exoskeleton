"""
Live EMG from a Delsys Trigno system (DS-T03 / SP-W02 base / SP-W06 Trigno
Avanti sensors) through the Trigno Control Utility (TCU) TCP/IP SDK.

Everything here follows the Trigno SDK User Guide MAN-025-3-5 (2021):

- The TCU must be running on the host; it is the TCP server (section 2-3).
- Command port 50040: ASCII commands, each terminated by <CR><LF>; a packet
  ends with two <CR><LF> pairs. The server sends its version on connect and
  replies "OK", "INVALID COMMAND", "CANNOT COMPLETE" or a value (6.2-6.4).
- EMG Data port 50043 (16 channels; legacy 50041): IEEE 4-byte floats,
  multiplexed, read in multiples of channels * 4 bytes (6.1.1). Byte order
  from "ENDIANNESS?" (6.3.24). Units from "SENSOR n CHANNEL m UNITS?" (6.3.20,
  appendix example "Volts").
- Sample rate: with Backwards Compatibility on, fixed per port and Upsample
  setting (6.1.2); otherwise (samples per frame) / (frame interval), from
  "MAX SAMPLES EMG?" and "FRAME INTERVAL?" (appendix).

Only queries and START / STOP / QUIT are sent. This module never changes the
sensor configuration (no pairing, SETMODE, SETRANGE, ...): configure the
system in the TCU.

NOT VERIFIED ON HARDWARE. Tested against source/mock_tcu.py, which implements
the same documented subset. See README for what to check on the real system.
"""

import socket
import time
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from .config import ChannelConfig, TrignoConfig
from .sensor import EMGBlock, EMGSource, SourceConnectionError, SourceError, SourceTimeoutError

LEGACY_EMG_PORT = 50041

# SDK 6.1.2: EMG port sample rates with Backwards Compatibility ON, by
# (legacy EMG port 50041?, upsample on). The EMG Data port (50043) gives 2000 Hz
# or 1111.111... Hz (15 / 0.0135); the legacy port 2000 Hz or 1925.925... Hz (26 / 0.0135).
BACKWARDS_COMPATIBLE_EMG_RATES: dict[tuple[bool, bool], float] = {
    (False, True): 2000.0,
    (False, False): 15 / 0.0135,
    (True, True): 2000.0,
    (True, False): 26 / 0.0135,
}

UNIT_SCALES_TO_MV = {"volts": 1000.0, "v": 1000.0, "millivolts": 1.0, "mv": 1.0}

ERROR_REPLIES = ("INVALID COMMAND", "CANNOT COMPLETE")


class TrignoProtocolError(SourceError):
    """The TCU sent a reply this client does not understand."""


class TrignoConfigurationError(SourceError):
    """The TCU or sensors are in a state this pipeline cannot use."""


class TrignoCommandChannel:
    """Line-based command connection to the TCU command port."""

    def __init__(
        self, host: str, port: int, connect_timeout: float, command_timeout: float
    ) -> None:
        self.host = host
        self.port = port
        self.connect_timeout = connect_timeout
        self.command_timeout = command_timeout
        self._socket: socket.socket | None = None
        self._buffer = bytearray()
        self.stopped_by_trigger = False

    def connect(self) -> str:
        """Connect and return the server's version banner (SDK 6.3.1)."""
        try:
            self._socket = socket.create_connection(
                (self.host, self.port), timeout=self.connect_timeout
            )
        except OSError as error:
            raise SourceConnectionError(
                f"Cannot connect to the Trigno Control Utility command port at "
                f"{self.host}:{self.port} ({error}). Is the TCU running and the base station "
                f"connected?"
            ) from error
        self._socket.settimeout(self.command_timeout)
        banner = [self._read_line()]
        # The banner may span several lines; take whatever follows within a moment
        self._socket.settimeout(0.2)
        try:
            while True:
                line = self._read_line()
                if line:
                    banner.append(line)
        except SourceTimeoutError:
            pass
        finally:
            self._socket.settimeout(self.command_timeout)
        return " ".join(banner)

    def _recv(self) -> bytes:
        assert self._socket is not None
        try:
            chunk = self._socket.recv(4096)
        except socket.timeout as error:
            raise SourceTimeoutError("The TCU did not reply in time") from error
        except OSError as error:
            raise SourceConnectionError(f"Command connection to the TCU failed: {error}") from error
        if not chunk:
            raise SourceConnectionError("The TCU closed the command connection")
        return chunk

    def _read_line(self) -> str:
        """Next non-empty reply line. Records an unsolicited STOPPED (SDK 6.3.26)."""
        while True:
            while b"\r\n" not in self._buffer:
                self._buffer += self._recv()
            raw, _, rest = bytes(self._buffer).partition(b"\r\n")
            self._buffer = bytearray(rest)
            line = raw.decode("ascii", errors="replace").strip()
            if line == "STOPPED":
                self.stopped_by_trigger = True
                continue
            if line:
                return line

    def _drain(self) -> None:
        """Discard stale bytes so the next reply belongs to the next command."""
        assert self._socket is not None
        self._socket.setblocking(False)
        try:
            while True:
                chunk = self._socket.recv(4096)
                if not chunk:
                    break
                self._buffer += chunk
        except (BlockingIOError, InterruptedError):
            pass
        except OSError as error:
            raise SourceConnectionError(f"Command connection to the TCU failed: {error}") from error
        finally:
            self._socket.settimeout(self.command_timeout)
        while b"\r\n" in self._buffer:
            raw, _, rest = bytes(self._buffer).partition(b"\r\n")
            self._buffer = bytearray(rest)
            if raw.strip() == b"STOPPED":
                self.stopped_by_trigger = True

    def send(self, command: str) -> str:
        """Send one command packet and return the reply line."""
        if self._socket is None:
            raise SourceConnectionError("Not connected to the TCU")
        self._drain()
        try:
            self._socket.sendall(f"{command}\r\n\r\n".encode("ascii"))
        except OSError as error:
            raise SourceConnectionError(f"Cannot send {command!r} to the TCU: {error}") from error
        return self._read_line()

    def close(self) -> None:
        """Close the connection. Safe to call twice."""
        if self._socket is not None:
            try:
                self._socket.close()
            finally:
                self._socket = None


@dataclass(frozen=True)
class TrignoSensor:
    """What the TCU reported for one configured slot."""

    slot: int
    channel_name: str
    start_index: int
    units: str
    sensor_type: str
    active: str


def _yes_no(reply: str, query: str) -> bool:
    """Parse a YES/NO (or ON/OFF) reply; the flag is the last word."""
    word = reply.split()[-1].upper() if reply.split() else ""
    if word in ("YES", "ON"):
        return True
    if word in ("NO", "OFF"):
        return False
    raise TrignoProtocolError(f"Unexpected reply to {query!r}: {reply!r}")


def _number(reply: str, query: str) -> float:
    """Parse a numeric reply."""
    try:
        return float(reply.split()[-1])
    except (ValueError, IndexError) as error:
        raise TrignoProtocolError(f"Unexpected reply to {query!r}: {reply!r}") from error


class TrignoSource(EMGSource):  # pylint: disable=too-many-instance-attributes
    """EMG from the Delsys Trigno Control Utility SDK (see module docstring)."""

    kind = "trigno"

    def __init__(self, config: TrignoConfig, channels: Sequence[ChannelConfig]) -> None:
        # The sample rate is unknown until the TCU is queried in start()
        super().__init__([channel.name for channel in channels], 0.0)
        self.config = config
        self.channels = tuple(channels)
        self.banner = ""
        self.base_firmware = ""
        self.backwards_compatibility: bool | None = None
        self.upsampling: bool | None = None
        self.frame_interval_s = 0.0
        self.max_samples_emg = 0
        self.endianness = "LITTLE"
        self.sensors: tuple[TrignoSensor, ...] = ()
        self.warnings: list[str] = []
        self._command: TrignoCommandChannel | None = None
        self._data: socket.socket | None = None
        self._buffer = bytearray()
        self._column_indices: list[int] = []
        self._scales = np.ones(0)
        self._samples_read = 0
        self._streaming = False

    def _query(self, command: str) -> str:
        assert self._command is not None
        reply = self._command.send(command)
        if reply.upper() in ERROR_REPLIES:
            raise TrignoProtocolError(f"TCU replied {reply!r} to {command!r}")
        return reply

    def _require_ok(self, command: str) -> None:
        assert self._command is not None
        reply = self._command.send(command)
        if reply.upper() != "OK":
            raise TrignoConfigurationError(f"TCU replied {reply!r} to {command!r}")

    def _query_system(self) -> None:
        """System-wide settings: backwards compatibility, upsampling, framing, byte order."""
        self.backwards_compatibility = _yes_no(
            self._query("BACKWARDS COMPATIBILITY?"), "BACKWARDS COMPATIBILITY?"
        )
        self.upsampling = _yes_no(self._query("UPSAMPLING?"), "UPSAMPLING?")
        self.frame_interval_s = _number(self._query("FRAME INTERVAL?"), "FRAME INTERVAL?")
        self.max_samples_emg = int(_number(self._query("MAX SAMPLES EMG?"), "MAX SAMPLES EMG?"))
        endianness = self._query("ENDIANNESS?").split()[-1].upper()
        if endianness not in ("LITTLE", "BIG"):
            raise TrignoProtocolError(f"Unexpected reply to 'ENDIANNESS?': {endianness!r}")
        self.endianness = endianness
        try:
            self.base_firmware = self._query("BASE FIRMWARE?")
        except TrignoProtocolError as error:
            self.warnings.append(str(error))

    def _query_sensors(self) -> None:
        """Per-slot pairing, channel position and units for every configured channel."""
        sensors = []
        for channel in self.channels:
            slot = channel.trigno_slot
            if not _yes_no(self._query(f"SENSOR {slot} PAIRED?"), "PAIRED?"):
                raise TrignoConfigurationError(
                    f"No sensor is paired in TCU slot {slot} (channel {channel.name}). "
                    f"Pair it in the Trigno Control Utility."
                )
            emg_channels = int(
                _number(self._query(f"SENSOR {slot} EMGCHANNELCOUNT?"), "EMGCHANNELCOUNT?")
            )
            if emg_channels < 1:
                raise TrignoConfigurationError(
                    f"Sensor in slot {slot} (channel {channel.name}) reports no EMG channel"
                )
            start_index = int(_number(self._query(f"SENSOR {slot} STARTINDEX?"), "STARTINDEX?"))
            column = start_index - self.config.start_index_base
            if not 0 <= column < self.config.channels_on_port:
                raise TrignoConfigurationError(
                    f"Slot {slot}: STARTINDEX {start_index} is outside the "
                    f"{self.config.channels_on_port}-channel EMG port "
                    f"(start_index_base = {self.config.start_index_base})"
                )
            sensors.append(
                TrignoSensor(
                    slot=slot,
                    channel_name=channel.name,
                    start_index=column,
                    units=self._query(f"SENSOR {slot} CHANNEL 1 UNITS?"),
                    sensor_type=self._query(f"SENSOR {slot} TYPE?"),
                    active=self._query(f"SENSOR {slot} ACTIVE?"),
                )
            )

        columns = [sensor.start_index for sensor in sensors]
        if len(set(columns)) != len(columns):
            raise TrignoConfigurationError(f"Configured slots share EMG port positions: {columns}")
        # SDK 6.1.3: without multi-channel sensors, slot n sits at 0-based position n - 1
        if all(sensor.start_index == sensor.slot for sensor in sensors):
            self.warnings.append(
                "Every STARTINDEX equals its slot number: the TCU may count from 1. "
                "Check trigno.start_index_base."
            )
        self.sensors = tuple(sensors)

    def _resolve_units(self) -> None:
        scales = []
        for sensor in self.sensors:
            units = sensor.units.strip().lower()
            if self.config.required_units and units != self.config.required_units.strip().lower():
                raise TrignoConfigurationError(
                    f"Slot {sensor.slot} reports units {sensor.units!r}, "
                    f"expected {self.config.required_units!r}"
                )
            if units not in UNIT_SCALES_TO_MV:
                raise TrignoConfigurationError(
                    f"Slot {sensor.slot}: unsupported units {sensor.units!r}"
                )
            scales.append(UNIT_SCALES_TO_MV[units])
        self._scales = np.asarray(scales)[:, None]

    def _resolve_sample_rate(self) -> float:
        """Sample rate of the EMG port, from the documented rules (SDK 6.1.2)."""
        if self.frame_interval_s <= 0 or self.max_samples_emg <= 0:
            raise TrignoProtocolError(
                f"Unusable framing: FRAME INTERVAL {self.frame_interval_s}, "
                f"MAX SAMPLES EMG {self.max_samples_emg}"
            )
        from_frame = self.max_samples_emg / self.frame_interval_s
        if self.backwards_compatibility:
            legacy = self.config.emg_port == LEGACY_EMG_PORT
            rate = BACKWARDS_COMPATIBLE_EMG_RATES[(legacy, bool(self.upsampling))]
            if abs(rate - from_frame) / rate > 0.01:
                self.warnings.append(
                    f"Documented rate {rate:.3f} Hz differs from MAX SAMPLES EMG / FRAME INTERVAL "
                    f"= {from_frame:.3f} Hz; using the documented rate"
                )
        else:
            rate = from_frame
        expected = self.config.expected_sample_rate_hz
        if expected > 0 and abs(rate - expected) / expected > 0.005:
            raise TrignoConfigurationError(
                f"TCU EMG sample rate is {rate:.3f} Hz but the configuration "
                f"expects {expected:g} Hz"
            )
        return rate

    def start(self) -> None:
        self.stop()
        self.warnings = []
        self._command = TrignoCommandChannel(
            self.config.host,
            self.config.command_port,
            self.config.connect_timeout_s,
            self.config.command_timeout_s,
        )
        try:
            self.banner = self._command.connect()
            self._query_system()
            self._query_sensors()
            self._resolve_units()
            self.sample_rate = self._resolve_sample_rate()
            self._column_indices = [sensor.start_index for sensor in self.sensors]

            try:
                self._data = socket.create_connection(
                    (self.config.host, self.config.emg_port), timeout=self.config.connect_timeout_s
                )
            except OSError as error:
                raise SourceConnectionError(
                    f"Cannot connect to the TCU EMG data port {self.config.host}:"
                    f"{self.config.emg_port} ({error})"
                ) from error
            self._data.settimeout(self.config.data_timeout_s)
            self._buffer = bytearray()
            self._require_ok("START")
            self._streaming = True
            self.connected = True
        except BaseException:
            self.stop()
            raise

    def read_block(self) -> EMGBlock | None:
        if self._data is None or not self.connected:
            raise SourceConnectionError("Trigno source is not started")
        row_bytes = self.config.channels_on_port * 4
        receive_time = time.monotonic()
        while len(self._buffer) < row_bytes:
            try:
                chunk = self._data.recv(65536)
            except socket.timeout as error:
                raise SourceTimeoutError(
                    f"No EMG data from the TCU for {self.config.data_timeout_s} s"
                ) from error
            except OSError as error:
                self.connected = False
                raise SourceConnectionError(f"EMG data connection failed: {error}") from error
            if not chunk:
                self.connected = False
                raise SourceConnectionError("The TCU closed the EMG data connection")
            self._buffer += chunk
            receive_time = time.monotonic()

        n_rows = len(self._buffer) // row_bytes
        payload = bytes(self._buffer[: n_rows * row_bytes])
        del self._buffer[: n_rows * row_bytes]
        dtype = np.dtype("<f4" if self.endianness == "LITTLE" else ">f4")
        frame = np.frombuffer(payload, dtype=dtype).reshape(n_rows, self.config.channels_on_port)
        samples = frame[:, self._column_indices].T.astype(np.float64) * self._scales

        first = self._samples_read
        self._samples_read += n_rows
        return EMGBlock(samples, first, receive_time)

    def stop(self) -> None:
        if self._command is not None:
            for command in (("STOP",) if self._streaming else ()) + ("QUIT",):
                try:
                    self._command.send(command)
                except SourceError:
                    break
            self._command.close()
            self._command = None
        if self._data is not None:
            try:
                self._data.close()
            finally:
                self._data = None
        self._streaming = False
        self.connected = False

    def info(self) -> dict[str, str]:
        info = super().info()
        info["host"] = f"{self.config.host}:{self.config.command_port}/{self.config.emg_port}"
        info["server"] = self.banner or "-"
        info["backwards compat."] = str(self.backwards_compatibility)
        info["upsampling"] = str(self.upsampling)
        info["slots"] = (
            ", ".join(
                f"{sensor.channel_name}=slot {sensor.slot}@{sensor.start_index}"
                for sensor in self.sensors
            )
            or "-"
        )
        if self.warnings:
            info["warnings"] = " | ".join(self.warnings)
        return info
