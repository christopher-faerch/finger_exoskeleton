"""
A stand-in for the Delsys Trigno Control Utility (TCU) SDK server, for tests
and for demonstrating the Trigno backend without hardware.

It implements only the documented subset of MAN-025-3-5 that
source/trigno.py uses: the command port with the replies from the guide's
appendix, and the EMG Data port streaming 16 multiplexed float32 channels per
sample in 13.5 ms frames. It is NOT a model of the real TCU's timing,
buffering or error behavior and does not prove hardware compatibility.
"""

import socket
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray

# Generates EMG in volts: (first_sample_index, n_samples) -> (n_sensors, n_samples)
SignalFunction = Callable[[int, int], NDArray[np.float64]]


@dataclass
class MockSensor:
    """
    One paired sensor slot. position is the 0-based column of its EMG channel
    in the 16-channel port buffer; STARTINDEX? reports position + the
    settings' start_index_base.
    """

    slot: int
    position: int
    paired: bool = True
    active: bool = True
    units: str = "Volts"
    sensor_type: str = "O"  # SDK section 7: Trigno Avanti sensor = type "O"


@dataclass
class MockTCUSettings:  # pylint: disable=too-many-instance-attributes
    """Behavior of the mock server."""

    sensors: list[MockSensor] = field(default_factory=list)
    backwards_compatibility: bool = True
    upsampling: bool = True
    frame_interval_s: float = 0.0135
    samples_per_frame: int = 27  # 27 / 0.0135 s = 2000 Hz
    endianness: str = "LITTLE"
    channels_on_port: int = 16
    realtime: bool = True
    fragment_sizes: tuple[
        int, ...
    ] = ()  # if set, frames are sent in chunks of these sizes (cycled)
    max_frames: int | None = None
    start_index_base: int = 0  # what STARTINDEX? adds to the 0-based position
    banner: str = "Mock Trigno SDK server (documented protocol subset)"


def _zero_signal(_first: int, n_samples: int, n_sensors: int) -> NDArray[np.float64]:
    return np.zeros((n_sensors, n_samples))


class MockTCU:  # pylint: disable=too-many-instance-attributes
    """Threaded mock server. Use as a context manager or call start()/stop()."""

    def __init__(
        self,
        settings: MockTCUSettings,
        signal: SignalFunction | None = None,
        host: str = "127.0.0.1",
        command_port: int = 0,
        emg_port: int = 0,
    ) -> None:
        self.settings = settings
        self.signal = signal
        self.host = host
        self._requested_ports = (command_port, emg_port)
        self.command_port = 0
        self.emg_port = 0
        self.received_commands: list[str] = []
        self.frames_sent = 0
        self._command_server: socket.socket | None = None
        self._data_server: socket.socket | None = None
        self._data_client: socket.socket | None = None
        self._threads: list[threading.Thread] = []
        self._stop = threading.Event()
        self._streaming = threading.Event()
        self._paused = threading.Event()
        self._lock = threading.Lock()

    # -- lifecycle ---------------------------------------------------------

    def start(self) -> "MockTCU":
        """Bind both ports and start serving."""
        self._command_server = self._listen(self._requested_ports[0])
        self._data_server = self._listen(self._requested_ports[1])
        self.command_port = self._command_server.getsockname()[1]
        self.emg_port = self._data_server.getsockname()[1]
        for target in (self._serve_commands, self._accept_data, self._stream):
            thread = threading.Thread(
                target=target, daemon=True, name=f"mock-tcu-{target.__name__}"
            )
            thread.start()
            self._threads.append(thread)
        return self

    def _listen(self, port: int) -> socket.socket:
        server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        server.bind((self.host, port))
        server.listen(1)
        server.settimeout(0.1)
        return server

    def stop(self) -> None:
        """Stop serving and close every socket."""
        self._stop.set()
        self._streaming.clear()
        for sock in (self._command_server, self._data_server, self._data_client):
            if sock is not None:
                try:
                    sock.close()
                except OSError:
                    pass
        for thread in self._threads:
            thread.join(timeout=2.0)

    def __enter__(self) -> "MockTCU":
        return self.start()

    def __exit__(self, *_exc: object) -> None:
        self.stop()

    # -- fault injection ---------------------------------------------------

    def drop_data_connection(self) -> None:
        """Close the EMG data connection, as if the TCU or USB link went away."""
        with self._lock:
            if self._data_client is not None:
                try:
                    self._data_client.shutdown(socket.SHUT_RDWR)
                except OSError:
                    pass
                self._data_client.close()
                self._data_client = None

    def pause_stream(self, paused: bool) -> None:
        """Stop or resume sending data without closing the connection."""
        if paused:
            self._paused.set()
        else:
            self._paused.clear()

    @property
    def streaming(self) -> bool:
        """True between START and STOP."""
        return self._streaming.is_set()

    # -- command port ------------------------------------------------------

    def _reply(self, command: str) -> str:  # pylint: disable=too-many-return-statements
        settings = self.settings
        words = command.upper().split()
        if command.upper() == "BACKWARDS COMPATIBILITY?":
            return "YES" if settings.backwards_compatibility else "NO"
        if command.upper() == "UPSAMPLING?":
            return "UPSAMPLING ON" if settings.upsampling else "UPSAMPLING OFF"
        if command.upper() == "FRAME INTERVAL?":
            return f"{settings.frame_interval_s}"
        if command.upper() == "MAX SAMPLES EMG?":
            return str(settings.samples_per_frame)
        if command.upper() == "ENDIANNESS?":
            return settings.endianness
        if command.upper() == "BASE FIRMWARE?":
            return "Firmware:  MOCK"
        if command.upper() == "START":
            self._streaming.set()
            return "OK"
        if command.upper() in ("STOP", "QUIT"):
            self._streaming.clear()
            return "OK"
        if len(words) >= 3 and words[0] == "SENSOR" and words[1].isdigit():
            return self._sensor_reply(int(words[1]), " ".join(words[2:]))
        return "INVALID COMMAND"

    def _sensor_reply(self, slot: int, query: str) -> str:
        sensor = next((s for s in self.settings.sensors if s.slot == slot), None)
        if query == "PAIRED?":
            return "YES" if sensor is not None and sensor.paired else "NO"
        if sensor is None:
            return "INVALID COMMAND"
        replies = {
            "ACTIVE?": "YES" if sensor.active else "NO",
            "EMGCHANNELCOUNT?": "1",
            "STARTINDEX?": str(sensor.position + self.settings.start_index_base),
            "TYPE?": sensor.sensor_type,
            "CHANNEL 1 UNITS?": sensor.units,
        }
        return replies.get(query, "INVALID COMMAND")

    def _serve_commands(self) -> None:
        assert self._command_server is not None
        while not self._stop.is_set():
            try:
                client, _ = self._command_server.accept()
            except (socket.timeout, OSError):
                continue
            with client:
                client.settimeout(0.1)
                self._handle_command_client(client)

    def _handle_command_client(self, client: socket.socket) -> None:
        client.sendall(f"{self.settings.banner}\r\n\r\n".encode("ascii"))
        buffer = b""
        while not self._stop.is_set():
            try:
                chunk = client.recv(4096)
            except socket.timeout:
                continue
            except OSError:
                return
            if not chunk:
                self._streaming.clear()
                return
            buffer += chunk
            # SDK 6.2: commands are processed when a packet ends with two CR LF
            while b"\r\n\r\n" in buffer:
                packet, buffer = buffer.split(b"\r\n\r\n", 1)
                for line in packet.decode("ascii").split("\r\n"):
                    if not line.strip():
                        continue
                    self.received_commands.append(line.strip())
                    client.sendall(f"{self._reply(line.strip())}\r\n\r\n".encode("ascii"))
                    if line.strip().upper() == "QUIT":
                        return

    # -- data port ---------------------------------------------------------

    def _accept_data(self) -> None:
        assert self._data_server is not None
        while not self._stop.is_set():
            try:
                client, _ = self._data_server.accept()
            except (socket.timeout, OSError):
                continue
            with self._lock:
                if self._data_client is not None:
                    self._data_client.close()
                self._data_client = client

    def _frame_bytes(self, first_index: int) -> bytes:
        settings = self.settings
        n = settings.samples_per_frame
        frame = np.zeros((n, settings.channels_on_port), dtype=np.float32)
        if settings.sensors:
            if self.signal is not None:
                values = self.signal(first_index, n)
            else:
                values = _zero_signal(first_index, n, len(settings.sensors))
            for row, sensor in enumerate(settings.sensors):
                frame[:, sensor.position] = values[row]
        byte_order = "<" if settings.endianness == "LITTLE" else ">"
        return frame.astype(np.dtype(f"{byte_order}f4")).tobytes()

    def _send(self, payload: bytes) -> bool:
        with self._lock:
            client = self._data_client
        if client is None:
            return False
        sizes = self.settings.fragment_sizes
        try:
            if not sizes:
                client.sendall(payload)
            else:
                position, i = 0, 0
                while position < len(payload):
                    size = max(1, sizes[i % len(sizes)])
                    client.sendall(payload[position : position + size])
                    position += size
                    i += 1
        except OSError:
            return False
        return True

    def _stream(self) -> None:
        next_time = time.monotonic()
        sample_index = 0
        while not self._stop.is_set():
            if not self._streaming.is_set() or self._paused.is_set():
                time.sleep(0.005)
                next_time = time.monotonic()
                continue
            limit = self.settings.max_frames
            if limit is not None and self.frames_sent >= limit:
                time.sleep(0.005)
                continue
            if self._send(self._frame_bytes(sample_index)):
                sample_index += self.settings.samples_per_frame
                self.frames_sent += 1
            else:
                time.sleep(0.002)  # no data client connected yet
            if self.settings.realtime:
                next_time += self.settings.frame_interval_s
                delay = next_time - time.monotonic()
                if delay > 0:
                    time.sleep(delay)
                else:
                    next_time = time.monotonic()
