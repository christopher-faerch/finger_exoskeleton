"""
EMG sources. Every source (synthetic, recorded, live Delsys Trigno in
source/trigno.py) has the same interface, EMGSource, and delivers EMGBlock
objects, so the downstream pipeline cannot tell them apart.

Units: all sources deliver samples in mV.
"""

import threading
import time
from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass
from types import TracebackType

import numpy as np
from numpy.typing import NDArray
from scipy.signal import butter, sosfilt

from .config import ChannelConfig, RecordedConfig, SyntheticConfig
from .import_data import load_columns, resolve_data_file


@dataclass(frozen=True)
class EMGBlock:
    """
    A block of consecutive samples from all channels.

    samples:            shape (n_channels, n_samples), in mV
    first_sample_index: index of samples[:, 0] since the source started; gaps
                        in the index mean samples were lost
    receive_time:       time.monotonic() when the block was received
    """

    samples: NDArray[np.float64]
    first_sample_index: int
    receive_time: float

    @property
    def n_samples(self) -> int:
        """Number of samples per channel."""
        return int(self.samples.shape[1])


class SourceError(RuntimeError):
    """Base class for acquisition errors."""


class SourceConnectionError(SourceError):
    """The source could not connect, or the connection was lost."""


class SourceTimeoutError(SourceError):
    """No data arrived within the timeout. The connection may still be alive."""


class EMGSource(ABC):
    """Interface shared by all EMG sources."""

    kind = "abstract"

    def __init__(self, channel_names: Sequence[str], sample_rate: float) -> None:
        self.channel_names: tuple[str, ...] = tuple(channel_names)
        self.sample_rate: float = sample_rate
        self.connected: bool = False

    @property
    def n_channels(self) -> int:
        """Number of EMG channels delivered."""
        return len(self.channel_names)

    @abstractmethod
    def start(self) -> None:
        """Connect and start streaming. Raises SourceConnectionError on failure."""

    @abstractmethod
    def read_block(self) -> EMGBlock | None:
        """
        Block until the next EMGBlock is available. Returns None when a finite
        source has no more data. Raises SourceTimeoutError or
        SourceConnectionError.
        """

    @abstractmethod
    def stop(self) -> None:
        """Stop streaming and release all connections. Safe to call twice."""

    def info(self) -> dict[str, str]:
        """Human-readable facts about the source, for the monitor."""
        return {"source": self.kind, "sample rate": f"{self.sample_rate:g} Hz"}

    def __enter__(self) -> "EMGSource":
        self.start()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self.stop()


class _Pacer:
    """Sleeps so blocks are released at the rate they would arrive from hardware."""

    def __init__(self, sample_rate: float) -> None:
        self.sample_rate = sample_rate
        self.start_time = time.monotonic()
        self.start_index = 0

    def restart(self, sample_index: int) -> None:
        """Pace from now, treating sample_index as the current position."""
        self.start_time = time.monotonic()
        self.start_index = sample_index

    def wait_until_available(self, end_sample_index: int) -> None:
        """Sleep until the sample before end_sample_index has 'happened'."""
        due = self.start_time + (end_sample_index - self.start_index) / self.sample_rate
        delay = due - time.monotonic()
        if delay > 0:
            time.sleep(delay)


# ---------------------------------------------------------------------------
# Synthetic EMG
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Segment:
    """
    One part of a synthetic scenario.

    flexor / extensor: voluntary drive, 0 (rest) to 1 (max_contraction_mv)
    tremor:            tremor modulation depth 0-1; flexors and extensors
                       burst in antiphase as in Parkinsonian tremor (design doc 7.5)
    """

    duration_s: float
    label: str
    flexor: float = 0.0
    extensor: float = 0.0
    tremor: float = 0.0


# Calibration part (played once). Matches [synthetic.calibration] in the config.
CALIBRATION_SCENARIO: tuple[Segment, ...] = (
    Segment(3.0, "rest (baseline calibration)"),
    Segment(2.0, "reference contraction", flexor=0.6, extensor=0.6),
    Segment(2.0, "rest"),
)

# Repeating part: exercises every controller path.
CYCLE_SCENARIO: tuple[Segment, ...] = (
    Segment(2.0, "rest"),
    Segment(4.0, "grasp", flexor=0.45, extensor=0.1),
    Segment(6.0, "grasp + tremor", flexor=0.45, extensor=0.15, tremor=0.8),
    Segment(2.0, "grasp, tremor subsides", flexor=0.45, extensor=0.1),
    Segment(2.0, "rest"),
    Segment(3.0, "wrist extension, no grasp", flexor=0.05, extensor=0.5),
    Segment(1.0, "rest"),
    Segment(3.0, "rest tremor, no grasp", flexor=0.05, extensor=0.05, tremor=0.8),
    Segment(1.0, "rest"),
)


class SyntheticSource(EMGSource):
    """
    Generates multi-channel surface EMG: band-limited (20-450 Hz) Gaussian
    noise whose amplitude follows a scripted scenario of rest, voluntary
    contractions of different strength and tremor-like 4-6 Hz bursting.
    Each channel has independent noise. Supports fault injection for testing
    the safety behavior.
    """

    kind = "synthetic"

    def __init__(
        self,
        config: SyntheticConfig,
        channels: Sequence[ChannelConfig],
        calibration_segments: Sequence[Segment] = CALIBRATION_SCENARIO,
        cycle_segments: Sequence[Segment] = CYCLE_SCENARIO,
    ) -> None:
        super().__init__([channel.name for channel in channels], config.sample_rate_hz)
        if not cycle_segments:
            raise ValueError("The synthetic scenario needs at least one cycle segment")
        self.config = config
        self.roles = tuple(channel.role for channel in channels)
        self._calibration = tuple(calibration_segments)
        self._cycle = tuple(cycle_segments)
        self._calibration_length = sum(segment.duration_s for segment in self._calibration)
        self._cycle_length = sum(segment.duration_s for segment in self._cycle)

        self._rng = np.random.default_rng(config.seed)
        nyquist = config.sample_rate_hz / 2
        self._noise_sos = butter(
            2,
            [20.0, min(450.0, 0.9 * nyquist)],
            btype="bandpass",
            fs=config.sample_rate_hz,
            output="sos",
        )
        self._noise_state = np.zeros((self._noise_sos.shape[0], self.n_channels, 2))
        self._noise_scale = self._unit_noise_scale()
        # Per-channel tremor phase: extensors burst in antiphase with flexors
        self._tremor_phase = np.array([np.pi if role == "extensor" else 0.0 for role in self.roles])
        self._powerline_phase = self._rng.uniform(0, 2 * np.pi, self.n_channels)

        self._lock = threading.Lock()
        self._next_index = 0
        self._pacer = _Pacer(self.sample_rate)
        self._disconnect_requested = False
        self._dropout_samples = 0

    def _unit_noise_scale(self) -> float:
        """Scale that makes the band-limited noise unit RMS."""
        sample = sosfilt(self._noise_sos, self._rng.standard_normal(int(10 * self.sample_rate)))
        return float(1.0 / np.std(sample[int(self.sample_rate) :]))

    def segment_at(self, time_s: float) -> Segment:
        """The scenario segment active at stream time time_s."""
        if time_s < self._calibration_length:
            segments, offset = self._calibration, time_s
        else:
            segments, offset = self._cycle, (time_s - self._calibration_length) % self._cycle_length
        for segment in segments:
            if offset < segment.duration_s:
                return segment
            offset -= segment.duration_s
        return segments[-1]

    def _drive(self, times: NDArray[np.float64]) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Per-channel voluntary drive and tremor depth, shape (n_channels, n)."""
        drive = np.zeros((self.n_channels, len(times)))
        tremor = np.zeros((self.n_channels, len(times)))
        for i, time_s in enumerate(times):
            segment = self.segment_at(float(time_s))
            for channel, role in enumerate(self.roles):
                drive[channel, i] = segment.flexor if role == "flexor" else segment.extensor
                tremor[channel, i] = segment.tremor
        return drive, tremor

    def generate(self, first_index: int, n_samples: int) -> NDArray[np.float64]:
        """Generate samples [first_index, first_index + n_samples) in mV."""
        config = self.config
        times = (first_index + np.arange(n_samples)) / self.sample_rate
        drive, tremor_depth = self._drive(times)

        # Tremor: rhythmic bursts, the drive is modulated by a raised sine
        burst = 0.5 * (
            1
            + np.sin(
                2 * np.pi * config.tremor_frequency_hz * times[None, :]
                + self._tremor_phase[:, None]
            )
        )
        modulation = (1 - tremor_depth) + tremor_depth * 2 * burst
        amplitude = config.rest_noise_mv + config.max_contraction_mv * drive * modulation

        white = self._rng.standard_normal((self.n_channels, n_samples))
        noise, self._noise_state = sosfilt(self._noise_sos, white, axis=1, zi=self._noise_state)
        signal: NDArray[np.float64] = amplitude * noise * self._noise_scale

        signal += config.powerline_mv * np.sin(
            2 * np.pi * config.powerline_hz * times[None, :] + self._powerline_phase[:, None]
        )
        return signal

    def start(self) -> None:
        with self._lock:
            self._disconnect_requested = False
            self.connected = True
        self._pacer.restart(self._next_index)

    def read_block(self) -> EMGBlock | None:
        with self._lock:
            if self._disconnect_requested:
                self.connected = False
                raise SourceConnectionError("Synthetic source: simulated disconnection")
            if not self.connected:
                raise SourceConnectionError("Synthetic source is not started")
            dropout = self._dropout_samples
            self._dropout_samples = 0

        if dropout:
            # Simulated dropout: samples are lost and nothing arrives meanwhile
            self._next_index += dropout
            if self.config.realtime:
                self._pacer.wait_until_available(self._next_index)

        first = self._next_index
        n_samples = self.config.block_size
        if self.config.realtime:
            self._pacer.wait_until_available(first + n_samples)
        samples = self.generate(first, n_samples)
        self._next_index += n_samples
        return EMGBlock(samples, first, time.monotonic())

    def stop(self) -> None:
        with self._lock:
            self.connected = False

    def inject_disconnect(self) -> None:
        """Make the next read_block raise SourceConnectionError (test/demo fault)."""
        with self._lock:
            self._disconnect_requested = True

    def inject_dropout(self, seconds: float) -> None:
        """Lose the next `seconds` of samples; with realtime pacing, nothing arrives meanwhile."""
        with self._lock:
            self._dropout_samples = int(seconds * self.sample_rate)

    def info(self) -> dict[str, str]:
        info = super().info()
        stream_time = self._next_index / self.sample_rate
        info["scenario"] = self.segment_at(stream_time).label
        return info


# ---------------------------------------------------------------------------
# Recorded EMG
# ---------------------------------------------------------------------------


class RecordedSource(EMGSource):
    """Replays a recorded CSV (one column per channel, in mV) in hardware-sized blocks."""

    kind = "recorded"

    def __init__(self, config: RecordedConfig, channels: Sequence[ChannelConfig]) -> None:
        super().__init__([channel.name for channel in channels], config.sample_rate_hz)
        self.config = config
        self.columns = [channel.recorded_column for channel in channels]
        self.file = resolve_data_file(config.file)
        self._data: NDArray[np.float64] | None = None
        self._position = 0
        self._next_index = 0
        self._pacer = _Pacer(self.sample_rate)

    def start(self) -> None:
        if self._data is None:
            try:
                columns = load_columns(self.file, self.columns)
            except FileNotFoundError as error:
                raise SourceConnectionError(f"Recording not found: {self.file}") from error
            except (IndexError, ValueError) as error:
                raise SourceConnectionError(
                    f"Cannot read columns {self.columns} from {self.file}: {error}"
                ) from error
            data = np.asarray(columns, dtype=np.float64)
            if data.ndim != 2 or data.shape[1] == 0:
                raise SourceConnectionError(f"Recording {self.file} has no samples")
            self._data = data
        self.connected = True
        self._pacer.restart(self._next_index)

    @property
    def n_recorded_samples(self) -> int:
        """Length of the recording in samples (0 before start)."""
        return 0 if self._data is None else int(self._data.shape[1])

    def read_block(self) -> EMGBlock | None:
        if not self.connected or self._data is None:
            raise SourceConnectionError("Recorded source is not started")
        if self._position >= self._data.shape[1]:
            if not self.config.loop:
                return None
            self._position = 0

        end = min(self._position + self.config.block_size, self._data.shape[1])
        samples = self._data[:, self._position : end].copy()
        first = self._next_index
        if self.config.realtime:
            self._pacer.wait_until_available(first + samples.shape[1])
        self._position = end
        self._next_index += samples.shape[1]
        return EMGBlock(samples, first, time.monotonic())

    def stop(self) -> None:
        self.connected = False

    def info(self) -> dict[str, str]:
        info = super().info()
        info["file"] = self.file.name
        info["position"] = f"{self._position / self.sample_rate:.1f} s"
        return info
