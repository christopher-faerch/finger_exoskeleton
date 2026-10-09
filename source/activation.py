"""
Per-user calibration and normalized muscle activation (design doc 7.4 step 4).

Calibration records a rest baseline and a reference contraction. Activation
is then expressed per channel as 0 = rest, 1 = reference contraction:

  slow activation   = (slow envelope - rest) / (reference - rest)   grasp intent
  fast activation   = same, from the fast envelope                  tremor bursts
  tremor activation = 4-6 Hz envelope amplitude / (reference - rest) of the fast envelope

The controller uses these summaries (design doc 6.5 / 8.3 / 7.7):
  grasp activation = mean slow activation of the flexor channels
  tremor level     = largest tremor activation over all channels
  tremor ratio     = rhythmicity (4-6 Hz share of the envelope fluctuation)
                     of the channel with the largest tremor activation
"""

import json
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from .config import CalibrationConfig
from .streaming import ProcessedBlock

FloatArray = NDArray[np.float64]


class CalibrationError(ValueError):
    """Calibration data is missing or unusable."""


@dataclass(frozen=True)
class Calibration:
    """Rest and reference levels per channel, in mV."""

    channel_names: tuple[str, ...]
    rest_slow: FloatArray
    reference_slow: FloatArray
    rest_fast: FloatArray
    reference_fast: FloatArray

    def validate(self, min_reference_ratio: float) -> None:
        """Raise CalibrationError if the reference is not clearly above rest on every channel."""
        arrays = (self.rest_slow, self.reference_slow, self.rest_fast, self.reference_fast)
        if any(array.shape != (len(self.channel_names),) for array in arrays):
            raise CalibrationError("Calibration arrays do not match the channel count")
        if not all(np.all(np.isfinite(array)) for array in arrays):
            raise CalibrationError("Calibration contains non-finite values")
        for name, rest, reference in zip(self.channel_names, self.rest_slow, self.reference_slow):
            if rest <= 0 or reference < min_reference_ratio * rest:
                raise CalibrationError(
                    f"Channel {name}: reference {reference:.4f} mV is not at least "
                    f"{min_reference_ratio:g}x the rest level {rest:.4f} mV. Repeat the "
                    f"calibration with a clear reference contraction."
                )

    def save(self, path: Path) -> None:
        """Write the calibration as JSON."""
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(
                {
                    "channel_names": list(self.channel_names),
                    "rest_slow": self.rest_slow.tolist(),
                    "reference_slow": self.reference_slow.tolist(),
                    "rest_fast": self.rest_fast.tolist(),
                    "reference_fast": self.reference_fast.tolist(),
                },
                indent=2,
            ),
            encoding="utf-8",
        )

    @classmethod
    def load(cls, path: Path) -> "Calibration":
        """Read a calibration written by save()."""
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            return cls(
                channel_names=tuple(data["channel_names"]),
                rest_slow=np.asarray(data["rest_slow"], dtype=np.float64),
                reference_slow=np.asarray(data["reference_slow"], dtype=np.float64),
                rest_fast=np.asarray(data["rest_fast"], dtype=np.float64),
                reference_fast=np.asarray(data["reference_fast"], dtype=np.float64),
            )
        except (OSError, KeyError, ValueError, TypeError) as error:
            raise CalibrationError(f"Cannot read calibration {path}: {error}") from error


class Calibrator:
    """
    Collects envelopes during the baseline and reference windows (seconds of
    stream time since restart()) and produces a Calibration.
    """

    def __init__(
        self, config: CalibrationConfig, channel_names: Sequence[str], sample_rate: float
    ) -> None:
        self.config = config
        self.channel_names = tuple(channel_names)
        self.sample_rate = sample_rate
        self.result: Calibration | None = None
        self.error: str | None = None
        self._start_index: int | None = None
        self._last_time = 0.0
        self._parts: dict[str, list[FloatArray]] = {}
        self.restart()

    def restart(self) -> None:
        """Start a new calibration from the next block."""
        self.result = None
        self.error = None
        self._start_index = None
        self._last_time = 0.0
        self._parts = {"rest_slow": [], "rest_fast": [], "reference_slow": [], "reference_fast": []}

    @property
    def done(self) -> bool:
        """True once a calibration result or an error is available."""
        return self.result is not None or self.error is not None

    @property
    def phase(self) -> str:
        """What the user should be doing right now."""
        if self.result is not None:
            return "done"
        if self.error is not None:
            return "failed"
        baseline, reference = self.config.baseline_window_s, self.config.reference_window_s
        if self._last_time < baseline[1]:
            return "rest (baseline)"
        if self._last_time < reference[0]:
            return "prepare reference contraction"
        return "reference contraction"

    def update(self, block: ProcessedBlock) -> None:
        """Feed one processed block. Finishes automatically after the reference window."""
        if self.done:
            return
        if self._start_index is None:
            self._start_index = block.first_sample_index
        times = (
            block.first_sample_index - self._start_index + np.arange(block.n_samples)
        ) / self.sample_rate
        self._last_time = float(times[-1])
        for key, window in (
            ("rest", self.config.baseline_window_s),
            ("reference", self.config.reference_window_s),
        ):
            mask = (times >= window[0]) & (times < window[1])
            if np.any(mask):
                self._parts[f"{key}_slow"].append(block.slow_envelope[:, mask])
                self._parts[f"{key}_fast"].append(block.fast_envelope[:, mask])
        if self._last_time >= self.config.reference_window_s[1]:
            self._finish()

    def _finish(self) -> None:
        if not self._parts["rest_slow"] or not self._parts["reference_slow"]:
            self.error = "No samples fell inside the calibration windows"
            return
        joined = {key: np.concatenate(parts, axis=1) for key, parts in self._parts.items()}
        percentile = self.config.reference_percentile
        calibration = Calibration(
            channel_names=self.channel_names,
            rest_slow=np.mean(joined["rest_slow"], axis=1),
            reference_slow=np.percentile(joined["reference_slow"], percentile, axis=1),
            rest_fast=np.mean(joined["rest_fast"], axis=1),
            reference_fast=np.percentile(joined["reference_fast"], percentile, axis=1),
        )
        try:
            calibration.validate(self.config.min_reference_ratio)
        except CalibrationError as error:
            self.error = str(error)
            return
        self.result = calibration


@dataclass(frozen=True)
class ActivationEstimate:
    """Normalized activation at the end of one block."""

    sample_index: int
    receive_time: float
    slow: FloatArray
    fast: FloatArray
    tremor: FloatArray
    tremor_ratio_per_channel: FloatArray
    grasp_activation: float
    tremor_level: float
    tremor_ratio: float


class ActivationEstimator:
    """Turns processed blocks into normalized activation using a Calibration."""

    def __init__(self, calibration: Calibration, roles: Sequence[str]) -> None:
        if len(roles) != len(calibration.channel_names):
            raise CalibrationError(
                "Calibration and channel configuration have different channel counts"
            )
        self.calibration = calibration
        self.flexors = np.array([role == "flexor" for role in roles])
        if not np.any(self.flexors):
            raise CalibrationError("At least one flexor channel is required for grasp detection")
        self._slow_span = calibration.reference_slow - calibration.rest_slow
        self._fast_span = calibration.reference_fast - calibration.rest_fast
        if np.any(self._slow_span <= 0) or np.any(self._fast_span <= 0):
            raise CalibrationError("Reference levels must be above rest levels")

    def normalize(self, block: ProcessedBlock) -> tuple[FloatArray, FloatArray, FloatArray]:
        """Normalized slow, fast and tremor activation for every sample, (n_channels, n)."""
        calibration = self.calibration
        slow = np.maximum(
            (block.slow_envelope - calibration.rest_slow[:, None]) / self._slow_span[:, None], 0.0
        )
        fast = np.maximum(
            (block.fast_envelope - calibration.rest_fast[:, None]) / self._fast_span[:, None], 0.0
        )
        tremor = block.tremor_amplitude / self._fast_span[:, None]
        return slow, fast, tremor

    def estimate(self, block: ProcessedBlock) -> ActivationEstimate:
        """Activation at the last sample of the block."""
        slow, fast, tremor = self.normalize(block)
        slow_now, fast_now, tremor_now = slow[:, -1], fast[:, -1], tremor[:, -1]
        ratio_now = block.tremor_ratio[:, -1]
        strongest = int(np.argmax(tremor_now))
        return ActivationEstimate(
            sample_index=block.first_sample_index + block.n_samples - 1,
            receive_time=block.receive_time,
            slow=slow_now,
            fast=fast_now,
            tremor=tremor_now,
            tremor_ratio_per_channel=ratio_now,
            grasp_activation=float(np.mean(slow_now[self.flexors])),
            tremor_level=float(tremor_now[strongest]),
            tremor_ratio=float(ratio_now[strongest]),
        )
