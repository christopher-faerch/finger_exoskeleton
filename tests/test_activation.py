"""Calibration and normalized activation."""

from pathlib import Path

import numpy as np
import pytest

from source.activation import ActivationEstimator, Calibration, CalibrationError, Calibrator
from source.config import CalibrationConfig
from source.streaming import ProcessedBlock

FS = 1000.0
NAMES = ("F1", "E1")


def block(
    first: int,
    slow: tuple[float, float],
    fast: tuple[float, float],
    tremor: tuple[float, float] = (0.0, 0.0),
    n: int = 100,
) -> ProcessedBlock:
    """A processed block with constant envelopes."""

    def constant(values: tuple[float, float]) -> np.ndarray:
        return np.repeat(np.array(values, dtype=float)[:, None], n, axis=1)

    return ProcessedBlock(
        raw=np.zeros((2, n)),
        filtered=np.zeros((2, n)),
        slow_envelope=constant(slow),
        fast_envelope=constant(fast),
        tremor_amplitude=constant(tremor),
        tremor_ratio=constant((0.9, 0.5)),
        first_sample_index=first,
        receive_time=0.0,
        sample_rate=FS,
    )


def run_calibration(rest: tuple[float, float], reference: tuple[float, float]) -> Calibrator:
    """Feed 1 s rest, then 1 s reference contraction."""
    calibrator = Calibrator(
        CalibrationConfig(
            baseline_window_s=(0.0, 1.0),
            reference_window_s=(1.0, 2.0),
            reference_percentile=95.0,
            min_reference_ratio=3.0,
        ),
        NAMES,
        FS,
    )
    for i in range(10):
        calibrator.update(block(i * 100, rest, rest))
    for i in range(10, 21):
        calibrator.update(block(i * 100, reference, reference))
    return calibrator


def test_calibration_from_windows() -> None:
    """Rest and reference levels come from their windows."""
    calibrator = run_calibration((0.01, 0.02), (0.5, 0.4))
    assert calibrator.done and calibrator.result is not None
    np.testing.assert_allclose(calibrator.result.rest_slow, [0.01, 0.02])
    np.testing.assert_allclose(calibrator.result.reference_slow, [0.5, 0.4])
    assert calibrator.phase == "done"


def test_weak_reference_is_rejected() -> None:
    """A reference contraction barely above rest fails calibration with a clear message."""
    calibrator = run_calibration((0.01, 0.02), (0.02, 0.4))
    assert calibrator.result is None
    assert calibrator.error is not None and "F1" in calibrator.error
    assert calibrator.phase == "failed"


def test_calibration_phase_follows_stream_time() -> None:
    """The phase tells the user what to do."""
    calibrator = Calibrator(
        CalibrationConfig(baseline_window_s=(0.0, 1.0), reference_window_s=(1.5, 2.0)), NAMES, FS
    )
    assert calibrator.phase == "rest (baseline)"
    for i in range(12):
        calibrator.update(block(i * 100, (0.01, 0.01), (0.01, 0.01)))
    assert calibrator.phase == "prepare reference contraction"


def test_normalization_and_summaries() -> None:
    """0 at rest, 1 at reference; grasp uses flexors only; tremor scaled by the fast span."""
    calibration = run_calibration((0.01, 0.02), (0.51, 0.42)).result
    assert calibration is not None
    estimator = ActivationEstimator(calibration, ("flexor", "extensor"))

    rest = estimator.estimate(block(0, (0.01, 0.02), (0.01, 0.02)))
    assert rest.grasp_activation == pytest.approx(0.0)
    full = estimator.estimate(block(0, (0.51, 0.02), (0.51, 0.02), tremor=(0.1, 0.2)))
    assert full.grasp_activation == pytest.approx(1.0)
    np.testing.assert_allclose(full.slow, [1.0, 0.0], atol=1e-12)
    # Extensor tremor 0.2 / (0.42 - 0.02) = 0.5 is the largest
    assert full.tremor_level == pytest.approx(0.5)
    assert full.tremor_ratio == pytest.approx(0.5)
    below_rest = estimator.estimate(block(0, (0.0, 0.0), (0.0, 0.0)))
    assert below_rest.grasp_activation == 0.0


def test_save_and_load(tmp_path: Path) -> None:
    """A saved calibration loads back identically."""
    calibration = run_calibration((0.01, 0.02), (0.5, 0.4)).result
    assert calibration is not None
    path = tmp_path / "calibration.json"
    calibration.save(path)
    loaded = Calibration.load(path)
    assert loaded.channel_names == NAMES
    np.testing.assert_allclose(loaded.reference_fast, calibration.reference_fast)
    with pytest.raises(CalibrationError):
        Calibration.load(tmp_path / "missing.json")


def test_estimator_requires_a_flexor() -> None:
    """Grasp detection is impossible without a flexor channel."""
    calibration = run_calibration((0.01, 0.02), (0.5, 0.4)).result
    assert calibration is not None
    with pytest.raises(CalibrationError):
        ActivationEstimator(calibration, ("extensor", "extensor"))
