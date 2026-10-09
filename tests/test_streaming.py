"""Streaming processor: filter state continuity, RMS, notch, envelopes and tremor ratio."""

from dataclasses import replace

import numpy as np
import pytest
from numpy.typing import NDArray
from scipy.signal import butter, sosfilt, sosfilt_zi

from source.config import ProcessingConfig
from source.import_data import EMGData
from source.process_signal import SignalProcess
from source.sensor import EMGBlock
from source.streaming import MovingRMS, StreamingFilter, StreamingProcessor

FS = 2000.0


def blocks_of(signal: NDArray[np.float64], sizes: list[int]) -> list[EMGBlock]:
    """Split (n_channels, n) into consecutive blocks of varying sizes."""
    blocks, start, i = [], 0, 0
    while start < signal.shape[1]:
        size = sizes[i % len(sizes)]
        blocks.append(EMGBlock(signal[:, start : start + size], start, 0.0))
        start += size
        i += 1
    return blocks


def test_filter_state_carries_across_blocks() -> None:
    """Blocks of any size give exactly the one-shot causal result."""
    rng = np.random.default_rng(0)
    signal = rng.standard_normal((3, 5000)) + np.array([[2.0], [-1.0], [0.0]])
    sos = butter(4, [20, 450], btype="bandpass", fs=FS, output="sos")

    streaming = StreamingFilter(sos, 3, steady_state_start=True)
    pieces = [streaming.process(block.samples) for block in blocks_of(signal, [1, 27, 13, 400, 2])]

    zi = sosfilt_zi(sos)[:, None, :] * signal[:, 0][None, :, None]
    expected, _ = sosfilt(sos, signal, axis=1, zi=zi)
    np.testing.assert_allclose(np.hstack(pieces), expected, rtol=1e-10, atol=1e-12)


def test_steady_state_start_has_no_offset_transient() -> None:
    """A constant electrode offset does not produce a start-up transient."""
    sos = butter(4, [20, 450], btype="bandpass", fs=FS, output="sos")
    output = StreamingFilter(sos, 1, steady_state_start=True).process(np.full((1, 2000), -3.0))
    assert np.max(np.abs(output)) < 1e-9


def test_moving_rms_matches_offline_smooth_signal() -> None:
    """The streaming RMS equals the existing offline SignalProcess.smooth_signal."""
    rng = np.random.default_rng(1)
    signal = rng.standard_normal((2, 3000))
    rms = MovingRMS(2, window=200)
    streamed = np.hstack([rms.process(block.samples) for block in blocks_of(signal, [27, 5, 333])])
    for channel in range(2):
        data = EMGData("x")
        data.data = signal[channel].tolist()
        offline = SignalProcess(data, FS).smooth_signal(window_size=200).get_data()
        np.testing.assert_allclose(streamed[channel], offline, rtol=1e-9)


def test_moving_rms_window_of_one_is_rectification() -> None:
    """A one-sample window is the absolute value."""
    signal = np.array([[1.0, -2.0, 3.0, -4.0]])
    np.testing.assert_allclose(MovingRMS(1, 1).process(signal), np.abs(signal))


def make_processor(**changes: object) -> StreamingProcessor:
    """Processor with the default configuration plus changes, two channels."""
    config = replace(ProcessingConfig(), **changes)  # type: ignore[arg-type]
    return StreamingProcessor(config, 2, FS)


def test_processor_blocks_equal_one_shot() -> None:
    """Every output of the full processor is independent of the block size."""
    rng = np.random.default_rng(2)
    signal = rng.standard_normal((2, 6000)) * 0.1
    one = make_processor().process(EMGBlock(signal, 0, 0.0))
    many = make_processor()
    parts = [many.process(block) for block in blocks_of(signal, [27, 100, 1, 64])]
    for name in ("filtered", "slow_envelope", "fast_envelope", "tremor_amplitude", "tremor_ratio"):
        np.testing.assert_allclose(
            np.hstack([getattr(part, name) for part in parts]),
            getattr(one, name),
            rtol=1e-9,
            atol=1e-12,
            err_msg=name,
        )


def test_notch_removes_power_line() -> None:
    """50 Hz interference is removed; 120 Hz EMG-band content is kept."""
    t = np.arange(20000) / FS
    signal = np.vstack([np.sin(2 * np.pi * 50 * t), np.sin(2 * np.pi * 120 * t)])
    out = make_processor(notch_enabled=True).process(EMGBlock(signal, 0, 0.0)).filtered[:, 10000:]
    assert np.std(out[0]) < 0.02
    assert np.std(out[1]) == pytest.approx(np.sqrt(0.5), rel=0.05)
    without = (
        make_processor(notch_enabled=False).process(EMGBlock(signal, 0, 0.0)).filtered[:, 10000:]
    )
    assert np.std(without[0]) > 0.5


def test_channels_are_processed_independently() -> None:
    """Silence on one channel stays silent while the other is active."""
    rng = np.random.default_rng(3)
    signal = np.vstack([rng.standard_normal(4000), np.zeros(4000)])
    out = make_processor().process(EMGBlock(signal, 0, 0.0))
    assert np.max(out.slow_envelope[1]) == 0.0
    assert np.mean(out.slow_envelope[0, 1000:]) > 0.5


def test_tremor_ratio_separates_rhythmic_from_random() -> None:
    """5 Hz bursting gives a ratio near 1; a steady contraction gives a low ratio."""
    rng = np.random.default_rng(4)
    t = np.arange(int(8 * FS)) / FS
    noise = rng.standard_normal((2, t.size))
    bursting = 0.5 * (1 + np.sin(2 * np.pi * 5 * t))
    signal = np.vstack([noise[0] * bursting, noise[1] * 0.5])
    out = make_processor().process(EMGBlock(signal, 0, 0.0))
    ratio = out.tremor_ratio[:, int(3 * FS) :]
    assert np.median(ratio[0]) > 0.85
    assert np.median(ratio[1]) < 0.7
    assert np.median(out.tremor_amplitude[0, int(3 * FS) :]) > 5 * np.median(
        out.tremor_amplitude[1, int(3 * FS) :]
    )


def test_rejects_band_above_nyquist() -> None:
    """A band-pass edge above Nyquist is a configuration error, not a silent failure."""
    with pytest.raises(ValueError, match="Nyquist"):
        StreamingProcessor(ProcessingConfig(), 2, sample_rate=800.0)


def test_reset_restarts_filters() -> None:
    """After reset() the processor behaves like a new one."""
    rng = np.random.default_rng(5)
    signal = rng.standard_normal((2, 1000))
    processor = make_processor()
    processor.process(EMGBlock(rng.standard_normal((2, 1000)), 0, 0.0))
    processor.reset()
    np.testing.assert_allclose(
        processor.process(EMGBlock(signal, 0, 0.0)).slow_envelope,
        make_processor().process(EMGBlock(signal, 0, 0.0)).slow_envelope,
    )
