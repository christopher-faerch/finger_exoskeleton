"""Synthetic and recorded sources share the EMGSource interface."""

from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
from conftest import offline_synthetic

from source.config import AppConfig, RecordedConfig
from source.sensor import (
    CALIBRATION_SCENARIO,
    EMGBlock,
    RecordedSource,
    Segment,
    SourceConnectionError,
    SyntheticSource,
)


def read(source: SyntheticSource | RecordedSource, seconds: float) -> list[EMGBlock]:
    """Read about `seconds` of blocks."""
    blocks: list[EMGBlock] = []
    while sum(b.n_samples for b in blocks) < seconds * source.sample_rate:
        block = source.read_block()
        if block is None:
            break
        blocks.append(block)
    return blocks


def test_synthetic_blocks_are_contiguous_and_shaped(config: AppConfig) -> None:
    """Blocks have the configured size, all channels, and contiguous indices."""
    source = offline_synthetic(config)
    source.start()
    blocks = read(source, 1.0)
    assert all(b.samples.shape == (4, config.synthetic.block_size) for b in blocks)
    assert [b.first_sample_index for b in blocks] == [
        i * config.synthetic.block_size for i in range(len(blocks))
    ]


def test_synthetic_levels_follow_the_scenario(config: AppConfig) -> None:
    """Rest is quiet, contractions are strong, and channels are independent."""
    source = offline_synthetic(config)
    source.start()
    samples = np.hstack([b.samples for b in read(source, 5.0)])
    fs = int(config.synthetic.sample_rate_hz)
    rest = samples[:, fs : 2 * fs]
    contraction = samples[:, int(3.5 * fs) : int(4.5 * fs)]  # reference contraction, drive 0.6
    assert np.all(np.std(rest, axis=1) < 0.01)
    assert np.all(np.std(contraction, axis=1) > 0.15)
    correlation = np.corrcoef(contraction)
    assert np.max(np.abs(correlation - np.eye(4))) < 0.1


def test_synthetic_tremor_bursts_alternate(config: AppConfig) -> None:
    """In the tremor segment, the amplitude envelope oscillates at 5 Hz, extensors in antiphase."""
    cycle = (Segment(4.0, "tremor", flexor=0.5, extensor=0.5, tremor=1.0),)
    source = SyntheticSource(
        replace(config.synthetic, realtime=False, powerline_mv=0.0),
        config.channels,
        calibration_segments=(),
        cycle_segments=cycle,
    )
    source.start()
    samples = np.hstack([b.samples for b in read(source, 4.0)])[:, :8000]
    fs = config.synthetic.sample_rate_hz
    envelope = np.abs(samples).reshape(4, -1, 40).mean(axis=2)  # 50 Hz envelope
    spectrum = np.abs(np.fft.rfft(envelope - envelope.mean(axis=1, keepdims=True), axis=1))
    freqs = np.fft.rfftfreq(envelope.shape[1], d=40 / fs)
    assert np.all(np.abs(freqs[np.argmax(spectrum, axis=1)] - 5.0) < 0.5)
    flexor, extensor = config.channel_roles.index("flexor"), config.channel_roles.index("extensor")
    assert np.corrcoef(envelope[flexor], envelope[extensor])[0, 1] < -0.5


def test_synthetic_is_deterministic(config: AppConfig) -> None:
    """The same seed gives the same signal."""
    first, second = offline_synthetic(config), offline_synthetic(config)
    first.start()
    second.start()
    first_block, second_block = first.read_block(), second.read_block()
    assert first_block is not None and second_block is not None
    np.testing.assert_array_equal(first_block.samples, second_block.samples)


def test_synthetic_fault_injection(config: AppConfig) -> None:
    """Disconnects raise SourceConnectionError; dropouts leave a gap in the sample index."""
    source = offline_synthetic(config)
    source.start()
    block = source.read_block()
    assert block is not None
    source.inject_dropout(0.5)
    after = source.read_block()
    assert after is not None
    assert after.first_sample_index == block.n_samples + int(0.5 * source.sample_rate)
    source.inject_disconnect()
    with pytest.raises(SourceConnectionError):
        source.read_block()
    assert not source.connected
    source.start()
    assert source.read_block() is not None


def test_synthetic_scenario_labels(config: AppConfig) -> None:
    """The scenario starts with the calibration part, then repeats the cycle."""
    source = offline_synthetic(config)
    assert source.segment_at(0.1).label == CALIBRATION_SCENARIO[0].label
    calibration_length = sum(s.duration_s for s in CALIBRATION_SCENARIO)
    assert source.segment_at(calibration_length + 0.1).label == "rest"


@pytest.fixture(name="csv_file")
def fixture_csv_file(tmp_path: Path) -> Path:
    """A small recording with a header: column k holds k*1000 + sample index."""
    path = tmp_path / "rec.csv"
    rows = ["a,b,c,d"] + [",".join(str(k * 1000 + i) for k in range(4)) for i in range(100)]
    path.write_text("\n".join(rows), encoding="utf-8")
    return path


def test_recorded_columns_order_and_end(config: AppConfig, csv_file: Path) -> None:
    """Columns follow the channel configuration; a non-looping recording ends with None."""
    source = RecordedSource(
        RecordedConfig(file=str(csv_file), block_size=27, loop=False, realtime=False),
        config.channels,
    )
    source.start()
    blocks = read(source, 1.0)
    samples = np.hstack([b.samples for b in blocks])
    assert samples.shape == (4, 100)
    for row, channel in enumerate(config.channels):
        np.testing.assert_array_equal(samples[row], channel.recorded_column * 1000 + np.arange(100))
    assert source.read_block() is None
    assert blocks[-1].n_samples == 100 - 3 * 27


def test_recorded_loops(config: AppConfig, csv_file: Path) -> None:
    """A looping recording wraps around while the sample index keeps increasing."""
    source = RecordedSource(
        RecordedConfig(file=str(csv_file), block_size=40, loop=True, realtime=False),
        config.channels,
    )
    source.start()
    blocks = []
    for _ in range(5):
        block = source.read_block()
        assert block is not None
        blocks.append(block)
    # 0, 40, 80 (20 samples to the end), then from the start again: 100, 140
    assert [b.first_sample_index for b in blocks] == [0, 40, 80, 100, 140]
    np.testing.assert_array_equal(blocks[3].samples, blocks[0].samples)


def test_recorded_missing_file(config: AppConfig, tmp_path: Path) -> None:
    """A missing recording is reported as a connection error with the path."""
    source = RecordedSource(RecordedConfig(file=str(tmp_path / "missing.csv")), config.channels)
    with pytest.raises(SourceConnectionError, match="missing.csv"):
        source.start()


def test_recorded_dataset_file(config: AppConfig) -> None:
    """The real dataset loads when present (skipped otherwise)."""
    recorded = RecordedSource(replace(config.recorded, realtime=False), config.channels)
    if not recorded.file.exists():
        pytest.skip("dataset not present")
    recorded.start()
    assert recorded.n_recorded_samples == 1_280_000
    block = recorded.read_block()
    assert block is not None and block.samples.shape == (4, config.recorded.block_size)
