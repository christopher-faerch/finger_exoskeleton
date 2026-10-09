"""TrignoSource against the mock TCU: protocol, decoding, configuration checks and faults."""

import socket
from collections.abc import Iterator
from dataclasses import replace

import numpy as np
import pytest
from numpy.typing import NDArray

from source.config import AppConfig, ChannelConfig, TrignoConfig
from source.mock_tcu import MockSensor, MockTCU, MockTCUSettings
from source.sensor import EMGBlock, SourceConnectionError, SourceTimeoutError
from source.trigno import (
    TrignoConfigurationError,
    TrignoProtocolError,
    TrignoSource,
    _number,
    _yes_no,
)

# Scrambled positions in the 16-channel EMG port buffer, as after channel reallocation (SDK 6.1.3)
POSITIONS = {1: 5, 2: 0, 3: 9, 4: 2}


def encoded_signal(first: int, n: int) -> NDArray[np.float64]:
    """Volts that encode sensor number and sample index: (sensor + 1) * 1e-6 * index."""
    index = first + np.arange(n)
    return np.vstack([(sensor + 1) * 1e-6 * index for sensor in range(4)])


def settings(**changes: object) -> MockTCUSettings:
    """Mock settings with four sensors at the scrambled positions."""
    base = MockTCUSettings(
        sensors=[MockSensor(slot=slot, position=index) for slot, index in POSITIONS.items()]
    )
    return replace(base, **changes)  # type: ignore[arg-type]


def trigno_config(tcu: MockTCU, **changes: object) -> TrignoConfig:
    """Trigno settings pointing at the mock, with short timeouts."""
    config = TrignoConfig(
        host="127.0.0.1",
        command_port=tcu.command_port,
        emg_port=tcu.emg_port,
        connect_timeout_s=1.0,
        command_timeout_s=1.0,
        data_timeout_s=0.3,
    )
    return replace(config, **changes)  # type: ignore[arg-type]


@pytest.fixture(name="channels")
def fixture_channels(config: AppConfig) -> tuple[ChannelConfig, ...]:
    """The four configured channels (slots 1-4)."""
    return config.channels


@pytest.fixture(name="tcu")
def fixture_tcu() -> Iterator[MockTCU]:
    """A mock TCU sending each 13.5 ms frame in awkward fragments."""
    with MockTCU(settings(fragment_sizes=(1, 7, 63, 64, 200, 3)), encoded_signal) as tcu:
        yield tcu


def read_samples(source: TrignoSource, n_samples: int) -> list[EMGBlock]:
    """Read blocks until at least n_samples per channel have arrived."""
    blocks: list[EMGBlock] = []
    while sum(block.n_samples for block in blocks) < n_samples:
        block = source.read_block()
        assert block is not None
        blocks.append(block)
    return blocks


def test_decodes_fragmented_multiplexed_stream(
    tcu: MockTCU, channels: tuple[ChannelConfig, ...]
) -> None:
    """Fragmented TCP data decodes into the right channels, in mV, with contiguous indices."""
    source = TrignoSource(trigno_config(tcu), channels)
    source.start()
    try:
        assert source.sample_rate == pytest.approx(2000.0)
        blocks = read_samples(source, 600)
    finally:
        source.stop()

    samples = np.hstack([block.samples for block in blocks])
    index = np.arange(samples.shape[1])
    for row, channel in enumerate(channels):
        # Channel order follows the configuration, not the port position; V -> mV
        expected = channel.trigno_slot * 1e-6 * index * 1000.0
        np.testing.assert_allclose(samples[row], expected, rtol=1e-6, atol=1e-7)
    for previous, block in zip(blocks, blocks[1:]):
        assert block.first_sample_index == previous.first_sample_index + previous.n_samples
    assert [s.start_index for s in source.sensors] == [POSITIONS[c.trigno_slot] for c in channels]
    assert not source.warnings


def test_big_endian_stream(channels: tuple[ChannelConfig, ...]) -> None:
    """The byte order reported by ENDIANNESS? is used for decoding."""
    with MockTCU(settings(endianness="BIG"), encoded_signal) as tcu:
        source = TrignoSource(trigno_config(tcu), channels)
        source.start()
        try:
            samples = np.hstack([b.samples for b in read_samples(source, 100)])
        finally:
            source.stop()
    np.testing.assert_allclose(samples[0, :100], np.arange(100) * 1e-3, atol=1e-7)


@pytest.mark.parametrize(
    ("backwards", "upsampling", "per_frame", "expected"),
    [
        (True, True, 27, 2000.0),  # SDK 6.1.2, EMG Data port, upsample on
        (True, False, 15, 15 / 0.0135),  # 1111.111 Hz
        (False, True, 30, 30 / 0.0135),  # backwards compatibility off: samples per frame / interval
    ],
)
def test_sample_rate_from_documented_rules(
    channels: tuple[ChannelConfig, ...],
    backwards: bool,
    upsampling: bool,
    per_frame: int,
    expected: float,
) -> None:
    """The sample rate is derived from the TCU replies, never assumed."""
    mock_settings = settings(
        backwards_compatibility=backwards, upsampling=upsampling, samples_per_frame=per_frame
    )
    with MockTCU(mock_settings, encoded_signal) as tcu:
        source = TrignoSource(trigno_config(tcu), channels)
        source.start()
        source.stop()
    assert source.sample_rate == pytest.approx(expected)
    assert not source.warnings


def test_expected_sample_rate_mismatch_is_refused(channels: tuple[ChannelConfig, ...]) -> None:
    """A configured expected rate protects against an unexpected TCU setting."""
    with MockTCU(settings(upsampling=False, samples_per_frame=15), encoded_signal) as tcu:
        source = TrignoSource(trigno_config(tcu, expected_sample_rate_hz=2000.0), channels)
        with pytest.raises(TrignoConfigurationError, match="sample rate"):
            source.start()
    assert not source.connected


def test_unpaired_sensor_is_refused(channels: tuple[ChannelConfig, ...]) -> None:
    """Every configured slot must be paired."""
    mock_settings = settings()
    mock_settings.sensors[2].paired = False
    with MockTCU(mock_settings, encoded_signal) as tcu:
        with pytest.raises(TrignoConfigurationError, match="slot 3"):
            TrignoSource(trigno_config(tcu), channels).start()


def test_unexpected_units_are_refused(channels: tuple[ChannelConfig, ...]) -> None:
    """Units other than the configured ones stop the start-up."""
    mock_settings = settings()
    mock_settings.sensors[0].units = "g"
    with MockTCU(mock_settings, encoded_signal) as tcu:
        with pytest.raises(TrignoConfigurationError, match="units"):
            TrignoSource(trigno_config(tcu), channels).start()


def test_one_based_start_index(channels: tuple[ChannelConfig, ...]) -> None:
    """start_index_base = 1 handles a TCU that counts buffer positions from 1."""
    mock_settings = MockTCUSettings(
        sensors=[MockSensor(slot=s, position=s - 1) for s in range(1, 5)], start_index_base=1
    )
    with MockTCU(mock_settings, encoded_signal) as tcu:
        source = TrignoSource(trigno_config(tcu, start_index_base=1), channels)
        source.start()
        try:
            samples = np.hstack([b.samples for b in read_samples(source, 50)])
        finally:
            source.stop()
    np.testing.assert_allclose(samples[1, :50], 2e-3 * np.arange(50), atol=1e-7)


def test_start_index_equal_to_slot_triggers_warning(channels: tuple[ChannelConfig, ...]) -> None:
    """A 1-based TCU read with the 0-based setting reports indices equal to the slots: warn."""
    mock_settings = MockTCUSettings(
        sensors=[MockSensor(slot=s, position=s - 1) for s in range(1, 5)], start_index_base=1
    )
    with MockTCU(mock_settings, encoded_signal) as tcu:
        source = TrignoSource(trigno_config(tcu), channels)
        source.start()
        source.stop()
    assert any("start_index_base" in warning for warning in source.warnings)


def test_connection_refused(channels: tuple[ChannelConfig, ...]) -> None:
    """No TCU listening gives a clear SourceConnectionError."""
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        free_port = probe.getsockname()[1]
    config = TrignoConfig(command_port=free_port, emg_port=free_port, connect_timeout_s=0.5)
    with pytest.raises(SourceConnectionError, match="Trigno Control Utility"):
        TrignoSource(config, channels).start()


def test_data_connection_loss(tcu: MockTCU, channels: tuple[ChannelConfig, ...]) -> None:
    """A dropped data connection raises SourceConnectionError and marks the source disconnected."""
    source = TrignoSource(trigno_config(tcu), channels)
    source.start()
    read_samples(source, 50)
    tcu.drop_data_connection()
    with pytest.raises(SourceConnectionError):
        for _ in range(10_000):
            source.read_block()
    assert not source.connected
    source.stop()


def test_stream_pause_times_out_then_recovers(
    tcu: MockTCU, channels: tuple[ChannelConfig, ...]
) -> None:
    """Silence on the data port raises SourceTimeoutError; data afterwards is read normally."""
    source = TrignoSource(trigno_config(tcu), channels)
    source.start()
    try:
        read_samples(source, 50)
        tcu.pause_stream(True)
        with pytest.raises(SourceTimeoutError):
            for _ in range(10_000):
                source.read_block()
        tcu.pause_stream(False)
        assert read_samples(source, 50)
        assert source.connected
    finally:
        source.stop()


def test_stop_sends_stop_and_quit_and_is_idempotent(
    tcu: MockTCU, channels: tuple[ChannelConfig, ...]
) -> None:
    """stop() ends streaming with documented commands and can be called twice."""
    source = TrignoSource(trigno_config(tcu), channels)
    source.start()
    assert tcu.streaming
    source.stop()
    source.stop()
    assert tcu.received_commands[-2:] == ["STOP", "QUIT"]
    assert "START" in tcu.received_commands
    assert not source.connected
    with pytest.raises(SourceConnectionError):
        source.read_block()


def test_only_documented_commands_are_sent(
    tcu: MockTCU, channels: tuple[ChannelConfig, ...]
) -> None:
    """The client only queries and sends START / STOP / QUIT (no configuration changes)."""
    source = TrignoSource(trigno_config(tcu), channels)
    source.start()
    source.stop()
    allowed = (
        "BACKWARDS COMPATIBILITY?",
        "UPSAMPLING?",
        "FRAME INTERVAL?",
        "MAX SAMPLES EMG?",
        "ENDIANNESS?",
        "BASE FIRMWARE?",
        "START",
        "STOP",
        "QUIT",
    )
    for command in tcu.received_commands:
        assert command in allowed or (
            command.startswith("SENSOR ") and command.endswith("?")
        ), command


def test_reply_parsing() -> None:
    """YES/NO, ON/OFF and numeric replies, and errors for anything else."""
    assert _yes_no("YES", "q") and not _yes_no("NO", "q")
    assert _yes_no("UPSAMPLING ON", "q") and not _yes_no("UPSAMPLING OFF", "q")
    assert _number("0.0135", "q") == pytest.approx(0.0135)
    with pytest.raises(TrignoProtocolError):
        _yes_no("MAYBE", "q")
    with pytest.raises(TrignoProtocolError):
        _number("INVALID COMMAND", "q")
