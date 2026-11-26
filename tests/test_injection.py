"""The test signal (key k): noise and simulated tremor over the EMG, and its evaluation."""

from pathlib import Path
from types import SimpleNamespace

import matplotlib
import numpy as np
import pytest

import evaluate_injection
import run_emg
import settings
from source.injection import INJECTIONS_FILE, Episode, Injector, parse_spans, read_episodes
from source.injection_evaluation import Run, evaluate_run
from source.pipeline import Acquisition, Pipeline
from source.recording import RecordedSource
from source.signal_processing import EMGBlock, SignalProcessor
from source.stiffness_controller import State

matplotlib.use("Agg")  # draw without opening a window

ROOT = Path(__file__).resolve().parent.parent
RECORDING = ROOT / "measurements" / "20261105-144738_trigno_to-lange-greb"
CUE_RECORDING = ROOT / "measurements" / "20261113-140546_trigno_chris_arm2"
RATE = 2000.0


def blocks(emg: np.ndarray, size: int = 27) -> list[EMGBlock]:
    """A (n_channels, n) signal cut into blocks as a source delivers them."""
    return [EMGBlock(emg[:, start:start + size], start, 0.0)
            for start in range(0, emg.shape[1], size)]


def test_off_leaves_the_block_alone() -> None:
    """While it is off, the injector gives back the very same block."""
    injector = Injector(["flexor"], RATE, message=lambda text: None)
    block = EMGBlock(np.ones((1, 27)), 0, 0.0)
    assert injector.apply(block) is block
    assert not injector.episodes


def test_key_toggles_episodes_with_a_fade() -> None:
    """Two presses give one episode; the signal fades in and back out to the original."""
    messages: list[str] = []
    injector = Injector(["flexor"], RATE, message=messages.append)
    emg = np.random.default_rng(1).standard_normal((1, 8000)) * 0.02
    out = []
    for block in blocks(emg):
        if block.first_index == 27 * 40:
            assert injector.toggle()
        if block.first_index == 27 * 150:
            assert not injector.toggle()
        out.append(injector.apply(block).samples)
    result = np.concatenate(out, axis=1)
    assert len(injector.episodes) == 1
    episode = injector.episodes[0]
    assert episode.start_s == pytest.approx(27 * 40 / RATE)
    assert episode.end_s == pytest.approx(27 * 150 / RATE)
    assert np.array_equal(result[:, :27 * 40], emg[:, :27 * 40])  # untouched before
    fade = round(settings.INJECT_RAMP_S * RATE)
    assert np.array_equal(result[:, 27 * 150 + fade + 27:], emg[:, 27 * 150 + fade + 27:])
    assert not np.allclose(result[:, 27 * 60:27 * 140], emg[:, 27 * 60:27 * 140])
    assert any("Test signal ON" in text for text in messages)


def test_injected_tremor_is_seen_by_the_signal_processing() -> None:
    """A steady contraction with the test signal has tremor depth and ratio above the on-levels."""
    emg = np.random.default_rng(2).standard_normal((1, 20000)) * 0.05  # 1 s rest, then a grip
    injector = Injector(["flexor"], RATE, message=lambda text: None)
    injector.set(True)
    processor = SignalProcessor(1, RATE)
    processed = [processor.process(injector.apply(block)) for block in blocks(emg)]
    depth = np.concatenate([block.depth for block in processed], axis=1)[0]
    ratio = np.concatenate([block.ratio for block in processed], axis=1)[0]
    late = slice(4 * int(RATE), None)  # after the fade and the 0.5 s windows have filled
    assert np.median(depth[late]) > settings.TREMOR_ON
    assert np.median(ratio[late]) > settings.RATIO_ON


def test_schedule_and_injections_file(tmp_path: Path) -> None:
    """--inject spans switch the signal; injections.csv is written and read back."""
    schedule = parse_spans(["0.5:1.0", "0.1:0.3"])
    assert [episode.start_s for episode in schedule] == [0.1, 0.5]
    injector = Injector(["flexor", "extensor"], RATE, message=lambda text: None,
                        schedule=schedule)
    for block in blocks(np.zeros((2, 3000))):
        injector.apply(block)
    path = injector.write(tmp_path, 1.5)
    assert path == tmp_path / INJECTIONS_FILE
    episodes = read_episodes(path)
    assert len(episodes) == 2
    assert episodes[0].start_s == pytest.approx(0.1, abs=27 / RATE)
    assert episodes[1].end_s == pytest.approx(1.0, abs=27 / RATE)
    with pytest.raises(ValueError):
        parse_spans(["3:2"])


def test_run_emg_records_the_test_signal(tmp_path: Path) -> None:
    """run_emg.py --inject with --out saves injections.csv next to emg.csv."""
    code = run_emg.main(["--source", "recorded", "--file", str(RECORDING), "--headless",
                         "--duration", "1.5", "--out", str(tmp_path), "--inject", "0.3:0.8"])
    assert code == 0
    (folder,) = tmp_path.iterdir()
    episodes = read_episodes(folder / INJECTIONS_FILE)
    assert len(episodes) == 1 and episodes[0].start_s == pytest.approx(0.3, abs=0.02)


def test_key_k_in_the_viewer() -> None:
    """k switches the test signal and no longer makes matplotlib's x axis logarithmic."""
    from source.viewer import LiveViewer  # pylint: disable=import-outside-toplevel
    source = RecordedSource(RECORDING, realtime=False)
    pipeline = Pipeline(source.channels, source.sample_rate)
    injector = Injector([channel.role for channel in source.channels], source.sample_rate)
    viewer = LiveViewer(pipeline, Acquisition(source, pipeline, injector=injector))
    viewer.on_key(SimpleNamespace(key="k"))
    assert injector.on
    assert "k" not in matplotlib.rcParams["keymap.xscale"]
    viewer.draw(0)
    assert "Test signal: ON" in viewer.text.get_text()


def test_metrics_of_a_made_up_run() -> None:
    """Detection, latency, release and false alarms from a hand-made state sequence."""
    times = np.arange(0, 20, 0.5)
    states = [State.IDLE] * len(times)
    for i, t in enumerate(times):
        if 2 <= t < 9:
            states[i] = State.GRASP_ARMED
        if 4 <= t < 7:
            states[i] = State.STIFFENED
        if 15 <= t < 16:  # no test signal here: a false alarm
            states[i] = State.STIFFENED
    run = Run(times, states, np.zeros(len(times)))
    result = evaluate_run(run, [Episode(3.0, 6.0), Episode(10.0, 12.0)])
    first, second = result["episodes"]
    assert first["gripping"] and first["stiffened"]
    assert first["latency_s"] == pytest.approx(1.0) and first["release_s"] == pytest.approx(1.0)
    assert not second["gripping"] and not second["stiffened"]
    assert result["detection_rate"] == 1.0 and result["stiffened_at_rest"] == 0
    assert result["false_stiffen_per_minute"] > 0


def test_evaluate_injection_on_a_cue_recording(capsys: pytest.CaptureFixture[str]) -> None:
    """The script replays a recording with and without the test signal and scores it."""
    code = evaluate_injection.main(["--sessions", str(CUE_RECORDING),
                                    "--inject", "51:54", "68.5:72"])
    assert code == 0
    output = capsys.readouterr().out
    assert "detection rate 1.00" in output and "grasp agreement" in output


def test_evaluate_injection_needs_spans() -> None:
    """Without --inject and without injections.csv there is nothing to score."""
    assert evaluate_injection.main(["--sessions", str(RECORDING)]) == 1
