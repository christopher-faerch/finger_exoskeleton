import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray
import pytest

from source.import_data import DATA_PATH, EMGData, Muscle
from source.process_signal import DATASET_SEGMENT_SECONDS, SignalProcess
from source.visualize_data import Visualizer

SAMPLE_RATE = 2000.0


def make_emg(name: str, samples: NDArray[np.float64]) -> EMGData:
    """Wrap a numpy array in a named EMGData object."""
    emg_data = EMGData(name)
    emg_data.data = samples.tolist()
    return emg_data


def sine(frequency: float, seconds: float = 2.0) -> NDArray[np.float64]:
    """A unit sine wave sampled at SAMPLE_RATE."""
    time = np.arange(int(seconds * SAMPLE_RATE)) / SAMPLE_RATE
    result: NDArray[np.float64] = np.sin(2 * np.pi * frequency * time)
    return result


def test_get_data_without_data_raises() -> None:
    """get_data() raises NoDataError before any data is loaded."""
    with pytest.raises(EMGData.NoDataError):
        EMGData("empty").get_data()


def test_smooth_signal_rms_of_constant() -> None:
    """The moving RMS of a constant signal is that constant."""
    emg_data = make_emg("ECR", np.full(500, 3.0))
    smoothed = SignalProcess(emg_data, SAMPLE_RATE).smooth_signal(window_size=50)
    assert np.allclose(smoothed.get_data(), 3.0)
    assert smoothed.name == "ECR RMS (50)"


def test_high_pass_filter_removes_low_frequencies() -> None:
    """A 20 Hz high-pass removes a 1 Hz drift but keeps a 100 Hz component."""
    emg_data = make_emg("ECR", sine(1.0) + sine(100.0))
    filtered = SignalProcess(emg_data, SAMPLE_RATE).high_pass_filter(20.0)

    # Ignore the edges, where the filter has not settled
    middle = np.asarray(filtered.get_data())[500:-500]
    assert np.allclose(middle, sine(100.0)[500:-500], atol=0.05)
    assert filtered.name == "ECR high-pass 20 Hz"


def test_band_pass_filter_keeps_only_the_emg_band() -> None:
    """A 20-450 Hz band-pass removes 1 Hz and 800 Hz but keeps 100 Hz."""
    emg_data = make_emg("ECR", sine(1.0) + sine(100.0) + sine(800.0))
    filtered = SignalProcess(emg_data, SAMPLE_RATE).band_pass_filter(20.0, 450.0)

    middle = np.asarray(filtered.get_data())[500:-500]
    assert np.allclose(middle, sine(100.0)[500:-500], atol=0.05)
    assert filtered.name == "ECR band-pass 20-450 Hz"


def test_notch_filter_removes_power_line() -> None:
    """A 50 Hz notch removes 50 Hz power line noise but keeps 100 Hz."""
    emg_data = make_emg("ECR", sine(50.0, seconds=6.0) + sine(100.0, seconds=6.0))
    filtered = SignalProcess(emg_data, SAMPLE_RATE).notch_filter(50.0)

    # A narrow notch takes ~1 s to settle, so ignore 2 s at each edge
    middle = np.asarray(filtered.get_data())[4000:-4000]
    assert np.allclose(middle, sine(100.0, seconds=6.0)[4000:-4000], atol=0.05)
    assert filtered.name == "ECR notch 50 Hz"


def test_power_spectral_density_peaks_at_signal_frequency() -> None:
    """The Welch PSD of a 120 Hz sine peaks at 120 Hz."""
    emg_data = make_emg("ECR", sine(120.0, seconds=10.0))
    frequencies, psd = SignalProcess(emg_data, SAMPLE_RATE).power_spectral_density()
    assert frequencies[np.argmax(psd)] == pytest.approx(120.0, abs=1.0)


@pytest.mark.skipif(
    not (DATA_PATH / "10_raw.csv").exists() or not (DATA_PATH / "10_filtered.csv").exists(),
    reason="needs data/sEMG_online/10_raw.csv and 10_filtered.csv",
)
@pytest.mark.parametrize("muscle", list(Muscle))
def test_filter_replication_matches_filtered_file(muscle: Muscle) -> None:
    """filter_replication() on 10_raw.csv reproduces 10_filtered.csv."""
    raw = EMGData("raw")
    raw.load_from_file("10_raw.csv", muscle)
    filtered = EMGData("filtered")
    filtered.load_from_file("10_filtered.csv", muscle)

    replicated = np.asarray(SignalProcess(raw, SAMPLE_RATE).filter_replication().get_data())
    expected = np.asarray(filtered.get_data())

    assert np.corrcoef(replicated, expected)[0, 1] > 0.998
    # Away from the first 200 ms after each segment join the match is within a few uV
    error = np.abs(replicated - expected)
    segment_length = int(DATASET_SEGMENT_SECONDS * SAMPLE_RATE)
    for start in range(0, len(error), segment_length):
        error[start:start + int(0.2 * SAMPLE_RATE)] = 0.0
    assert error.max() < 0.005


def test_plot_one_subplot_per_argument_labelled_by_name() -> None:
    """plot() makes one subplot per argument and labels each line with its name."""
    raw = make_emg("raw", sine(100.0))
    filtered = make_emg("filtered", sine(100.0))
    rms = make_emg("rms", np.ones(10))

    Visualizer(SAMPLE_RATE).plot([raw, filtered], rms)

    axes = plt.gcf().get_axes()
    labels = [[line.get_label() for line in ax.get_lines()] for ax in axes]
    assert labels == [["raw", "filtered"], ["rms"]]
    plt.close("all")


def test_plot_labels_y_axis_with_unit() -> None:
    """plot() puts each panel's unit on its y-axis."""
    derivative = EMGData("derivative", "mV/s")
    derivative.data = [0.0, 1.0]

    Visualizer(SAMPLE_RATE).plot(make_emg("raw", np.zeros(2)), derivative)

    ylabels = [ax.get_ylabel() for ax in plt.gcf().get_axes()]
    assert ylabels == ["Amplitude [mV]", "Amplitude [mV/s]"]
    plt.close("all")


def test_plot_spectrum_labels_each_signal() -> None:
    """plot_spectrum() overlays one line per signal, labelled by name."""
    Visualizer(SAMPLE_RATE).plot_spectrum(make_emg("a", sine(50.0)), make_emg("b", sine(80.0)))

    labels = [line.get_label() for line in plt.gca().get_lines()]
    assert labels == ["a", "b"]
    plt.close("all")
