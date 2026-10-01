from source.import_data import EMGData, Muscle
from source.process_signal import SignalProcess
from source.visualize_data import Visualizer


def show_data() -> None:
    """Load one EMG recording and plot it raw, filtered and smoothed."""
    sample_rate: float = 2000.0
    window_size: int = 200

    emg_data = EMGData("ECR")
    emg_data.load_from_file("10_raw.csv", Muscle.ECR)

    # Same cleanup as the paper's online filter: band-pass + 50 Hz power line notch
    processer = SignalProcess(emg_data, sample_rate)
    band_passed_data: EMGData = processer.band_pass_filter(20.0, 450.0)
    filtered_data: EMGData = SignalProcess(band_passed_data, sample_rate).notch_filter(50.0)
    filtered_data.set_name("ECR filtered")

    smooth_data: EMGData = processer.smooth_signal(window_size)
    filtered_processer = SignalProcess(filtered_data, sample_rate)
    filtered_smooth_data: EMGData = filtered_processer.smooth_signal(window_size)

    # calculate_jerk() is the first derivative. Of an EMG voltage this is not a
    # physical jerk, so it is labelled as what it is
    derivative_data = EMGData("ECR derivative", "mV/s")
    derivative_data.data = SignalProcess.calculate_jerk(emg_data.get_data(), sample_rate)

    # Raw ECR sits at a large DC offset, so raw and filtered each get their own panel
    plotter = Visualizer(sample_rate)
    plotter.plot(emg_data, filtered_data, derivative_data, smooth_data, filtered_smooth_data)
    plotter.plot_spectrum(emg_data, filtered_data)
    plotter.show()


def flitered_vs_raw() -> None:
    """Plot the dataset's filtered ECR signal against its raw signal and filter_replication()."""
    emg_data_raw = EMGData("ECR RAW")
    emg_data_raw.load_from_file("10_raw.csv", Muscle.ECR)
    emg_data_filtered = EMGData("ECR FILTERED")
    emg_data_filtered.load_from_file("10_filtered.csv", Muscle.ECR)

    sample_rate: float = 2000.0
    emg_data_replicated = SignalProcess(emg_data_raw, sample_rate).filter_replication()

    difference = EMGData("FILTERED - REPLICATION")
    difference.data = [
        filtered - replicated
        for filtered, replicated
        in zip(emg_data_filtered.get_data(), emg_data_replicated.get_data())
    ]

    tol = 0.001
    max_diff = 0
    cnt = 0
    for data in difference.data:
        if data > tol: 
            cnt += 1 
        if abs(data) > abs(max_diff):
            max_diff = data

    print(f"Max diff: {max_diff} amount over tol: {cnt/len(difference.data) * 100:.4f} %")

    plotter = Visualizer(sample_rate)
    plotter.plot(emg_data_raw, emg_data_filtered, emg_data_replicated)
    plotter.show()


def main() -> None:
    """Entry point."""
    flitered_vs_raw()


if __name__ == '__main__':
    main()
