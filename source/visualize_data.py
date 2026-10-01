from collections.abc import Sequence

import matplotlib.pyplot as plt
import numpy as np

from source.import_data import EMGData
from source.process_signal import SignalProcess


class Visualizer:
    """Plots EMGData signals, labelling each one with its name."""

    def __init__(self, sample_rate: float = 2000.0) -> None:
        self.sample_rate: float = sample_rate

    def plot(self, *panels: EMGData | Sequence[EMGData], title: str = "EMG") -> None:
        """
        Plot EMG signals over time, one subplot per argument.
        Pass a list of EMGData to overlay several signals in the same subplot.
        All subplots share the time axis, so zooming one zooms all.

        Example: plot([raw, filtered], jerk, [raw_rms, filtered_rms])
        """
        fig, axes = plt.subplots(len(panels), 1, figsize=(12, 2.5 * len(panels)),
                                 sharex=True, squeeze=False)
        fig.suptitle(title)

        for ax, panel in zip(axes[:, 0], panels):
            signals = [panel] if isinstance(panel, EMGData) else panel

            for emg_data in signals:
                data = emg_data.get_data()
                time = np.arange(len(data)) / self.sample_rate
                ax.plot(time, data, label=emg_data.name, alpha=0.8)

            units = ", ".join(dict.fromkeys(emg_data.unit for emg_data in signals))
            ax.set_ylabel(f"Amplitude [{units}]")
            ax.legend(loc="upper right")
            ax.grid(True)

        axes[-1, 0].set_xlabel("Time [s]")
        fig.tight_layout()

    def plot_spectrum(self, *signals: EMGData, title: str = "Power spectral density") -> None:
        """
        Plot the power spectral density (Welch's method) of each signal in dB,
        overlaid in one figure. Comparable to Fig. 6 of the dataset paper.
        """
        fig, ax = plt.subplots(figsize=(12, 4))

        for emg_data in signals:
            processer = SignalProcess(emg_data, self.sample_rate)
            frequencies, psd = processer.power_spectral_density()
            # Tiny floor so a perfect notch does not give log10(0)
            ax.plot(frequencies, 10 * np.log10(psd + 1e-30), label=emg_data.name, alpha=0.8)

        units = ", ".join(dict.fromkeys(emg_data.unit for emg_data in signals))
        ax.set_title(title)
        ax.set_xlabel("Frequency [Hz]")
        ax.set_ylabel(f"PSD [dB {units}²/Hz]")
        ax.legend(loc="upper right")
        ax.grid(True)
        fig.tight_layout()

    @staticmethod
    def show() -> None:
        """
        Show all figures made so far, in separate windows at the same time.
        """
        plt.show()
