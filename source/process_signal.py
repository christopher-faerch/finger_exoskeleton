from collections import deque
from math import sqrt
import numpy as np
from numpy.typing import NDArray
from scipy.signal import butter, filtfilt, iirnotch, lfilter, lfiltic, sosfiltfilt, welch
from .import_data import EMGData

# The dataset was recorded as one segment per cycle: 104 s cycle + 30 s rest
# (docs/dataset_emg_hand_gestures.pdf, Fig. 4)
DATASET_SEGMENT_SECONDS = 134.0


class SignalProcess:
    """Signal processing on one EMGData signal sampled at sample_rate Hz."""

    def __init__(self, data: EMGData, sample_rate: float = 2000.0) -> None:
        self.data = data
        self.sample_rate = sample_rate

    def smooth_signal(self, window_size: int = 100) -> EMGData:
        """
        Smoothening the signal using moving RMS
        """
        raw_data = self.data.get_data()

        smoothed_data = []
        window: deque[float] = deque()
        sum_squares = 0.0

        for sample in raw_data:
            # Add newest sample
            window.append(sample)
            sum_squares += sample ** 2

            # Remove oldest sample
            if len(window) > window_size:
                old_sample = window.popleft()
                sum_squares -= old_sample ** 2

            # Calculate RMS
            rms = sqrt(sum_squares / len(window))
            smoothed_data.append(rms)

        result = EMGData(f"{self.data.name} RMS ({window_size})", self.data.unit)
        result.data = smoothed_data

        return result

    def high_pass_filter(self, cutoff: float = 20.0, order: int = 4) -> EMGData:
        """
        Remove baseline drift and motion artifacts below the cutoff frequency
        using a zero-phase Butterworth high-pass filter
        """
        raw_data = self.data.get_data()

        sos = butter(order, cutoff, btype="highpass", fs=self.sample_rate, output="sos")
        filtered_data = sosfiltfilt(sos, raw_data)

        result = EMGData(f"{self.data.name} high-pass {cutoff:g} Hz", self.data.unit)
        result.data = filtered_data.tolist()

        return result

    def band_pass_filter(self, low: float = 20.0, high: float = 450.0, order: int = 4) -> EMGData:
        """
        Keep only the sEMG band between low and high using a zero-phase
        Butterworth band-pass filter. Removes baseline drift and motion
        artifacts below low and high-frequency noise above high.
        """
        raw_data = self.data.get_data()

        sos = butter(order, [low, high], btype="bandpass", fs=self.sample_rate, output="sos")
        filtered_data = sosfiltfilt(sos, raw_data)

        result = EMGData(f"{self.data.name} band-pass {low:g}-{high:g} Hz", self.data.unit)
        result.data = filtered_data.tolist()

        return result

    def notch_filter(self, frequency: float = 50.0, quality: float = 30.0) -> EMGData:
        """
        Remove power line interference at frequency (50 Hz in the dataset's
        country) using a zero-phase IIR notch filter. Higher quality gives a
        narrower notch.
        """
        raw_data = self.data.get_data()

        b, a = iirnotch(frequency, quality, fs=self.sample_rate)
        filtered_data = filtfilt(b, a, raw_data)

        result = EMGData(f"{self.data.name} notch {frequency:g} Hz", self.data.unit)
        result.data = filtered_data.tolist()

        return result

    def filter_replication(self) -> EMGData:
        """
        Recreate the dataset's *_filtered.csv signal from its *_raw.csv signal.

        Source: docs/dataset_emg_hand_gestures.pdf (Ozdemir et al., Data in Brief
        41 (2022) 107921). The paper describes the BIOPAC BSL 4.0 online filter as
        a "sixth-order" Butterworth 5-500 Hz band-pass plus a second-order 50 Hz
        notch. Measured against 10_raw.csv / 10_filtered.csv, it actually is:

        - 2nd-order Butterworth high-pass at 5 Hz
        - 2nd-order Butterworth low-pass at 500 Hz
        - 2nd-order notch at 50 Hz with Q = 1 (a very wide notch)

        (2 + 2 + 2 = the paper's "sixth order"). It is causal (online), its
        history starts filled with the first sample, and the steps in the raw
        signal where the per-cycle segments were joined (every 134 s) are not
        passed through. Matches the filtered file to within ~0.003 mV, except for
        the first ~200 ms after each join, where BIOPAC handles the step slightly
        differently (up to ~0.07 mV).
        """
        signal = np.asarray(self.data.get_data(), dtype=np.float64)

        # Remove the step at each segment join
        segment_length = int(DATASET_SEGMENT_SECONDS * self.sample_rate)
        for start in range(segment_length, len(signal), segment_length):
            signal[start:] -= signal[start] - signal[start - 1]

        filters = [
            butter(2, 5.0, btype="highpass", fs=self.sample_rate),
            butter(2, 500.0, btype="lowpass", fs=self.sample_rate),
            iirnotch(50.0, 1.0, fs=self.sample_rate),
        ]
        for b, a in filters:
            history = [signal[0], signal[0]]
            initial_state = lfiltic(b, a, y=history, x=history)
            signal, _ = lfilter(b, a, signal, zi=initial_state)

        result = EMGData(f"{self.data.name} filter replication", self.data.unit)
        result.data = signal.tolist()

        return result

    @staticmethod
    def calculate_jerk(signal: list[float], sample_rate: float) -> list[float]:
        """First time derivative of the signal."""
        dt = 1.0 / sample_rate
        jerk = np.gradient(signal, dt)

        return jerk.tolist()

    @staticmethod
    def calculate_third_derivative(signal: list[float], sample_rate: float) -> list[float]:
        """Third time derivative of the signal."""
        dt = 1.0 / sample_rate

        first_derivative = np.gradient(signal, dt)
        second_derivative = np.gradient(first_derivative, dt)
        third_derivative = np.gradient(second_derivative, dt)

        return third_derivative.tolist()

    def fourier_transform(self) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Return (frequencies in Hz, magnitude) of the mean-removed signal."""
        signal = np.asarray(self.data.get_data())

        signal = signal - np.mean(signal)
        fft = np.fft.rfft(signal)

        frequencies = np.fft.rfftfreq(
            len(signal),
            d=1 / self.sample_rate
        )

        magnitude = np.abs(fft)
        return frequencies, magnitude

    def power_spectral_density(
        self, segment_length: int = 4096
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """
        Return (frequencies in Hz, PSD in unit^2/Hz) estimated with Welch's method,
        as in Fig. 6 of the dataset paper. Averaging over segments of
        segment_length samples gives a much less noisy spectrum than one FFT
        of the whole recording.
        """
        signal = np.asarray(self.data.get_data())
        segment_length = min(segment_length, len(signal))
        frequencies, psd = welch(signal, fs=self.sample_rate, nperseg=segment_length)
        return frequencies, psd
