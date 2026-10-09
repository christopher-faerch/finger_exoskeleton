"""
Monitors for a running Pipeline.

LiveMonitor: a matplotlib window that refreshes ~10 times per second with
  raw and filtered EMG, activation envelopes, controller state, stiffness,
  connection status, sample rate, timing, faults and the e-stop.
  Keys: a = arm, e = emergency stop, r = reset e-stop, c = recalibrate,
        q = quit, plus any fault-injection keys passed in (d, s, ...).

ConsoleMonitor: one status line per period, for headless runs.
"""

from collections.abc import Callable
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation
from matplotlib.backend_bases import Event, KeyEvent

from .runtime import MonitorSnapshot, Pipeline

FaultInjectors = dict[str, tuple[str, Callable[[], None]]]

STATE_COLORS = {
    "DISARMED": "#9e9e9e",
    "IDLE": "#4caf50",
    "GRASP_ARMED": "#2196f3",
    "STIFFENED": "#ff9800",
    "FAULT": "#e53935",
}


def _stacked(values: np.ndarray[Any, Any]) -> tuple[np.ndarray[Any, Any], float]:
    """Remove each channel's mean and offset the channels for a stacked plot."""
    if values.size == 0:
        return values, 1.0
    centred = values - values.mean(axis=1, keepdims=True)
    spread = float(np.percentile(np.abs(centred), 99)) if centred.size else 1.0
    spacing = max(spread * 2.5, 1e-6)
    offsets = spacing * np.arange(values.shape[0])[::-1, None]
    return centred + offsets, spacing


def _last(values: np.ndarray[Any, Any]) -> float:
    """Last value of a history, 0 when empty."""
    return float(values[-1]) if values.size else 0.0


def status_lines(snapshot: MonitorSnapshot) -> list[str]:
    """Text status shared by both monitors."""
    timings = snapshot.timings
    counters = snapshot.counters

    def ms(name: str) -> str:
        last, mean, peak = timings.get(name, (0.0, 0.0, 0.0))
        return f"{mean * 1000:6.2f} avg {peak * 1000:6.2f} max (last {last * 1000:.2f}) ms"

    lines = [
        f"State:       {snapshot.state}{'  (armed)' if snapshot.armed else ''}",
        f"Stiffness:   requested {_last(snapshot.requested_stiffness):.2f}"
        f"  applied {_last(snapshot.applied_stiffness):.2f}",
        f"E-STOP:      {'ACTIVE - ' + snapshot.estop_reason if snapshot.estop else 'off'}",
        f"Faults:      {', '.join(snapshot.faults) or 'none'}",
        f"Connection:  {'connected' if snapshot.connected else 'DISCONNECTED'}",
        f"Calibration: {snapshot.calibration_phase}",
        f"Sample rate: {snapshot.sample_rate:.3f} Hz",
        f"Transition:  {snapshot.last_transition}",
        "",
        f"Acquisition read  {ms('acquisition read')}",
        f"Block interval    {ms('block interval')}",
        f"Processing/block  {ms('processing per block')}",
        f"Block age         {ms('block age at processing')}",
        f"Controller tick   {ms('controller tick')}",
        f"Control period    {ms('control period')}",
        "",
        "Counters: " + ", ".join(f"{name} {value}" for name, value in sorted(counters.items())),
        "",
    ]
    lines += [f"{key}: {value}" for key, value in snapshot.source_info.items()]
    lines += [""] + list(snapshot.messages)
    return lines


class LiveMonitor:  # pylint: disable=too-many-instance-attributes
    """matplotlib live view; run() blocks until the window is closed or q is pressed."""

    def __init__(self, pipeline: Pipeline, fault_injectors: FaultInjectors | None = None) -> None:
        self.pipeline = pipeline
        self.fault_injectors = fault_injectors or {}
        self.figure = plt.figure(figsize=(15, 9))
        self.figure.canvas.manager.set_window_title(  # type: ignore[union-attr]
            "EMG exoskeleton monitor (DRY RUN - no physical actuation)"
        )
        grid = self.figure.add_gridspec(4, 2, width_ratios=(2.2, 1.0), hspace=0.45)
        self.ax_raw = self.figure.add_subplot(grid[0, 0])
        self.ax_filtered = self.figure.add_subplot(grid[1, 0])
        self.ax_activation = self.figure.add_subplot(grid[2, 0])
        self.ax_control = self.figure.add_subplot(grid[3, 0])
        self.ax_text = self.figure.add_subplot(grid[:, 1])
        self.ax_text.axis("off")
        self._animation: FuncAnimation | None = None
        self.figure.canvas.mpl_connect("key_press_event", self._on_key)

    def help_text(self) -> str:
        """Key bindings."""
        keys = ["a arm", "e E-STOP", "r reset e-stop", "c recalibrate", "q quit"]
        keys += [f"{key} {label}" for key, (label, _) in self.fault_injectors.items()]
        return "Keys: " + " | ".join(keys)

    def _on_key(self, event: Event) -> None:
        if not isinstance(event, KeyEvent) or event.key is None:
            return
        key = event.key.lower()
        actions: dict[str, Callable[[], None]] = {
            "a": self.pipeline.arm,
            "e": lambda: self.pipeline.trigger_estop("operator key"),
            "r": self.pipeline.reset_estop,
            "c": self.pipeline.recalibrate,
            "q": lambda: plt.close(self.figure),
        }
        actions.update({k: action for k, (_, action) in self.fault_injectors.items()})
        if key in actions:
            actions[key]()

    def _draw(self, _frame: int) -> None:
        if not self.pipeline.running:
            self.figure.suptitle(
                "Control loop stopped: " + (self.pipeline.error or "finished"), color="#e53935"
            )
        snapshot = self.pipeline.snapshot()
        names = snapshot.channel_names
        emg_t = snapshot.emg_time - (snapshot.emg_time[-1] if snapshot.emg_time.size else 0.0)

        for ax, data, title in (
            (self.ax_raw, snapshot.raw, "Raw EMG (mean removed)"),
            (self.ax_filtered, snapshot.filtered, "Filtered EMG (20-450 Hz + notch)"),
        ):
            ax.clear()
            stacked, spacing = _stacked(data)
            for i, name in enumerate(names):
                if stacked.size:
                    ax.plot(emg_t, stacked[i], linewidth=0.6, label=name)
            ax.set_title(f"{title}   [channel spacing {spacing:.3f} mV]", fontsize=9)
            ax.set_yticks([])
            ax.legend(loc="upper left", fontsize=7, ncol=len(names))

        ax = self.ax_activation
        ax.clear()
        for i, name in enumerate(names):
            if snapshot.activation.size:
                ax.plot(emg_t, snapshot.activation[i], linewidth=0.8, label=name)
        control_t = snapshot.control_time - (
            snapshot.control_time[-1] if snapshot.control_time.size else 0
        )
        ax.plot(control_t, snapshot.grasp, color="black", linewidth=1.8, label="grasp (flexors)")
        ax.axhline(snapshot.thresholds["grasp on"], color="black", linestyle="--", linewidth=0.8)
        ax.axhline(snapshot.thresholds["grasp off"], color="black", linestyle=":", linewidth=0.8)
        ax.set_title("Normalized slow activation (0 = rest, 1 = reference)", fontsize=9)
        ax.set_ylim(-0.05, max(1.5, float(np.max(snapshot.grasp, initial=0)) * 1.1))
        ax.legend(loc="upper left", fontsize=7, ncol=len(names) + 1)

        ax = self.ax_control
        ax.clear()
        ax.plot(control_t, snapshot.tremor, color="purple", label="tremor level (4-6 Hz)")
        ax.axhline(snapshot.thresholds["tremor on"], color="purple", linestyle="--", linewidth=0.8)
        ax.axhline(snapshot.thresholds["tremor off"], color="purple", linestyle=":", linewidth=0.8)
        ax.plot(
            control_t,
            snapshot.requested_stiffness,
            color="#ff9800",
            linewidth=1.8,
            label="stiffness requested",
        )
        ax.plot(
            control_t,
            snapshot.applied_stiffness,
            color="#e53935",
            linewidth=1.0,
            linestyle="--",
            label="stiffness applied (dry run)",
        )
        ax.set_ylim(-0.05, 1.1)
        ax.set_xlabel("Seconds before now")
        ax.set_title("Controller", fontsize=9)
        ax.legend(loc="upper left", fontsize=7, ncol=2)

        color = "#e53935" if snapshot.estop else STATE_COLORS.get(snapshot.state, "white")
        self.ax_text.clear()
        self.ax_text.axis("off")
        self.ax_text.text(
            0.0,
            1.0,
            f" {('E-STOP  ' if snapshot.estop else '')}{snapshot.state} ",
            fontsize=18,
            weight="bold",
            color="white",
            va="top",
            bbox={"facecolor": color, "edgecolor": "none"},
            transform=self.ax_text.transAxes,
        )
        self.ax_text.text(
            0.0,
            0.93,
            "\n".join(status_lines(snapshot) + ["", self.help_text()]),
            fontsize=7.5,
            family="monospace",
            va="top",
            wrap=True,
            transform=self.ax_text.transAxes,
        )

    def run(self) -> None:
        """Show the window and refresh until it is closed."""
        self._animation = FuncAnimation(
            self.figure, self._draw, interval=100, cache_frame_data=False
        )
        plt.show()


class ConsoleMonitor:  # pylint: disable=too-few-public-methods
    """Prints one compact status line per call."""

    def __init__(self, pipeline: Pipeline) -> None:
        self.pipeline = pipeline

    def __call__(self) -> None:
        snapshot = self.pipeline.snapshot()
        grasp = snapshot.grasp[-1] if snapshot.grasp.size else 0.0
        tremor = snapshot.tremor[-1] if snapshot.tremor.size else 0.0
        stiffness = snapshot.requested_stiffness[-1] if snapshot.requested_stiffness.size else 0.0
        period = snapshot.timings.get("control period", (0.0, 0.0, 0.0))[1]
        scenario = snapshot.source_info.get("scenario", "")
        print(
            f"{snapshot.state:12s} grasp {grasp:5.2f} tremor {tremor:5.3f} "
            f"stiffness {stiffness:4.2f} | calib {snapshot.calibration_phase:22s} "
            f"| faults {','.join(snapshot.faults) or '-':30s} "
            f"| loop {1 / period if period else 0:5.1f} Hz | {scenario}",
            flush=True,
        )
