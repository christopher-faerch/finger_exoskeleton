"""
The live view of a running pipeline, and the status line for --headless runs.

LiveViewer shows, about ten times per second:
  - per channel: the filtered EMG with its slow envelope (effort) and fast envelope (bursts)
  - the controller: grasp activation, tremor depth and stiffness, with the on-thresholds
  - a text panel: the latest values, the controller state, the calibration and messages
Keys: c = calibrate again, k = test signal on/off (noise + simulated tremor), q = quit.
"""

from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation

import settings
from source.cues import CueProtocol
from source.pipeline import Acquisition, Pipeline

STATE_COLORS = {"DISARMED": "gray", "IDLE": "green", "GRASP_ARMED": "orange", "STIFFENED": "red"}


def last(values: np.ndarray) -> float:
    """The last value of a (n,) array, or 0 when it is empty."""
    return float(values[-1]) if values.size else 0.0


def calibration_text(pipeline: Pipeline) -> str:
    """What the calibration is doing, or why it failed."""
    if pipeline.calibrator.error:
        return f"FAILED: {pipeline.calibrator.error}"
    if pipeline.phase in ("done", "relax to arm"):
        return pipeline.phase
    if pipeline.waiting_to_start:
        return f"starts in {pipeline.start_delay_s - pipeline.stream_time:.0f} s"
    return f"{pipeline.phase} ({pipeline.calibrator.seconds_left:.0f} s left)"


def method_text(pipeline: Pipeline) -> str:
    """Which method drives the controller now: the signal processing (DSP) or the GRU."""
    if pipeline.estimator is None:
        return "DSP"
    if pipeline.using_gru:
        return "GRU (grasp and tremor)" if settings.GRU_USE_TREMOR else "GRU grasp + DSP tremor"
    return f"DSP (GRU not used: {pipeline.gru_note or 'waiting for an open finger'})"


def injection_text(acquisition: Acquisition) -> str:
    """Whether the test signal (key 'k') is on."""
    if acquisition.injector is None:
        return "not available"
    return "ON (noise + tremor)" if acquisition.injector.on else "off"


def status_line(pipeline: Pipeline) -> str:
    """One line with the latest values, for --headless runs."""
    recent = pipeline.recent()
    channels = "  ".join(
        f"{name} {last(recent['slow'][i]) * 1000:6.1f} uV depth {last(recent['depth'][i]):4.2f} "
        f"ratio {last(recent['ratio'][i]):4.2f}" for i, name in enumerate(pipeline.names))
    line = (f"{channels} | {pipeline.state.value:11s} stiffness {last(recent['stiffness']):.2f} "
            f"grasp {last(recent['grasp']):.2f} | cal {calibration_text(pipeline)}")
    if pipeline.estimator is not None:
        line += (f" | {method_text(pipeline)}: intention {last(recent['gru_intention']):+.2f} "
                 f"tremor p {last(recent['gru_probability']):.2f}")
    return line


class LiveViewer:
    """A matplotlib window; run() blocks until the window is closed or q is pressed."""

    def __init__(self, pipeline: Pipeline, acquisition: Acquisition,
                 protocol: CueProtocol | None = None) -> None:
        self.pipeline = pipeline
        self.acquisition = acquisition
        self.protocol = protocol  # cues to show in big letters, or None
        n = len(pipeline.names)
        for name in ("keymap.xscale", "keymap.yscale"):  # matplotlib's own k / l: log axes
            plt.rcParams[name] = [key for key in plt.rcParams[name] if key not in ("k", "K")]
        self.figure = plt.figure(figsize=(14, 2.5 * n + 3))
        grid = self.figure.add_gridspec(n + 1, 2, width_ratios=(3, 1), hspace=0.4)

        self.emg_lines = []
        for row, name in enumerate(pipeline.names):
            ax = self.figure.add_subplot(grid[row, 0])
            lines = (ax.plot([], [], linewidth=0.5, label="filtered EMG")[0],
                     ax.plot([], [], linewidth=1.6, color="red", label="slow envelope")[0],
                     ax.plot([], [], linewidth=0.9, color="orange", label="fast envelope")[0])
            self.emg_lines.append(lines)
            ax.set_xlim(-settings.MONITOR_SECONDS, 0)
            ax.set_ylim(*settings.EMG_Y_LIMITS)
            ax.set_ylabel(f"{name}\nmV")
            ax.grid(alpha=0.3)
        self.figure.axes[0].legend(loc="upper left", fontsize=7, ncol=3)

        self.control_ax = self.figure.add_subplot(grid[n, 0])
        self.control_lines = (
            self.control_ax.plot([], [], color="red", label="grasp activation")[0],
            self.control_ax.plot([], [], color="purple", label="tremor depth")[0],
            self.control_ax.plot([], [], color="black", linewidth=2, label="stiffness")[0])
        self.gru_lines = None
        if pipeline.estimator is not None:  # the GRU's two outputs, dashed
            self.gru_lines = (
                self.control_ax.plot([], [], "b--", label="GRU intention")[0],
                self.control_ax.plot([], [], "m--", label="GRU tremor probability")[0])
        self.control_ax.axhline(settings.GRASP_ON, color="red", linestyle="--", linewidth=0.8)
        self.control_ax.axhline(settings.TREMOR_ON, color="purple", linestyle="--", linewidth=0.8)
        self.control_ax.set_xlim(-settings.MONITOR_SECONDS, 0)
        self.control_ax.set_ylim(-1.05 if pipeline.estimator is not None else 0, 1.05)
        self.control_ax.set_xlabel("Seconds before now")
        self.control_ax.legend(loc="upper left", fontsize=7, ncol=3)
        self.control_ax.grid(alpha=0.3)

        text_ax = self.figure.add_subplot(grid[:, 1])
        text_ax.axis("off")
        self.text = text_ax.text(0, 1, "", fontsize=8, family="monospace", va="top",
                                 transform=text_ax.transAxes)
        self.figure.canvas.mpl_connect("key_press_event", self.on_key)
        self.animation: FuncAnimation | None = None

    def on_key(self, event: Any) -> None:
        """c: calibrate again, k: test signal on/off, q: quit."""
        if event.key == "c":
            self.pipeline.recalibrate()
        elif event.key == "k" and self.acquisition.injector is not None:
            self.acquisition.injector.toggle()
        elif event.key == "q":
            plt.close(self.figure)

    def draw(self, _frame: int) -> None:
        """Copy the newest data into the plots."""
        pipeline = self.pipeline
        if self.protocol is not None:
            cue = self.protocol.cue_at(pipeline.stream_time)
            self.figure.suptitle(cue, fontsize=28, color="darkblue")
        recent = pipeline.recent()
        newest = last(recent["time"])
        seconds_ago = recent["time"] - newest
        for row, (emg, slow, fast) in enumerate(self.emg_lines):
            emg.set_data(seconds_ago, recent["filtered"][row])
            slow.set_data(seconds_ago, recent["slow"][row])
            fast.set_data(seconds_ago, recent["fast"][row])

        control_ago = recent["control_time"] - newest
        for line, values in zip(self.control_lines,
                                (recent["grasp"], recent["control_depth"], recent["stiffness"])):
            line.set_data(control_ago, values)
        if self.gru_lines is not None:
            self.gru_lines[0].set_data(control_ago, recent["gru_intention"])
            self.gru_lines[1].set_data(control_ago, recent["gru_probability"])
        state = pipeline.state.value
        title = f"Controller {state}  [{pipeline.controller.last_transition}]"
        injector = self.acquisition.injector
        if injector is not None and injector.on:
            title = "TEST SIGNAL ON (noise + tremor)   " + title
        self.control_ax.set_title(title, fontsize=8, color=STATE_COLORS[state])

        lines = [f"{'channel':8s} {'RMS uV':>7s} {'depth':>6s} {'ratio':>6s}"]
        for i, name in enumerate(pipeline.names):
            lines.append(f"{name:8s} {last(recent['slow'][i]) * 1000:7.1f} "
                         f"{last(recent['depth'][i]):6.2f} {last(recent['ratio'][i]):6.2f}")
        lines += ["", f"Method:      {method_text(pipeline)}",
                  f"Controller:  {state}", f"Stiffness:   {last(recent['stiffness']):.2f}",
                  f"Grasp:       {last(recent['grasp']):.2f}",
                  f"Calibration: {calibration_text(pipeline)}",
                  f"Test signal: {injection_text(self.acquisition)}", "",
                  "Keys: c calibrate again | k test signal | q quit", ""] + recent["messages"]
        if not self.acquisition.running:
            lines += ["", "ACQUISITION STOPPED", self.acquisition.error or ""]
        self.text.set_text("\n".join(lines))

    def run(self) -> None:
        """Show the window and keep it updated until it is closed."""
        self.animation = FuncAnimation(self.figure, self.draw, interval=100,
                                       cache_frame_data=False)
        plt.show()
