"""
The live pipeline: what happens to every block of EMG.

  Pipeline.step(block):
    1. filter it (SignalProcessor)
    2. keep the last few seconds for the viewer
    3. calibrate (after start_delay_s, if one is given), or compute the activation
       with the calibration
    4. after calibration: arm the controller once the hand has relaxed
       (grasp <= GRASP_OFF for GRASP_OFF_TIME_S), so a grip still held at
       the end of the calibration does not count as a grasp
    5. update the stiffness controller

  With a GRU estimator (run_emg.py --method gru) step 3 also asks the GRU, and its
  intention replaces the grasp when it is valid; its tremor probability replaces
  the tremor decision only with GRU_USE_TREMOR (settings.py). The signal processing
  (DSP) still runs: it calibrates, it arms the controller, and it takes over at once
  when the GRU is not valid.

  Acquisition: a thread that reads blocks from a source, lays the test signal over
  them while it is switched on (key 'k', see source/injection.py), records them
  (with --out) and gives them to Pipeline.step until it is stopped.

The controller runs on stream time (sample number / sample rate), so a replay
gives the same result as the live run it was recorded from.
"""

import threading
from collections import deque
from pathlib import Path

import numpy as np

import settings
from settings import Channel
from source.calibration import Activation, Calibration, Calibrator, compute_activation
from source.calibration import load_calibration
from source.gru.inference import GRUEstimate, GRUEstimator
from source.injection import Injector
from source.recording import RecordedSource, SessionWriter
from source.signal_processing import EMGBlock, ProcessedBlock, SignalProcessor
from source.stiffness_controller import HoldTimer, State, StiffnessController
from source.trigno import TrignoSource


class Pipeline:  # pylint: disable=too-many-instance-attributes
    """Filtering, calibration and control of one stream of EMG blocks."""

    def __init__(self, channels: list[Channel], sample_rate: float,
                 calibration_file: Path | None = None, use_saved: bool = True,
                 estimator: GRUEstimator | None = None, start_delay_s: float = 0.0) -> None:
        self.channels = channels
        self.names = [channel.name for channel in channels]
        self.sample_rate = sample_rate
        self.calibration_file = calibration_file
        self.start_delay_s = start_delay_s  # the calibration routine waits this long first
        self.processor = SignalProcessor(len(channels), sample_rate)
        self.calibrator = Calibrator(self.names, sample_rate)
        self.controller = StiffnessController()
        self.relax_timer = HoldTimer(settings.GRASP_OFF_TIME_S)

        flexors = np.array([channel.role == "flexor" for channel in channels])
        self.flexors = flexors if flexors.any() else np.ones(len(channels), dtype=bool)

        self.calibration: Calibration | None = None  # in use
        self.activation: Activation | None = None  # of the newest block
        self.waiting_to_arm = False  # calibrated, waiting for the hand to relax
        self.loaded_calibration = False
        self.reported_error: str | None = None
        self.recalibrate_requested = False
        self.samples_received = 0

        self.estimator = estimator  # the GRU method, or None for signal processing only
        self.dsp_activation: Activation | None = None  # always from the signal processing
        self.gru_estimate: GRUEstimate | None = None  # newest answer of the GRU
        self.using_gru = False  # True while the GRU's numbers drive the controller
        self.gru_note = ""  # the last reason the GRU was not used

        self.lock = threading.Lock()  # the viewer reads the history from another thread
        self.messages: list[tuple[float, str]] = []  # (seconds of EMG received, text)
        blocks = int(settings.MONITOR_SECONDS * sample_rate / settings.BLOCK_SIZE) * 2
        self.history: deque = deque(maxlen=blocks)  # recent ProcessedBlocks
        # (time, grasp, depth, stiffness, GRU intention, GRU tremor probability) per block
        self.control_history: deque = deque(maxlen=blocks)

        if calibration_file is not None and use_saved:
            saved, note = load_calibration(calibration_file, self.names)
            if note:
                self.message(note)
            if saved is not None:
                self.loaded_calibration = True
                self.calibrator.result = saved

    # --- what the viewer and the motor read ---------------------------------------------------

    @property
    def state(self) -> State:
        """The controller state."""
        return self.controller.state

    @property
    def stream_time(self) -> float:
        """Seconds of EMG received so far: the clock of emg.csv, the cues and the labels."""
        return self.samples_received / self.sample_rate

    @property
    def waiting_to_start(self) -> bool:
        """True during the start delay before a calibration routine."""
        return (self.calibration is None and not self.loaded_calibration
                and self.stream_time < self.start_delay_s)

    @property
    def phase(self) -> str:
        """What the user should be doing: the calibration phase, or "relax to arm"."""
        if self.waiting_to_start:
            return "waiting to start"
        return "relax to arm" if self.waiting_to_arm else self.calibrator.phase

    def message(self, text: str) -> None:
        """Print a message and keep it for the viewer and the session file."""
        print(text, flush=True)
        with self.lock:
            self.messages.append((self.stream_time, text))

    def recent(self) -> dict:
        """Copies of the last MONITOR_SECONDS of signals, for the viewer."""
        with self.lock:
            blocks = list(self.history)
            control = np.array(list(self.control_history)).reshape(-1, 6).T
            messages = [text for _, text in self.messages[-8:]]
        n_keep = int(settings.MONITOR_SECONDS * self.sample_rate)
        n_channels = len(self.channels)

        def join(name: str) -> np.ndarray:
            if not blocks:
                return np.zeros((n_channels, 0))
            return np.concatenate([getattr(block, name) for block in blocks], axis=1)[:, -n_keep:]

        if blocks:
            first = blocks[0].first_index
            n_samples = sum(block.raw.shape[1] for block in blocks)
            times = (first + np.arange(n_samples)) / self.sample_rate
        else:
            times = np.zeros(0)
        return {
            "time": times[-n_keep:], "filtered": join("filtered"), "slow": join("slow"),
            "fast": join("fast"), "depth": join("depth"), "ratio": join("ratio"),
            "control_time": control[0], "grasp": control[1], "control_depth": control[2],
            "stiffness": control[3], "gru_intention": control[4],
            "gru_probability": control[5], "messages": messages,
        }

    def recalibrate(self) -> None:
        """Start a new calibration from the next block (called from the viewer)."""
        self.recalibrate_requested = True

    def signal_interrupted(self) -> None:
        """The EMG stopped for a while: the GRU's memory no longer fits, so empty it."""
        if self.estimator is not None:
            self.estimator.reset()

    # --- the work for every block -------------------------------------------------------------

    def step(self, block: EMGBlock) -> None:
        """Process one block of raw EMG; see the top of this file."""
        # NaN or infinity would break the filters for good
        block.samples = np.nan_to_num(block.samples, nan=0.0, posinf=0.0, neginf=0.0)
        processed = self.processor.process(block)
        n = block.samples.shape[1]
        self.samples_received += n
        with self.lock:
            self.history.append(processed)

        if self.recalibrate_requested:
            self.recalibrate_requested = False
            self.restart_calibration()
        if self.calibration is None and not self.waiting_to_start:
            self.calibrator.update(processed)  # its clock starts with the first block it gets
            self.check_calibration()

        now = (block.first_index + n) / self.sample_rate
        if self.calibration is not None:
            dsp = compute_activation(processed, self.calibration, self.flexors)
            self.dsp_activation = dsp
            if self.waiting_to_arm:  # arming always follows the signal processing
                self.arm_when_relaxed(dsp, now)
            self.activation = self.choose_activation(dsp, processed)
        self.controller.update(self.activation, now)

        grasp = self.activation.grasp if self.activation is not None else 0.0
        depth = self.activation.tremor_depth if self.activation is not None else 0.0
        estimate = self.gru_estimate
        intention = estimate.intention if estimate is not None and estimate.valid else np.nan
        probability = estimate.tremor_probability \
            if estimate is not None and estimate.valid else np.nan
        with self.lock:
            self.control_history.append(
                (now, grasp, depth, self.controller.stiffness, intention, probability))

    def choose_activation(self, dsp: Activation, processed: ProcessedBlock) -> Activation:
        """The activation the controller gets: from the GRU when it is valid, else the DSP's."""
        if self.estimator is None or self.calibration is None:
            return dsp
        estimate = self.estimator.update(processed, self.calibration.rest_slow)
        self.gru_estimate = estimate
        if not estimate.valid:
            if estimate.reason != self.gru_note:
                self.gru_note = estimate.reason
                self.message(f"GRU not used ({estimate.reason}); using the signal processing")
            self.using_gru = False  # back to the signal processing at once
        elif not self.using_gru and self.state in (State.DISARMED, State.IDLE):
            # Only switch to the GRU while the finger is open, so the motor target cannot jump
            self.using_gru = True
            self.gru_note = ""
            self.message("GRU estimates in use")
        if not self.using_gru:
            return dsp
        # The position controller closes the finger in proportion to the grasp (0-1), so
        # only the closing part of the intention is used; opening is "no grasp".
        # Without GRU_USE_TREMOR the tremor stays the DSP's (tremor_probability None).
        probability = estimate.tremor_probability if settings.GRU_USE_TREMOR else None
        return Activation(grasp=max(estimate.intention, 0.0), tremor_depth=dsp.tremor_depth,
                          tremor_ratio=dsp.tremor_ratio, receive_time=dsp.receive_time,
                          tremor_probability=probability)

    def check_calibration(self) -> None:
        """Start using a finished calibration, or report why it failed."""
        result = self.calibrator.result
        if result is not None:
            self.calibration = result
            self.waiting_to_arm = True
            self.relax_timer.reset()
            if self.loaded_calibration:
                start = "Using the saved calibration"
            else:
                start = "Calibration complete"
            self.message(f"{start}; relax the hand to arm the controller")
            levels = " ".join(f"{name} {rest * 1000:.0f}/{grip * 1000:.0f}" for name, rest, grip
                              in zip(self.names, result.rest_slow, result.reference_slow))
            self.message(f"  rest/reference uV: {levels}")
            if not self.loaded_calibration and self.calibration_file is not None:
                result.save(self.calibration_file)
                self.message(f"Calibration saved to {self.calibration_file}; later runs use it")
        elif self.calibrator.error is not None and self.calibrator.error != self.reported_error:
            self.reported_error = self.calibrator.error
            self.message(f"Calibration failed: {self.calibrator.error} Press 'c' to repeat it.")

    def arm_when_relaxed(self, activation: Activation, now: float) -> None:
        """Arm the controller once the grasp has stayed at rest for GRASP_OFF_TIME_S."""
        if self.relax_timer.update(activation.grasp <= settings.GRASP_OFF, now):
            self.waiting_to_arm = False
            self.controller.arm()
            self.message("Hand relaxed; controller armed")

    def restart_calibration(self) -> None:
        """Forget the calibration and run the calibration routine again."""
        self.controller.disarm("recalibration")
        self.calibration = None
        self.activation = None
        self.waiting_to_arm = False
        self.loaded_calibration = False
        self.reported_error = None
        self.dsp_activation = None
        self.gru_estimate = None
        self.using_gru = False
        if self.estimator is not None:
            self.estimator.reset(forget_broken=True)
        self.calibrator.restart()
        rest, grip = settings.REST_WINDOW_S, settings.GRIP_WINDOW_S
        self.message(f"Calibration restarted: rest {rest[0]:g}-{rest[1]:g} s, "
                     f"reference grip {grip[0]:g}-{grip[1]:g} s")


class Acquisition:
    """Reads blocks from a source in a background thread and feeds them to the pipeline."""

    def __init__(self, source: RecordedSource | TrignoSource, pipeline: Pipeline,
                 session: SessionWriter | None = None, injector: Injector | None = None) -> None:
        self.source = source
        self.pipeline = pipeline
        self.session = session  # records the raw EMG with --out
        self.injector = injector  # the test signal (key 'k'), or None
        self.error: str | None = None
        self.stop_requested = False
        self.thread = threading.Thread(target=self.loop, name="acquisition", daemon=True)

    @property
    def running(self) -> bool:
        """True while the thread is reading blocks."""
        return self.thread.is_alive()

    def start(self) -> None:
        """Start reading blocks."""
        self.thread.start()

    def stop(self) -> None:
        """Stop the thread and the source."""
        self.stop_requested = True
        if self.thread.is_alive():
            self.thread.join(timeout=5.0)
        self.source.stop()

    def loop(self) -> None:
        """The thread: read, record and process blocks until stopped."""
        try:
            while not self.stop_requested:
                try:
                    block = self.source.read_block()
                except TimeoutError:
                    self.pipeline.message("No EMG data from the source; still waiting")
                    self.pipeline.signal_interrupted()
                    continue
                if block is None:
                    self.pipeline.message("Source finished (end of recording)")
                    return
                if self.injector is not None:  # recorded as the pipeline sees it
                    block = self.injector.apply(block)
                if self.session is not None:
                    self.session.write(block)
                self.pipeline.step(block)
        except Exception as error:  # pylint: disable=broad-exception-caught
            # Keep the error for the viewer and the session file instead of losing it in the thread
            self.error = f"Acquisition stopped: {error!r}"
            self.pipeline.message(self.error)
