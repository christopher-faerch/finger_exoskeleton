"""
All settings of the exoskeleton software in one place. Change the values here.

Units: EMG in mV, times in seconds, frequencies in Hz, motor positions in
raw Dynamixel ticks (4096 per turn).
"""

from dataclasses import dataclass


@dataclass
class Channel:
    """One EMG sensor: its name, its muscle role ("flexor" or "extensor") and its Trigno slot."""

    name: str
    role: str
    slot: int


# The sensors in use. The flexor channels detect the grasp; all channels are checked for tremor.
CHANNELS = [
    Channel("FDS", "flexor", 15),  # underside (palm side) of the forearm
]

# --- Delsys Trigno Control Utility (TCU) ----------------------------------------------------
TRIGNO_HOST = "127.0.0.1"  # the PC running the TCU
TRIGNO_COMMAND_PORT = 50040
TRIGNO_EMG_PORT = 50043  # 16 EMG channels as 32-bit floats
TRIGNO_CHANNELS_ON_PORT = 16
TRIGNO_CONNECT_TIMEOUT_S = 3.0
TRIGNO_COMMAND_TIMEOUT_S = 5.0  # the TCU takes about 2.6 s to answer STOP
TRIGNO_DATA_TIMEOUT_S = 0.5  # no EMG data for this long: a message, then keep waiting

# Live runs wait this long (seconds of EMG) before the calibration routine and the cues
# begin: the window needs time to open, and the person time to get ready.
START_DELAY_S = 20.0

# --- Replay and live view -----------------------------------------------------------------
BLOCK_SIZE = 27  # samples per block when replaying (one Trigno frame of 13.5 ms at 2000 Hz)
MONITOR_SECONDS = 5.0  # seconds of EMG shown in the live view
EMG_Y_LIMITS = (0.0, 0.15)  # fixed y range of the EMG plots in mV

# --- Signal processing (see source/signal_processing.py) ----------------------------------
BANDPASS_HZ = (20.0, 450.0)
BANDPASS_ORDER = 4
NOTCH_HZ = 50.0  # power line (60 Hz in the Americas)
NOTCH_QUALITY = 30.0
SLOW_WINDOW_S = 0.2  # moving RMS: the slow envelope (voluntary effort)
FAST_CUTOFF_HZ = 10.0  # low-pass of the rectified EMG: the fast envelope (tremor bursts)
TREMOR_BAND_HZ = (4.0, 6.0)  # Parkinsonian tremor
RHYTHM_BAND_HZ = (1.0, 9.0)  # all envelope fluctuation, for the tremor ratio
TREMOR_WINDOW_S = 0.5  # RMS window of the tremor and fluctuation amplitudes
DEPTH_MIN_ENVELOPE_MV = 0.01  # floor of the slow envelope when computing tremor depth

# --- Calibration (see source/calibration.py) ----------------------------------------------
REST_WINDOW_S = (0.5, 3.0)  # rest during this window...
GRIP_WINDOW_S = (3.5, 10.0)  # ...then hold a firm grip during this one
GRIP_PERCENTILE = 95.0  # grip level = this percentile of the slow envelope in the grip window
MIN_GRIP_RATIO = 3.0  # the grip must be at least this many times the rest level
CALIBRATION_FILE = "calibration.json"  # default file for live Trigno runs

# --- Stiffness controller (see source/stiffness_controller.py) ----------------------------
# Grasp activation: 0 = rest, 1 = the calibration grip. Tremor is measured as tremor depth.
GRASP_ON = 0.25
GRASP_OFF = 0.15
GRASP_ON_TIME_S = 0.10
GRASP_OFF_TIME_S = 0.30  # also how long the hand must rest before the controller arms
TREMOR_ON = 0.15
TREMOR_OFF = 0.08
RATIO_ON = 0.8  # tremor only counts when the fluctuation is rhythmic (ratio near 1)
RATIO_OFF = 0.6
TREMOR_ON_TIME_S = 0.3  # 0.05 gave false tremor on fast grips
TREMOR_OFF_TIME_S = 0.75
ENGAGE_STIFFNESS = 0.3  # stiffness when tremor is first detected
STIFFNESS_GAIN = 2.0  # extra stiffness per unit of tremor depth above TREMOR_OFF
MAX_STIFFNESS = 1.0
ENGAGE_RATE_PER_S = 20.0
RELEASE_RATE_PER_S = 1.5

# --- Tendon motor (see source/motor.py), used with run_emg.py --motor ---------------------
# Positions measured 2026-11-06 with torque off; only their difference and their value
# modulo 4096 matter (the turn the motor is in is found at start).
MOTOR_PORT = "COM5"
MOTOR_BAUDRATE = 3000000
MOTOR_ID = 2  # the tendon motor; other motors on the bus are left alone
OPEN_POS = 4659
CLOSED_POS = 2833
MAX_SPEED = 1500.0  # ticks per second
MOTOR_RATE_HZ = 50.0
SIGNAL_TIMEOUT_S = 0.5  # no grasp signal for this long: the finger opens
STALE_S = 0.3  # newest EMG older than this counts as no signal
HOLD_WINDOW_S = 1.0  # while STIFFENED: hold the mean grasp of this long before
DEADBAND = 0.1
GAMMA = 1.0
MOTOR_CONTROL = "threshold"  # "position": the finger follows the grasp; "velocity": the grasp
# sets the closing speed, relaxing opens the finger, and tremor holds it where it is;
# "threshold": open or closed only (see below), the steadiest while a grip is held
# Threshold control (grasp 0 = rest, 1 = the calibrated firm grip). The finger closes when the
# grasp stays above ON for ON_TIME_S and opens when it stays below OFF for OFF_TIME_S. The gap
# between ON and OFF is what keeps a held grip from flickering: the EMG of a steady grip varies
# by about 20 % around its mean (measured on the cue recordings of 2026-11-13).
THRESHOLD_ON = 0.30
THRESHOLD_OFF = 0.15
THRESHOLD_ON_TIME_S = 0.10
THRESHOLD_OFF_TIME_S = 0.30
# Position control against flicker: the grasp is low-pass filtered (time constant SMOOTH_S)
# and the finger only moves when the filtered grasp has changed by STEP (share of open-closed)
# since its last move. Chosen by replaying 6 recordings (2026-11-13 and 2026-11-27): direction
# changes 62 -> 15 per minute (threshold control: 12, the real grips), 1 s ripple of a held
# grip 14 % -> 3 % of the range, opening 0.16 s.
POSITION_SMOOTH_S = 0.3
POSITION_STEP = 0.10
VELOCITY_CLOSE_SPEED = 1000.0  # ticks per second at full grasp (velocity control)
VELOCITY_OPEN_SPEED = 1000.0  # ticks per second when the hand relaxes (velocity control)
# Velocity control hysteresis, as threshold control: close only while the grasp has been above
# CLOSE_ABOVE for CLOSE_TIME_S (faster the harder the grip), open only once it has been below
# OPEN_BELOW for OPEN_TIME_S, and in between the finger stands still.
VELOCITY_CLOSE_ABOVE = 0.30
VELOCITY_OPEN_BELOW = 0.15
VELOCITY_CLOSE_TIME_S = 0.10
VELOCITY_OPEN_TIME_S = 0.30

# --- Test signal: noise and simulated tremor over the EMG (see source/injection.py) -------
# Key 'k' in the live window (k + Enter with --headless) switches it on and off.
INJECT_TREMOR_HZ = 5.0  # Parkinsonian rest tremor is 4-6 Hz
INJECT_TREMOR_DEPTH = 0.8  # 0 = no tremor, 1 = the EMG is switched fully on and off
INJECT_TREMOR_BURST_MV = 0.005  # noise bursts at the tremor rhythm, so tremor shows at rest
INJECT_NOISE_MV = 0.003  # white noise (RMS); the Trigno's own noise is below 0.00075
INJECT_HUM_HZ = 50.0  # mains hum...
INJECT_HUM_MV = 0.005  # ...and its amplitude
INJECT_RAMP_S = 0.3  # the test signal fades in and out over this long

# --- GRU: the neural-network alternative to the signal processing (see GRU_README.md) ------
# Used with run_emg.py --method gru. A trained model file stores the values it was trained
# with; the values here are for training a NEW model with train_gru.py.
GRU_CHANNEL_ROLES = ["flexor"]  # one sensor; or ["flexor", "extensor"] for two sensors
GRU_FEATURE_RATE_HZ = 100.0  # the network gets new features every 10 ms
GRU_FEATURE_WINDOW_S = 0.05  # each feature is computed over the last 50 ms
GRU_HIDDEN_UNITS = 64  # size of the GRU's memory
GRU_SEQUENCE_SECONDS = 4.0  # length of one training example
GRU_BURN_IN_SECONDS = 0.5  # start of every training example that is not scored
GRU_LEARNING_RATE = 0.001
GRU_EPOCHS = 30
GRU_BATCH_SIZE = 32
# After every cue change the label is unknown for this long: the person needs time to react.
# Measured on the cue recordings of 2026-11-13: the grip starts 0.9 s (0.5-1.7 s) and the
# release 0.75 s (0.6-1.0 s) after the cue changes. 1.5 s covers nearly all of them.
GRU_LABEL_MARGIN_S = 1.5
GRU_WARMUP_S = 1.0  # after a reset the estimates are not valid for this long
# Use the GRU's tremor output for the controller? False: the GRU gives the grasp and the
# signal processing (DSP) still decides tremor. The tremor output has only been trained on
# synthetic tremor and gave false alarms on real EMG (2026-11-13), so it is off for now.
GRU_USE_TREMOR = False
GRU_TREMOR_ON = 0.6  # tremor probability that counts as tremor...
GRU_TREMOR_OFF = 0.4  # ...and below which the tremor has stopped
GRU_MODEL_FILE = "models/gru.pt"


def processing_settings() -> dict:
    """The signal processing settings, as stored in a calibration file."""
    return {
        "bandpass_low_hz": BANDPASS_HZ[0],
        "bandpass_high_hz": BANDPASS_HZ[1],
        "bandpass_order": BANDPASS_ORDER,
        "notch_enabled": True,
        "notch_hz": NOTCH_HZ,
        "notch_quality": NOTCH_QUALITY,
        "slow_window_s": SLOW_WINDOW_S,
        "fast_cutoff_hz": FAST_CUTOFF_HZ,
        "tremor_band_hz": list(TREMOR_BAND_HZ),
        "tremor_window_s": TREMOR_WINDOW_S,
        "rhythm_band_hz": list(RHYTHM_BAND_HZ),
        "depth_min_envelope_mv": DEPTH_MIN_ENVELOPE_MV,
    }
