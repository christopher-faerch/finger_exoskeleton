"""
How the whole system reacts to the test signal (noise + simulated tremor, see injection.py).

A recording is replayed through the same Pipeline as live (filters, calibration,
stiffness controller), as fast as possible, and the controller state is logged
after every block. The facit is when the test signal was on:

  - a live recording made with key 'k' (or --inject) has it in injections.csv,
    and its emg.csv already holds the test signal;
  - a clean recording can get the test signal laid on now, in given spans
    (evaluate_injection.py --inject). Then the clean recording is replayed as
    well, as the reference for what the controller should have done.

Per episode (one span with the test signal on):
  gripping       the controller was in a grasp state (GRASP_ARMED or STIFFENED) at some
                 time during the episode; only then can the tremor make the finger stiff.
                 With a clean reference this is judged on the clean replay, so a grasp
                 the test signal causes by itself does not count as gripping
  stiffened      the controller reached STIFFENED during the episode
  latency_s      from the start of the episode to STIFFENED (none when the controller
                 was still STIFFENED from the episode before)
  release_s      from the end of the episode until the controller left STIFFENED

Over the whole recording:
  detection_rate             share of the gripping episodes that reached STIFFENED
  stiffened_at_rest          episodes at rest that reached STIFFENED anyway (the test
                             signal alone looked like a grasp: should be 0)
  false_stiffen_per_minute   STIFFENED onsets outside the episodes (and outside
                             MARGIN_S after them) per minute of armed controller
  grasp_agreement            with a clean reference: share of the time inside the
                             episodes where the grasp decision (grasping or not) is
                             the same as without the test signal: does the noise
                             make the controller grasp or release by mistake?
  grasp_accuracy_in/out      with a labels.csv (cue recordings): share of the time
                             where the grasp decision matches the cue, inside and
                             outside the episodes

The motor, from motor.csv (runs with --motor and --out; see source/motor.py). Positions
are the closure: 0 = open, 1 = closed, so position and velocity control compare directly.
The motor log holds the states the controller had live.
  motor_tremor_in            RMS of the closure in the tremor band (MOTOR_TREMOR_BAND_HZ)
                             during the gripping episodes: how much the finger shook with
                             the test signal (0 = not at all)
  motor_tremor_out           the same while gripping outside the episodes: the baseline
  motor_range_in             mean max-min closure during a gripping episode: how far the
                             finger moved while the tremor was on
  closing_time_s             median time from a grasp onset (IDLE -> GRASP_ARMED) until
                             the finger is CLOSED_LEVEL closed (grasps that never get there
                             are not counted; see n_closings)
  opening_time_s             median time from the end of a grasp (-> IDLE) until the finger
                             is back below OPEN_LEVEL
  mean_closure_gripping      mean closure while gripping outside the episodes
"""

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from scipy.signal import butter, sosfiltfilt

import settings
from source.calibration import Calibration
from source.gru.inference import GRUEstimator
from source.gru.labels import LABELS_FILE, read_labels
from source.injection import INJECTIONS_FILE, Episode, Injector, read_episodes
from source.motor import MOTOR_LOG_FILE
from source.pipeline import Pipeline
from source.recording import RecordedSource
from source.stiffness_controller import State

MARGIN_S = 1.0  # a STIFFENED onset this soon after an episode still counts as caused by it
GRASPING = (State.GRASP_ARMED, State.STIFFENED)
MOTOR_TREMOR_BAND_HZ = (3.0, 8.0)  # around the 5 Hz of the test signal
CLOSED_LEVEL = 0.5  # closure that counts as "closed" for closing_time_s
OPEN_LEVEL = 0.1  # closure that counts as "open again" for opening_time_s


@dataclass
class Run:
    """The controller state after every block of one replay."""

    times: np.ndarray  # stream time at the end of every block, s
    states: list[State]
    grasp: np.ndarray  # grasp activation the controller got


def replay(folder: Path, calibration: dict | None, schedule: list[Episode] | None,
           model: Any = None) -> Run:
    """
    Replay a session folder through the live pipeline, with the test signal laid on
    in `schedule` (None: as recorded). calibration: the levels to use, or None to run
    the calibration routine on the recording as live.
    """
    source = RecordedSource(folder, loop=False, realtime=False)
    names = [channel.name for channel in source.channels]
    estimator = None
    if model is not None:
        estimator = GRUEstimator(model, source.channels, source.sample_rate)
    delay = float(source.recording.metadata.get("start_delay_s", 0.0))
    pipeline = Pipeline(source.channels, source.sample_rate, estimator=estimator,
                        start_delay_s=delay)
    pipeline.message = lambda text: None  # type: ignore[method-assign]  # keep the output short
    if calibration is not None:
        pipeline.calibrator.result = Calibration(
            channel_names=names,
            rest_slow=levels(calibration, "rest_slow", names),
            reference_slow=levels(calibration, "reference_slow", names),
            rest_fast=levels(calibration, "rest_fast", names),
            reference_fast=levels(calibration, "reference_fast", names))
        pipeline.loaded_calibration = True
    injector = None
    if schedule:
        roles = [channel.role for channel in source.channels]
        injector = Injector(roles, source.sample_rate, message=lambda text: None,
                            schedule=schedule)

    times, states, grasp = [], [], []
    while (block := source.read_block()) is not None:
        if injector is not None:
            block = injector.apply(block)
        pipeline.step(block)
        times.append((block.first_index + block.samples.shape[1]) / source.sample_rate)
        states.append(pipeline.state)
        grasp.append(pipeline.activation.grasp if pipeline.activation is not None else 0.0)
    return Run(np.array(times), states, np.array(grasp))


@dataclass
class MotorRun:
    """motor.csv read back: one row per motor step."""

    times: np.ndarray  # stream time, s
    states: list[str]  # controller state names
    closure: np.ndarray  # 0 = open, 1 = closed (NaN where the position could not be read)


def read_motor_log(folder: Path) -> MotorRun | None:
    """The motor log of a session folder, or None if it has none (or too few rows)."""
    path = folder / MOTOR_LOG_FILE
    if not path.is_file():
        return None
    with open(path, newline="", encoding="utf-8") as file:
        rows = list(csv.DictReader(file))
    if len(rows) < 2:
        return None
    return MotorRun(np.array([float(row["time_s"]) for row in rows]),
                    [row["state"] for row in rows],
                    np.array([float(row["closure"]) for row in rows]))


def rms(values: np.ndarray) -> float | None:
    """Root mean square, or None for no values."""
    return float(np.sqrt(np.mean(values ** 2))) if values.size else None


def tremor_band(motor: MotorRun) -> np.ndarray:
    """The closure band-passed to MOTOR_TREMOR_BAND_HZ (zero phase, offline)."""
    closure = motor.closure.copy()
    missing = np.isnan(closure)
    if missing.all():
        return np.zeros_like(closure)
    if missing.any():  # fill missed reads from their neighbours
        closure[missing] = np.interp(motor.times[missing], motor.times[~missing],
                                     closure[~missing])
    rate = 1.0 / float(np.median(np.diff(motor.times)))
    low, high = MOTOR_TREMOR_BAND_HZ
    if high >= rate / 2 or len(closure) < 30:
        return np.zeros_like(closure)
    sos = butter(2, (low, high), btype="band", fs=rate, output="sos")
    return np.asarray(sosfiltfilt(sos, closure))


def time_until(motor: MotorRun, start: int, reached: np.ndarray, until: np.ndarray) -> float | None:
    """Seconds from row start until `reached` first holds, looking only while `until` holds."""
    for index in range(start, len(motor.times)):
        if not until[index]:
            return None
        if reached[index]:
            return float(motor.times[index] - motor.times[start])
    return None


def motor_metrics(motor: MotorRun, episodes: list[Episode], per_episode: list[dict]) -> dict:
    """What the finger did; see the top of this file."""
    band = tremor_band(motor)
    gripping = np.array([state in (State.GRASP_ARMED.name, State.STIFFENED.name)
                         for state in motor.states])
    _, affected = episode_masks(motor.times, episodes)
    tremor_in: list[float] = []
    range_in: list[float] = []
    for episode in per_episode:
        if not episode["gripping"]:
            continue
        mask = (motor.times >= episode["start_s"]) & (motor.times < episode["end_s"])
        closure = motor.closure[mask & ~np.isnan(motor.closure)]
        if closure.size:
            tremor_in.append(float(np.sqrt(np.mean(band[mask] ** 2))))
            range_in.append(float(np.ptp(closure)))
    steady = gripping & ~affected & ~np.isnan(motor.closure)
    return {
        "motor_tremor_in": float(np.mean(tremor_in)) if tremor_in else None,
        "motor_tremor_out": rms(band[gripping & ~affected]),
        "motor_range_in": float(np.mean(range_in)) if range_in else None,
        **closing_and_opening(motor, gripping),
        "mean_closure_gripping": float(np.mean(motor.closure[steady])) if steady.any() else None,
    }


def closing_and_opening(motor: MotorRun, gripping: np.ndarray) -> dict:
    """closing_time_s, n_closings and opening_time_s; see the top of this file."""
    closure = np.nan_to_num(motor.closure, nan=-1.0)  # a missed read never counts as reached
    onsets = np.flatnonzero(gripping[1:] & ~gripping[:-1]) + 1
    ends = np.flatnonzero(~gripping[1:] & gripping[:-1]) + 1
    closings = [time_until(motor, int(i), closure >= CLOSED_LEVEL, gripping) for i in onsets]
    openings = [time_until(motor, int(i), (closure >= 0) & (closure <= OPEN_LEVEL), ~gripping)
                for i in ends]
    closings_done = [value for value in closings if value is not None]
    openings_done = [value for value in openings if value is not None]
    return {
        "closing_time_s": float(np.median(closings_done)) if closings_done else None,
        "n_closings": f"{len(closings_done)}/{len(closings)}",
        "opening_time_s": float(np.median(openings_done)) if openings_done else None,
    }


def levels(calibration: dict, key: str, names: list[str]) -> np.ndarray:
    """One kind of calibration level, in the order of the channels."""
    by_name = dict(zip(calibration["channel_names"], calibration[key]))
    return np.array([by_name[name] for name in names], dtype=float)


def episode_metrics(run: Run, episode: Episode, reference: Run | None = None) -> dict:
    """The reaction to one episode; see the top of this file."""
    end = episode.end_s if episode.end_s is not None else run.times[-1]
    inside = (run.times >= episode.start_s) & (run.times < end)
    stiff = np.array([state == State.STIFFENED for state in run.states])
    judged = reference if reference is not None else run
    grasping = np.array([state in GRASPING for state in judged.states])
    result: dict = {"start_s": episode.start_s, "end_s": end,
                    "gripping": bool(np.any(grasping & inside)),
                    "stiffened": bool(np.any(stiff & inside))}
    first = int(np.argmax(inside)) if inside.any() else 0
    result["already_stiff"] = bool(inside.any() and stiff[first])  # from an episode before
    if result["stiffened"] and not result["already_stiff"]:
        onset = int(np.flatnonzero(stiff & inside)[0])
        result["latency_s"] = float(run.times[onset] - episode.start_s)
    if result["stiffened"]:
        after = np.flatnonzero((run.times >= end) & ~stiff)
        if after.size and stiff[int(np.flatnonzero(inside)[-1])]:
            result["release_s"] = float(run.times[after[0]] - end)
    return result


def episode_masks(times: np.ndarray, episodes: list[Episode]) -> tuple[np.ndarray, np.ndarray]:
    """(inside an episode, inside or up to MARGIN_S after one) at every time."""
    inside = np.zeros(len(times), dtype=bool)
    affected = np.zeros(len(times), dtype=bool)
    for episode in episodes:
        end = episode.end_s if episode.end_s is not None else np.inf
        inside |= (times >= episode.start_s) & (times < end)
        affected |= (times >= episode.start_s) & (times < end + MARGIN_S)
    return inside, affected


def median_of(episodes: list[dict], key: str) -> float | None:
    """The median of one per-episode value over the episodes that have it."""
    values = [episode[key] for episode in episodes if key in episode]
    return float(np.median(values)) if values else None


def evaluate_run(run: Run, episodes: list[Episode], reference: Run | None = None,
                 labels: np.ndarray | None = None) -> dict:
    """All metrics of one replay; see the top of this file."""
    if reference is not None and len(reference.states) != len(run.states):
        reference = None
    per_episode = [episode_metrics(run, episode, reference) for episode in episodes]
    gripping = [episode for episode in per_episode if episode["gripping"]]

    inside, affected = episode_masks(run.times, episodes)
    stiff = np.array([state == State.STIFFENED for state in run.states])
    onsets = np.flatnonzero(stiff[1:] & ~stiff[:-1]) + 1
    armed = np.array([state != State.DISARMED for state in run.states])
    block_s = float(np.median(np.diff(run.times))) if len(run.times) > 1 else 0.0
    armed_minutes = np.sum(armed & ~affected) * block_s / 60

    result: dict = {
        "episodes": per_episode,
        "n_episodes": len(episodes),
        "n_gripping": len(gripping),
        "detection_rate": (sum(episode["stiffened"] for episode in gripping) / len(gripping)
                           if gripping else None),
        "stiffened_at_rest": sum(episode["stiffened"] for episode in per_episode
                                 if not episode["gripping"]),
        "median_latency_s": median_of(gripping, "latency_s"),
        "median_release_s": median_of(gripping, "release_s"),
        "false_stiffen_per_minute": (float(np.sum(~affected[onsets]) / armed_minutes)
                                     if armed_minutes > 0 else None),
        "armed_minutes_outside": float(armed_minutes),
    }
    grasping = np.array([state in GRASPING for state in run.states])
    if reference is not None:
        clean = np.array([state in GRASPING for state in reference.states])
        result["grasp_agreement"] = float(np.mean(grasping[inside] == clean[inside])) \
            if inside.any() else None
    if labels is not None:
        for key, where in (("grasp_accuracy_in", inside), ("grasp_accuracy_out", ~inside)):
            known = where & armed & ~np.isnan(labels)
            result[key] = float(np.mean(grasping[known] == (labels[known] > 0))) \
                if known.any() else None
    return result


def grasp_labels(folder: Path, times: np.ndarray) -> np.ndarray | None:
    """The cue labels (intention) at the given times, or None without a labels.csv."""
    path = folder / LABELS_FILE
    if not path.is_file():
        return None
    intention, _ = read_labels(path, times, settings.GRU_LABEL_MARGIN_S)
    return intention


def evaluate_session(folder: Path, calibration: dict | None, schedule: list[Episode] | None,
                     model: Any = None) -> dict:
    """
    Evaluate one session folder: with a schedule, the test signal is laid on now and
    the clean replay is the reference; without, the episodes come from injections.csv.
    """
    if schedule:
        run = replay(folder, calibration, schedule, model)
        reference = replay(folder, calibration, None, model)
        episodes = schedule
    else:
        episodes = read_episodes(folder / INJECTIONS_FILE)
        if not episodes:
            raise ValueError(f"{folder} has no {INJECTIONS_FILE}; give spans with --inject")
        run = replay(folder, calibration, None, model)
        reference = None
    result = evaluate_run(run, episodes, reference, grasp_labels(folder, run.times))
    motor = read_motor_log(folder)
    if motor is not None:
        result.update(motor_metrics(motor, episodes, result["episodes"]))
    result["session"] = folder.name
    result["test_signal_added_now"] = bool(schedule)
    return result


def format_result(result: dict) -> str:
    """One session's metrics as a few lines of text."""
    def number(value: Any, unit: str = "", digits: int = 2) -> str:
        return "-" if value is None else f"{value:.{digits}f}{unit}"

    lines = [f"{result['session']}: {result['n_episodes']} episodes, "
             f"{result['n_gripping']} while gripping"]
    for episode in result["episodes"]:
        what = "STIFFENED" if episode["stiffened"] else "not stiffened"
        lines.append(f"  {episode['start_s']:6.1f}-{episode['end_s']:6.1f} s  "
                     f"{'gripping' if episode['gripping'] else 'at rest ':8s}  {what:13s} "
                     f"latency {number(episode.get('latency_s'), ' s')}  "
                     f"release {number(episode.get('release_s'), ' s')}")
    lines.append(f"  detection rate {number(result['detection_rate'])}  "
                 f"median latency {number(result['median_latency_s'], ' s')}  "
                 f"stiffened at rest {result['stiffened_at_rest']}  "
                 f"false STIFFENED {number(result['false_stiffen_per_minute'], '/min')} "
                 f"(over {result['armed_minutes_outside']:.1f} armed min)")
    extra = []
    if "grasp_agreement" in result:
        extra.append(f"grasp agreement with clean replay {number(result['grasp_agreement'])}")
    if "grasp_accuracy_in" in result:
        extra.append(f"grasp vs cues: inside {number(result['grasp_accuracy_in'])}, "
                     f"outside {number(result['grasp_accuracy_out'])}")
    if extra:
        lines.append("  " + "  ".join(extra))
    if "motor_tremor_in" in result:
        lines.append(f"  motor: tremor in {number(result['motor_tremor_in'], digits=3)} "
                     f"out {number(result['motor_tremor_out'], digits=3)}  "
                     f"range in {number(result['motor_range_in'])}  "
                     f"closing {number(result['closing_time_s'], ' s')} "
                     f"({result['n_closings']})  "
                     f"opening {number(result['opening_time_s'], ' s')}  "
                     f"mean closure {number(result['mean_closure_gripping'])}")
    return "\n".join(lines)


# The comparison table: (key in a result, column header, digits)
COMPARISON_COLUMNS = [
    ("detection_rate", "Detect", 2), ("median_latency_s", "Lat[s]", 2),
    ("median_release_s", "Rel[s]", 2), ("stiffened_at_rest", "StRest", 1),
    ("false_stiffen_per_minute", "False/min", 2), ("grasp", "Grasp", 2),
    ("motor_tremor_in", "MTrIn", 3), ("motor_tremor_out", "MTrOut", 3),
    ("motor_range_in", "MRange", 2), ("closing_time_s", "Close[s]", 2),
    ("opening_time_s", "Open[s]", 2)]


def combination(result: dict) -> str:
    """Which of the four combinations a session is, e.g. "velocity + gru"."""
    return f"{result.get('motor_control') or 'no motor'} + {result.get('method', 'dsp')}"


def compare(results: list[dict]) -> dict[str, dict]:
    """The mean of every metric per combination (sessions without a value are left out)."""
    table: dict[str, dict] = {}
    for name in sorted({combination(result) for result in results}):
        group = [result for result in results if combination(result) == name]
        row: dict = {"sessions": len(group)}
        for key, _, _ in COMPARISON_COLUMNS:
            if key == "grasp":  # with a clean reference, else against the cues
                values = [result.get("grasp_agreement", result.get("grasp_accuracy_in"))
                          for result in group]
            else:
                values = [result.get(key) for result in group]
            known = [float(value) for value in values if value is not None]
            row[key] = float(np.mean(known)) if known else None
        table[name] = row
    return table


def format_comparison(table: dict[str, dict]) -> str:
    """The comparison as a text table, one row per combination."""
    header = f"{'Combination':20s} {'N':>3s} " + " ".join(
        f"{title:>9s}" for _, title, _ in COMPARISON_COLUMNS)
    lines = [header, "-" * len(header)]
    for name, row in table.items():
        cells = ["-" if row[key] is None else f"{row[key]:.{digits}f}"
                 for key, _, digits in COMPARISON_COLUMNS]
        lines.append(f"{name:20s} {row['sessions']:3d} " + " ".join(f"{c:>9s}" for c in cells))
    return "\n".join(lines)
