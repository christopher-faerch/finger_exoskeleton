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
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

import settings
from source.calibration import Calibration
from source.gru.inference import GRUEstimator
from source.gru.labels import LABELS_FILE, read_labels
from source.injection import INJECTIONS_FILE, Episode, Injector, read_episodes
from source.pipeline import Pipeline
from source.recording import RecordedSource
from source.stiffness_controller import State

MARGIN_S = 1.0  # a STIFFENED onset this soon after an episode still counts as caused by it
GRASPING = (State.GRASP_ARMED, State.STIFFENED)


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


def evaluate_run(run: Run, episodes: list[Episode], reference: Run | None = None,
                 labels: np.ndarray | None = None) -> dict:
    """All metrics of one replay; see the top of this file."""
    if reference is not None and len(reference.states) != len(run.states):
        reference = None
    per_episode = [episode_metrics(run, episode, reference) for episode in episodes]
    gripping = [episode for episode in per_episode if episode["gripping"]]
    latencies = [episode["latency_s"] for episode in gripping if "latency_s" in episode]

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
        "median_latency_s": float(np.median(latencies)) if latencies else None,
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
    return "\n".join(lines)
