"""
Test signals laid over the EMG: noise and simulated tremor, switched on and off with a key.

Press 'k' in the live window (or type k and Enter in a --headless run) to switch the
test signal on; press it again to switch it off. While it is on, every raw EMG block
gets, per channel:

  tremor   the EMG is switched on and off in bursts INJECT_TREMOR_HZ times per second,
           with depth INJECT_TREMOR_DEPTH, and the extensors burst half a period after
           the flexors (the same model as add_tremor in source/gru/synthetic.py).
           Bursts of noise of INJECT_TREMOR_BURST_MV are added as well, so the tremor
           also shows while the hand rests (as Parkinsonian rest tremor).
  noise    white noise of INJECT_NOISE_MV (RMS) and a INJECT_HUM_HZ hum of
           INJECT_HUM_MV (amplitude), as from a poor electrode contact and the mains

Both fade in and out over INJECT_RAMP_S. The test signal is laid on before the block
is recorded and processed (see Acquisition in source/pipeline.py), so emg.csv holds
exactly what the pipeline saw, and a replay gives the same result. The times it was
on are saved in injections.csv: that is the facit for evaluate_injection.py.

Instead of the key, a schedule of fixed spans can switch it (run_emg.py --inject 30:40,
and evaluate_injection.py --inject): then a replay with a test signal can be repeated.
"""

import csv
import threading
from collections.abc import Callable
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np

import settings
from source.signal_processing import EMGBlock

INJECTIONS_FILE = "injections.csv"


@dataclass
class Episode:
    """One period with the test signal on, in seconds of stream time (end None: still on)."""

    start_s: float
    end_s: float | None = None


class Injector:
    """Lays noise and simulated tremor over EMG blocks while it is switched on."""

    def __init__(self, roles: list[str], sample_rate: float, seed: int = 0,
                 message: Callable[[str], None] = print,
                 schedule: list[Episode] | None = None) -> None:
        self.sample_rate = sample_rate
        self.schedule = schedule or []  # fixed spans: then they decide, not the key
        # Extensors burst half a period after the flexors
        self.phase = np.array([np.pi if role == "extensor" else 0.0 for role in roles])
        self.rng = np.random.default_rng(seed)
        self.message = message
        self.requested = False  # set from the viewer's thread, read by the acquisition thread
        self.active = False  # what the blocks get now
        self.weight = 0.0  # 0 = off, 1 = fully on; fades over INJECT_RAMP_S
        self.episodes: list[Episode] = []
        self.lock = threading.Lock()

    @property
    def on(self) -> bool:
        """True while the test signal is switched on (or about to be)."""
        return self.requested

    def toggle(self) -> bool:
        """Switch the test signal on or off from the next block; returns the new state."""
        if self.schedule:
            self.message("The test signal follows --inject; the key does nothing")
            return self.requested
        self.requested = not self.requested
        return self.requested

    def set(self, on: bool) -> None:
        """Switch the test signal on or off from the next block."""
        self.requested = on

    def apply(self, block: EMGBlock) -> EMGBlock:
        """The block with the test signal laid over it (the block itself while it is off)."""
        n = block.samples.shape[1]
        start_s = block.first_index / self.sample_rate
        if self.schedule:
            self.requested = any(episode.start_s <= start_s < (episode.end_s or np.inf)
                                 for episode in self.schedule)
        self.follow_request(start_s)
        if not self.active and self.weight == 0.0:
            return block

        # The weight ramps towards 1 (on) or 0 (off), sample by sample
        step = 1.0 / (settings.INJECT_RAMP_S * self.sample_rate)
        direction = 1.0 if self.active else -1.0
        weight = np.clip(self.weight + direction * step * np.arange(1, n + 1), 0.0, 1.0)
        self.weight = float(weight[-1])

        times = (block.first_index + np.arange(n)) / self.sample_rate
        emg = np.asarray(block.samples, dtype=np.float64)
        depth = settings.INJECT_TREMOR_DEPTH * weight
        burst = 0.5 * (1 + np.sin(2 * np.pi * settings.INJECT_TREMOR_HZ * times
                                  + self.phase[:, None]))  # 0 ... 1
        # Where the muscle is active the EMG is switched on and off (the mean power stays
        # the same); at rest bursts of noise are added
        tremor = emg * ((1 - depth) + depth * 2 * burst)
        tremor += (self.rng.standard_normal(emg.shape) * settings.INJECT_TREMOR_BURST_MV
                   * depth * 2 * burst)
        noise = (self.rng.standard_normal(emg.shape) * settings.INJECT_NOISE_MV
                 + settings.INJECT_HUM_MV * np.sin(2 * np.pi * settings.INJECT_HUM_HZ * times))
        return replace(block, samples=tremor + noise * weight)

    def follow_request(self, now_s: float) -> None:
        """Start or end an episode when the key has been pressed since the last block."""
        requested = self.requested
        if requested == self.active:
            return
        self.active = requested
        with self.lock:
            if requested:
                self.episodes.append(Episode(now_s))
            elif self.episodes:
                self.episodes[-1].end_s = now_s
        if requested:
            self.message(f"Test signal ON at {now_s:.1f} s: {settings.INJECT_TREMOR_HZ:g} Hz "
                         f"tremor (depth {settings.INJECT_TREMOR_DEPTH:g}) + noise")
        else:
            self.message(f"Test signal OFF at {now_s:.1f} s")

    def finish(self, end_s: float) -> list[Episode]:
        """End an episode that is still on at end_s; returns all episodes."""
        with self.lock:
            if self.episodes and self.episodes[-1].end_s is None:
                self.episodes[-1].end_s = end_s
            return list(self.episodes)

    def write(self, folder: Path, end_s: float) -> Path | None:
        """Write injections.csv into a session folder (nothing if it was never on)."""
        episodes = self.finish(end_s)
        if not episodes:
            return None
        path = folder / INJECTIONS_FILE
        write_episodes(path, episodes)
        return path


def write_episodes(path: Path, episodes: list[Episode]) -> None:
    """injections.csv: one line per episode, with the test-signal settings that were used."""
    with open(path, "w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["start_s", "end_s", "tremor_hz", "tremor_depth", "tremor_burst_mv",
                         "noise_mv", "hum_hz", "hum_mv"])
        for episode in episodes:
            writer.writerow([f"{episode.start_s:.4f}", f"{episode.end_s:.4f}",
                             settings.INJECT_TREMOR_HZ, settings.INJECT_TREMOR_DEPTH,
                             settings.INJECT_TREMOR_BURST_MV, settings.INJECT_NOISE_MV,
                             settings.INJECT_HUM_HZ, settings.INJECT_HUM_MV])


def read_episodes(path: Path) -> list[Episode]:
    """The episodes of an injections.csv (an empty list if there is no such file)."""
    if not path.is_file():
        return []
    with open(path, newline="", encoding="utf-8") as file:
        return [Episode(float(row["start_s"]), float(row["end_s"]))
                for row in csv.DictReader(file)]


def parse_spans(texts: list[str]) -> list[Episode]:
    """Episodes from "start:end" texts in seconds, e.g. ["30:40", "55:62"]."""
    episodes = []
    for text in texts:
        start, end = (float(value) for value in text.split(":"))
        if end <= start:
            raise ValueError(f"{text}: the end must come after the start")
        episodes.append(Episode(start, end))
    return sorted(episodes, key=lambda episode: episode.start_s)
