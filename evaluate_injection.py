"""
Score how the whole system reacts to the test signal (noise + simulated tremor).

Examples:
  A live recording where k was pressed (it has an injections.csv):
    python evaluate_injection.py --sessions measurements/<folder>

  A clean recording, with the test signal laid on now from 20 to 26 s and 40 to 46 s
  (the clean replay is then the reference):
    python evaluate_injection.py --sessions measurements/<folder> --inject 20:26 40:46

  The same with the GRU for the grasp:
    python evaluate_injection.py --sessions ... --method gru --model models/gru.pt

The recording is replayed through the same pipeline and controller as live. See
source/injection_evaluation.py for what the numbers mean.
"""

import argparse
import json
from pathlib import Path

from source.gru.inference import load_model
from source.injection import parse_spans
from source.injection_evaluation import evaluate_session, format_result
from source.recording import Recording


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """The command-line options."""
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--sessions", nargs="+", type=Path, required=True,
                        help="session folders")
    parser.add_argument("--inject", nargs="+", metavar="START:END",
                        help="lay the test signal on in these spans (seconds) now; without "
                             "it, the spans come from each session's injections.csv")
    parser.add_argument("--calibration", type=Path,
                        help="calibration file; default: the one stored in session.json, "
                             "else the calibration routine at the start of the recording")
    parser.add_argument("--method", choices=["dsp", "gru"], default="dsp")
    parser.add_argument("--model", type=Path, default=Path("models/gru.pt"),
                        help="with --method gru: the model file")
    parser.add_argument("--out", type=Path, help="also write the results to this JSON file")
    args = parser.parse_args(argv)
    if args.inject is not None:
        try:
            args.inject = parse_spans(args.inject)
        except ValueError as error:
            parser.error(f"--inject: {error}")
    return args


def main(argv: list[str] | None = None) -> int:
    """Replay every session and print its metrics."""
    args = parse_args(argv)
    model = None
    if args.method == "gru":
        model, note = load_model(args.model)
        print(note)
        if model is None:
            return 1

    results = []
    for folder in args.sessions:
        if args.calibration is not None:
            calibration = json.loads(args.calibration.read_text(encoding="utf-8"))
        else:
            calibration = Recording(folder).metadata.get("calibration")
        try:
            result = evaluate_session(folder, calibration, args.inject, model)
        except ValueError as error:
            print(error)
            return 1
        result["method"] = args.method
        results.append(result)
        print(format_result(result))
    if args.out is not None:
        args.out.write_text(json.dumps(results, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
