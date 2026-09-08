"""Load a safe artifact and predict one audio file."""

from __future__ import annotations

import argparse
from pathlib import Path

from ser_lib.artifacts import load_model_artifact
from ser_lib.inference import EmotionPredictor


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("artifact", type=Path)
    parser.add_argument("audio", type=Path)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    loaded = load_model_artifact(args.artifact, map_location=args.device)
    predictor = EmotionPredictor(
        loaded.model,
        loaded.audio_loader,
        loaded.pipeline,
        loaded.collator,
        labels=loaded.manifest.labels,
        device=args.device,
    )
    result = predictor.predict_file(args.audio)
    print(f"emotion={result.emotion} confidence={result.confidence:.4f}")
    print(result.probabilities)


if __name__ == "__main__":
    main()
