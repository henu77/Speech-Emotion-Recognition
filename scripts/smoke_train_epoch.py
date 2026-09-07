"""Run a self-contained, end-to-end one-epoch SER training smoke test.

The script creates a tiny temporary WAV dataset and exercises the production
path from audio decoding through feature extraction, batching, model forward,
backpropagation, and optimizer updates. No external dataset is required.
"""

from __future__ import annotations

import argparse
import math
import struct
import tempfile
import wave
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from ser_lib.data.audio import AudioLoader, AudioLoaderConfig
from ser_lib.data.collate import SERCollator
from ser_lib.data.config import BatchingConfig
from ser_lib.data.dataset import SERDataset
from ser_lib.data.pipeline import SamplePipeline
from ser_lib.data.representations.spectral import LogMelRepresentation
from ser_lib.data.types import AudioRecord
from ser_lib.engine.trainer import Trainer, TrainerConfig
from ser_lib.models.cnn_models import CNNBaseline


def _write_test_wav(
    path: Path,
    *,
    sample_rate: int,
    frequency: float,
    duration_seconds: float,
) -> None:
    """Write a deterministic mono 16-bit PCM sine wave."""
    frame_count = round(sample_rate * duration_seconds)
    frames = bytearray()
    for index in range(frame_count):
        fade = min(1.0, index / 160, (frame_count - index) / 160)
        value = round(10_000 * fade * math.sin(2 * math.pi * frequency * index / sample_rate))
        frames.extend(struct.pack("<h", value))

    with wave.open(str(path), "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(frames)


def run_smoke_test(device: str) -> None:
    sample_rate = 16_000
    torch.manual_seed(2026)

    with tempfile.TemporaryDirectory(prefix="ser-one-epoch-") as temp_dir:
        root = Path(temp_dir)
        records: list[AudioRecord] = []
        # Different durations verify dynamic padding; different frequency bands
        # give the two synthetic labels a learnable distinction.
        for index in range(8):
            label = index % 2
            wav_path = root / f"sample-{index}.wav"
            _write_test_wav(
                wav_path,
                sample_rate=sample_rate,
                frequency=(220.0 if label == 0 else 660.0) + index * 3,
                duration_seconds=0.24 + index * 0.015,
            )
            records.append(
                AudioRecord(
                    uid=f"smoke-{index}",
                    audio_path=wav_path,
                    label=label,
                    speaker_id=f"synthetic-{index}",
                )
            )

        audio_loader = AudioLoader(AudioLoaderConfig(target_sample_rate=sample_rate))
        representation = LogMelRepresentation(
            sample_rate=sample_rate,
            n_fft=256,
            win_length=256,
            hop_length=80,
            n_mels=16,
            f_max=8_000,
        )
        pipeline = SamplePipeline(representation)
        dataset = SERDataset(records, audio_loader, pipeline, strict=True)
        collator = SERCollator(pipeline.output_specs, BatchingConfig(type="dynamic"))
        batches = DataLoader(
            dataset,
            batch_size=4,
            shuffle=False,
            num_workers=0,
            collate_fn=collator,
        )

        model = CNNBaseline(
            feature_dim=16,
            num_classes=2,
            hidden_dim=16,
            dropout=0.0,
        )
        parameters_before = {
            name: parameter.detach().clone() for name, parameter in model.named_parameters()
        }
        trainer = Trainer(
            model,
            TrainerConfig(
                epochs=10,
                device=device,
                seed=2026,
                deterministic=True,
                learning_rate=1e-3,
            ),
        )
        history = trainer.fit(batches)

        if len(history) != 10:
            raise RuntimeError(f"expected one epoch result, got {len(history)}")
        result = history[0]
        if result.sample_count != len(dataset):
            raise RuntimeError(
                f"expected {len(dataset)} trained samples, got {result.sample_count}"
            )
        if result.optimizer_steps != len(batches):
            raise RuntimeError(
                f"expected {len(batches)} optimizer steps, got {result.optimizer_steps}"
            )
        if not math.isfinite(result.loss):
            raise RuntimeError(f"training produced non-finite loss: {result.loss}")
        if not any(
            not torch.equal(parameters_before[name], parameter.detach().cpu())
            for name, parameter in model.named_parameters()
        ):
            raise RuntimeError("training completed but did not update any model parameter")

        print("ONE_EPOCH_SMOKE_TEST=PASS")
        print(f"device={trainer.device}")
        print(f"samples={result.sample_count}")
        print(f"batches={len(batches)}")
        print(f"optimizer_steps={result.optimizer_steps}")
        print(f"loss={result.loss:.6f}")
        print(f"accuracy={result.accuracy:.4f}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--device",
        default="cpu",
        help="PyTorch device used for training (default: cpu; for example cuda or cuda:0)",
    )
    args = parser.parse_args()
    run_smoke_test(args.device)


if __name__ == "__main__":
    main()
