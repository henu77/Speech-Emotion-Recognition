"""Train an SER experiment through the public Python API."""

from __future__ import annotations

import argparse
from pathlib import Path

from torch.utils.data import DataLoader

from ser_lib.data import DatasetManifest, SERDataset
from ser_lib.engine import Trainer, build_experiment_components, load_experiment_config


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path)
    parser.add_argument("--split", default="train")
    parser.add_argument("--batch-size", type=int, default=16)
    args = parser.parse_args()

    config = load_experiment_config(args.config)
    components = build_experiment_components(config, train=True)
    manifest = DatasetManifest.load(config.data.manifest)
    dataset = SERDataset(
        manifest.resolved_records(args.split),
        components.audio_loader,
        components.pipeline,
    )
    batches = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=components.collator,
    )
    trainer = Trainer.from_experiment(components.model, config)
    for result in trainer.fit(lambda: batches):
        print(
            f"epoch={result.epoch} loss={result.loss:.6f} "
            f"accuracy={result.accuracy:.4f} samples={result.sample_count}"
        )


if __name__ == "__main__":
    main()
