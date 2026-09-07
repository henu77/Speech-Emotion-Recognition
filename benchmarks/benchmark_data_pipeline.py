"""数据管线人工性能基准；不作为 pytest 硬门槛。

示例：
    python benchmarks/benchmark_data_pipeline.py path/to/dataset.yaml --split train
"""
from __future__ import annotations
import argparse
import time
from pathlib import Path
from torch.utils.data import DataLoader
from ser_lib.data import DatasetManifest, SERDataset, build_collator, build_components, load_data_config

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=Path)
    parser.add_argument("--split", default="train")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--max-batches", type=int, default=50)
    args = parser.parse_args()

    config = load_data_config(args.config)
    manifest = DatasetManifest.load(config.manifest)
    loader, pipeline = build_components(config, train=args.split == "train")
    dataset = SERDataset(manifest.resolved_records(args.split), loader, pipeline)
    batches = DataLoader(
        dataset, batch_size=args.batch_size, num_workers=args.num_workers,
        collate_fn=build_collator(pipeline.output_specs, config.batching),
        pin_memory=False,
    )
    started = time.perf_counter()
    sample_count = batch_count = 0
    first_batch_seconds = None
    for batch_count, batch in enumerate(batches, start=1):
        if first_batch_seconds is None:
            first_batch_seconds = time.perf_counter() - started
        sample_count += len(batch.uids)
        if batch_count >= args.max_batches:
            break
    elapsed = time.perf_counter() - started
    print({
        "samples": sample_count,
        "batches": batch_count,
        "seconds": round(elapsed, 4),
        "first_batch_seconds": round(first_batch_seconds or elapsed, 4),
        "samples_per_second": round(sample_count / elapsed, 2) if elapsed else None,
        "num_workers": args.num_workers,
    })

if __name__ == "__main__":
    main()
