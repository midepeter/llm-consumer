"""
Hybrid Dataset Merger
======================
Merges evaluated open-source records (70%) with evaluated synthetic records (30%)
into a single, shuffled, final hybrid dataset.

Ratio enforcement strategy:
  - Take ALL synthetic records that passed evaluation.
  - Calculate the open-source quota as:  os_quota = synthetic_count * (70 / 30)
  - If more open-source records exist than os_quota, sample them (stratified by source).
  - If fewer open-source records exist, use all and log the actual ratio.

Output: hybrid_dataset.jsonl — the final training-ready file.

Each record in the output has a normalised top-level field 'split' added:
  "synthetic"   or   "open_source"
"""

import json
import logging
import random
from collections import defaultdict
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

TARGET_SYNTHETIC_RATIO = 0.30    # 30 % synthetic
TARGET_OS_RATIO        = 0.70    # 70 % open-source
RANDOM_SEED            = 42


# ---------------------------------------------------------------------------
# Stratified sampling helpers
# ---------------------------------------------------------------------------

def _stratified_sample(records: list[dict], n: int) -> list[dict]:
    """
    Sample n records from a pool, preserving the source-wise distribution
    as closely as possible.
    """
    if n >= len(records):
        return records

    # Group by source
    by_source: dict[str, list[dict]] = defaultdict(list)
    for r in records:
        by_source[r.get("source", "unknown")].append(r)

    total  = len(records)
    result = []
    for source, group in by_source.items():
        quota = round(n * len(group) / total)
        quota = min(quota, len(group))
        result.extend(random.sample(group, quota))

    # top-up or trim due to rounding
    remaining = [r for r in records if r not in result]
    if len(result) < n:
        extra = min(n - len(result), len(remaining))
        result.extend(random.sample(remaining, extra))
    elif len(result) > n:
        result = result[:n]

    return result


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def merge_datasets(
    synthetic_records: list[dict[str, Any]],
    open_source_records: list[dict[str, Any]],
    output_file: Path = Path("hybrid_dataset.jsonl"),
    seed: int = RANDOM_SEED,
) -> list[dict[str, Any]]:
    """
    Merge synthetic and open-source records into a hybrid dataset.

    Args:
        synthetic_records:    Evaluated synthetic records (eval_passed=True).
        open_source_records:  Soft-tagged open-source records (all kept).
        output_file:          Where to write the final JSONL.
        seed:                 Random seed for reproducibility.

    Returns:
        The merged, shuffled dataset as a list of dicts.
    """
    random.seed(seed)

    n_synthetic = len(synthetic_records)

    if n_synthetic == 0:
        logger.warning("No synthetic records provided — output will be 100%% open-source.")
        os_sample = open_source_records
    else:
        # Target: synthetic is 30%, so open-source should be synthetic * (70/30)
        os_quota = round(n_synthetic * (TARGET_OS_RATIO / TARGET_SYNTHETIC_RATIO))
        logger.info(
            "Targeting %d open-source records to pair with %d synthetic records (%.0f%%/%.0f%% split).",
            os_quota, n_synthetic,
            TARGET_OS_RATIO * 100, TARGET_SYNTHETIC_RATIO * 100,
        )
        os_sample = _stratified_sample(open_source_records, os_quota)

    # Tag splits
    for r in synthetic_records:
        r["split"] = "synthetic"
    for r in os_sample:
        r["split"] = "open_source"

    merged = synthetic_records + os_sample
    random.shuffle(merged)

    # Write
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with output_file.open("w", encoding="utf-8") as f:
        for record in merged:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    # Stats
    actual_synthetic_pct = len(synthetic_records) / len(merged) * 100 if merged else 0
    actual_os_pct        = len(os_sample) / len(merged) * 100 if merged else 0

    logger.info("─" * 60)
    logger.info("Hybrid dataset written → %s", output_file)
    logger.info("  Total records  : %d", len(merged))
    logger.info("  Synthetic      : %d  (%.1f%%)", len(synthetic_records), actual_synthetic_pct)
    logger.info("  Open-source    : %d  (%.1f%%)", len(os_sample), actual_os_pct)

    # Source breakdown
    source_counts: dict[str, int] = defaultdict(int)
    for r in merged:
        source_counts[r.get("source", "unknown")] += 1
    logger.info("  Source breakdown:")
    for src, count in sorted(source_counts.items(), key=lambda x: -x[1]):
        logger.info("    %-30s %d", src, count)
    logger.info("─" * 60)

    return merged


# ---------------------------------------------------------------------------
# Convenience: load already-written JSONL files and merge
# ---------------------------------------------------------------------------

def merge_from_files(
    synthetic_file: Path,
    open_source_file: Path,
    output_file: Path = Path("hybrid_dataset.jsonl"),
) -> list[dict[str, Any]]:
    """
    Load synthetic and open-source JSONL files from disk and merge them.
    Useful when running merge as a standalone step.
    """
    def _load(path: Path) -> list[dict]:
        records = []
        with path.open(encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        records.append(json.loads(line))
                    except json.JSONDecodeError as e:
                        logger.warning("Skipping malformed line in %s: %s", path, e)
        return records

    logger.info("Loading synthetic records from %s …", synthetic_file)
    synthetic = [r for r in _load(synthetic_file) if r.get("eval_passed", False)]
    logger.info("  → %d passed synthetic records", len(synthetic))

    logger.info("Loading open-source records from %s …", open_source_file)
    open_source = _load(open_source_file)
    logger.info("  → %d open-source records", len(open_source))

    return merge_datasets(synthetic, open_source, output_file=output_file)
