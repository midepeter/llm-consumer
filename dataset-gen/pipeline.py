"""
Hybrid Dataset Pipeline — Orchestrator
========================================
Runs the full chain-of-thought knowledge distillation pipeline:

  1. Load prompts               (prompt.py  → load_prompts / clean_prompts)
  2. Generate synthetic data    (generator.py → generate_synthetic_dataset)
  3. Load + tag open-source     (open_source.py → load_open_source_datasets)
  4. Soft-evaluate open-source  (evaluator.py → evaluate_open_source_record)
  5. Merge 70/30                (merger.py → merge_datasets)
  6. Write final hybrid dataset → hybrid_dataset.jsonl

Run:
  export HF_TOKEN="hf_..."
  python pipeline.py

Optional env vars:
  TEACHER_MODEL       HuggingFace model ID   (default: Qwen/Qwen3-Coder)
  JUDGE_MODEL         HuggingFace model ID   (default: same as TEACHER_MODEL)
  PASS_THRESHOLD      Min score to keep      (default: 3.0)
  MAX_NEW_TOKENS      Max tokens per call    (default: 2048)
  TEMPERATURE         Sampling temperature   (default: 0.6)
  SYNTHETIC_OUTPUT    Synthetic JSONL path   (default: synthetic_data.jsonl)
  OS_OUTPUT           Open-source JSONL path (default: open_source_data.jsonl)
  HYBRID_OUTPUT       Final output path      (default: hybrid_dataset.jsonl)
  MAX_OS_PER_DATASET  Cap per OS dataset     (default: no cap)
  SKIP_LLM_JUDGE      "1" to skip judge      (default: 0)
"""

import json
import logging
import os
from pathlib import Path

# ── local modules ────────────────────────────────────────────────────────────
from prompt import load_prompts, clean_prompts
from generator import generate_synthetic_dataset
from open_source import load_open_source_datasets, DATASET_REGISTRY
from evaluator import evaluate_open_source_record
from merger import merge_datasets

# ── logging ──────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

# ── paths ────────────────────────────────────────────────────────────────────
SYNTHETIC_OUTPUT = Path(os.getenv("SYNTHETIC_OUTPUT", "synthetic_data.jsonl"))
OS_OUTPUT        = Path(os.getenv("OS_OUTPUT",        "open_source_data.jsonl"))
HYBRID_OUTPUT    = Path(os.getenv("HYBRID_OUTPUT",    "hybrid_dataset.jsonl"))

MAX_OS_PER_DATASET = (
    int(os.getenv("MAX_OS_PER_DATASET"))
    if os.getenv("MAX_OS_PER_DATASET")
    else None
)
SKIP_LLM_JUDGE = os.getenv("SKIP_LLM_JUDGE", "0") == "1"

# ── which open-source datasets to pull (all by default) ──────────────────────
OPEN_SOURCE_DATASETS: list[str] = list(DATASET_REGISTRY.keys())
# Uncomment to limit:
# OPEN_SOURCE_DATASETS = ["leetcode", "mbpp", "humanevalpack"]


# =============================================================================
# Step helpers
# =============================================================================

def step_synthetic(prompts: list[str]) -> list[dict]:
    """Generate + evaluate synthetic records from cleaned prompts."""
    logger.info("=" * 60)
    logger.info("STEP 2 — Generating synthetic data (%d prompts)", len(prompts))
    logger.info("=" * 60)
    passed = generate_synthetic_dataset(
        prompts,
        output_file=SYNTHETIC_OUTPUT,
        skip_llm_judge=SKIP_LLM_JUDGE,
        resume=True,
    )
    logger.info("Synthetic step done. %d records passed.", len(passed))
    return passed


def step_open_source() -> list[dict]:
    """Load, normalise, and soft-evaluate open-source datasets."""
    logger.info("=" * 60)
    logger.info("STEP 3 — Loading open-source datasets")
    logger.info("=" * 60)
    raw_records = load_open_source_datasets(
        datasets=OPEN_SOURCE_DATASETS,
        max_records_per_dataset=MAX_OS_PER_DATASET,
    )

    logger.info("STEP 4 — Soft-evaluating open-source records …")
    evaluated = [evaluate_open_source_record(r) for r in raw_records]

    # Stream to disk for audit / future incremental runs
    with OS_OUTPUT.open("w", encoding="utf-8") as f:
        for record in evaluated:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    logger.info("Open-source records written to %s", OS_OUTPUT)

    return evaluated


def step_merge(synthetic: list[dict], open_source: list[dict]) -> list[dict]:
    """Merge 70% open-source + 30% synthetic into the final hybrid dataset."""
    logger.info("=" * 60)
    logger.info("STEP 5 — Merging into hybrid dataset (70 / 30)")
    logger.info("=" * 60)
    return merge_datasets(synthetic, open_source, output_file=HYBRID_OUTPUT)


# =============================================================================
# Main
# =============================================================================

def main():
    logger.info("╔══════════════════════════════════════════════════════════╗")
    logger.info("║   Chain-of-Thought Knowledge Distillation Pipeline       ║")
    logger.info("╚══════════════════════════════════════════════════════════╝")

    # ── STEP 1: Load + clean prompts ─────────────────────────────────────────
    logger.info("STEP 1 — Loading and cleaning prompts")

    # ── Configure your prompt source here ────────────────────────────────────
    # Option A — load from a file:
    # raw_prompts = load_prompts("prompts.txt")     # or .json / .jsonl / .csv

    # Option B — inline list:
    raw_prompts = load_prompts([
        "Implement a function to find the longest common subsequence of two strings. Explain each step.",
        "Given a binary tree, write an algorithm to determine if it is height-balanced. Include complexity analysis.",
        "Write a Python class implementing a thread-safe LRU cache with O(1) get and put.",
        "Solve the 0/1 knapsack problem using dynamic programming. Trace through an example.",
        "Implement Dijkstra's shortest path algorithm and explain why it works.",
    ])
    # ─────────────────────────────────────────────────────────────────────────

    prompts = clean_prompts(raw_prompts, deduplicate=True)
    logger.info("Cleaned prompts ready: %d", len(prompts))

    # ── STEP 2: Synthetic data generation ────────────────────────────────────
    synthetic_records = step_synthetic(prompts)

    # ── STEPS 3 & 4: Open-source load + soft eval ────────────────────────────
    open_source_records = step_open_source()

    # ── STEP 5: Merge ────────────────────────────────────────────────────────
    hybrid = step_merge(synthetic_records, open_source_records)

    logger.info("Pipeline complete. Final dataset: %d records → %s", len(hybrid), HYBRID_OUTPUT)


if __name__ == "__main__":
    main()
