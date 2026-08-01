"""
Exemplar-guided distillation pipeline.

Flow:
    open-source task -> unlabeled input
    unlabeled input + exemplar triplets -> teacher
    teacher output + held-out reference -> validation/filtering
    accepted (input, rationale, answer) -> JSONL

Run a small pilot first (the default):
    export INFERENCE_BACKEND="vllm"
    export VLLM_BASE_URL="http://<gpu-host>:8000/v1"
    export TEACHER_MODEL="your-served-model-name"
    python pipeline.py

Set RUN_MODE=main only after inspecting pilot_distillation.jsonl.

Inference configuration:
    INFERENCE_BACKEND   "vllm" for a self-hosted OpenAI-compatible server,
                        or "huggingface" (default)
    VLLM_BASE_URL       e.g. http://<gpu-host>:8000/v1
    VLLM_API_KEY        optional server key (default: EMPTY)
    TEACHER_MODEL       exact model ID returned by GET /v1/models
    JUDGE_MODEL         optional; defaults to TEACHER_MODEL

The pilot uses 25 generation requests by default. A main run uses 2,000
generation requests. Enabling the LLM judge adds one completion request per
generated record; use SKIP_LLM_JUDGE=1 for the initial generation run.

Scale configuration:
MAIN_CANDIDATE_COUNT      unique tasks to submit in a main run
TARGET_ACCEPTED_RECORDS   final accepted-record goal; unset to process all candidates
MAX_CONCURRENT_REQUESTS   bounded in-flight requests (start at 8-16 for vLLM)
MAX_OS_PER_DATASET        per-source loading cap; must support the candidate count
"""

import logging
import os
from pathlib import Path

from exemplars import load_exemplars
from generator import generate_distillation_dataset
from open_source import DATASET_REGISTRY, load_open_source_datasets, sample_unlabeled_inputs


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

RUN_MODE = os.getenv("RUN_MODE", "pilot").strip().lower()
PILOT_SAMPLE_SIZE = int(os.getenv("PILOT_SAMPLE_SIZE", "25"))
MAIN_CANDIDATE_COUNT = int(
    os.getenv("MAIN_CANDIDATE_COUNT", os.getenv("MAIN_SAMPLE_SIZE", "2000"))
)
MAX_OS_PER_DATASET = int(os.getenv("MAX_OS_PER_DATASET", "1000"))
SAMPLE_SEED = int(os.getenv("SAMPLE_SEED", "42"))
EXEMPLAR_LIMIT = int(os.getenv("EXEMPLAR_LIMIT", "3"))
SKIP_LLM_JUDGE = os.getenv("SKIP_LLM_JUDGE", "0") == "1"
MAX_CONCURRENT_REQUESTS = int(os.getenv("MAX_CONCURRENT_REQUESTS", "1"))
TARGET_ACCEPTED_RECORDS = (
    int(os.getenv("TARGET_ACCEPTED_RECORDS"))
    if os.getenv("TARGET_ACCEPTED_RECORDS")
    else None
)
EXEMPLAR_FILE = Path(os.getenv("EXEMPLAR_FILE", "exemplars.jsonl"))
PILOT_OUTPUT = Path(os.getenv("PILOT_OUTPUT", "pilot_distillation.jsonl"))
MAIN_OUTPUT = Path(os.getenv("MAIN_OUTPUT", "main_distillation.jsonl"))


def _selected_datasets() -> list[str]:
    configured = os.getenv("OPEN_SOURCE_DATASETS", "")
    if not configured.strip():
        return list(DATASET_REGISTRY)

    datasets = [item.strip() for item in configured.split(",") if item.strip()]
    unknown = sorted(set(datasets) - DATASET_REGISTRY.keys())
    if unknown:
        raise ValueError(
            f"Unknown OPEN_SOURCE_DATASETS values: {', '.join(unknown)}. "
            f"Available: {', '.join(DATASET_REGISTRY)}."
        )
    return datasets


def _run_configuration() -> tuple[int, Path]:
    if RUN_MODE == "pilot":
        return PILOT_SAMPLE_SIZE, PILOT_OUTPUT
    if RUN_MODE == "main":
        return MAIN_CANDIDATE_COUNT, MAIN_OUTPUT
    raise ValueError("RUN_MODE must be either 'pilot' or 'main'.")


def main() -> None:
    sample_size, output_file = _run_configuration()
    if TARGET_ACCEPTED_RECORDS is not None and TARGET_ACCEPTED_RECORDS > sample_size:
        raise ValueError(
            "TARGET_ACCEPTED_RECORDS cannot exceed the current candidate budget. "
            "Increase MAIN_CANDIDATE_COUNT to allow for validation rejections."
        )
    datasets = _selected_datasets()
    exemplars = load_exemplars(EXEMPLAR_FILE, limit=EXEMPLAR_LIMIT)

    logger.info(
        "Starting %s run: %d candidates, target=%s, %d exemplar(s), datasets=%s",
        RUN_MODE,
        sample_size,
        TARGET_ACCEPTED_RECORDS or "all accepted",
        len(exemplars),
        ", ".join(datasets),
    )
    source_records = load_open_source_datasets(
        datasets=datasets,
        max_records_per_dataset=MAX_OS_PER_DATASET,
    )
    unlabeled_inputs = sample_unlabeled_inputs(
        source_records,
        sample_size=sample_size,
        seed=SAMPLE_SEED,
    )
    accepted = generate_distillation_dataset(
        unlabeled_inputs,
        exemplars,
        output_file=output_file,
        skip_llm_judge=SKIP_LLM_JUDGE,
        resume=True,
        target_accepted=TARGET_ACCEPTED_RECORDS,
        max_workers=MAX_CONCURRENT_REQUESTS,
    )
    logger.info(
        "%s complete: %d accepted records written to %s.",
        RUN_MODE.capitalize(),
        len(accepted),
        output_file,
    )
    if RUN_MODE == "pilot":
        logger.info(
            "Inspect the pilot output before scaling: RUN_MODE=main "
            "MAIN_CANDIDATE_COUNT=<desired-count> python pipeline.py"
        )


if __name__ == "__main__":
    main()
