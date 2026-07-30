"""
Open-Source Dataset Loader
==========================
Loads and normalises records from the following HuggingFace datasets:

  • mppk/leetcode_problems
  • deepmind/code_contests
  • swe-bench/SWE-bench  /  princeton-nlp/SWE-bench_Lite
  • bigcode/humanevalpack
  • google-research-datasets/mbpp  (or 'mbpp' shortname)

Each dataset is normalised to the shared schema:
  {
    "prompt":   str,     # the question / task description
    "answer":   str,     # reference solution / answer
    "thinking": str,     # "" for open-source (no CoT available)
    "raw":      str,     # same as answer for open-source
    "source":   str,     # dataset name
    "metadata": dict,    # any extra fields kept for traceability
  }

Records are soft-tagged by evaluate_open_source_record() from evaluator.py
but are never hard-filtered.
"""

import logging
from typing import Any, Callable

from datasets import load_dataset, Dataset

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Shared schema helper
# ---------------------------------------------------------------------------

def _make_record(
    prompt: str,
    answer: str,
    source: str,
    metadata: dict | None = None,
) -> dict[str, Any]:
    return {
        "prompt":   prompt.strip(),
        "answer":   answer.strip(),
        "thinking": "",
        "raw":      answer.strip(),
        "source":   source,
        "metadata": metadata or {},
    }


# ---------------------------------------------------------------------------
# Per-dataset loaders
# ---------------------------------------------------------------------------

def _load_leetcode(split: str = "train", max_records: int | None = None) -> list[dict]:
    logger.info("Loading mppk/leetcode_problems …")
    ds = load_dataset("mppk/leetcode_problems", split=split, trust_remote_code=True)
    records = []
    for row in _iter(ds, max_records):
        prompt = (
            f"Problem: {row.get('title', '')}\n\n"
            f"Difficulty: {row.get('difficulty', 'Unknown')}\n\n"
            f"{row.get('description', row.get('content', ''))}"
        )
        answer = row.get("solution", row.get("python_solution", ""))
        if not prompt.strip() or not answer.strip():
            continue
        records.append(_make_record(
            prompt, answer, "leetcode_problems",
            metadata={"title": row.get("title"), "difficulty": row.get("difficulty")},
        ))
    logger.info("  → %d records from leetcode_problems", len(records))
    return records


def _load_code_contests(split: str = "train", max_records: int | None = None) -> list[dict]:
    logger.info("Loading deepmind/code_contests …")
    ds = load_dataset("deepmind/code_contests", split=split, trust_remote_code=True)
    records = []
    for row in _iter(ds, max_records):
        prompt  = row.get("description", "")
        # solutions field is a dict with keys 'language', 'solution'
        solutions = row.get("solutions", {})
        sol_list  = solutions.get("solution", []) if isinstance(solutions, dict) else []
        answer    = sol_list[0] if sol_list else ""
        if not prompt.strip() or not answer.strip():
            continue
        records.append(_make_record(
            prompt, answer, "code_contests",
            metadata={"name": row.get("name"), "difficulty": row.get("difficulty")},
        ))
    logger.info("  → %d records from code_contests", len(records))
    return records


def _load_swe_bench(
    repo: str = "swe-bench/SWE-bench",
    split: str = "test",
    max_records: int | None = None,
) -> list[dict]:
    logger.info("Loading %s …", repo)
    ds = load_dataset(repo, split=split, trust_remote_code=True)
    records = []
    for row in _iter(ds, max_records):
        prompt = (
            f"Repository: {row.get('repo', '')}\n"
            f"Issue: {row.get('problem_statement', '')}"
        )
        answer = row.get("patch", row.get("hints_text", ""))
        if not prompt.strip() or not answer.strip():
            continue
        records.append(_make_record(
            prompt, answer, repo.split("/")[-1],
            metadata={"instance_id": row.get("instance_id"), "repo": row.get("repo")},
        ))
    logger.info("  → %d records from %s", len(records), repo)
    return records


def _load_humanevalpack(split: str = "test", max_records: int | None = None) -> list[dict]:
    logger.info("Loading bigcode/humanevalpack …")
    records = []
    for lang in ("python", "js", "java", "cpp", "go", "rust"):
        try:
            ds = load_dataset("bigcode/humanevalpack", lang, split=split, trust_remote_code=True)
        except Exception as exc:
            logger.warning("  humanevalpack/%s failed: %s", lang, exc)
            continue
        for row in _iter(ds, max_records):
            prompt = row.get("prompt", row.get("declaration", ""))
            answer = row.get("canonical_solution", "")
            if not prompt.strip() or not answer.strip():
                continue
            records.append(_make_record(
                prompt, answer, f"humanevalpack_{lang}",
                metadata={"task_id": row.get("task_id"), "language": lang},
            ))
    logger.info("  → %d records from humanevalpack (all languages)", len(records))
    return records


def _load_mbpp(split: str = "train", max_records: int | None = None) -> list[dict]:
    logger.info("Loading google-research-datasets/mbpp …")
    ds = load_dataset("google-research-datasets/mbpp", split=split, trust_remote_code=True)
    records = []
    for row in _iter(ds, max_records):
        prompt = row.get("text", "")
        answer = row.get("code", "")
        tests  = row.get("test_list", [])
        if tests:
            prompt += "\n\nTest cases:\n" + "\n".join(tests)
        if not prompt.strip() or not answer.strip():
            continue
        records.append(_make_record(
            prompt, answer, "mbpp",
            metadata={"task_id": row.get("task_id")},
        ))
    logger.info("  → %d records from mbpp", len(records))
    return records


# ---------------------------------------------------------------------------
# Registry — add new datasets here without changing other files
# ---------------------------------------------------------------------------

DATASET_REGISTRY: dict[str, Callable[..., list[dict]]] = {
    "leetcode":        _load_leetcode,
    "code_contests":   _load_code_contests,
    "swe_bench":       lambda **kw: _load_swe_bench("swe-bench/SWE-bench", **kw),
    "swe_bench_lite":  lambda **kw: _load_swe_bench("princeton-nlp/SWE-bench_Lite", **kw),
    "humanevalpack":   _load_humanevalpack,
    "mbpp":            _load_mbpp,
}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def load_open_source_datasets(
    datasets: list[str] | None = None,
    max_records_per_dataset: int | None = None,
) -> list[dict[str, Any]]:
    """
    Load and normalise open-source datasets.

    Args:
        datasets:  list of keys from DATASET_REGISTRY (default: all).
        max_records_per_dataset: cap per dataset (useful for testing).

    Returns:
        Combined list of normalised records.
    """
    keys = datasets or list(DATASET_REGISTRY.keys())
    all_records: list[dict] = []

    for key in keys:
        if key not in DATASET_REGISTRY:
            logger.warning("Unknown dataset key '%s'. Available: %s", key, list(DATASET_REGISTRY))
            continue
        loader = DATASET_REGISTRY[key]
        try:
            records = loader(max_records=max_records_per_dataset)
            all_records.extend(records)
        except Exception as exc:
            logger.error("Failed to load dataset '%s': %s", key, exc)

    logger.info("Total open-source records loaded: %d", len(all_records))
    return all_records


# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------

def _iter(ds: Dataset, max_records: int | None):
    if max_records is not None:
        ds = ds.select(range(min(max_records, len(ds))))
    yield from ds
