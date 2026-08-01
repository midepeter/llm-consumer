"""
Open-Source Dataset Loader
==========================
Loads and normalises records from the following HuggingFace datasets:

  • kaysss/leetcode-problem-set
  • deepmind/code_contests
  • swe-bench/SWE-bench  /  princeton-nlp/SWE-bench_Lite
  • bigcode/humanevalpack
  • google-research-datasets/mbpp  (or 'mbpp' shortname)
  • codeparrot/apps

Each dataset is normalised to the shared schema:
  {
    "prompt":   str,     # the question / task description
    "answer":   str,     # optional reference solution / answer
    "thinking": str,     # "" for open-source (no CoT available)
    "raw":      str,     # same as answer for open-source
    "source":   str,     # dataset name
    "metadata": dict,    # any extra fields kept for traceability
  }

Records are soft-tagged by evaluate_open_source_record() from evaluator.py
but are never hard-filtered.
"""

import hashlib
import json
import logging
import random
from collections import defaultdict
from typing import Any, Callable

from datasets import load_dataset, Dataset

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Shared schema helper
# ---------------------------------------------------------------------------

def _load_hf_dataset(*args, **kwargs):
    """
    Wrapper around datasets.load_dataset compatible with datasets>=4.
    trust_remote_code is intentionally unsupported and removed.
    """
    kwargs.pop("trust_remote_code", None)
    try:
        return load_dataset(*args, **kwargs)
    except Exception as exc:
        message = str(exc)
        if "trust_remote_code" in message and "not supported anymore" in message:
            raise RuntimeError(
                "This dataset appears to rely on a loading script, which is not "
                "supported in datasets>=4. Use a static-format dataset (Parquet/Arrow) "
                "or ask the dataset author to migrate it."
            ) from exc
        raise

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
    logger.info("Loading kaysss/leetcode-problem-set …")
    ds = _load_hf_dataset("kaysss/leetcode-problem-set", split=split)
    records = []
    for row in _iter(ds, max_records):
        prompt = (
            f"LeetCode problem: {row.get('title', '')}\n\n"
            f"Difficulty: {row.get('difficulty', 'Unknown')}\n\n"
            f"Topics: {row.get('topicTags', 'Unknown')}\n\n"
            "Develop a correct solution. State the algorithm, provide an implementation, "
            "and explain the time and space complexity."
        )
        answer = row.get("solution", row.get("python_solution", ""))
        if not str(row.get("title", "")).strip():
            continue
        records.append(_make_record(
            prompt, answer, "leetcode_problem_set",
            metadata={
                "question_id": row.get("frontendQuestionId"),
                "title": row.get("title"),
                "title_slug": row.get("titleSlug"),
                "difficulty": row.get("difficulty"),
                "topic_tags": row.get("topicTags"),
            },
        ))
    logger.info("  → %d records from leetcode_problem_set", len(records))
    return records


def _load_code_contests(split: str = "train", max_records: int | None = None) -> list[dict]:
    logger.info("Loading deepmind/code_contests …")
    ds = _load_hf_dataset("deepmind/code_contests", split=split)
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
    ds = _load_hf_dataset(repo, split=split)
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
            ds = _load_hf_dataset("bigcode/humanevalpack", lang, split=split)
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
    ds = _load_hf_dataset("google-research-datasets/mbpp", split=split)
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


def _load_apps(split: str = "train", max_records: int | None = None) -> list[dict]:
    """Load APPS programming problems, retaining solutions only for validation."""
    logger.info("Loading codeparrot/apps …")
    ds = _load_hf_dataset("codeparrot/apps", split=split)
    records = []
    for row in _iter(ds, max_records):
        prompt = row.get("question", "")
        solutions = row.get("solutions", "")
        if isinstance(solutions, str):
            try:
                parsed_solutions = json.loads(solutions)
            except json.JSONDecodeError:
                parsed_solutions = []
        else:
            parsed_solutions = solutions if isinstance(solutions, list) else []
        answer = parsed_solutions[0] if parsed_solutions else ""
        if not str(prompt).strip():
            continue
        records.append(_make_record(
            prompt,
            str(answer),
            "apps",
            metadata={
                "problem_id": row.get("problem_id"),
                "difficulty": row.get("difficulty"),
            },
        ))
    logger.info("  → %d records from apps", len(records))
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
    "apps":            _load_apps,
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


def sample_unlabeled_inputs(
    records: list[dict[str, Any]],
    sample_size: int,
    seed: int = 42,
) -> list[dict[str, Any]]:
    """
    Select source tasks for teacher generation without exposing their answers.

    The returned records retain ``reference_answer`` only for downstream
    validation. Callers must pass only ``input`` to the teacher.
    """
    if sample_size <= 0:
        raise ValueError("sample_size must be greater than zero.")

    usable = [record for record in records if str(record.get("prompt", "")).strip()]
    if not usable:
        raise ValueError("No open-source records contain a usable prompt.")

    if sample_size > len(usable):
        raise ValueError(
            f"Requested {sample_size} unique source tasks, but only {len(usable)} "
            "usable tasks were loaded. Increase MAX_OS_PER_DATASET, add source "
            "datasets, or lower the candidate budget."
        )

    rng = random.Random(seed)
    by_source: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in usable:
        by_source[str(record.get("source", "open_source"))].append(record)
    for group in by_source.values():
        rng.shuffle(group)

    selected: list[dict[str, Any]] = []
    positions = {source: 0 for source in by_source}
    sources = sorted(by_source)
    while len(selected) < sample_size:
        added = False
        for source in sources:
            position = positions[source]
            group = by_source[source]
            if position >= len(group):
                continue
            selected.append(group[position])
            positions[source] += 1
            added = True
            if len(selected) == sample_size:
                break
        if not added:
            raise RuntimeError("Source-balanced sampling exhausted unexpectedly.")

    source_counts = {
        source: sum(1 for record in selected if record.get("source") == source)
        for source in sources
    }
    logger.info("Source-balanced candidate sample: %s", source_counts)
    inputs: list[dict[str, Any]] = []
    for record in selected:
        prompt = str(record["prompt"]).strip()
        source = str(record.get("source", "open_source"))
        metadata = dict(record.get("metadata", {}))
        identity = f"{source}\n{metadata}\n{prompt}".encode("utf-8")
        inputs.append({
            "sample_id": hashlib.sha256(identity).hexdigest()[:16],
            "input": prompt,
            "source": source,
            "source_metadata": metadata,
            "reference_answer": str(record.get("answer", "")).strip(),
        })
    return inputs


# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------

def _iter(ds: Dataset, max_records: int | None):
    if max_records is not None:
        ds = ds.select(range(min(max_records, len(ds))))
    yield from ds
