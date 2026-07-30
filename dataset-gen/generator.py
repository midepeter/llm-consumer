"""
Synthetic Dataset Generator
============================
Takes cleaned prompts (output of prompt.py's clean_prompts()),
passes each through the teacher model (Qwen3 Coder via HF Inference API)
to produce chain-of-thought rationales, then evaluates every record
through the two-layer evaluator (rule-based + LLM-as-judge).

Only records that pass eval_passed=True are written to the synthetic
output file and returned for the merge step.

Output schema per record:
  {
    "prompt":      str,
    "thinking":    str,    # <think>…</think> block
    "answer":      str,    # final answer
    "raw":         str,    # full model output
    "source":      "synthetic",
    "eval_rule":   dict,
    "eval_llm":    dict,
    "eval_final":  float,
    "eval_passed": bool,
  }
"""

import json
import logging
import os
import re
import time
from pathlib import Path
from typing import Any

from huggingface_hub import InferenceClient

from evaluator import evaluate_synthetic_record, build_judge_client

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

HF_TOKEN       = os.getenv("HF_TOKEN", "")
TEACHER_MODEL  = os.getenv("TEACHER_MODEL", "Qwen/Qwen3-Coder")
MAX_NEW_TOKENS = int(os.getenv("MAX_NEW_TOKENS", "2048"))
TEMPERATURE    = float(os.getenv("TEMPERATURE", "0.6"))

SYNTHETIC_OUTPUT = Path(os.getenv("SYNTHETIC_OUTPUT", "synthetic_data.jsonl"))

COT_SYSTEM_PROMPT = (
    "You are an expert reasoning assistant specialised in software engineering, "
    "algorithms, and code. For every question or task you receive, first think "
    "step-by-step inside <think>…</think> tags, then provide a clear, concise "
    "final answer — including a complete code solution where relevant."
)


# ---------------------------------------------------------------------------
# Teacher model call
# ---------------------------------------------------------------------------

def _build_teacher_client() -> InferenceClient:
    if not HF_TOKEN:
        raise EnvironmentError("HF_TOKEN environment variable is not set.")
    return InferenceClient(token=HF_TOKEN)


def _generate_one(
    client: InferenceClient,
    prompt: str,
    retries: int = 3,
    backoff: float = 5.0,
) -> dict[str, Any]:
    messages = [
        {"role": "system", "content": COT_SYSTEM_PROMPT},
        {"role": "user",   "content": prompt},
    ]
    for attempt in range(1, retries + 1):
        try:
            response = client.chat.completions.create(
                model=TEACHER_MODEL,
                messages=messages,
                max_tokens=MAX_NEW_TOKENS,
                temperature=TEMPERATURE,
            )
            raw      = response.choices[0].message.content.strip()
            thinking, answer = _parse_cot(raw)
            return {"prompt": prompt, "thinking": thinking, "answer": answer, "raw": raw}
        except Exception as exc:
            logger.warning("Teacher call attempt %d/%d failed: %s", attempt, retries, exc)
            if attempt < retries:
                time.sleep(backoff * attempt)

    logger.error("All retries exhausted for prompt: %s…", prompt[:60])
    return {"prompt": prompt, "thinking": "", "answer": "", "raw": "", "error": "generation_failed"}


def _parse_cot(text: str) -> tuple[str, str]:
    match = re.search(r"<think>(.*?)</think>", text, re.DOTALL)
    if match:
        return match.group(1).strip(), text[match.end():].strip()
    return "", text


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def generate_synthetic_dataset(
    prompts: list[str],
    output_file: Path = SYNTHETIC_OUTPUT,
    skip_llm_judge: bool = False,
    resume: bool = True,
) -> list[dict[str, Any]]:
    """
    Generate and evaluate synthetic CoT records for a list of cleaned prompts.

    Args:
        prompts:        Cleaned prompt strings (from prompt.py clean_prompts()).
        output_file:    Where to stream evaluated records (JSONL, append mode).
        skip_llm_judge: If True, skip the LLM-as-judge layer (faster, lower quality).
        resume:         Skip prompts already written to output_file.

    Returns:
        List of records that passed evaluation (eval_passed=True).
    """
    teacher_client = _build_teacher_client()
    judge_client   = build_judge_client() if not skip_llm_judge else None

    # Resume: find already-processed prompts
    completed: set[str] = set()
    if resume and output_file.exists():
        with output_file.open(encoding="utf-8") as f:
            for line in f:
                try:
                    obj = json.loads(line)
                    if obj.get("prompt"):
                        completed.add(obj["prompt"])
                except json.JSONDecodeError:
                    pass
        logger.info("Resume: %d prompts already processed.", len(completed))

    pending = [p for p in prompts if p not in completed]
    logger.info("Generating synthetic data for %d prompt(s) …", len(pending))

    passed_records: list[dict[str, Any]] = []
    rejected_count = 0

    with output_file.open("a", encoding="utf-8") as out:
        for idx, prompt in enumerate(pending, 1):
            logger.info("[%d/%d] Generating → evaluating: %s…", idx, len(pending), prompt[:60])

            # Step 1: generate CoT from teacher
            raw_record = _generate_one(teacher_client, prompt)

            # Step 2: evaluate (rule-based + optional LLM judge)
            evaluated = evaluate_synthetic_record(
                judge_client,
                {**raw_record, "source": "synthetic"},
                skip_llm_judge=skip_llm_judge,
            )

            # Step 3: stream to file regardless of pass/fail (for audit)
            out.write(json.dumps(evaluated, ensure_ascii=False) + "\n")
            out.flush()

            if evaluated["eval_passed"]:
                passed_records.append(evaluated)
            else:
                rejected_count += 1
                logger.info(
                    "  ✗ REJECTED (score=%.2f, flags=%s)",
                    evaluated["eval_final"],
                    evaluated["eval_rule"].get("rule_flags", []),
                )

    pass_rate = (
        len(passed_records) / len(pending) * 100 if pending else 0.0
    )
    logger.info(
        "Synthetic generation complete. Passed: %d | Rejected: %d | Pass rate: %.1f%%",
        len(passed_records), rejected_count, pass_rate,
    )
    logger.info("All records (pass + fail) written to %s for audit.", output_file)

    return passed_records
