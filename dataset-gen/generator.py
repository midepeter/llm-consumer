"""
Exemplar-guided teacher generation for distillation samples.

The teacher receives only an unlabeled source input and few-shot exemplar
triplets. Source reference answers remain local and are available only to the
validation stage.
"""

import json
import logging
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

from huggingface_hub.utils import HfHubHTTPError
from openai import OpenAIError

from evaluator import build_judge_client, evaluate_synthetic_record
from exemplars import format_exemplars
from inference import build_inference_client, thinking_extra_body, validate_model_access


logger = logging.getLogger(__name__)

TEACHER_MODEL = os.getenv("TEACHER_MODEL", "")
MAX_NEW_TOKENS = int(os.getenv("MAX_NEW_TOKENS", "2048"))
THINKING_BUDGET = int(os.getenv("THINKING_BUDGET", "1024"))
TEMPERATURE = float(os.getenv("TEMPERATURE", "0.4"))
DISTILLATION_OUTPUT = Path(os.getenv("DISTILLATION_OUTPUT", "distillation_data.jsonl"))

DISTILLATION_SYSTEM_PROMPT = """\
You are generating high-quality supervised distillation data for coding tasks.
Use the demonstrations to infer the expected standard. Solve the new task
independently; do not copy an example answer. Give a concise but rigorous
rationale followed by a complete final answer. For code tasks, include working
code and complexity analysis when relevant.
"""


def _build_teacher_client() -> Any:
    if not TEACHER_MODEL.strip():
        raise EnvironmentError(
            "TEACHER_MODEL is not set. Provide the model ID served by the selected backend."
        )
    return build_inference_client()


def _validate_teacher_model_access(client: Any) -> None:
    validate_model_access(client, TEACHER_MODEL)


def build_distillation_messages(
    unlabeled_input: str,
    exemplars: list[dict[str, str]],
) -> list[dict[str, str]]:
    """Build the teacher request without any source reference answer."""
    if not unlabeled_input.strip():
        raise ValueError("The unlabeled input cannot be empty.")
    if not exemplars:
        raise ValueError("At least one exemplar triplet is required.")

    return [
        {"role": "system", "content": DISTILLATION_SYSTEM_PROMPT},
        {
            "role": "user",
            "content": (
                f"{format_exemplars(exemplars)}\n\n"
                "NEW TASK INPUT:\n"
                f"{unlabeled_input.strip()}\n\n"
                "Produce the rationale and final answer for the new task."
            ),
        },
    ]


def _extract_thinking_and_answer(message: Any) -> tuple[str, str, str]:
    """Extract native Qwen thinking or a tagged fallback from a completion."""
    content = (message.content or "").strip()
    reasoning = getattr(message, "reasoning_content", None)
    if reasoning and reasoning.strip():
        thinking = reasoning.strip()
        return thinking, content, f"<think>\n{thinking}\n</think>\n{content}"

    match = re.search(r"<think>(.*?)</think>", content, re.DOTALL)
    if match:
        return match.group(1).strip(), content[match.end():].strip(), content
    return "", content, content


def _generate_one(
    client: Any,
    unlabeled_input: str,
    exemplars: list[dict[str, str]],
    retries: int = 3,
    backoff: float = 5.0,
) -> dict[str, str]:
    messages = build_distillation_messages(unlabeled_input, exemplars)
    extra_body = thinking_extra_body(THINKING_BUDGET)

    for attempt in range(1, retries + 1):
        try:
            response = client.chat.completions.create(
                model=TEACHER_MODEL,
                messages=messages,
                max_tokens=MAX_NEW_TOKENS,
                temperature=TEMPERATURE,
                extra_body=extra_body,
            )
            thinking, answer, raw = _extract_thinking_and_answer(
                response.choices[0].message
            )
            return {"thinking": thinking, "answer": answer, "raw": raw}
        except (HfHubHTTPError, OpenAIError) as exc:
            response = getattr(exc, "response", None)
            if response is not None and response.status_code == 404:
                raise ValueError(
                    f"Teacher model '{TEACHER_MODEL}' was not found by the configured backend."
                ) from exc
            logger.warning("Teacher attempt %d/%d failed: %s", attempt, retries, exc)
            if attempt < retries:
                time.sleep(backoff * attempt)

    return {
        "thinking": "",
        "answer": "",
        "raw": "",
        "error": "generation_failed",
    }


def _existing_output_state(output_file: Path) -> tuple[set[str], int]:
    if not output_file.exists():
        return set(), 0
    completed: set[str] = set()
    accepted_count = 0
    with output_file.open(encoding="utf-8") as handle:
        for line in handle:
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                logger.warning("Ignoring malformed existing output line in %s.", output_file)
                continue
            if isinstance(record, dict) and record.get("sample_id"):
                completed.add(str(record["sample_id"]))
                if record.get("validation", {}).get("passed", False):
                    accepted_count += 1
    return completed, accepted_count


def _generate_and_evaluate(
    teacher_client: Any,
    judge_client: Any,
    source_record: dict[str, Any],
    exemplars: list[dict[str, str]],
    skip_llm_judge: bool,
) -> dict[str, Any]:
    task_input = str(source_record.get("input", "")).strip()
    sample_id = str(source_record.get("sample_id", "")).strip()
    if not sample_id or not task_input:
        raise ValueError("Every source record needs non-empty sample_id and input fields.")

    generated = _generate_one(teacher_client, task_input, exemplars)
    evaluation_input = {
        "prompt": task_input,
        **generated,
        "source": source_record.get("source", "open_source"),
        "reference_answer": source_record.get("reference_answer", ""),
    }
    evaluated = evaluate_synthetic_record(
        judge_client,
        evaluation_input,
        skip_llm_judge=skip_llm_judge,
    )
    evaluated.pop("reference_answer", None)
    dataset_record = {
        "sample_id": sample_id,
        "input": task_input,
        "rationale": evaluated.pop("thinking"),
        "answer": evaluated.pop("answer"),
        "source": evaluated.pop("source"),
        "source_metadata": source_record.get("source_metadata", {}),
        "validation": {
            "rule": evaluated.pop("eval_rule"),
            "llm": evaluated.pop("eval_llm"),
            "score": evaluated.pop("eval_final"),
            "passed": evaluated.pop("eval_passed"),
        },
        "raw": evaluated.pop("raw"),
    }
    if "error" in evaluated:
        dataset_record["error"] = evaluated["error"]
    return dataset_record


def generate_distillation_dataset(
    unlabeled_inputs: list[dict[str, Any]],
    exemplars: list[dict[str, str]],
    output_file: Path = DISTILLATION_OUTPUT,
    skip_llm_judge: bool = False,
    resume: bool = True,
    target_accepted: int | None = None,
    max_workers: int = 1,
) -> list[dict[str, Any]]:
    """
    Generate validated ``(input, rationale, answer)`` records.

    ``reference_answer`` may optionally be present on an input for validation, but is
    deliberately excluded from persisted output and teacher messages. When
    ``target_accepted`` is set, generation stops after that many accepted records
    exist in ``output_file``; ``max_workers`` bounds concurrent API requests.
    """
    if target_accepted is not None and target_accepted <= 0:
        raise ValueError("target_accepted must be greater than zero when set.")
    if max_workers <= 0:
        raise ValueError("max_workers must be greater than zero.")

    teacher_client = _build_teacher_client()
    _validate_teacher_model_access(teacher_client)
    judge_client = build_judge_client() if not skip_llm_judge else None
    completed, existing_accepted = (
        _existing_output_state(output_file) if resume else (set(), 0)
    )
    if target_accepted is not None and existing_accepted >= target_accepted:
        logger.info(
            "Accepted-record target already met: %d/%d.",
            existing_accepted,
            target_accepted,
        )
        return []

    pending = [
        record for record in unlabeled_inputs
        if str(record.get("sample_id", "")) not in completed
    ]
    output_file.parent.mkdir(parents=True, exist_ok=True)
    accepted: list[dict[str, Any]] = []

    with (
        output_file.open("a", encoding="utf-8") as handle,
        ThreadPoolExecutor(max_workers=max_workers) as executor,
    ):
        for batch_start in range(0, len(pending), max_workers):
            if (
                target_accepted is not None
                and existing_accepted + len(accepted) >= target_accepted
            ):
                break

            batch = pending[batch_start:batch_start + max_workers]
            logger.info(
                "Generating candidates %d-%d of %d (%d worker(s)).",
                batch_start + 1,
                batch_start + len(batch),
                len(pending),
                max_workers,
            )
            results = executor.map(
                lambda record: _generate_and_evaluate(
                    teacher_client,
                    judge_client,
                    record,
                    exemplars,
                    skip_llm_judge,
                ),
                batch,
            )
            for dataset_record in results:
                if (
                    target_accepted is not None
                    and existing_accepted + len(accepted) >= target_accepted
                ):
                    break
                handle.write(json.dumps(dataset_record, ensure_ascii=False) + "\n")
                handle.flush()
                if dataset_record["validation"]["passed"]:
                    accepted.append(dataset_record)

    logger.info(
        "Generation complete: %d/%d newly generated samples passed validation.",
        len(accepted),
        len(pending),
    )
    total_accepted = existing_accepted + len(accepted)
    if target_accepted is not None and total_accepted < target_accepted:
        raise RuntimeError(
            f"Accepted-record target not met: {total_accepted}/{target_accepted}. "
            "Increase the candidate budget with more unique source tasks, then resume."
        )
    return accepted
