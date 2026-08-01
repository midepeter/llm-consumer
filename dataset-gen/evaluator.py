"""
Evaluator Module
================
Two-layer evaluation for synthetic CoT records:

  Layer 1 — Rule-based scoring  (fast, no model call)
    • Checks presence of <think> block
    • Checks minimum answer length
    • Penalises refusal / boilerplate phrases
    • Checks code blocks for coding prompts

  Layer 2 — LLM-as-judge via teacher model (high signal)
    • Sends prompt + response back to the teacher with a structured rubric
    • Returns a 1–5 score + short critique

Final score = weighted combination of both layers.
Records below SYNTHETIC_PASS_THRESHOLD are rejected for the synthetic split.
Open-source records are soft-tagged only (never hard-filtered here).
"""

import re
import json
import logging
import os
import time
from typing import Any

from huggingface_hub.utils import HfHubHTTPError
from openai import OpenAIError

from inference import build_inference_client, validate_model_access

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

JUDGE_MODEL           = os.getenv("JUDGE_MODEL", os.getenv("TEACHER_MODEL", ""))
SYNTHETIC_PASS_THRESHOLD = float(os.getenv("PASS_THRESHOLD", "3.0"))   # out of 5.0
RULE_WEIGHT           = 0.40   # weight of rule-based score in final score
LLM_WEIGHT            = 0.60   # weight of LLM-judge score in final score

REFUSAL_PATTERNS = re.compile(
    r"\b(I cannot|I can't|I'm sorry|As an AI|I am unable|I don't have the ability"
    r"|I'm not able|I apologize|I must decline)\b",
    re.IGNORECASE,
)

JUDGE_SYSTEM_PROMPT = """\
You are a strict but fair dataset quality evaluator for a code reasoning dataset.

Given a PROMPT, an optional REFERENCE ANSWER, and a RESPONSE (which may include
a <think> chain-of-thought block followed by a final answer), score the response
on the following rubric and reply with ONLY valid JSON — no extra text.

Rubric (each dimension 1-5):
  reasoning_depth   : Is the thinking thorough, correct, and step-by-step?
  answer_quality    : Is the final answer correct, complete, and directly addresses the prompt?
  code_quality      : (for coding tasks) Is the code syntactically valid and idiomatic? Rate 3 if N/A.
  no_hallucination  : Does the response avoid fabricating facts / APIs? (5 = no hallucinations)
  reference_alignment: When a reference answer is provided, is the response materially consistent with it? Rate 3 if N/A.

Reply format (JSON only):
{
  "reasoning_depth":  <int 1-5>,
  "answer_quality":   <int 1-5>,
  "code_quality":     <int 1-5>,
  "no_hallucination": <int 1-5>,
  "critique":         "<one sentence>"
}
"""

# ---------------------------------------------------------------------------
# Layer 1 — Rule-based scorer
# ---------------------------------------------------------------------------

def rule_based_score(record: dict[str, Any]) -> dict[str, Any]:
    """
    Returns a score dict:
      rule_score        : float 0–5
      rule_flags        : list[str]  (issues found)
      has_think_block   : bool
      has_code_block    : bool
    """
    thinking  = record.get("thinking", "")
    answer    = record.get("answer",   "")
    raw       = record.get("raw",      "")
    prompt    = record.get("prompt",   "")

    flags: list[str] = []
    score = 5.0

    # — thinking block presence
    has_think = bool(thinking.strip())
    if not has_think:
        flags.append("missing_think_block")
        score -= 1.5

    # — minimum thinking depth
    if has_think and len(thinking.split()) < 30:
        flags.append("shallow_thinking")
        score -= 0.75

    # — answer existence
    if not answer.strip():
        flags.append("empty_answer")
        score -= 2.0

    # — minimum answer length
    if answer.strip() and len(answer.split()) < 10:
        flags.append("very_short_answer")
        score -= 0.5

    # — refusal patterns
    if REFUSAL_PATTERNS.search(raw):
        flags.append("refusal_detected")
        score -= 2.0

    # — code block check for coding prompts
    coding_keywords = re.compile(
        r"\b(function|implement|write|code|algorithm|class|def |leetcode|solve)\b",
        re.IGNORECASE,
    )
    is_coding_prompt = bool(coding_keywords.search(prompt))
    has_code_block   = bool(re.search(r"```[\w]*\n.*?```", raw, re.DOTALL))
    if is_coding_prompt and not has_code_block:
        flags.append("coding_prompt_missing_code_block")
        score -= 0.75

    score = max(0.0, min(5.0, score))

    return {
        "rule_score":      round(score, 2),
        "rule_flags":      flags,
        "has_think_block": has_think,
        "has_code_block":  has_code_block,
    }


# ---------------------------------------------------------------------------
# Layer 2 — LLM-as-judge
# ---------------------------------------------------------------------------

def llm_judge_score(
    client: Any,
    record: dict[str, Any],
    retries: int = 2,
    backoff: float = 4.0,
) -> dict[str, Any]:
    """
    Calls the teacher model as judge. Returns a score dict:
      llm_score        : float 1–5 (average of rubric dims)
      llm_dimensions   : dict
      llm_critique     : str
    """
    prompt   = record.get("prompt", "")
    response = record.get("raw", "")
    reference_answer = record.get("reference_answer", "")
    reference_section = (
        f"\n\nREFERENCE ANSWER (validation only):\n{reference_answer}"
        if reference_answer
        else ""
    )

    user_message = (
        f"PROMPT:\n{prompt}\n\n"
        f"RESPONSE:\n{response}"
        f"{reference_section}\n\n"
        "Evaluate and reply with JSON only."
    )

    for attempt in range(1, retries + 1):
        try:
            resp = client.chat.completions.create(
                model=JUDGE_MODEL,
                messages=[
                    {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
                    {"role": "user",   "content": user_message},
                ],
                max_tokens=256,
                temperature=0.1,
            )
            raw_json = resp.choices[0].message.content.strip()
            # strip markdown code fences if model wraps in ```json
            raw_json = re.sub(r"^```[\w]*\n?|```$", "", raw_json.strip(), flags=re.MULTILINE)
            dims = json.loads(raw_json)
            numeric = {
                k: float(v)
                for k, v in dims.items()
                if k != "critique" and isinstance(v, (int, float))
            }
            avg = sum(numeric.values()) / len(numeric) if numeric else 3.0
            return {
                "llm_score":      round(avg, 2),
                "llm_dimensions": numeric,
                "llm_critique":   dims.get("critique", ""),
            }
        except (HfHubHTTPError, OpenAIError) as exc:
            response = getattr(exc, "response", None)
            if response is not None and response.status_code == 404:
                raise ValueError(
                    f"Judge model '{JUDGE_MODEL}' was not found by the configured backend."
                ) from exc
            logger.warning("LLM judge attempt %d/%d failed: %s", attempt, retries, exc)
            if attempt < retries:
                time.sleep(backoff)
        except Exception as exc:
            logger.warning("LLM judge attempt %d/%d failed: %s", attempt, retries, exc)
            if attempt < retries:
                time.sleep(backoff)

    # fallback — neutral score so rule-based score still decides
    return {"llm_score": 3.0, "llm_dimensions": {}, "llm_critique": "judge_failed"}


# ---------------------------------------------------------------------------
# Combined evaluation
# ---------------------------------------------------------------------------

def evaluate_synthetic_record(
    client: Any,
    record: dict[str, Any],
    skip_llm_judge: bool = False,
) -> dict[str, Any]:
    """
    Evaluate one synthetic record. Returns the record enriched with:
      eval_rule, eval_llm, eval_final, eval_passed
    """
    rule_result = rule_based_score(record)
    rule_score  = rule_result["rule_score"]

    if record.get("error") or not record.get("answer", "").strip():
        llm_result = {
            "llm_score": 0.0,
            "llm_dimensions": {},
            "llm_critique": "generation_failed",
        }
        llm_score = 0.0
    elif skip_llm_judge:
        llm_result = {"llm_score": 3.0, "llm_dimensions": {}, "llm_critique": "skipped"}
        llm_score  = 3.0
    else:
        llm_result = llm_judge_score(client, record)
        llm_score  = llm_result["llm_score"]

    final_score = round(RULE_WEIGHT * rule_score + LLM_WEIGHT * llm_score, 3)
    passed      = (
        not record.get("error")
        and bool(record.get("answer", "").strip())
        and final_score >= SYNTHETIC_PASS_THRESHOLD
    )

    return {
        **record,
        "eval_rule":    rule_result,
        "eval_llm":     llm_result,
        "eval_final":   final_score,
        "eval_passed":  passed,
    }


def evaluate_open_source_record(record: dict[str, Any]) -> dict[str, Any]:
    """
    Soft-tag an open-source record with a rule-based quality score.
    Never hard-filters — just enriches with eval metadata.
    """
    rule_result = rule_based_score(record)
    return {
        **record,
        "eval_rule":   rule_result,
        "eval_final":  rule_result["rule_score"],
        "eval_passed": True,   # always kept for open-source
        "eval_source": "open_source",
    }


def build_judge_client() -> Any:
    if not JUDGE_MODEL.strip():
        raise EnvironmentError(
            "JUDGE_MODEL is not set. Set JUDGE_MODEL or TEACHER_MODEL to a served model ID."
        )
    client = build_inference_client()
    validate_model_access(client, JUDGE_MODEL)
    return client
