"""
Chain-of-Thought Knowledge Distillation Pipeline
Teacher model: Qwen3 Coder Next via a configurable inference backend

Flow:
  load_prompts()  →  clean_prompts()  →  generate_rationale()  →  save_dataset()
"""

import os
import re
import csv
import json
import time
import logging
from pathlib import Path
from typing import Any

from huggingface_hub.utils import HfHubHTTPError
from openai import OpenAIError

from inference import build_inference_client, thinking_extra_body, validate_model_access

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

TEACHER_MODEL   = os.getenv("TEACHER_MODEL", "")
MAX_NEW_TOKENS  = 2048
TEMPERATURE     = 0.6
OUTPUT_FILE     = Path("dataset.jsonl")

COT_SYSTEM_PROMPT = (
    "You are an expert reasoning assistant. "
    "For every question or task you receive, first think step-by-step inside "
    "<think>...</think> tags, then provide a clear, concise final answer after them."
)

# ---------------------------------------------------------------------------
# 1. Prompt Loading — supports .txt, .json/.jsonl, .csv, or a Python list
# ---------------------------------------------------------------------------

def load_prompts(source: str | list[str] | Path) -> list[str]:
    """
    Load prompts from:
      - a Python list[str]
      - a .txt file  (one prompt per line)
      - a .json file (list of strings or list of dicts with a 'prompt' key)
      - a .jsonl file (one JSON object per line with a 'prompt' key)
      - a .csv file  (first column or column named 'prompt')
    """
    if isinstance(source, list):
        return [str(p) for p in source]

    path = Path(source)
    if not path.exists():
        raise FileNotFoundError(f"Prompt file not found: {path}")

    suffix = path.suffix.lower()

    if suffix == ".txt":
        lines = path.read_text(encoding="utf-8").splitlines()
        return [l for l in lines if l.strip()]

    if suffix == ".json":
        data = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(data, list):
            return [item["prompt"] if isinstance(item, dict) else str(item) for item in data]
        raise ValueError("JSON file must contain a top-level list.")

    if suffix == ".jsonl":
        prompts = []
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line:
                obj = json.loads(line)
                prompts.append(obj.get("prompt", obj.get("text", str(obj))))
        return prompts

    if suffix == ".csv":
        prompts = []
        with path.open(newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            if reader.fieldnames and "prompt" in reader.fieldnames:
                for row in reader:
                    if row["prompt"].strip():
                        prompts.append(row["prompt"].strip())
            else:
                # fall back to first column
                f.seek(0)
                plain = csv.reader(f)
                for row in plain:
                    if row and row[0].strip():
                        prompts.append(row[0].strip())
        return prompts

    raise ValueError(f"Unsupported file format: {suffix}. Use .txt, .json, .jsonl, or .csv")


# ---------------------------------------------------------------------------
# 2. Prompt Cleaning
# ---------------------------------------------------------------------------

def clean_prompt(text: str) -> str:
    """Normalize whitespace, remove control characters, and strip the prompt."""
    # remove non-printable / control chars (except newline/tab)
    text = re.sub(r"[^\x09\x0A\x20-\x7E\u00A0-\uFFFF]", "", text)
    # collapse multiple blank lines into one
    text = re.sub(r"\n{3,}", "\n\n", text)
    # collapse multiple spaces (but preserve newlines)
    text = re.sub(r"[ \t]{2,}", " ", text)
    return text.strip()


def clean_prompts(prompts: list[str], deduplicate: bool = True) -> list[str]:
    """
    Clean and optionally deduplicate a list of prompts.
    Prompts that are empty after cleaning are dropped.
    """
    cleaned: list[str] = []
    seen: set[str] = set()

    for raw in prompts:
        text = clean_prompt(raw)
        if not text:
            logger.warning("Dropping empty prompt after cleaning.")
            continue
        if deduplicate:
            key = text.lower()
            if key in seen:
                logger.info("Dropping duplicate prompt: %s…", text[:60])
                continue
            seen.add(key)
        cleaned.append(text)

    logger.info("Prompts after cleaning: %d (dropped %d)", len(cleaned), len(prompts) - len(cleaned))
    return cleaned


# ---------------------------------------------------------------------------
# 3. Teacher Model — Chain-of-Thought generation
# ---------------------------------------------------------------------------

def build_client() -> Any:
    if not TEACHER_MODEL.strip():
        raise EnvironmentError(
            "TEACHER_MODEL is not set. Provide the model ID served by the selected backend."
        )
    client = build_inference_client()
    validate_model_access(client, TEACHER_MODEL)
    return client


def generate_rationale(
    client: Any,
    prompt: str,
    retries: int = 3,
    backoff: float = 5.0,
) -> dict[str, Any]:
    """
    Send a prompt to the teacher model and return a dict with:
      - prompt:    the original prompt
      - thinking:  extracted <think>…</think> block (chain-of-thought)
      - answer:    the final answer after the thinking block
      - raw:       full model output
    """
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
                extra_body=thinking_extra_body(
                    int(os.getenv("THINKING_BUDGET", "1024"))
                ),
            )
            message = response.choices[0].message
            answer = (message.content or "").strip()
            thinking = (getattr(message, "reasoning_content", None) or "").strip()
            if thinking:
                raw = f"<think>\n{thinking}\n</think>\n{answer}"
            else:
                raw = answer
                thinking, answer = _parse_cot_output(raw)
            return {
                "prompt":   prompt,
                "thinking": thinking,
                "answer":   answer,
                "raw":      raw,
            }
        except (HfHubHTTPError, OpenAIError) as exc:
            response = getattr(exc, "response", None)
            if response is not None and response.status_code == 404:
                raise ValueError(
                    f"Teacher model '{TEACHER_MODEL}' was not found by the configured backend."
                ) from exc
            logger.warning("Attempt %d/%d failed: %s", attempt, retries, exc)
            if attempt < retries:
                time.sleep(backoff * attempt)
            else:
                logger.error("All retries exhausted for prompt: %s…", prompt[:60])
                return {
                    "prompt":   prompt,
                    "thinking": "",
                    "answer":   "",
                    "raw":      "",
                    "error":    str(exc),
                }
        except Exception as exc:
            logger.warning("Attempt %d/%d failed: %s", attempt, retries, exc)
            if attempt < retries:
                time.sleep(backoff * attempt)
            else:
                logger.error("All retries exhausted for prompt: %s…", prompt[:60])
                return {
                    "prompt":   prompt,
                    "thinking": "",
                    "answer":   "",
                    "raw":      "",
                    "error":    str(exc),
                }


def _parse_cot_output(text: str) -> tuple[str, str]:
    """Split <think>…</think> from the final answer."""
    match = re.search(r"<think>(.*?)</think>", text, re.DOTALL)
    if match:
        thinking = match.group(1).strip()
        answer   = text[match.end():].strip()
    else:
        thinking = ""
        answer   = text
    return thinking, answer


# ---------------------------------------------------------------------------
# 4. Dataset Generation — run all prompts through the teacher
# ---------------------------------------------------------------------------

def generate_dataset(
    prompts: list[str],
    output_file: Path = OUTPUT_FILE,
    resume: bool = True,
) -> list[dict[str, Any]]:
    """
    Generate chain-of-thought rationales for every prompt and write them to
    a JSONL file. Supports resuming from where it left off if the output file
    already exists.
    """
    client = build_client()

    # resume: skip prompts already in output file
    completed: set[str] = set()
    if resume and output_file.exists():
        with output_file.open(encoding="utf-8") as f:
            for line in f:
                try:
                    obj = json.loads(line)
                    if "prompt" in obj:
                        completed.add(obj["prompt"])
                except json.JSONDecodeError:
                    pass
        logger.info("Resuming — %d prompts already done.", len(completed))

    results: list[dict[str, Any]] = []
    pending = [p for p in prompts if p not in completed]
    total   = len(pending)
    logger.info("Generating rationales for %d prompt(s)…", total)

    with output_file.open("a", encoding="utf-8") as out:
        for idx, prompt in enumerate(pending, 1):
            logger.info("[%d/%d] Processing: %s…", idx, total, prompt[:60])
            record = generate_rationale(client, prompt)
            results.append(record)
            out.write(json.dumps(record, ensure_ascii=False) + "\n")
            out.flush()

    logger.info("Dataset written to %s (%d records).", output_file, len(results) + len(completed))
    return results


# ---------------------------------------------------------------------------
# 5. Entry point — edit source / prompts list to suit your dataset
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # ── Option A: load from a file ──────────────────────────────────────────
    # raw_prompts = load_prompts("prompts.txt")     # .txt / .json / .jsonl / .csv

    # ── Option B: inline list ───────────────────────────────────────────────
    raw_prompts = load_prompts([
        "Explain how gradient descent works step by step.",
        "What is the difference between a process and a thread?",
        "Write a Python function to reverse a linked list and explain why it works.",
    ])

    # Clean
    prompts = clean_prompts(raw_prompts, deduplicate=True)

    # Generate dataset via teacher model
    dataset = generate_dataset(prompts, output_file=OUTPUT_FILE, resume=True)

    print(f"\n✅  Done. {len(dataset)} new record(s) written to {OUTPUT_FILE}")
