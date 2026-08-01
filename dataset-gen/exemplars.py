"""Load and format few-shot exemplar triplets for distillation."""

import json
from pathlib import Path
from typing import Any


REQUIRED_FIELDS = ("input", "rationale", "answer")


def load_exemplars(path: Path, limit: int | None = None) -> list[dict[str, str]]:
    """
    Load exemplar triplets from JSONL.

    Each line must contain either input/rationale/answer or the legacy
    prompt/thinking/answer field names.
    """
    if not path.exists():
        raise FileNotFoundError(
            f"Exemplar file not found: {path}. Create it with JSONL records "
            "containing input, rationale, and answer fields."
        )

    exemplars: list[dict[str, str]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Invalid JSON on line {line_number} of {path}."
                ) from exc
            if not isinstance(item, dict):
                raise ValueError(f"Line {line_number} of {path} must be a JSON object.")

            exemplar = {
                "input": str(item.get("input", item.get("prompt", ""))).strip(),
                "rationale": str(item.get("rationale", item.get("thinking", ""))).strip(),
                "answer": str(item.get("answer", "")).strip(),
            }
            missing = [field for field in REQUIRED_FIELDS if not exemplar[field]]
            if missing:
                raise ValueError(
                    f"Line {line_number} of {path} is missing non-empty fields: "
                    f"{', '.join(missing)}."
                )
            exemplars.append(exemplar)
            if limit is not None and len(exemplars) >= limit:
                break

    if not exemplars:
        raise ValueError(f"No usable exemplar triplets were found in {path}.")
    return exemplars


def format_exemplars(exemplars: list[dict[str, str]]) -> str:
    """Format exemplars as explicit input/rationale/final-answer demonstrations."""
    return "\n\n".join(
        (
            f"EXAMPLE {index}\n"
            f"INPUT:\n{exemplar['input']}\n\n"
            f"RATIONALE:\n{exemplar['rationale']}\n\n"
            f"FINAL ANSWER:\n{exemplar['answer']}"
        )
        for index, exemplar in enumerate(exemplars, 1)
    )
