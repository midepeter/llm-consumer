"""Inference-backend configuration shared by generation and evaluation."""

import os
from typing import Any

from huggingface_hub import HfApi, InferenceClient
from huggingface_hub.utils import HfHubHTTPError
from openai import OpenAI


INFERENCE_BACKEND = os.getenv("INFERENCE_BACKEND", "huggingface").strip().lower()
HF_TOKEN = os.getenv("HF_TOKEN", "")
VLLM_BASE_URL = os.getenv("VLLM_BASE_URL", "").rstrip("/")
VLLM_API_KEY = os.getenv("VLLM_API_KEY", "EMPTY")


def build_inference_client() -> Any:
    """Create the configured Hugging Face or self-hosted vLLM client."""
    if INFERENCE_BACKEND == "huggingface":
        if not HF_TOKEN:
            raise EnvironmentError("HF_TOKEN environment variable is not set.")
        return InferenceClient(token=HF_TOKEN)

    if INFERENCE_BACKEND == "vllm":
        if not VLLM_BASE_URL:
            raise EnvironmentError(
                "VLLM_BASE_URL is not set. Use the OpenAI-compatible endpoint, "
                "for example: http://<gpu-host>:8000/v1."
            )
        return OpenAI(base_url=VLLM_BASE_URL, api_key=VLLM_API_KEY)

    raise ValueError(
        "INFERENCE_BACKEND must be either 'huggingface' or 'vllm'."
    )


def validate_model_access(client: Any, model: str) -> None:
    """Fail early when the selected model is unavailable from the backend."""
    if not model.strip():
        raise EnvironmentError("A non-empty model ID is required.")

    if INFERENCE_BACKEND == "huggingface":
        api = HfApi(token=HF_TOKEN)
        try:
            api.model_info(model)
        except HfHubHTTPError as exc:
            if exc.response is not None and exc.response.status_code == 404:
                raise ValueError(
                    f"Model '{model}' was not found on Hugging Face."
                ) from exc
            raise
        return

    available_models = {item.id for item in client.models.list().data}
    if model not in available_models:
        raise ValueError(
            f"Model '{model}' is not served by {VLLM_BASE_URL}. Available models: "
            f"{', '.join(sorted(available_models)) or 'none'}."
        )


def thinking_extra_body(thinking_budget: int) -> dict[str, Any]:
    """Return backend-specific request options for Qwen native thinking."""
    if INFERENCE_BACKEND == "vllm":
        return {"chat_template_kwargs": {"enable_thinking": True}}

    body: dict[str, Any] = {"enable_thinking": True}
    if thinking_budget > 0:
        body["thinking_budget"] = thinking_budget
    return body
