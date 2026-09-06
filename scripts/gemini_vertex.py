"""Shared authenticated Vertex AI access for Gemini benchmark scripts."""

from __future__ import annotations

import json
from pathlib import Path

from google import genai
from google.genai import types
from google.oauth2 import service_account


CLOUD_SCOPE = "https://www.googleapis.com/auth/cloud-platform"


def create_client(credentials_path: Path, location: str = "global") -> genai.Client:
    metadata = json.loads(credentials_path.read_text(encoding="utf-8"))
    credentials = service_account.Credentials.from_service_account_file(
        credentials_path,
        scopes=[CLOUD_SCOPE],
    )
    return genai.Client(
        vertexai=True,
        project=metadata["project_id"],
        location=location,
        credentials=credentials,
        http_options=types.HttpOptions(api_version="v1"),
    )


def generation_config(system: str, max_output_tokens: int) -> types.GenerateContentConfig:
    return types.GenerateContentConfig(
        system_instruction=system,
        temperature=0,
        max_output_tokens=max_output_tokens,
        thinking_config=types.ThinkingConfig(thinking_budget=0, include_thoughts=False),
        automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),
    )


def usage_dict(response) -> dict:
    usage = response.usage_metadata
    if usage is None:
        return {}
    return {
        "input_tokens": usage.prompt_token_count,
        "output_tokens": usage.candidates_token_count,
        "reasoning_tokens": usage.thoughts_token_count,
        "total_tokens": usage.total_token_count,
    }
