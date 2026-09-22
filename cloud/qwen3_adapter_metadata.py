"""Prepare SWIFT LoRA checkpoints for portable Hugging Face publication."""

from __future__ import annotations

import json
from pathlib import Path


def sanitize_adapter(
    adapter: Path,
    model_id: str,
    *,
    title: str = "GovInfo Qwen3-Embedding-8B adapter",
    description: str = (
        "LoRA checkpoint for the Lean RAG GovInfo retrieval experiment. "
        "The base model revision and validation metrics are recorded in "
        "`adapter_aware_run2/validation_manifest.json`."
    ),
    tags: tuple[str, ...] = ("lora", "text-embeddings"),
) -> None:
    """Replace the machine-local base path without changing adapter weights."""
    config_path = adapter / "adapter_config.json"
    config = json.loads(config_path.read_text())
    config["base_model_name_or_path"] = model_id
    config_path.write_text(json.dumps(config, indent=2) + "\n")
    (adapter / "README.md").write_text(
        "---\n"
        f"base_model: {model_id}\n"
        "library_name: peft\n"
        "tags:\n" + "".join(f"  - {tag}\n" for tag in tags) +
        "---\n\n"
        f"# {title}\n\n"
        f"{description}\n"
    )
