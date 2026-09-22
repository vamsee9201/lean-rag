#!/usr/bin/env bash
set -euo pipefail

python -m pip install -q sentence-transformers==6.0.1
python /mnt/code/train_qwen3_embedding_adapter_aware.py
