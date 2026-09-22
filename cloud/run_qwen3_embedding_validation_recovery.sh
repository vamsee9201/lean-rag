#!/usr/bin/env bash
set -euo pipefail

python -m pip install -q sentence-transformers==6.0.1
python /mnt/code/recover_qwen3_embedding_validation.py
