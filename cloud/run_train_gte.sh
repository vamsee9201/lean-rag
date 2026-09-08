#!/usr/bin/env bash
set -euo pipefail

python -m pip install -q sentence-transformers==6.0.1 datasets==4.5.0
python /mnt/code/train_gte_govinfo.py
