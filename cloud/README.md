# Hugging Face Qwen3.5 LoRA smoke test

This compatibility gate uses the official ModelScope `ms-swift` 4.4.1 image,
one Hugging Face L40S GPU, a ten-step LoRA run, adapter reload, and three held-out
generations. It mounts the validated pilot dataset read-only and sends the Hub
token as an encrypted Job secret.

Maximum billable runtime: one hour. At the current L40S price of $1.80/hour, the
hard maximum compute charge is $1.80.

```sh
hf jobs run \
  --name qwen35-rag-lora-smoke \
  --flavor l40sx1 \
  --timeout 1h \
  --secrets HF_TOKEN \
  -v ./cloud:/mnt/code:ro \
  -v hf://datasets/vamsee9201/lean-rag-training-data:/mnt/data:ro \
  modelscope-registry.us-west-1.cr.aliyuncs.com/modelscope-repo/modelscope:ubuntu22.04-cuda13.0.3-py312-torch2.11.0-vllm0.23.0-modelscope1.38.1-swift4.4.1 \
  python /mnt/code/train_qwen35_smoke.py
```

Success requires all of the following:

- CUDA, Transformers 5.9+, FlashAttention, flash-linear-attention, and
  causal-conv1d import successfully.
- Ten backward/optimizer steps have finite loss and gradient norm.
- A LoRA adapter checkpoint is saved.
- The adapter reloads and generates three validation answers.
- The checkpoint, logs, manifest, and validation generations are uploaded to
  `vamsee9201/qwen35-9b-rag-lora-pilot`.
