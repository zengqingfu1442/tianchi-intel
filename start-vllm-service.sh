#!/bin/bash
model="/llm/models/Qwen2-7B-Instruct"
served_model_name="Qwen/Qwen2-7B-Instruct"


python -m ipex_llm.vllm.cpu.entrypoints.openai.api_server \
  --served-model-name $served_model_name \
  --port 8000 \
  --model $model \
  --trust-remote-code \
  --device cpu \
  --dtype bfloat16 \
  --enforce-eager \
  --load-in-low-bit bf16 \
  --max-model-len 4096 \
  --max-num-batched-tokens 10240 \
  --max-num-seqs 12 \
  --tensor-parallel-size 1
