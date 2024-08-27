#!/bin/bash
served_model_name="Qwen/Qwen2-7B-Instruct"

python3 -m ipex_llm.vllm.cpu.entrypoints.openai.api_server \
        --model qwen2/Qwen/Qwen2-7B-Instruct --port 8000  \
        --served-model-name 'Qwen/Qwen2-7B-Instruct' \
        --load-format 'auto' --device cpu --dtype bfloat16 \
        --load-in-low-bit sym_int4 \
        --trust-remote-code \
        --max-num-batched-tokens 32768
