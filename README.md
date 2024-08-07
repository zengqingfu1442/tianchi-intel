# Prerequisite
1. docker
2. docker-compose

# Start llm with ipex-llm.vllm
```bash
bash llm.sh
```

# Download bge-small-zh-v1.5 model
```bash
bash download-bge-small-zh-v1.5.sh
```

# Build docker image
```bash
DOCKEr_BUILDKIT=1 docker build -t demo:1.0 -f Dockerfile .
```

# Start frontend and RAG system
```bash
docker-compose up -d
```

# Open the chatbox in your browser

http://your-host-ip:8501
