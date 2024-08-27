# 启动前端

可以使用docker/docker-compose启动前端或者直接在ecs.g8i6xlarg上启动前端
## 方法一：使用docker/docker-compose部署前端（推荐）
注意这里使用的是Ubuntu22.04的镜像，docker-compose启动之前记得将IP地址改成你自己ecs服务器的内网IP地址
```bash
apt-get udpate && apt-get install docker.io docker-buildx -y
wget https://github.com/docker/compose/releases/download/v2.26.0/docker-compose-linux-x86_64 -O /usr/local/bin/docker-compose
chmod +x /usr/local/bin/docker-compose
bash download-bge-small-zh-v1.5.sh
DOCKER_BUILDKIT=1 docker build -f Dockerfile  -t demo:1.0 .
docker-compose up -d
```
![docker_images](img/docker_images.png)
启动后可以看到前端容器
![前端容器](img/demo-fe-docker.png)

## 方法二： 直接在ecs.g8i6xlarg上启动前端
```bash
pip3 install torch==2.3.1+cpu --extra-index-url https://download.pytorch.org/whl/cpu
pip3 install streamlit langchain langchain-core langchain-community PyPDF2 langchain-experimental faiss-cpu python-docx sentence-transformers openai
bash download-bge-small-zh-v1.5.sh
export OPENAI_BASE_URL="http://YOUR-SERVER-INNER-IP:8000/v1"
export OPENAI_API_KEY="YWRtaW46dGVzdEAxMjMuY29tCg=="
export OPENAI_MODEL="Qwen/Qwen2-7B-Instruct"
export EMB_MODEL_PATH=`pwd`"/bge-small-zh-v1.5"
streamlit run demo.py
```


# 启动后端

## 安装ipex-llm
```bash
pip install ipex-llm==2.1.0b20240805
```

## 安装vllm for CPU
```bash
pip3 install psutil
pip3 install sentencepiece  # Required for LLaMA tokenizer.
pip3 install fastapi
pip3 install "uvicorn[standard]"
pip3 install "pydantic<2"  # Required for OpenAI server.

git clone https://github.com/vllm-project/vllm.git && \
cd ./vllm && \
git checkout v0.4.2 && \
pip install wheel packaging ninja setuptools>=49.4.0 numpy && \
pip install -v -r requirements-cpu.txt --extra-index-url https://download.pytorch.org/whl/cpu && \
VLLM_TARGET_DEVICE=cpu python3 setup.py install
pip install jsonschema Jinja2 -U
```

## 准备Qwen2-7B模型
安装modelscope
```bash
pip install modelscope
```

```ipython
import torch
from modelscope import snapshot_download, AutoModel, AutoTokenizer
import os
# 第一个参数表示下载模型的型号，第二个参数是下载后存放的缓存地址，第三个表示版本号，默认 master
model_dir = snapshot_download('Qwen/Qwen2-7B-Instruct', cache_dir='qwen2', revision='master')
```
可以在当前目录下的qwen2目录下看到下载的模型
![模型权重](img/qwen2-7b.png)

## 启动大模型推理服务
```bash
python3 -m ipex_llm.vllm.cpu.entrypoints.openai.api_server \
        --model qwen2/Qwen/Qwen2-7B-Instruct --port 8000  \
        --api-key 'YWRtaW46dGVzdEAxMjMuY29tCg==' \
        --served-model-name 'Qwen/Qwen2-7B-Instruct' \
        --load-format 'auto' --device cpu --dtype bfloat16 \
        --load-in-low-bit sym_int4 \
        --max-num-batched-tokens 32768
```
![大模型推理服务](img/vllm_server.png)]

### 测试大模型推理服务是否work
```bash
curl http://localhost:8000/v1/completions \
         -H "Content-Type: application/json" \
         -H "Authorization: Bearer                      YWRtaW46dGVzdEAxMjMuY29tCg== \
         -d '{
                 "model": "Qwen/Qwen2-7B-Instruct",
                 "prompt": "San Francisco is a",
                 "max_tokens": 128,
                 "temperature": 0
 }'
 ```

 ```bash
 curl http://localhost:8000/v1/chat/completions   -H "Content-Type: application/json"   -H "Authorization: Bearer YWRtaW46dGVzdEAxMjMuY29tCg=="   -d '{
    "model": "Qwen/Qwen2-7B-Instruct",
    "messages": [
      {
        "role": "system",
        "content": "You are a helpful assistant."
      },
      {
        "role": "user",
        "content": "以《构建人类命运共同体》为主题，写一篇不少于1000字的作文。"
      }
    ],
    "stream": true }'
```
![本地测试接口效果](img/curl-demo.gif)

# 最终结合RAG效果展示
<video src="img/demo.mp4" controls="controls" style="max-width: 730px;">
</video>