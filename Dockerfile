FROM python:3.11 AS base

ENV TINI_VERSION v0.19.0
ADD https://github.com/krallin/tini/releases/download/${TINI_VERSION}/tini /mnt/tini
RUN chmod +x /mnt/tini
COPY bge-small-zh-v1.5 /mnt/bge-small-zh-v1.5

FROM python:3.11

ENV TINI_VERSION v0.19.0
ENTRYPOINT ["/usr/bin/tini", "--"]

WORKDIR /workspace
COPY . .
RUN --mount=type=bind,from=base,source=/mnt,target=/mnt,ro \
    cp /mnt/tini /usr/bin/ \
    && mkdir -p /models \
    && cp -r /mnt/bge-small-zh-v1.5 /models/ \
    && pip3 install streamlit langchain langchain-core langchain-community PyPDF2 langchain-experimental faiss-cpu python-docx sentence-transformers openai \
    && rm -rf /root/.cache /var/lib/apt/lists/*
CMD ["streamlit", "run", "/workspace/demo.py"]

