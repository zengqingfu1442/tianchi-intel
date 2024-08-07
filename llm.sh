#/bin/bash
export DOCKER_IMAGE=intelanalytics/ipex-llm-serving-cpu:latest
export CONTAINER_NAME=ipex-llm-serving-cpu-container
docker run -itd \
        --net=host \
        --cpuset-cpus="0-47" \
        --cpuset-mems="0" \
        -v /path/to/models:/llm/models \
        -e no_proxy=localhost,127.0.0.1 \
        --memory="64G" \
        --name=$CONTAINER_NAME \
        --shm-size="16g" \
        $DOCKER_IMAGE

sleep 3

docker exec -it ipex-llm-serving-cpu-container /bin/bash /llm/start-vllm-service.sh