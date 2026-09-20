#!/bin/bash

PROJECT=$HOME/moe_offload
SSD_ROOT=$HOME/moe_ssd

GPU=${1:-all}

if [ "$GPU" = "all" ]; then
    GPU_OPT="all"
else
    GPU_OPT="device=${GPU}"
fi

echo "Project : ${PROJECT}"
echo "SSD     : ${SSD_ROOT}"
echo "GPU     : ${GPU}"

docker run -it \
    --rm \
    --name moe-llama-dev \
    --gpus "${GPU_OPT}" \
    --ipc=host \
    --network=host \
    --ulimit memlock=-1 \
    --ulimit stack=67108864 \
    --cap-add SYS_PTRACE \
    -v ${PROJECT}:/workspace \
    -v ${SSD_ROOT}:/ssd \
    -w /workspace/llama.cpp \
    moe-llama-dev:cuda12.9 \
    bash