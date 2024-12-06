#!/bin/bash

set -x  # print commands and their arguments as they are executed

PROJECT_PATH=$(cd $(dirname $0)/..; pwd)  # project root
IMAGE_NAME="aiaa-5047:base"

docker run -it --rm --gpus all --shm-size="10g" --cap-add=SYS_ADMIN \
       --mount type=bind,src=$PROJECT_PATH,dst=/aiaa-5047 \
       --mount type=bind,src=$HOME/.cache/,dst=/root/.cache \
       --env PYTHONPATH=/aiaa-5047 \
       $IMAGE_NAME /bin/bash
