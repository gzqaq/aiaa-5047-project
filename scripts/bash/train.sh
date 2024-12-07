#!/bin/bash

PROJECT_HOME=$(cd $(dirname $0)/../..; pwd)
DOCKER_PROJECT="/aiaa-5047"
IMAGE_NAME="aiaa-5047:base"
PY_SCRIPT="scripts/train-sae.py"

HELP_MSG="
Usage:  $0 [OPTIONS] [ARG...]

Run $PROJECT_HOME/$PY_SCRIPT using docker run.

Arguments:
  [ARG...]
        Arguments for $PY_SCRIPT, whose help message can be shown only with --help

Options:
  -h
         Show this message and quit
  -cuda, --cuda-device <UINT>
         Which cuda device to use for $PY_SCRIPT
"

CUDA_DEVICE=""
DOCKER_CMD="python $DOCKER_PROJECT/$PY_SCRIPT"
while [ $# -gt 0 ]; do
    case $1 in
        -h)
            echo "$HELP_MSG"
            exit
            ;;
        -cuda | --cuda-device)
            if [[ -z $2 || $2 = -* ]]; then
                CUDA_DEVICE=0
            else
                CUDA_DEVICE=$2
                shift
            fi
            shift
            ;;
        *)
            DOCKER_CMD="$DOCKER_CMD $1"
            shift
    esac
done

if [[ $CUDA_DEVICE =~ ^[0-9]+$ ]]; then
    echo "Use cuda device $CUDA_DEVICE"
else
    echo "Cuda device not specified. Abort!" >&2
    exit 1
fi

set -x

docker run -d -it --rm --gpus all --shm-size="10g" --cap-add=SYS_ADMIN \
       --mount type=bind,src=$PROJECT_HOME,dst=$DOCKER_PROJECT \
       --mount type=bind,src=$HOME/.cache,dst=/root/.cache \
       --env PYTHONPATH=/aiaa-5047 \
       --env CUDA_VISIBLE_DEVICES=$CUDA_DEVICE \
       --env XLA_PYTHON_CLIENT_PREALLOCATE=false \
       $IMAGE_NAME $DOCKER_CMD
