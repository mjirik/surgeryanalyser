#!/bin/bash

DATADIR="/home/dnn-user/data_detectron2"

# Launch (require GPUs):
docker run --gpus all -it --rm \
  --shm-size=8gb --env="DISPLAY" --volume="/tmp/.X11-unix:/tmp/.X11-unix:rw" \
  -v $DATADIR:/home/appuser/data \
  --name=detectron2 detectron2:v0 \
  $1
