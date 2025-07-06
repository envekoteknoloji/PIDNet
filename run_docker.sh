#!/bin/bash

xhost +local:

sudo docker run \
    --name pidnet_container \
    --privileged \
    --net=host \
    --ipc=host \
    --shm-size=4gb \
    --gpus all \
    -it --rm \
    -v ./:/home/user/workspace/PIDNet \
    -v /tmp/.X11-unix:/tmp/.X11-unix \
    pidnet_image
