#!/bin/bash

xhost +local:docker
sudo nvidia-docker run -v /home/michael/Michael/robotics/ObjectNav/SemExp-3DTransformer/:/code -v /media/michael/sdd1/gibson:/code/data --net=host --privileged -v=/dev:/dev -e DISPLAY=$DISPLAY -v $HOME/.Xauthority:/root/.Xauthority:ro -e NVIDIA_DRIVER_CAPABILITIES=all -e NVIDIA_VISIBLE_DEVICES=all -it -d --name semexp3d_trans semexp3d_trans /bin/bash