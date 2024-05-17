#!/bin/bash

version=$1
file="${2:=Dockerfile}"
sudo nvidia-docker image prune
sudo nvidia-docker build -t semexp3d_trans:$version -f $file .
sudo nvidia-docker tag semexp3d_trans:$version michaelkim0606/semexp3d_trans:$version
sudo nvidia-docker push michaelkim0606/semexp3d_trans:$version