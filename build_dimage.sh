#!/bin/bash

version=$1
sudo docker image prune
sudo docker build -t semexp3d_trans:$version .
sudo docker tag semexp3d_trans:$version michaelkim0606/semexp3d_trans:$version
sudo docker push michaelkim0606/semexp3d_trans:$version