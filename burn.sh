#!/bin/bash
sudo git clone https://github.com/wilicc/gpu-burn
cd gpu-burn
sudo docker build -t gpu_burn .
cd ..
sudo rm -rf gpu-burn
# docker run --rm --gpus all gpu_burn