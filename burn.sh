#!/bin/bash
sudo git clone https://github.com/wilicc/gpu-burn
cd gpu-burn
make

# sudo docker build -t gpu_burn .
# cd ..
# sudo rm -rf gpu-burn
# # docker run --rm --gpus all gpu_burn
# # sudo docker run -it --gpus '"device=0"' gpu_burn  /bin/bash