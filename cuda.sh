#!/bin/bash
# FILE1="/etc/apt/apt.conf.d/20auto-upgrades"
# FILE2="/etc/apt/apt.conf.d/10periodic"
# sed -i 's/"1"/"0"/g' $FILE1
# sed -i 's/"1"/"0"/g' $FILE2
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb
sudo apt-get update
sudo apt-get -y install cuda-toolkit-12-2
echo "export PATH=/usr/local/cuda-12.2/bin${PATH:+:${PATH}}" >> ~/.bashrc
echo "export LD_LIBRARY_PATH=/usr/local/cuda-12.2/lib64\${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}" >> ~/.bashrc
sudo reboot
