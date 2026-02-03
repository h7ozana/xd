#!/bin/bash
sudo wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb
sudo apt update
sudo apt -y install cuda-toolkit-12-4
echo "export PATH=/usr/local/cuda-12.4/bin${PATH:+:${PATH}}" >> ~/.bashrc
echo "export LD_LIBRARY_PATH=/usr/local/cuda-12.4/lib64\${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}" >> ~/.bashrc
# sudo reboot
sudo apt -y install cudnn
sudo mv ./burn.sh ..
history -c
#sudo reboot


# echo "export PATH=/usr/local/cuda-12.6/bin${PATH:+:${PATH}}" >> ~/.bashrc
# echo "export LD_LIBRARY_PATH=/usr/local/cuda-12.6/lib64\${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}" >> ~/.bashrc

#버전 중복될경우
# echo 'export PATH=/usr/local/cuda-12.6/bin:$PATH' >> ~/.bashrc
# echo 'export LD_LIBRARY_PATH=/usr/local/cuda-12.6/lib64:$LD_LIBRARY_PATH' >> ~/.bashrc



#24v에서 오류, 브로큰 / 아래는 22에서 따오는거
#wget http://security.ubuntu.com/ubuntu/pool/universe/n/ncurses/libtinfo5_6.3-2ubuntu0.1_amd64.deb
#sudo dpkg -i libtinfo5_6.3-2ubuntu0.1_amd64.deb
#sudo apt -y install cuda-toolkit-12-4 민철중3ㅇ32