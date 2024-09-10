#!/bin/bash
sudo apt install -y python3 python3-pip
sudo pip3 install torch torchvision torchaudio

cd pytorch-classification

machine_name="xd"
#for i in 0 2 4 /// for {0..1} = {0~1}
for i in 0
do
  python cifar_new.py -a vgg19_bn --gpu-id $i --epochs 100 --schedule 30 40 --gamma 0.1 --workers 2 --checkpoint checkpoints/cifar10/vgg19_bn_${machine_name}_gpu$i
done