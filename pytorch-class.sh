#!/bin/bash
sudo apt install -y python3 python3-pip python3.10-venv ghostscript

python3 -m venv test1
source test1/bin/activate

pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip3 install matplotlib progress
cd pytorch-classification

machine_name="xd"
sudo mkdir -p checkpoints/cifar10
#for i in 0 2 4 /// for {0..1} = {0~1}
for i in 1
do
  python cifar_new.py -a vgg19_bn --gpu-id $i --epochs 7 --schedule 3 5 --gamma 0.1 --workers 2 --checkpoint checkpoints/cifar10/vgg19_bn_${machine_name}_gpu$i
done

# H100 기준 epochs 7 / schedule 3 5 >> 1분미만 vgg19_bn


# 타 모델 변경시 모델명만 기입하고 뎁스에 레이어를 입력하는 구조
# for i in 0
# do
#   python cifar_new.py -a preresnet --depth 110 --gpu-id $i --epochs 7 --schedule 3 5 --gamma 0.1 --workers 2 --checkpoint checkpoints/cifar10/preresnet_${machine_name}_gpu$i
# done



#오류시 고스트삭제


# cd ~/pytorch-classification/checkpoints/cifar10/vgg19_bn_xd_gpu0
# sudo gs -dNOPAUSE -dBATCH -sDEVICE=pdfwrite -sOutputFile=output.pdf log.eps
# git remote add origin http://github.com/h7ozana/data

#윈도우에서 파일 떠오기
# 예시 scp user@<서버_IP>:~/pytorch-classification/checkpoints/cifar10/vgg19_bn_xd_gpu0/output.pdf C:\Users\<사용자명>\Desktop\
# scp user@192.168.39.39:~/pytorch-classification/checkpoints/cifar10/vgg19_bn_xd_gpu0/output.pdf C:\Users\user\Desktop\
