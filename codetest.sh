#!/bin/bash

# 필요한 패키지 설치
sudo apt install -y python3 python3-pip python3.10-venv ghostscript

# 가상 환경 생성 및 활성화
python3 -m venv test
source test/bin/activate

# PyTorch 설치 (필요한 버전으로 변경)
pip install torch torchvision torchaudio

# Python 코드 실행
python -c '
import torch
import time

# CUDA 정보 출력
print("CUDA Available: ", torch.cuda.is_available())

if torch.cuda.is_available():
    print("CUDA Device Name: ", torch.cuda.get_device_name(0))
    print("CUDA Device Count: ", torch.cuda.device_count())

# CPU 연산
cpu_tensor = torch.rand(10000, 10000)
start_time = time.time()
cpu_sum = cpu_tensor.sum()
end_time = time.time()
print("CPU Sum:", cpu_sum)
print("CPU Time taken: {:.6f} seconds".format(end_time - start_time))

# GPU 연산
if torch.cuda.is_available():
    gpu_tensor = torch.rand(10000, 10000).to("cuda")
    start_time = time.time()
    gpu_sum = gpu_tensor.sum()
    end_time = time.time()
    print("GPU Sum:", gpu_sum)
    print("GPU Time taken: {:.6f} seconds".format(end_time - start_time))
else:
    print("GPU is not available.")
'