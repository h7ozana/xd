#!/bin/bash

nvidia-docker run -it --rm -e PYTHONWARNINGS="ignore" nvcr.io/nvidia/pytorch:24.10-py3 python -c "
import torch
import time
print('PyTorch Version:', torch.__version__)
print('CUDA Available:', torch.cuda.is_available())

if torch.cuda.is_available():
    device_name = torch.cuda.get_device_name(0)
    print('CUDA Device Name:', device_name)
    print('CUDA Device Count:', torch.cuda.device_count())

# CPU test
cpu_tensor = torch.rand(10000, 10000)
start_time = time.time()
cpu_sum = torch.sum(cpu_tensor)
end_time = time.time()
print('CPU Sum:', cpu_sum.item())
print('CPU Time taken: {:.6f} seconds'.format(end_time - start_time))

# GPU test
if torch.cuda.is_available():
    gpu_tensor = torch.rand(10000, 10000, device='cuda')
    start_time = time.time()
    gpu_sum = torch.sum(gpu_tensor)
    end_time = time.time()
    print('GPU Sum:', gpu_sum.item())
    print('GPU Time taken: {:.6f} seconds'.format(end_time - start_time))
else:
    print('GPU is not available.')
"
