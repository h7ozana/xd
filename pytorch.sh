#!/bin/bash
sudo apt install -y python3 python3-pip

pip3 install torch torchvision torchaudio

python3 --version

#동작확인 / 토치 쿠다버전
python3 -c "import torch; print(torch.__version__)"
python3 -c "import torch; print(torch.version.cuda)"

#GPU확인
python3 -c "import torch; print(f'Total GPUs: {torch.cuda.device_count()}'); print([torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())])"
python3 -c "import torch; print([torch.cuda.is_available() for i in range(torch.cuda.device_count())])"
