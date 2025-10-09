import torch
print(torch.version.cuda)  # PyTorch构建时使用的CUDA版本
print(torch.cuda.is_available())  # 应为True