import torch


var_ckpt = "local_output/ar-ckpt-last.pth"

ckpt = torch.load(var_ckpt, map_location='cpu')
ckpt = ckpt['trainer']['var_wo_ddp']
# print(ckpt.keys())
torch.save(ckpt, "local_output/ar-ckpt-last-slim.pth")