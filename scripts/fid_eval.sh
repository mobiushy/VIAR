
# python -m pytorch_fid \
#     data/VIRTUAL_imagenet256_labeled.npz \
#     samples/d30-var_d30.pth-cfg-1.5-seed-1 \
#     --device cuda \
#     --batch-size 512

python -m pytorch_fid \
    data/VIRTUAL_imagenet256_labeled.npz \
    samples/implicit-viar.pth-cfg-1.5-seed-1 \
    --device cuda \
    --batch-size 512
