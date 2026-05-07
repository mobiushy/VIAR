
# For VAR
# python sampling.py \
#     --ckpt_path ckpts/var_d30.pth \
#     --vae_ckpt ckpts/vae_ch160v4096z32.pth \
#     --cfg 1.5 \
#     --depth 30 \
#     --sample_dir samples

# python evaluator.py \
#     data/VIRTUAL_imagenet256_labeled.npz \
#     samples/d30-var_d30.pth-cfg-1.5-seed-1.npz


# For VIAR
python sampling.py \
    --ckpt_path ckpts/viar.pth \
    --vae_ckpt ckpts/vae_ch160v4096z32.pth \
    --cfg 1.5 \
    --depth 30 \
    --sample_dir samples \
    --use_implicit True \
    --iter_left 10 \
    --iter_right 10

python evaluator.py \
    data/VIRTUAL_imagenet256_labeled.npz \
    samples/implicit-viar.pth-cfg-1.5-seed-1.npz