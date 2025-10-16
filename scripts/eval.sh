
python sampling.py \
    --ckpt_path /gemini/space/jiangpf/VIAR/var_d30.pth \
    --vae_ckpt /gemini/space/jiangpf/VIAR/vae_ch160v4096z32.pth \
    --cfg 1.5 \
    --depth 30 \
    --sample_dir ./samples

python evaluator.py \
    /gemini/space/jiangpf/data/VIRTUAL_imagenet256_labeled.npz \
    admnet_guided_upsampled_imagenet256.npz