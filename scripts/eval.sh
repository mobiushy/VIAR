
python sampling.py \
    --ckpt_path /gemini/space/jiangpf/VIAR/ar-ckpt-best.pth \
    --vae_ckpt /gemini/space/jiangpf/VIAR/vae_ch160v4096z32.pth \
    --cfg 2 \
    --depth 30 \
    --use_implicit True \
    --sample_dir ./samples

# python evaluator.py \
#     /gemini/space/jiangpf/data/VIRTUAL_imagenet256_labeled.npz \
#     /gemini/space/jiangpf/VIAR/samples/d30-var_d30.pth-cfg-2.0-seed-1.npz