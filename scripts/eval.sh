
export ORION_GMEM_CONTROL=v1

# python sampling.py \
#     --ckpt_path /gemini/space/jiangpf/VIAR/var_d30.pth \
#     --vae_ckpt /gemini/space/jiangpf/VIAR/vae_ch160v4096z32.pth \
#     --cfg 2 \
#     --depth 30 \
#     --sample_dir ./samples

# python evaluator.py \
#     /gemini/space/jiangpf/data/VIRTUAL_imagenet256_labeled.npz \
#     /gemini/space/jiangpf/VIAR/samples/d30-var_d30.pth-cfg-2.0-seed-1.npz

python sampling.py \
    --ckpt_path /gemini/space/jiangpf/VIAR/local_output/ar-ckpt-last-slim.pth \
    --vae_ckpt /gemini/space/jiangpf/VIAR/vae_ch160v4096z32.pth \
    --cfg 2 \
    --depth 20 \
    --sample_dir ./samples \
    --use_implicit True

python evaluator.py \
    /gemini/space/jiangpf/data/VIRTUAL_imagenet256_labeled.npz \
    /gemini/space/jiangpf/VIAR/samples/implicit-ar-ckpt-last-slim.pth-cfg-2.0-seed-1.npz