import os
import torch
import torchvision
import random
import numpy as np
import PIL.Image as PImage
import argparse

setattr(torch.nn.Linear, 'reset_parameters', lambda self: None)
setattr(torch.nn.LayerNorm, 'reset_parameters', lambda self: None)

from models import VQVAE, build_vae_var


def parse_args():
    parser = argparse.ArgumentParser(description="VAR/VIAR image generation script")
    
    # model config
    parser.add_argument("--depth", type=int, default=30, choices=[16, 20, 24, 30])
    parser.add_argument("--use_implicit", action="store_true")
    parser.add_argument("--pre_depth", type=int, default=5)
    parser.add_argument("--post_depth", type=int, default=5)
    
    # ckpt path
    parser.add_argument("--var_ckpt", type=str, default=None)
    parser.add_argument("--vae_ckpt", type=str, default='ckpts/vae_ch160v4096z32.pth')
    
    # generation config
    parser.add_argument("--class_label", type=int, default=970)
    parser.add_argument("--seeds", type=int, nargs='+', default=[s for s in range(10, 20)])
    parser.add_argument("--cfg", type=float, default=1.5)
    parser.add_argument("--top_k", type=int, default=900)
    parser.add_argument("--top_p", type=float, default=0.96)
    parser.add_argument("--more_smooth", action="store_true")
    
    # output config
    parser.add_argument("--output_dir", type=str, default="gen_images")
    parser.add_argument("--save_grid", action="store_true")
    parser.add_argument("--nrow", type=int, default=4)
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    tf32 = True
    torch.backends.cudnn.allow_tf32 = bool(tf32)
    torch.backends.cuda.matmul.allow_tf32 = bool(tf32)
    torch.set_float32_matmul_precision('high' if tf32 else 'highest')
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"[INFO] Using device: {device}")
    print(f"[INFO] Mode: {'VIAR (implicit)' if args.use_implicit else 'VAR (explicit)'}")
    print(f"[INFO] Model depth: {args.depth}")
    print(f"[INFO] Class label: {args.class_label}")
    print(f"[INFO] Seeds: {args.seeds}")
    print(f"[INFO] Number of images to generate: {len(args.seeds)}")
    
    # auto select ckpts
    if args.var_ckpt is None:
        if args.use_implicit:
            args.var_ckpt = 'ckpts/viar.pth'
        else:
            args.var_ckpt = f'ckpts/var_d{args.depth}.pth'
    print(f"[INFO] VAR checkpoint: {args.var_ckpt}")
    print(f"[INFO] VAE checkpoint: {args.vae_ckpt}")
    
    patch_nums = (1, 2, 3, 4, 5, 6, 8, 10, 13, 16)
    print("[INFO] Building models...")
    
    vae, var = build_vae_var(
        V=4096, Cvae=32, ch=160, share_quant_resi=4,
        device=device, patch_nums=patch_nums,
        num_classes=1000, depth=args.depth, shared_aln=False,
        use_implicit=args.use_implicit,
        pre_depth=args.pre_depth,
        post_depth=args.post_depth
    )
    
    vae.load_state_dict(torch.load(args.vae_ckpt, map_location='cpu'), strict=True)
    var.load_state_dict(torch.load(args.var_ckpt, map_location='cpu'), strict=True)
    
    vae.eval()
    var.eval()
    for p in vae.parameters():
        p.requires_grad_(False)
    for p in var.parameters():
        p.requires_grad_(False)
    
    print(f"[INFO] Model loaded successfully.")
    
    output_subdir = os.path.join(args.output_dir, os.path.basename(args.var_ckpt).replace('.pth', ''))
    os.makedirs(output_subdir, exist_ok=True)
    
    all_images = []
    
    print(f"\n[INFO] Generating {len(args.seeds)} images for class {args.class_label}...")
    
    for i, seed in enumerate(args.seeds):
        print(f"  Generating image {i+1}/{len(args.seeds)} with seed {seed}...")
        
        torch.manual_seed(seed)
        random.seed(seed)
        np.random.seed(seed)
        
        label_B = torch.tensor([args.class_label], device=device)
        
        with torch.inference_mode():
            with torch.autocast('cuda', enabled=True, dtype=torch.float16, cache_enabled=True):
                recon_B3HW = var.autoregressive_infer_cfg(
                    B=1,
                    label_B=label_B,
                    cfg=args.cfg,
                    top_k=args.top_k,
                    top_p=args.top_p,
                    g_seed=seed,
                    more_smooth=args.more_smooth,
                )
        
        all_images.append(recon_B3HW)
        
        img_np = recon_B3HW[0].permute(1, 2, 0).mul(255).cpu().numpy().astype(np.uint8)
        img = PImage.fromarray(img_np)
        img_path = os.path.join(output_subdir, f"class{args.class_label}_seed{seed}.png")
        img.save(img_path)
    
    print(f"[INFO] Saved {len(args.seeds)} individual images to {output_subdir}")
    
    if args.save_grid and len(all_images) > 1:
        all_images_tensor = torch.cat(all_images, dim=0)
        grid = torchvision.utils.make_grid(all_images_tensor, nrow=args.nrow, padding=2, pad_value=1.0)
        grid_np = grid.permute(1, 2, 0).mul(255).cpu().numpy().astype(np.uint8)
        grid_img = PImage.fromarray(grid_np)
        
        seeds_str = '_'.join(map(str, args.seeds[:4])) + (f"_etc{len(args.seeds)}" if len(args.seeds) > 4 else "")
        grid_path = os.path.join(output_subdir, f"class{args.class_label}_seeds{seeds_str}_grid.png")
        grid_img.save(grid_path)
        print(f"[INFO] Saved grid image to {grid_path}")
    
    print("\n[INFO] Done!")


if __name__ == "__main__":
    main()
